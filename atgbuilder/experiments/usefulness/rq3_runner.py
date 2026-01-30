# -*- coding: utf-8 -*-
"""
RQ3 Online Mining Runner (ADB + uiautomator dump, no extra deps)

(Performance PATCH)
  - Try uiautomator dump --compressed (auto-detect support; fallback if not supported)
  - Prefer lighter current-activity source (dumpsys window windows) before dumpsys activity activities
  - Parse UI XML only ONCE per sample: produce ui_hash + clickables from the same parsed root
  - Avoid debug-time re-parse (reuse candidates count from current sample)

(HARD PERF PATCH v2)
  - Guided: remove "tap -> second dump/get_current_activity" within the same second.
    Instead, record a PendingAction and settle it on the NEXT second’s sample.
    => per wall-clock second: at most ONE (get_current_activity + dump_ui_xml + parse).

(Goal-first FIX)
  - per (act, ui_hash) maintain per-category stats (tries/success/last_try),
    and use soft penalty + cooldown to allow re-entering About/Settings/etc.
  - allow goal-first nodes to be re-tapped with a shorter cooldown.

(NEW CRITICAL FIX v3 - runtime package adopt)
  - If config.package mismatches the actual app package shown by mCurrentFocus,
    auto-adopt the observed package (e.g., com.alfred.home) to avoid "events=0 + grace loop".
  - After adopt: use pkg_run for in_app/pidof/force-stop/launch/candidate filtering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any


# ---------------- ADB helpers ----------------

def sh(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out, err


def adb_base(serial: Optional[str]) -> List[str]:
    base = ["adb"]
    if serial:
        base += ["-s", serial]
    return base


def adb(serial: Optional[str], args: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    return sh(adb_base(serial) + args, timeout=timeout)


def adb_shell(serial: Optional[str], shell_cmd: str, timeout: int = 30) -> Tuple[int, str, str]:
    return adb(serial, ["shell", shell_cmd], timeout=timeout)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def log_write(path: str, msg: str, also_console: bool = True) -> None:
    ts = time.strftime("%H:%M:%S", time.localtime())
    line = f"[{ts}] {msg.rstrip()}"
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_console:
        print(line, flush=True)


def keyevent(serial: Optional[str], code: int) -> None:
    adb_shell(serial, f"input keyevent {code}", timeout=5)


def back(serial: Optional[str], n: int = 1) -> None:
    for _ in range(max(1, n)):
        keyevent(serial, 4)  # KEYCODE_BACK
        time.sleep(0.25)


# ---------------- Parsing predicted edges ----------------

EDGE_LINE_RE = re.compile(r"^\s*(.*?)\s*->\s*(.*?)\s*\|\s*p\s*=\s*([0-9.]+)\s*$")


def iter_wallclock_seconds(duration: int, sample_interval: int):
    """
    Wall-clock scheduler.
    Yields (t, start_ts, end_ts) where t is integer seconds since start.
    """
    start_ts = time.time()
    end_ts = start_ts + float(duration)
    next_ts = start_ts
    last_t = -1

    si = float(max(1, sample_interval))
    while True:
        now = time.time()
        if now >= end_ts:
            break

        if now < next_ts:
            time.sleep(min(0.2, next_ts - now))
            continue

        t = int(now - start_ts)
        if t >= duration:
            break

        if t == last_t:
            next_ts += si
            continue

        yield t, start_ts, end_ts
        last_t = t
        next_ts += si


def _norm_app_key(s: str) -> str:
    s = s.strip()
    if s.endswith(".apk"):
        s = s[:-4]
    return s


def parse_edges_file(path: str) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Returns:
      key -> list of (src, dst, p)
    key is normalized: strip .apk
    """
    mp: Dict[str, List[Tuple[str, str, float]]] = {}
    if not path or (not os.path.exists(path)):
        return mp

    cur_app = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("APK:"):
                cur_app = _norm_app_key(s.split("APK:", 1)[1].strip())
                mp.setdefault(cur_app, [])
                continue
            m = EDGE_LINE_RE.match(s)
            if m and cur_app:
                src = m.group(1).strip()
                dst = m.group(2).strip()
                pval = float(m.group(3))
                mp[cur_app].append((src, dst, pval))
    return mp


def lookup_edges(edges_map: Dict[str, List[Tuple[str, str, float]]], app_id: str, package: str) -> List[Tuple[str, str, float]]:
    """
    Fix common key mismatches:
      - app_id vs app_id.apk
      - APK: line might have basename only
      - sometimes people use package as key
    """
    keys_try = []
    app_id_n = _norm_app_key(app_id)
    keys_try.append(app_id_n)
    keys_try.append(app_id_n + ".apk")
    keys_try.append(package)
    keys_try.append(_norm_app_key(os.path.basename(app_id_n)))
    keys_try.append(_norm_app_key(os.path.basename(app_id_n)) + ".apk")
    for k in keys_try:
        k2 = _norm_app_key(k)
        if k2 in edges_map:
            return edges_map[k2]
    return []


# ---------------- Activity / UI state ----------------

def _normalize_activity(cur: str, package: str) -> str:
    """
    dumpsys may return:
      com.pkg/.MainActivity
      com.pkg/com.pkg.MainActivity
    Normalize to full class name: com.pkg.MainActivity
    """
    cur = cur.strip()
    if not cur:
        return ""
    if "/" in cur:
        pkg, cls = cur.split("/", 1)
        cls = cls.strip()
        if cls.startswith("."):
            return pkg + cls
        if cls.startswith(pkg):
            return cls
        if "." not in cls:
            return pkg + "." + cls
        return cls
    if cur.startswith("."):
        return package + cur
    return cur


def normalize_activity_list(activity_list: List[str], package: str) -> List[str]:
    out: List[str] = []
    for a in activity_list or []:
        a = (a or "").strip()
        if not a:
            continue
        out.append(_normalize_activity(a, package))
    return out


# -------- NEW: get focus component AND package --------

FOCUS_RE = re.compile(r"mCurrentFocus=Window\{.*?\s([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)}")
FOCUSEDAPP_RE = re.compile(r"mFocusedApp=AppWindowToken\{.*? ([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)}")
RESUMED_RE = re.compile(r"mResumedActivity:.*?\s([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)")
RESUMED2_RE = re.compile(r"ResumedActivity:.*?\s([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)")


def get_current_focus(serial: Optional[str], expected_package: str, timeout: int = 10) -> Tuple[str, str]:
    """
    Return (full_activity_class, focus_pkg).
    focus_pkg is the package part from 'pkg/cls' if available, else derived.
    """
    code, out, _ = adb_shell(serial, "dumpsys window windows", timeout=max(1, min(timeout, 8)))
    if code == 0 and out:
        m = FOCUS_RE.search(out)
        if m:
            comp = m.group(1).strip()
            pkg = comp.split("/", 1)[0].strip()
            return _normalize_activity(comp, expected_package).strip(), pkg
        m2 = FOCUSEDAPP_RE.search(out)
        if m2:
            comp = m2.group(1).strip()
            pkg = comp.split("/", 1)[0].strip()
            return _normalize_activity(comp, expected_package).strip(), pkg

    code, out, _ = adb_shell(serial, "dumpsys activity activities", timeout=timeout)
    if code == 0 and out:
        m = RESUMED_RE.search(out)
        if m:
            comp = m.group(1).strip()
            pkg = comp.split("/", 1)[0].strip()
            return _normalize_activity(comp, expected_package).strip(), pkg
        m2 = RESUMED2_RE.search(out)
        if m2:
            comp = m2.group(1).strip()
            pkg = comp.split("/", 1)[0].strip()
            return _normalize_activity(comp, expected_package).strip(), pkg

    return "", ""


def get_current_focus_fast(serial: Optional[str], expected_package: str, timeout: int = 6) -> Tuple[str, str]:
    """
    FAST version: only dumpsys window windows.
    """
    code, out, _ = adb_shell(serial, "dumpsys window windows", timeout=max(1, min(timeout, 8)))
    if code == 0 and out:
        m = FOCUS_RE.search(out)
        if m:
            comp = m.group(1).strip()
            pkg = comp.split("/", 1)[0].strip()
            return _normalize_activity(comp, expected_package).strip(), pkg
        m2 = FOCUSEDAPP_RE.search(out)
        if m2:
            comp = m2.group(1).strip()
            pkg = comp.split("/", 1)[0].strip()
            return _normalize_activity(comp, expected_package).strip(), pkg
    return "", ""


# -------- uiautomator dump (compressed auto-detect) --------

_UIA_COMPRESSED_SUPPORTED: Optional[bool] = None


def _try_uia_dump(serial: Optional[str], compressed: bool, timeout: int) -> Tuple[int, str, str]:
    flag = "--compressed " if compressed else ""
    cmd = f"uiautomator dump {flag}/sdcard/uidump.xml"
    return adb_shell(serial, cmd, timeout=timeout)


def dump_ui_xml(serial: Optional[str], timeout: int = 15) -> str:
    """
    PERF: try 'uiautomator dump --compressed' if supported.
    Auto-detect once; fallback safely.
    """
    global _UIA_COMPRESSED_SUPPORTED

    if _UIA_COMPRESSED_SUPPORTED is None:
        code, out, err = _try_uia_dump(serial, compressed=True, timeout=timeout)
        bad = (code != 0) or ("unknown option" in (out + err).lower()) or (
                "--compressed" in (out + err).lower() and "unknown" in (out + err).lower()
        )
        if bad:
            _UIA_COMPRESSED_SUPPORTED = False
            _try_uia_dump(serial, compressed=False, timeout=timeout)
        else:
            _UIA_COMPRESSED_SUPPORTED = True
    else:
        if _UIA_COMPRESSED_SUPPORTED:
            code, _, _ = _try_uia_dump(serial, compressed=True, timeout=timeout)
            if code != 0:
                _try_uia_dump(serial, compressed=False, timeout=timeout)
        else:
            _try_uia_dump(serial, compressed=False, timeout=timeout)

    code, out, _ = adb_shell(serial, "cat /sdcard/uidump.xml", timeout=timeout)
    if code != 0:
        return ""
    return out


# -------- Parse once per sample: root + hash + clickables --------

def parse_ui_root(xml_text: str) -> Optional[ET.Element]:
    if not xml_text or (not xml_text.lstrip().startswith("<")):
        return None
    try:
        return ET.fromstring(xml_text)
    except Exception:
        return None


def ui_hash_normalized_from_root(root: Optional[ET.Element], keep_clickable_text: bool = True, list_item_cap: int = 3) -> str:
    if root is None:
        return "empty"

    LIST_HINT = ("RecyclerView", "ListView")

    def norm_node(node: ET.Element) -> str:
        if node.tag != "node":
            return ""

        cls = node.attrib.get("class", "")
        rid = node.attrib.get("resource-id", "")
        clickable = node.attrib.get("clickable", "false")
        enabled = node.attrib.get("enabled", "false")
        scrollable = node.attrib.get("scrollable", "false")
        checked = node.attrib.get("checked", "false")
        selected = node.attrib.get("selected", "false")

        text = node.attrib.get("text", "") or ""
        desc = node.attrib.get("content-desc", "") or ""

        if keep_clickable_text and clickable == "true":
            text = " ".join(text.split())[:32]
            desc = " ".join(desc.split())[:32]
        else:
            text = ""
            desc = ""

        return f"C={cls}|ID={rid}|clk={clickable}|en={enabled}|scr={scrollable}|chk={checked}|sel={selected}|T={text}|D={desc}"

    parts: List[str] = []

    def walk(node: ET.Element):
        sig = norm_node(node)
        if sig:
            parts.append(sig)

        cls = node.attrib.get("class", "") or ""
        is_list = any(h in cls for h in LIST_HINT)

        children = list(node)
        if is_list and list_item_cap > 0:
            children = children[:list_item_cap]

        for ch in children:
            walk(ch)

    walk(root)
    canon = "\n".join(parts)
    return hashlib.md5(canon.encode("utf-8", errors="ignore")).hexdigest()


def parse_scrollables_from_root(root: Optional[ET.Element]) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = []
    if root is None:
        return res

    def parse_bounds(b: str) -> Optional[Tuple[int, int, int, int]]:
        m = re.match(r"\[(\d+),(\d+)]\[(\d+),(\d+)]", b)
        if not m:
            return None
        l, t, r, b2 = map(int, m.groups())
        return l, t, r, b2

    for node in root.iter():
        if node.tag != "node":
            continue

        enabled = node.attrib.get("enabled", "false") == "true"
        if not enabled:
            continue

        scrollable = node.attrib.get("scrollable", "false") == "true"
        if not scrollable:
            continue

        pkg = node.attrib.get("package", "") or ""
        clazz = node.attrib.get("class", "") or ""
        rid = node.attrib.get("resource-id", "") or ""
        bounds = node.attrib.get("bounds", "") or ""
        bb = parse_bounds(bounds)
        if not bb:
            continue
        l, t, r, b2 = bb
        w = r - l
        h = b2 - t
        if w < 80 or h < 160:
            continue

        cx = (l + r) // 2
        cy = (t + b2) // 2
        res.append({
            "package": pkg,
            "resource_id": rid,
            "clazz": clazz,
            "bounds": bounds,
            "l": l, "t": t, "r": r, "b": b2,
            "w": w, "h": h,
            "cx": cx, "cy": cy,
            "scrollable": True,
        })

    res.sort(key=lambda x: (x.get("w", 0) * x.get("h", 0)), reverse=True)
    return res


# ---------------- Numeric-trap filtering ----------------

PURE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")  # 1~4 digits
MONTH_WORD_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", re.I)
DATE_DESC_RE = re.compile(r"\b\d{1,2}\b.*\b\d{4}\b", re.I)  # e.g., "01 January 2025" / "1 2025"

def _is_pure_number_text(s: str) -> bool:
    if not s:
        return False
    return PURE_NUM_RE.match(s.strip()) is not None

def _is_calendar_day_like(text: str, desc: str, rid: str, clazz: str) -> bool:
    if not _is_pure_number_text(text):
        return False
    try:
        v = int(text.strip())
        if v < 1 or v > 31:
            return False
    except Exception:
        return False

    blob = f"{(desc or '').lower()} {(rid or '').lower()} {(clazz or '').lower()}"
    if "calendar" in blob or "datepicker" in blob:
        return True
    if MONTH_WORD_RE.search(desc or ""):
        return True
    if DATE_DESC_RE.search(desc or ""):
        return True
    return False

def is_numeric_trap_node(n: Dict[str, Any]) -> bool:
    text = (n.get("text") or "").strip()
    desc = (n.get("desc") or "").strip()
    rid = (n.get("resource_id") or "").strip().lower()
    clazz = (n.get("clazz") or "").strip()

    if rid in {"android:id/button1", "android:id/button2", "android:id/button3"}:
        return False

    if _is_calendar_day_like(text, desc, rid, clazz):
        return True

    if _is_pure_number_text(text) and (_is_pure_number_text(desc) or (not desc)):
        ACTION_ID_HINTS = ("btn", "button", "action", "ok", "cancel", "apply", "next", "prev", "save", "done",
                           "confirm", "settings")
        if any(h in rid for h in ACTION_ID_HINTS):
            return False
        if not rid:
            return True
        return True
    return False


# ---------------- Garbage node filtering ----------------

BAD_RIDS_GLOBAL = {
    "android:id/content",
    "android:id/list_container",
    "android:id/list",
    "android:id/action_bar_container",
    "android:id/action_bar_root",
    "android:id/statusBarBackground",
    "android:id/navigationBarBackground",
}

BAD_RID_SUBSTR = (
    "recycler_view",
    "list_container",
    "action_bar_root",
)

BAD_CLASS_HINT = (
    "FrameLayout",
    "LinearLayout",
    "RecyclerView",
    "ListView",
    "ViewGroup",
)


def is_garbage_node(n: Dict[str, Any]) -> bool:
    rid = (n.get("resource_id") or "").strip().lower()
    clazz = (n.get("clazz") or "").strip()
    txt = (n.get("text") or "").strip()
    desc = (n.get("desc") or "").strip()

    if rid in BAD_RIDS_GLOBAL:
        return True

    rid_l = rid.lower()
    for sub in BAD_RID_SUBSTR:
        if sub in rid_l:
            return True

    if any(h in clazz for h in BAD_CLASS_HINT) and (not txt) and (not desc):
        return True

    if rid.endswith(":id/content") and (not txt) and (not desc):
        return True

    return False


# ---------------- Clickables (FIXED) ----------------

def node_sig(n: Dict[str, Any]) -> str:
    base = f"{n.get('resource_id', '')}|{n.get('text', '')}|{n.get('desc', '')}|{n.get('clazz', '')}|{n.get('bounds', '')}"
    return hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()


def parse_clickables_from_root(root: Optional[ET.Element]) -> List[Dict[str, Any]]:
    """
    include normal enabled+clickable nodes + popup-menu items (tap clickable parent)
    """
    res: List[Dict[str, Any]] = []
    if root is None:
        return res

    def parse_bounds(b: str):
        m = re.match(r"\[(\d+),(\d+)]\[(\d+),(\d+)]", b)
        if not m:
            return None
        l, t, r, b2 = map(int, m.groups())
        return l, t, r, b2

    def mk_cand(node_for_meta: ET.Element, tap_node: ET.Element) -> Optional[Dict[str, Any]]:
        pkg = node_for_meta.attrib.get("package", "") or ""
        clazz = node_for_meta.attrib.get("class", "") or ""
        text = node_for_meta.attrib.get("text", "") or ""
        desc = node_for_meta.attrib.get("content-desc", "") or ""
        rid = node_for_meta.attrib.get("resource-id", "") or ""

        b = tap_node.attrib.get("bounds", "") or ""
        bb = parse_bounds(b)
        if not bb:
            return None
        l, t, r, b2 = bb
        if (r - l) <= 2 or (b2 - t) <= 2:
            return None
        cx, cy = (l + r) // 2, (t + b2) // 2

        cand = {
            "package": pkg,
            "resource_id": rid,
            "text": text,
            "desc": desc,
            "clazz": clazz,
            "bounds": b,
            "cx": cx,
            "cy": cy,
            "clickable": (tap_node.attrib.get("clickable", "false") == "true"),
        }
        if is_numeric_trap_node(cand):
            return None
        return cand

    def is_menu_text_like(node: ET.Element) -> bool:
        cls = (node.attrib.get("class", "") or "").lower()
        rid = (node.attrib.get("resource-id", "") or "").lower()
        txt = (node.attrib.get("text", "") or "").strip()
        if not txt:
            return False
        if ("textview" in cls) or ("checkedtextview" in cls):
            return True
        if rid.endswith("/title") or rid.endswith("/text1") or rid in {"android:id/title", "android:id/text1"}:
            return True
        return False

    def is_clickable(node: ET.Element) -> bool:
        return (node.attrib.get("clickable", "false") == "true")

    def is_enabled(node: ET.Element) -> bool:
        return (node.attrib.get("enabled", "false") == "true")

    seen: Set[str] = set()

    def push(c: Optional[Dict[str, Any]]):
        if not c:
            return
        sig = node_sig(c)
        if sig in seen:
            return
        seen.add(sig)
        res.append(c)

    def walk(node: ET.Element, parent: Optional[ET.Element]):
        if node.tag == "node":
            if is_enabled(node):
                if is_clickable(node):
                    push(mk_cand(node, node))
                if is_menu_text_like(node) and parent is not None and is_clickable(parent) and is_enabled(parent):
                    push(mk_cand(node, parent))
        for ch in list(node):
            walk(ch, node)

    walk(root, None)
    return res


# ---------------- Escape control / runtime package ----------------

SYSTEM_PKG_PREFIX = (
    "com.android.", "android", "com.google.android.", "com.miui.", "com.samsung.", "com.oneplus.",
)

def is_system_pkg(pkg: str) -> bool:
    p = (pkg or "").strip()
    if not p:
        return True
    return p.startswith(SYSTEM_PKG_PREFIX)

def in_app_pkg(focus_pkg: str, pkg_run: str) -> bool:
    return (focus_pkg or "").strip() == (pkg_run or "").strip()

def recover_if_escaped(
        serial: Optional[str],
        pkg_run: str,
        cur_act: str,
        cur_pkg: str,
        logp: str,
        reason: str,
) -> bool:
    if not cur_act or in_app_pkg(cur_pkg, pkg_run):
        return False

    log_write(logp, f"[escape] {reason}: act={cur_act} pkg={cur_pkg} -> BACK x2", also_console=True)
    back(serial, 2)
    time.sleep(0.4)

    act2, pkg2 = get_current_focus(serial, pkg_run, timeout=8)
    if in_app_pkg(pkg2, pkg_run):
        log_write(logp, f"[escape] recovered by BACK -> act={act2} pkg={pkg2}", also_console=True)
        return True

    log_write(logp, f"[escape] still out after BACK -> force-stop + relaunch pkg={pkg_run}", also_console=True)
    adb_shell(serial, f"am force-stop {pkg_run}", timeout=10)
    time.sleep(0.5)
    adb_shell(serial, f"monkey -p {pkg_run} -c android.intent.category.LAUNCHER 1", timeout=20)
    time.sleep(1.0)
    return True


# ---------------- keep_candidate_by_package ----------------

def keep_candidate_by_package(n: Dict[str, Any], target_package: str) -> bool:
    pkg = (n.get("package") or "").strip()
    rid = (n.get("resource_id") or "").strip().lower()
    txt = (n.get("text") or "").strip()
    desc = (n.get("desc") or "").strip()
    clazz = (n.get("clazz") or "").strip().lower()

    if pkg == target_package:
        return True

    if pkg != "android":
        return False

    allow_rids = {
        "android:id/button1", "android:id/button2", "android:id/button3",
        "android:id/home", "android:id/up",
        "android:id/title", "android:id/text1",
    }
    if rid in allow_rids:
        return True

    dl = desc.lower()
    if "navigate up" in dl or dl.strip() == "up":
        return True

    blob = f"{txt.lower()} {desc.lower()} {rid}"
    hard_bad = ("play.google", "market://", "com.android.vending", "open in browser", "browser", "chrome",
                "http://", "https://", "www.", "rate", "donate", "github")
    if any(w in blob for w in hard_bad):
        return False

    if txt and (("textview" in clazz) or ("checkedtextview" in clazz)):
        if len(txt.strip()) <= 48:
            return True

    if rid.startswith("android:id/") and (txt.strip() or desc.strip()):
        if len((txt + " " + desc).strip()) <= 80:
            return True

    return False


# ---------------- App control ----------------

def is_installed(serial: Optional[str], package: str) -> bool:
    code, out, _ = adb_shell(serial, f"pm path {package}", timeout=10)
    return (code == 0) and (package in out)

def install_apk(serial: Optional[str], apk_path: str, logp: str) -> bool:
    code, out, err = adb(serial, ["install", "-r", "-g", apk_path], timeout=180)
    log_write(logp, f"[install] code={code}\n{out}\n{err}")
    return code == 0

def uninstall_app(serial: Optional[str], package: str, logp: str) -> bool:
    code, out, err = adb(serial, ["uninstall", package], timeout=120)
    log_write(logp, f"[uninstall] code={code}\n{out}\n{err}")
    return code == 0

def clear_app_data(serial: Optional[str], package: str, logp: str) -> bool:
    code, out, err = adb_shell(serial, f"pm clear {package}", timeout=30)
    log_write(logp, f"[pm clear] code={code} out={out.strip()} err={err.strip()}")
    return code == 0


def force_stop(serial: Optional[str], package: str) -> None:
    adb_shell(serial, f"am force-stop {package}", timeout=10)

def launch_app(serial: Optional[str], package: str) -> None:
    adb_shell(serial, f"monkey -p {package} -c android.intent.category.LAUNCHER 1", timeout=20)

def app_pid(serial: Optional[str], package: str) -> str:
    code, out, _ = adb_shell(serial, f"pidof {package}", timeout=5)
    if code != 0:
        return ""
    return out.strip()

def tap(serial: Optional[str], x: int, y: int) -> None:
    adb_shell(serial, f"input tap {x} {y}", timeout=5)

def swipe(serial: Optional[str], x1: int, y1: int, x2: int, y2: int, duration_ms: int = 350) -> None:
    adb_shell(serial, f"input swipe {x1} {y1} {x2} {y2} {duration_ms}", timeout=5)

def reset_app_for_run(
        serial: Optional[str],
        package: str,
        apk_path: str,
        logp: str,
        mode: str = "reinstall",
) -> bool:
    try:
        force_stop(serial, package)
    except Exception:
        pass
    time.sleep(0.4)

    if mode == "keep":
        if not is_installed(serial, package):
            log_write(logp, "[reset][keep] app not installed -> install once", True)
            ok = install_apk(serial, apk_path, logp)
            if not ok:
                return False
            time.sleep(0.8)

        launch_app(serial, package)
        time.sleep(1.0)
        log_write(logp, "[reset][keep] force-stop + cleanup + launch (no clear/reinstall)", True)
        return True

    if mode == "clear":
        if not is_installed(serial, package):
            ok = install_apk(serial, apk_path, logp)
            if not ok:
                return False
        clear_app_data(serial, package, logp)
        time.sleep(0.6)
        return True

    uninstall_app(serial, package, logp)
    ok = install_apk(serial, apk_path, logp)
    if not ok:
        if is_installed(serial, package):
            log_write(logp, "[reset] reinstall failed -> fallback pm clear", True)
            clear_app_data(serial, package, logp)
            time.sleep(0.6)
            return True
        return False
    time.sleep(0.8)
    return True


# ---------------- EdgePool scheduling ----------------

@dataclass
class EdgeInfo:
    src: str
    dst: str
    p: float
    tried: int = 0
    success: int = 0
    cooldown_until: float = 0.0
    first_hit_ts: float = 0.0
    first_hit_tries: int = 0

class EdgePool:
    def __init__(self, edges: List[Tuple[str, str, float]]):
        self.by_src: Dict[str, List[EdgeInfo]] = {}
        for s, t, p in edges:
            self.by_src.setdefault(s, []).append(EdgeInfo(src=s, dst=t, p=p))
        for s in self.by_src:
            self.by_src[s].sort(key=lambda e: e.p, reverse=True)

    def pick(self, cur_src: str, now: float) -> Optional[EdgeInfo]:
        cands = self.by_src.get(cur_src, [])
        if not cands:
            return None
        best = None
        best_score = -1e18
        for e in cands:
            if now < e.cooldown_until:
                continue
            score = (e.p * 1000.0) - (e.tried * 5.0) + (e.success * -50.0)
            if score > best_score:
                best_score = score
                best = e
        return best

    def mark_try(self, e: EdgeInfo) -> None:
        e.tried += 1

    def mark_success(self, e: EdgeInfo, now: float) -> None:
        e.success += 1
        if e.first_hit_ts <= 0:
            e.first_hit_ts = now
            e.first_hit_tries = e.tried
        e.cooldown_until = now + 20.0

    def freeze_if_bad(self, e: EdgeInfo) -> None:
        if e.tried >= 15 and e.success == 0:
            e.cooldown_until = time.time() + 999999.0


# ---------------- Metrics ----------------

@dataclass
class PerSecondRow:
    t: int
    cur_activity: str
    ui_hash: str
    unique_activities: int
    unique_ui_states: int
    unique_transitions: int
    events_total: int
    events_per_sec: float
    activity_change_rate: float
    ui_state_change_rate: float
    crash_count: int

def auc_from_series(vals: List[int]) -> float:
    return float(sum(vals))

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


# ---------------- Guided scoring helpers ----------------

def camel_tokens(s: str) -> List[str]:
    if not s:
        return []
    s = s.split(".")[-1]
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", s)
    return [p.lower() for p in parts if p]

def _init_first_reach(activity_list: List[str]) -> Dict[str, Optional[int]]:
    mp: Dict[str, Optional[int]] = {}
    for a in activity_list or []:
        mp[a] = None
    return mp

def _mark_first_reach(first_reach: Dict[str, Optional[int]], cur_act: str, t: int) -> None:
    if not cur_act:
        return
    if cur_act in first_reach and first_reach[cur_act] is None:
        first_reach[cur_act] = int(t)


# ---------------- Guided Runner ----------------

def run_guided(
        serial: Optional[str],
        app_id: str,
        package: str,
        pool: EdgePool,
        duration: int,
        out_dir: str,
        sample_interval: int,
        logp: str,
        activity_list: List[str],
) -> None:
    """
    Guided runner with runtime package adopt (CRITICAL FIX).
    """
    from collections import deque

    ensure_dir(out_dir)
    timeseries_path = os.path.join(out_dir, "timeseries_1s.csv")
    trans_path = os.path.join(out_dir, "discovered_transitions.txt")

    ver_root = os.path.dirname(out_dir)
    action_map_path = os.path.join(ver_root, "action_map.json")

    TAP_COOLDOWN_SEC = 60
    GOAL_TAP_COOLDOWN_SEC = 12
    EDGE_COOLDOWN_SEC = 45
    POST_INPUT_SETTLE_SEC = 0.25

    ESCAPE_SETTINGS_DWELL_SEC = 18
    ESCAPE_COOLDOWN_SEC = 10
    ESCAPE_BACK_STEPS = 3
    ESCAPE_FORCE_RESTART_SEC = 60

    escape_back_remaining = 0
    last_escape_sec = -10 ** 9
    last_restart_sec = -10 ** 9
    settings_dwell = 0

    STARTUP_GRACE_SEC = 25
    RETRY_ACT_SEC = 0.8
    SCROLL_COOLDOWN_SEC = max(4, int(sample_interval) * 2)
    MAX_SCROLLS_PER_STATE = 3

    RID_BLOCK_SEC = 60
    rid_outcome: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    rid_block_until: Dict[Tuple[str, str, str], int] = {}

    recent_states = deque(maxlen=6)

    @dataclass
    class PendingAction:
        sec: int
        src_act: str
        src_ui: str
        node: Dict[str, Any]
        ek: Optional[str] = None
        target_edge: Optional[EdgeInfo] = None
        kind: str = ""

    # runtime package (may adopt)
    pkg_run = package
    activity_list = normalize_activity_list(activity_list, pkg_run)
    first_reach = _init_first_reach(activity_list)

    tap_last_t: Dict[Tuple[str, str], int] = {}
    edge_last_t: Dict[str, int] = {}
    scroll_last_t: Dict[Tuple[str, str], int] = {}
    scroll_cnt: Dict[Tuple[str, str], int] = {}

    @dataclass
    class CatStat:
        tries: int = 0
        success: int = 0
        last_try_sec: int = -10 ** 9

    goal_first_stat: Dict[Tuple[str, str], Dict[str, CatStat]] = {}

    CAT_RETRY_COOLDOWN_SEC = max(10, int(sample_interval) * 3)
    CAT_TRY_PENALTY = 4.0
    CAT_SUCCESS_BONUS = 1.0
    CAT_PRIORITY_BONUS = 3.0

    GOAL_FIRST_STATE_COOLDOWN_SEC = max(6, int(sample_interval) * 2)
    goal_first_last_sec: Dict[Tuple[str, str], int] = {}

    def edge_key(src: str, dst: str) -> str:
        return f"{src}||{dst}"

    def _rid_key_from_node(n: Dict[str, Any]) -> str:
        ridk = ((n.get("resource_id") or "").strip().lower())
        if ridk:
            return ridk
        return ((n.get("clazz") or "").strip().lower())

    def recently_tapped(act: str, n: Dict[str, Any], sec: int, cooldown_sec: int = TAP_COOLDOWN_SEC) -> bool:
        k = (act, node_sig(n))
        last = tap_last_t.get(k)
        return (last is not None) and (sec - last < int(max(1, cooldown_sec)))

    def mark_tapped(act: str, n: Dict[str, Any], sec: int) -> None:
        tap_last_t[(act, node_sig(n))] = int(sec)

    def edge_on_cooldown(ek: str, sec: int) -> bool:
        last = edge_last_t.get(ek)
        return (last is not None) and (sec - last < EDGE_COOLDOWN_SEC)

    def mark_edge(ek: str, sec: int) -> None:
        edge_last_t[ek] = int(sec)

    def load_action_map(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_action_map(path: str, mp: Dict[str, Any]) -> None:
        ensure_dir(os.path.dirname(path))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(mp, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def is_settings_like(act: str, root0: Optional[ET.Element]) -> bool:
        a = (act or "").lower()
        if any(k in a for k in ["settings", "preference", "prefs", "about"]):
            return True
        if root0 is None:
            return False
        c = 0
        for nd in root0.iter():
            if nd.tag != "node":
                continue
            cls = (nd.attrib.get("class", "") or "").lower()
            rid = (nd.attrib.get("resource-id", "") or "").lower()
            if "preference" in cls or "settings" in rid:
                return True
            c += 1
            if c >= 200:
                break
        return False

    def find_nav_up(cands_all: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for n in cands_all:
            rid = ((n.get("resource_id") or "").strip().lower())
            desc = ((n.get("desc") or "").strip().lower())
            if rid in {"android:id/home", "android:id/up"}:
                return n
            if "navigate up" in desc:
                return n
        return None

    def find_overflow_menu(cands_all: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for n in cands_all:
            desc = ((n.get("desc") or "").strip().lower())
            txt = ((n.get("text") or "").strip().lower())
            rid = ((n.get("resource_id") or "").strip().lower())
            blob = f"{desc} {txt} {rid}"
            if ("more options" in blob) or ("overflow" in blob) or ("更多选项" in blob) or (rid.endswith(":id/menu")) or ("menu" in blob):
                if n.get("cx") is not None and n.get("cy") is not None:
                    return n
            if ("btnpopup" in blob) or ("popup" in blob):
                return n
        return None

    def find_node_for_action(cands: List[Dict[str, Any]], actrec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rid = (actrec.get("resource_id") or "").strip().lower()
        txt = (actrec.get("text") or "").strip()
        desc = (actrec.get("desc") or "").strip()
        bounds = (actrec.get("bounds") or "").strip()

        best = None
        best_score = -1e9
        for n in cands:
            s = 0.0
            rid_n = (n.get("resource_id") or "").strip().lower()
            if rid and rid_n == rid:
                s += 10.0
            if bounds and (n.get("bounds") or "").strip() == bounds:
                s += 4.0
            if txt and (n.get("text") or "").strip() == txt:
                s += 2.0
            if desc and (n.get("desc") or "").strip() == desc:
                s += 2.0
            if s > best_score:
                best_score = s
                best = n

        return best if best_score >= 4.0 else None

    action_map = load_action_map(action_map_path)
    last_save_ts = time.time()

    pending: Optional[PendingAction] = None

    unique_acts: Set[str] = set()
    unique_states: Set[str] = set()
    unique_trans: Set[Tuple[str, str]] = set()

    target_hits: Dict[str, Dict[str, Any]] = {}

    events_total = 0
    act_change = 0
    ui_change = 0
    crash_count = 0
    escape_count = 0

    slow_sample_count = 0
    slow_sample_max_sec = 0.0
    slow_sample_total_sec = 0.0

    # -------- runtime adopt logic --------
    adopt_done = False
    observed_pkg_cnt: Dict[str, int] = {}

    def maybe_adopt_pkg(focus_pkg: str) -> None:
        nonlocal pkg_run, adopt_done
        fp = (focus_pkg or "").strip()
        if adopt_done:
            return
        if not fp or fp == pkg_run or is_system_pkg(fp):
            return
        observed_pkg_cnt[fp] = observed_pkg_cnt.get(fp, 0) + 1
        if observed_pkg_cnt[fp] >= 3:
            pid_old = app_pid(serial, pkg_run)
            pid_new = app_pid(serial, fp)
            # adopt if old pid empty but new exists OR we never get in-app for old within grace
            if (not pid_old) and pid_new:
                log_write(logp, f"[pkg-adopt] adopt pkg_run: {pkg_run} -> {fp} (pid_old empty, pid_new={pid_new})", True)
                try:
                    force_stop(serial, pkg_run)
                except Exception:
                    pass
                pkg_run = fp
                adopt_done = True
                # relaunch with adopted package
                try:
                    launch_app(serial, pkg_run)
                    time.sleep(1.0)
                except Exception:
                    pass

    def post_input_escape_guard(sec: int, reason: str) -> bool:
        nonlocal escape_count
        actp, pkgp = get_current_focus_fast(serial, pkg_run, timeout=3)
        maybe_adopt_pkg(pkgp)
        if actp and (not in_app_pkg(pkgp, pkg_run)) and (not is_system_pkg(pkgp)):
            escape_count += 1
            recover_if_escaped(serial, pkg_run, actp, pkgp, logp, reason=f"after_{reason} t={sec}")
            return True
        return False

    # initial launch
    force_stop(serial, pkg_run)
    launch_app(serial, pkg_run)

    launch_ts = time.time()
    last_activity = ""
    last_pkg = ""

    # startup grace with adopt
    while time.time() - launch_ts < STARTUP_GRACE_SEC:
        last_activity, last_pkg = get_current_focus(serial, pkg_run, timeout=8)
        maybe_adopt_pkg(last_pkg)
        if last_activity and in_app_pkg(last_pkg, pkg_run):
            break
        time.sleep(0.6)

    if last_activity and (not in_app_pkg(last_pkg, pkg_run)) and (not is_system_pkg(last_pkg)):
        escape_count += 1
        recover_if_escaped(serial, pkg_run, last_activity, last_pkg, logp, reason="guided init after grace")
        last_activity, last_pkg = get_current_focus(serial, pkg_run, timeout=8)
        maybe_adopt_pkg(last_pkg)

    last_xml = dump_ui_xml(serial, timeout=15)
    last_root = parse_ui_root(last_xml)
    last_state = ui_hash_normalized_from_root(last_root)

    if last_activity and in_app_pkg(last_pkg, pkg_run):
        unique_acts.add(last_activity)
        _mark_first_reach(first_reach, last_activity, 0)
    unique_states.add(last_state)

    tried_nodes: Dict[Tuple[str, str], Set[str]] = {}

    series_acts: List[int] = []
    series_states: List[int] = []
    series_trans: List[int] = []

    no_candidate_streak = 0
    no_transition_streak = 0
    last_ut = 0

    HARD_BAD = [
        "go to the store", "store", "play.google", "market://", "com.android.vending",
        "open in browser", "browser", "chrome", "http://", "https://", "www.",
        "rate", "donate", "github"
    ]
    SOFT_BAD = [
        "how to", "tutorial", "help", "privacy policy", "terms"
    ]

    GOAL_CATS = {
        "settings": {"settings", "setting", "preference", "prefs", "options", "设置", "einstellungen", "optionen"},
        "about": {"about", "info", "license", "version", "关于", "über", "info", "lizenz"},
        "help": {"help", "tutorial", "guide", "faq", "帮助", "教程", "指南", "常见问题", "hilfe", "anleitung"},
        "privacy": {"privacy", "policy", "terms", "隐私", "条款", "datenschutz", "bedingungen"},
        "account": {"account", "profile", "login", "signup", "register", "账户", "个人", "登录", "注册",
                    "konto", "profil", "anmelden", "registrieren"},
        "data": {"backup", "import", "export", "备份", "导入", "导出", "sicherung", "import", "export"},
        "lang": {"language", "locale", "语言", "sprache", "lokal"},
        "log": {"log", "日志"},
        "infor": {"infor", "信息", "统计信息"}
    }
    GOAL_PRIORITY = ["settings", "about", "privacy", "help", "account", "data", "lang", "log", "infor"]
    STRICT_GOAL_ORDER = False

    GOAL_STATIC_WORDS: Set[str] = set()
    for _kws in GOAL_CATS.values():
        GOAL_STATIC_WORDS |= set(_kws)
    GOAL_STATIC_WORDS |= {
        "menu", "more", "options", "option", "setting", "settings",
        "about", "help", "privacy", "account", "profile",
        "设置", "更多", "菜单", "选项", "关于", "帮助", "隐私", "账户",
        "einstellungen", "über", "hilfe", "datenschutz", "konto", "profil", "menü",
    }

    def build_goal_words() -> Tuple[Set[str], Set[str]]:
        goal_all: Set[str] = set()
        goal_unreached: Set[str] = set()

        for act, t0 in (first_reach or {}).items():
            toks = camel_tokens(act)
            for tk in toks:
                if tk:
                    goal_all.add(tk)
                    if t0 is None:
                        goal_unreached.add(tk)

        goal_all |= set([w.lower() for w in GOAL_STATIC_WORDS if w])
        goal_unreached |= set([w.lower() for w in GOAL_STATIC_WORDS if w])

        if len(goal_all) > 300:
            goal_all = set(sorted(goal_all)[:300])
        if len(goal_unreached) > 300:
            goal_unreached = set(sorted(goal_unreached)[:300])

        return goal_all, goal_unreached

    def _get_cat_stat(key_ui: Tuple[str, str], cat: str) -> CatStat:
        mp = goal_first_stat.setdefault(key_ui, {})
        st = mp.get(cat)
        if st is None:
            st = CatStat()
            mp[cat] = st
        return st

    def _priority_idx(cat: str) -> int:
        try:
            return GOAL_PRIORITY.index(cat)
        except Exception:
            return 999

    def _priority_bonus(cat: str) -> float:
        idx = _priority_idx(cat)
        if idx >= 999:
            return 0.0
        return CAT_PRIORITY_BONUS * float(max(0, (len(GOAL_PRIORITY) - idx))) / float(max(1, len(GOAL_PRIORITY)))

    def _cat_allowed(key_ui: Tuple[str, str], cat: str, sec: int) -> bool:
        st = _get_cat_stat(key_ui, cat)
        return (sec - st.last_try_sec) >= CAT_RETRY_COOLDOWN_SEC

    def _strict_pick_cat(key_ui: Tuple[str, str], sec: int) -> Optional[str]:
        mp = goal_first_stat.get(key_ui, {})
        for cat in GOAL_PRIORITY:
            st = mp.get(cat, CatStat())
            if st.tries == 0 and _cat_allowed(key_ui, cat, sec):
                return cat
        best = None
        best_key = (10 ** 9, 10 ** 9)
        for cat in GOAL_PRIORITY:
            st = mp.get(cat, CatStat())
            if not _cat_allowed(key_ui, cat, sec):
                continue
            key = (st.tries, _priority_idx(cat))
            if key < best_key:
                best_key = key
                best = cat
        return best

    def pick_goal_first_node(cands: List[Dict[str, Any]], key_ui: Tuple[str, str], sec: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        goal_all, goal_unreached = build_goal_words()

        best_node = None
        best_cat = None
        best_s = -1e18

        strict_cat = _strict_pick_cat(key_ui, sec) if STRICT_GOAL_ORDER else None

        for n in cands:
            text_l = (n.get("text", "") or "").lower()
            desc_l = (n.get("desc", "") or "").lower()
            rid_l = (n.get("resource_id", "") or "").lower()
            blob = f"{text_l} {desc_l} {rid_l}"

            if any(w in blob for w in HARD_BAD):
                continue

            local_best = -1e18
            local_cat = None

            cats_iter = [strict_cat] if strict_cat else list(GOAL_CATS.keys())
            for cat in cats_iter:
                if cat is None:
                    continue
                if not _cat_allowed(key_ui, cat, sec):
                    continue

                kws = GOAL_CATS.get(cat, set())
                hit = 0
                for w0 in kws:
                    wl = (w0 or "").lower()
                    if wl and (wl in blob):
                        hit += 1
                if hit <= 0:
                    continue

                s = 0.0
                s += 12.0 + 2.0 * float(hit)

                for wkw in goal_unreached:
                    if wkw and (wkw in blob):
                        s += 2.0
                for wkw in goal_all:
                    if wkw and (wkw in blob):
                        s += 0.4

                if n.get("resource_id"):
                    s += 0.5
                if n.get("clickable"):
                    s += 0.2

                s += _priority_bonus(cat)

                st = _get_cat_stat(key_ui, cat)
                s -= float(st.tries) * CAT_TRY_PENALTY
                s += float(min(3, st.success)) * CAT_SUCCESS_BONUS

                if s > local_best:
                    local_best = s
                    local_cat = cat

            if local_cat is None:
                continue
            if local_best < 12.0:
                continue

            if local_best > best_s:
                best_s = local_best
                best_node = n
                best_cat = local_cat

        return best_node, best_cat

    def score_candidates(candidates: List[Dict[str, Any]], target_edge: Optional[EdgeInfo]) -> List[Tuple[float, Dict[str, Any]]]:
        dst_l = (target_edge.dst.lower() if target_edge else "")

        target_words: Set[str] = set()
        if target_edge:
            target_words.update(camel_tokens(target_edge.dst))
            target_words.update({
                "setting", "settings", "about", "register", "signup", "login",
                "help", "tutorial", "option", "privacy", "license", "info",
                "language", "locale", "account", "profile", "clear", "history",
                "einstellungen", "über", "hilfe", "datenschutz", "konto", "profil", "sprache",
            })

        goal_all: Set[str] = set()
        goal_unreached: Set[str] = set()
        if not target_edge:
            goal_all, goal_unreached = build_goal_words()

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for n in candidates:
            text_l = (n.get("text", "") or "").lower()
            desc_l = (n.get("desc", "") or "").lower()
            rid_l = (n.get("resource_id", "") or "").lower()
            cls_l = (n.get("clazz", "") or "").lower()

            s = 0.0
            if _is_pure_number_text(n.get("text", "")) and not (n.get("desc", "") or "").strip():
                s -= 10.0

            blob = f"{text_l} {desc_l} {rid_l}"

            if any(w in blob for w in HARD_BAD):
                s -= 12.0

            if any(w in blob for w in SOFT_BAD):
                if target_edge and (("help" in dst_l) or ("tutorial" in dst_l)):
                    pass
                else:
                    s -= 5.0

            is_overflow = (
                    "more options" in desc_l or "更多选项" in desc_l or "overflow" in desc_l
                    or "btnpopup" in rid_l or "popup" in rid_l
                    or "menu" in rid_l or "overflow" in rid_l
            )

            if target_edge and ("settings" in dst_l or "einstellungen" in dst_l) and is_overflow:
                s += 50.0

            if target_edge:
                if is_overflow:
                    s += 2.0
            else:
                if is_overflow:
                    s += 0.8

            if target_edge:
                for wkw in target_words:
                    if not wkw:
                        continue
                    if wkw in text_l:
                        s += 1.2
                    if wkw in desc_l:
                        s += 1.2
                    if wkw in rid_l:
                        s += 0.8
            else:
                for wkw in goal_unreached:
                    if not wkw:
                        continue
                    if wkw in text_l:
                        s += 1.0
                    if wkw in desc_l:
                        s += 1.0
                    if wkw in rid_l:
                        s += 0.7

                for wkw in goal_all:
                    if not wkw:
                        continue
                    if wkw in text_l:
                        s += 0.35
                    if wkw in desc_l:
                        s += 0.35
                    if wkw in rid_l:
                        s += 0.2

            if n.get("resource_id"):
                s += 0.5
            if n.get("clickable"):
                s += 0.2
            if ("button" in cls_l) or ("imagebutton" in cls_l):
                s += 0.2

            scored.append((s, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def note_target_try(target_edge: EdgeInfo) -> None:
        pool.mark_try(target_edge)
        ek = edge_key(target_edge.src, target_edge.dst)
        target_hits.setdefault(ek, {
            "src": target_edge.src,
            "dst": target_edge.dst,
            "p": target_edge.p,
            "tries": 0,
            "success": 0,
            "first_hit_t": None,
            "first_hit_tries": None,
        })
        target_hits[ek]["tries"] += 1

    def note_target_success(target_edge: EdgeInfo, sec: int) -> None:
        pool.mark_success(target_edge, time.time())
        ek = edge_key(target_edge.src, target_edge.dst)
        th = target_hits.get(ek)
        if th is None:
            th = {
                "src": target_edge.src,
                "dst": target_edge.dst,
                "p": target_edge.p,
                "tries": 0,
                "success": 0,
                "first_hit_t": None,
                "first_hit_tries": None,
            }
            target_hits[ek] = th

        th["success"] += 1
        if th["first_hit_t"] is None:
            th["first_hit_t"] = sec
            th["first_hit_tries"] = th["tries"]

    def learn_action_for_transition(src_act: str, dst_act: str, n: Dict[str, Any], sec: int) -> None:
        if (not src_act) or (not dst_act) or (src_act == dst_act):
            return
        ek = edge_key(src_act, dst_act)
        rec = action_map.get(ek, {})
        rec2 = {
            "resource_id": n.get("resource_id", ""),
            "text": n.get("text", ""),
            "desc": n.get("desc", ""),
            "clazz": n.get("clazz", ""),
            "bounds": n.get("bounds", ""),
            "cx": int(n.get("cx", 0)),
            "cy": int(n.get("cy", 0)),
            "success": int(rec.get("success", 0)) + 1,
            "first_learn_t": rec.get("first_learn_t", sec),
            "last_hit_t": sec,
        }
        action_map[ek] = rec2

    def pick_cache_any(cur_act: str, sec: int) -> Optional[Tuple[str, Dict[str, Any]]]:
        best = None
        best_score = -1e18
        for ek, rec in action_map.items():
            if "||" not in ek:
                continue
            s, d = ek.split("||", 1)
            if s != cur_act:
                continue
            if edge_on_cooldown(ek, sec):
                continue

            unreached_bonus = 0.0
            if d in first_reach and first_reach[d] is None:
                unreached_bonus = 1000.0

            succ = float(rec.get("success", 0))
            score = unreached_bonus + succ * 10.0
            if score > best_score:
                best_score = score
                best = (ek, rec)
        return best

    def can_scroll(cur_act: str, ui_h: str, sec: int) -> bool:
        k = (cur_act, ui_h)
        last = scroll_last_t.get(k)
        if last is not None and (sec - last) < SCROLL_COOLDOWN_SEC:
            return False
        c = scroll_cnt.get(k, 0)
        return c < MAX_SCROLLS_PER_STATE

    def mark_scroll(cur_act: str, ui_h: str, sec: int) -> None:
        k = (cur_act, ui_h)
        scroll_last_t[k] = int(sec)
        scroll_cnt[k] = int(scroll_cnt.get(k, 0) + 1)

    with open(timeseries_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(list(PerSecondRow.__annotations__.keys()))

        last_written_t = -1
        prev_t = -1
        next_progress_ts = None

        last_snapshot = (
            last_activity, last_state, len(unique_acts), len(unique_states), len(unique_trans),
            events_total, act_change, ui_change, crash_count
        )

        try:
            for sec, start_ts, end_ts in iter_wallclock_seconds(duration, sample_interval):
                if next_progress_ts is None:
                    next_progress_ts = start_ts + 10.0

                if time.time() >= end_ts - 0.2:
                    break

                input_used = False
                acted = False

                delta = 1 if prev_t < 0 else max(1, sec - prev_t)
                if delta > 1:
                    log_write(logp, f"[slow-sample] skipped {delta - 1}s (prev_t={prev_t} -> t={sec})", True)
                prev_t = sec

                sample_start = time.time()
                now = time.time()

                pid = app_pid(serial, pkg_run)
                if not pid:
                    crash_count += 1
                    log_write(logp, f"[crash] pid missing for pkg_run={pkg_run} at t={sec}, relaunch", True)
                    launch_app(serial, pkg_run)
                    time.sleep(1.0)

                remaining = end_ts - time.time()
                if remaining <= 0:
                    break
                    adb_to = min(8, max(1, int(remaining * 0.5)))

            cur_activity, cur_pkg = get_current_focus_fast(serial, pkg_run, timeout=adb_to)
            maybe_adopt_pkg(cur_pkg)

            if not cur_activity:
                cur_activity, cur_pkg = get_current_focus(serial, pkg_run, timeout=min(12, adb_to + 4))
                maybe_adopt_pkg(cur_pkg)

            if (not cur_activity) and (sec > STARTUP_GRACE_SEC):
                log_write(logp, f"[empty-act] t={sec} pkg={cur_pkg} -> relaunch", True)
                launch_app(serial, pkg_run)
                time.sleep(1.0)
                cur_activity, cur_pkg = get_current_focus(serial, pkg_run, timeout=10)
                maybe_adopt_pkg(cur_pkg)

            if cur_activity and (not in_app_pkg(cur_pkg, pkg_run)):
                if time.time() - last_escape_sec < ESCAPE_COOLDOWN_SEC:
                    pass
                else:
                    escape_count += 1
                    recover_if_escaped(serial, pkg_run, cur_activity, cur_pkg, logp, reason=f"t={sec}")
                    cur_activity, cur_pkg = get_current_focus_fast(serial, pkg_run, timeout=8)
                    maybe_adopt_pkg(cur_pkg)
                last_escape_sec = time.time()

            if cur_activity and in_app_pkg(cur_pkg, pkg_run):
                unique_acts.add(cur_activity)
                _mark_first_reach(first_reach, cur_activity, sec)

            xml_text = dump_ui_xml(serial, timeout=min(12, adb_to + 4))
            root = parse_ui_root(xml_text)
            ui_hash = ui_hash_normalized_from_root(root)

            if ui_hash:
                unique_states.add(ui_hash)
            recent_states.append((cur_activity, ui_hash))

            sample_elapsed = time.time() - sample_start
            if sample_elapsed > 1.0:
                slow_sample_count += 1
                slow_sample_total_sec += sample_elapsed
                slow_sample_max_sec = max(slow_sample_max_sec, sample_elapsed)
                log_write(logp, f"[slow-sample] t={sec} took {sample_elapsed:.2f}s (max={slow_sample_max_sec:.2f} total={slow_sample_total_sec:.2f} cnt={slow_sample_count})", True)

            if cur_activity != last_activity:
                act_change += 1
                unique_trans.add((last_activity, cur_activity))
                no_transition_streak = 0
            else:
                no_transition_streak += 1

            if ui_hash != last_state:
                ui_change += 1
            last_activity = cur_activity
            last_state = ui_hash

            clickables = parse_clickables_from_root(root) if root else []
            clickables = [n for n in clickables if not is_garbage_node(n)]
            clickables = [n for n in clickables if keep_candidate_by_package(n, pkg_run)]

            scrollables = parse_scrollables_from_root(root) if root else []
            scrollables = [n for n in scrollables if keep_candidate_by_package(n, pkg_run)]

            target_edge = pool.pick(cur_activity, now) if cur_activity else None
            ek = edge_key(target_edge.src, target_edge.dst) if target_edge else None

            candidates = clickables
            scored = score_candidates(candidates, target_edge)
            scored = [(s, n) for s, n in scored if s > -5.0]

            key_ui = (cur_activity, ui_hash)
            goal_node, goal_cat = pick_goal_first_node(candidates, key_ui, sec)

            cache_hit = pick_cache_any(cur_activity, sec) if cur_activity else None
            cache_ek, cache_rec = cache_hit if cache_hit else (None, None)

            action_taken = None
            scroll_used = False

            if pending is not None and pending.sec < sec:
                p = pending
                pending = None
                if p.src_act == cur_activity and p.src_ui == ui_hash:
                    pass
                else:
                    log_write(logp, f"[pending-drop] t={sec} src_act mismatch (pending={p.src_act} cur={cur_activity})", True)

                if p.kind == "tap":
                    if p.node and (not recently_tapped(p.src_act, p.node, sec)):
                        log_write(logp, f"[pending-tap] t={sec} ek={p.ek} target={p.target_edge.dst if p.target_edge else 'none'} (cx={p.node['cx']} cy={p.node['cy']})", True)
                        tap(serial, p.node["cx"], p.node["cy"])
                        events_total += 1
                        input_used = True
                        acted = True
                        mark_tapped(p.src_act, p.node, sec)
                        time.sleep(POST_INPUT_SETTLE_SEC)
                        post_input_escape_guard(sec, "pending_tap")

            if not acted and target_edge and cur_activity == target_edge.src and (not edge_on_cooldown(ek, sec)):
                note_target_try(target_edge)
                best_n = None
                for s, n in scored:
                    if not recently_tapped(cur_activity, n, sec):
                        best_n = n
                        break

                if best_n is None and goal_node and (not recently_tapped(cur_activity, goal_node, sec)):
                    best_n = goal_node

                if best_n is not None:
                    log_write(logp, f"[guided-edge] t={sec} {target_edge.src} -> {target_edge.dst} (p={target_edge.p:.3f}) tap (cx={best_n['cx']} cy={best_n['cy']})", True)
                    pending = PendingAction(
                        sec=sec,
                        src_act=cur_activity,
                        src_ui=ui_hash,
                        node=best_n,
                        ek=ek,
                        target_edge=target_edge,
                        kind="tap"
                    )
                    mark_tapped(cur_activity, best_n, sec)
                    mark_edge(ek, sec)
                    input_used = True
                    acted = True
                else:
                    pool.freeze_if_bad(target_edge)
                    log_write(logp, f"[guided-edge-no-cand] t={sec} {target_edge.src} -> {target_edge.dst} (no untapped candidates)", True)

            if not acted and cache_rec and cur_activity == cache_rec.get("src", "") and (not edge_on_cooldown(cache_ek, sec)):
                cache_node = find_node_for_action(candidates, cache_rec)
                if cache_node and (not recently_tapped(cur_activity, cache_node, sec)):
                    log_write(logp, f"[cache-hit] t={sec} {cache_ek} tap (cx={cache_node['cx']} cy={cache_node['cy']})", True)
                    pending = PendingAction(
                        sec=sec,
                        src_act=cur_activity,
                        src_ui=ui_hash,
                        node=cache_node,
                        ek=cache_ek,
                        kind="tap"
                    )
                    mark_tapped(cur_activity, cache_node, sec)
                    mark_edge(cache_ek, sec)
                    input_used = True
                    acted = True

            if not acted and goal_node and (not recently_tapped(cur_activity, goal_node, sec)):
                st = _get_cat_stat(key_ui, goal_cat) if goal_cat else None
                if st:
                    st.tries += 1
                    st.last_try_sec = sec
                log_write(logp, f"[goal-first] t={sec} cat={goal_cat} tap (cx={goal_node['cx']} cy={goal_node['cy']})", True)
                pending = PendingAction(
                    sec=sec,
                    src_act=cur_activity,
                    src_ui=ui_hash,
                    node=goal_node,
                    kind="goal-tap"
                )
                mark_tapped(cur_activity, goal_node, sec)
                goal_first_last_sec[key_ui] = sec
                input_used = True
                acted = True

            if not acted and can_scroll(cur_activity, ui_hash, sec) and scrollables:
                scroll_node = scrollables[0]
                log_write(logp, f"[scroll] t={sec} scroll (cx={scroll_node['cx']} cy={scroll_node['cy']}) down", True)
                swipe(serial, scroll_node["cx"], scroll_node["cy"] - 50, scroll_node["cx"], scroll_node["cy"] + 200, 400)
                events_total += 1
                mark_scroll(cur_activity, ui_hash, sec)
                input_used = True
                acted = True
                time.sleep(POST_INPUT_SETTLE_SEC)
                post_input_escape_guard(sec, "scroll")

            if not acted and len(scored) > 0 and no_candidate_streak < 3:
                best_s, best_n = scored[0]
                if not recently_tapped(cur_activity, best_n, sec):
                    log_write(logp, f"[fallback-tap] t={sec} score={best_s:.2f} tap (cx={best_n['cx']} cy={best_n['cy']})", True)
                    pending = PendingAction(
                        sec=sec,
                        src_act=cur_activity,
                        src_ui=ui_hash,
                        node=best_n,
                        kind="fallback-tap"
                    )
                    mark_tapped(cur_activity, best_n, sec)
                    input_used = True
                    acted = True
                else:
                    no_candidate_streak += 1
            else:
                no_candidate_streak = 0

            if cur_activity and last_snapshot[0] != cur_activity:
                if pending and pending.target_edge and pending.src_act == last_snapshot[0] and cur_activity == pending.target_edge.dst:
                    note_target_success(pending.target_edge, sec)
                    learn_action_for_transition(pending.src_act, cur_activity, pending.node, sec)

            series_acts.append(len(unique_acts))
            series_states.append(len(unique_states))
            series_trans.append(len(unique_trans))

            events_per_sec = safe_div(events_total, max(1, sec))
            act_change_rate = safe_div(act_change, max(1, sec))
            ui_change_rate = safe_div(ui_change, max(1, sec))

            row = PerSecondRow(
                t=sec,
                cur_activity=cur_activity,
                ui_hash=ui_hash,
                unique_activities=len(unique_acts),
                unique_ui_states=len(unique_states),
                unique_transitions=len(unique_trans),
                events_total=events_total,
                events_per_sec=events_per_sec,
                activity_change_rate=act_change_rate,
                ui_state_change_rate=ui_change_rate,
                crash_count=crash_count
            )

            if sec > last_written_t:
                w.writerow([getattr(row, k) for k in PerSecondRow.__annotations__.keys()])
                last_written_t = sec

            last_snapshot = (
                cur_activity, ui_hash, len(unique_acts), len(unique_states), len(unique_trans),
                events_total, act_change, ui_change, crash_count
            )

            if time.time() >= next_progress_ts:
                progress = (sec / duration) * 100.0
                log_write(logp, f"[progress] t={sec}/{duration}s ({progress:.1f}%) | acts={len(unique_acts)} | ui={len(unique_states)} | trans={len(unique_trans)} | events={events_total} | crashes={crash_count} | escapes={escape_count}", True)
                next_progress_ts = time.time() + 10.0

            if time.time() - last_save_ts > 30.0:
                save_action_map(action_map_path, action_map)
                last_save_ts = time.time()

    except KeyboardInterrupt:
    log_write(logp, "[interrupt] user pressed Ctrl+C, stopping gracefully", True)
except Exception as e:
log_write(logp, f"[error] unexpected exception: {type(e).__name__}: {e}", True)
import traceback
log_write(logp, traceback.format_exc(), True)
finally:
save_action_map(action_map_path, action_map)
log_write(logp, f"[done] duration={duration}s | acts={len(unique_acts)} | ui={len(unique_states)} | trans={len(unique_trans)} | events={events_total} | crashes={crash_count} | escapes={escape_count}", True)
log_write(logp, f"[slow-sample] total slow samples: {slow_sample_count} | max={slow_sample_max_sec:.2f}s | total slow time={slow_sample_total_sec:.2f}s", True)

with open(trans_path, "w", encoding="utf-8") as f:
    f.write("# Discovered transitions (src -> dst)\n")
    for (s, d) in sorted(unique_trans):
        f.write(f"{s} -> {d}\n")

auc_acts = auc_from_series(series_acts)
auc_states = auc_from_series(series_states)
auc_trans = auc_from_series(series_trans)

log_write(logp, f"[metrics] AUC(acts)={auc_acts:.2f} | AUC(states)={auc_states:.2f} | AUC(trans)={auc_trans:.2f}", True)

goal_reached = sum(1 for v in first_reach.values() if v is not None)
goal_total = len(first_reach)
goal_rate = safe_div(goal_reached, goal_total) * 100.0
log_write(logp, f"[goals] reached {goal_reached}/{goal_total} ({goal_rate:.1f}%)", True)

for ek, th in sorted(target_hits.items()):
    log_write(logp, f"[target-edge] {ek} | tries={th['tries']} | success={th['success']} | first_hit_t={th['first_hit_t']} | first_hit_tries={th['first_hit_tries']}", True)


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Guided Testing Runner")
    parser.add_argument("--serial", help="ADB device serial (optional)")
    parser.add_argument("--app-id", required=True, help="App ID (e.g., com.example.app)")
    parser.add_argument("--package", required=True, help="Target package name")
    parser.add_argument("--edges", required=True, help="Path to edges file (predicted transitions)")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds (default: 300)")
    parser.add_argument("--out-dir", required=True, help="Output directory for logs/metrics")
    parser.add_argument("--sample-interval", type=int, default=1, help="Sample interval in seconds (default: 1)")
    parser.add_argument("--activity-list", help="Path to activity list file (one activity per line)")
    parser.add_argument("--log", default="runner.log", help="Log file path (default: runner.log)")

    args = parser.parse_args()

    log_write(args.log, f"[start] Guided runner started | app-id={args.app_id} | package={args.package} | duration={args.duration}s", True)

    # Load edges
    edges_map = parse_edges_file(args.edges)
    edges = lookup_edges(edges_map, args.app_id, args.package)
    log_write(args.log, f"[edges] loaded {len(edges)} edges for app-id={args.app_id} (package={args.package})", True)

    # Load activity list
    activity_list = []
    if args.activity_list and os.path.exists(args.activity_list):
        with open(args.activity_list, "r", encoding="utf-8") as f:
            activity_list = [line.strip() for line in f if line.strip()]
    log_write(args.log, f"[activity] loaded {len(activity_list)} target activities", True)

    # Create edge pool
    pool = EdgePool(edges)

    # Run guided testing
    run_guided(
        serial=args.serial,
        app_id=args.app_id,
        package=args.package,
        pool=pool,
        duration=args.duration,
        out_dir=args.out_dir,
        sample_interval=args.sample_interval,
        logp=args.log,
        activity_list=activity_list
    )

    log_write(args.log, "[finish] Guided runner completed successfully", True)


if __name__ == "__main__":
    main()