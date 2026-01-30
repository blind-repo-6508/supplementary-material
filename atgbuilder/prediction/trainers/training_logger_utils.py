# src/prediction/trainers/training_logger_utils.py
import logging
import os
from datetime import datetime


class TrainingLoggerUtils:

    @staticmethod
    def setup_logger(log_root: str, log_tag: str, log_class: str) -> logging.Logger:
        sub_dir = os.path.join(log_root, log_tag)
        os.makedirs(sub_dir, exist_ok=True)

        log_file = os.path.join(
            sub_dir,
            f"{log_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        if root.handlers:
            root.handlers.clear()
        root.addHandler(fh)
        root.addHandler(ch)

        logger = logging.getLogger(log_class)
        logger.setLevel(logging.INFO)
        logger.propagate = True  # 让它也走 root handlers

        logger.info("📂 日志文件: %s", log_file)
        return logger