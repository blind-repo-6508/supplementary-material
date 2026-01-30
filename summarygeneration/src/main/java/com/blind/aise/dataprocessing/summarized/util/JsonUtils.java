package com.blind.aise.dataprocessing.summarized.util;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class JsonUtils {
    public static Object parseFileToJson(File file) {
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] data = new byte[(int) file.length()];
            fis.read(data);
            String jsonStr = new String(data, StandardCharsets.UTF_8);
            if (jsonStr.trim().startsWith("[")) {
                return JSON.parseArray(jsonStr);
            } else {
                return JSON.parseObject(jsonStr);
            }
        } catch (IOException e) {
            throw new RuntimeException("解析JSON文件失败: " + file.getAbsolutePath(), e);
        }
    }

    public static JSONArray parseFileToJsonArray(File file) {
        Object json = parseFileToJson(file);
        if (json instanceof JSONArray) {
            return (JSONArray) json;
        } else {
            throw new RuntimeException(file.getAbsolutePath());
        }
    }

    public static JSONObject parseFileToJsonObject(File file) {
        Object json = parseFileToJson(file);
        if (json instanceof JSONObject) {
            return (JSONObject) json;
        } else {
            throw new RuntimeException(file.getAbsolutePath());
        }
    }

    public static <T> T convert(JSONObject json, Class<T> clazz) {
        return JSON.toJavaObject(json, clazz);
    }
}