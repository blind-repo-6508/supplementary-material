package com.blind.aise.dataprocessing.summarized.config;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class ATGConfig {
    private static final Properties props = new Properties();

    static {
        try (InputStream is = ATGConfig.class.getClassLoader().getResourceAsStream("application.properties")) {

            props.load(is);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static final String RAW_DATA_DIR = getProperty("raw.data.dir");

    public static final String ATG_INDEX_DIR = getProperty("atg.index.dir");
    public static final String ATG_ACTIVITIES_DIR = getProperty("atg.activities.dir");
    public static final String ATG_WIDGETS_DIR = getProperty("atg.widgets.dir");

    public static final String GPT_API_URL = getProperty("gpt.api.url");
    public static final String API_KEY = getProperty("gpt.api.key");
    public static final String MODEL_TYPE = getProperty("gpt.model.type");
    public static final int MAX_RETRY = Integer.parseInt(getProperty("gpt.max.retry"));

    public static final String PROCESSED_RECORD_FILE = getProperty("processed.record.file");
    public static final String TARGET_FILELIST = getProperty("target.filelist");

    public static final String ERROR_LOG_FILE = getProperty("error.log.file");

    private static String getProperty(String key) {
        String value = props.getProperty(key);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException(key);
        }
        return value.trim();
    }
}