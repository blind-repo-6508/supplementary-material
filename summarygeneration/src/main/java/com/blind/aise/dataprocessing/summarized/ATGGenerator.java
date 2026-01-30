package com.blind.aise.dataprocessing.summarized;

import com.alibaba.fastjson.JSONArray;
import com.blind.aise.dataprocessing.summarized.config.ATGConfig;
import com.blind.aise.dataprocessing.summarized.data.Activity;
import com.blind.aise.dataprocessing.summarized.data.Edge;
import com.blind.aise.dataprocessing.summarized.data.Widget;
import com.blind.aise.dataprocessing.summarized.gpt.GPTClient;
import com.blind.aise.dataprocessing.summarized.io.ATGIndexWriter;
import com.blind.aise.dataprocessing.summarized.io.ActivityInfoWriter;
import com.blind.aise.dataprocessing.summarized.io.AppDataReader;
import com.blind.aise.dataprocessing.summarized.io.WidgetInfoWriter;
import com.blind.aise.dataprocessing.summarized.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class ATGGenerator {
    private static final Logger logger = LoggerFactory.getLogger(ATGGenerator.class);

    public static void main(String[] args) {
        try {
            List<String> targetFileNames = loadTargetFileNames();
            if (targetFileNames.isEmpty()) {
                return;
            }

            Set<String> processedApps = loadProcessedApps();

            File rawDir = new File(ATGConfig.RAW_DATA_DIR);
            List<File> toProcessFiles = filterToProcessFiles(rawDir, targetFileNames, processedApps);
            if (toProcessFiles.isEmpty()) {
                return;
            }

            int totalCount = toProcessFiles.size();
            for (int i = 0; i < totalCount; i++) {
                File jsonFile = toProcessFiles.get(i);
                String appPackage = jsonFile.getName().replace(".json", "");
                int currentIndex = i + 1;

                if (processedApps.contains(appPackage)) {
                    continue;
                }

                try {

                    var appData = AppDataReader.readAppRawData(jsonFile);
                    List<Activity> activities = AppDataReader.parseActivities(appData);
                    List<Edge> edges = AppDataReader.parseEdges(appData);
                    List<Widget> widgets = AppDataReader.parseWidgets(edges);
                    ATGIndexWriter.writeIndexFile(appPackage, activities, edges);
                    ActivityInfoWriter.writeActivityInfoFile(appPackage, activities);
                    WidgetInfoWriter.writeWidgetInfoFile(appPackage, widgets);
                    markAsProcessed(appPackage);

                } catch (IOException e) {
                    String errorMsg = buildErrorMsg(currentIndex, totalCount, appPackage, e);
                    logger.error(errorMsg, e);
                    writeErrorLog(errorMsg, e);

                    if (e instanceof GPTClient.TokenExhaustedException) {
                        return;
                    }
                } catch (Exception e) {
                    String errorMsg = buildErrorMsg(currentIndex, totalCount, appPackage, e);
                    logger.error(errorMsg, e);
                    writeErrorLog(errorMsg, e);
                }
            }


        } catch (GPTClient.TokenExhaustedException e) {
            String errorMsg = e.getMessage();
            logger.error(errorMsg, e);
            writeErrorLog(errorMsg, e);
        } catch (Exception e) {
            String errorMsg = e.getMessage();
            logger.error(errorMsg, e);
            writeErrorLog(errorMsg, e);
        }
    }

    private static String buildErrorMsg(int currentIndex, int totalCount, String appPackage, Throwable e) {
        return new StringJoiner(" ")
                .add(appPackage)
                .add(e.getMessage())
                .toString();
    }

    private static void writeErrorLog(String errorMsg, Throwable e) {
        File errorFile = new File(ATGConfig.ERROR_LOG_FILE);
        if (!errorFile.exists()) {
            errorFile.getParentFile().mkdirs();
        }

        try (PrintWriter writer = new PrintWriter(new FileWriter(errorFile, true))) {
            writer.println("[" + new java.util.Date() + "] " + errorMsg);
            e.printStackTrace(writer);
            writer.println("----------------------------------------");
        } catch (IOException ex) {
            logger.error(ex.getMessage(), ex);
        }
    }

    private static List<String> loadTargetFileNames() throws IOException {
        File fileListJson = new File(ATGConfig.TARGET_FILELIST);

        JSONArray jsonArray = JsonUtils.parseFileToJsonArray(fileListJson);
        return jsonArray.toJavaList(String.class);
    }

    private static Set<String> loadProcessedApps() throws IOException {
        Set<String> processedApps = new HashSet<>();
        File recordFile = new File(ATGConfig.PROCESSED_RECORD_FILE);

        if (!recordFile.exists()) {
            recordFile.getParentFile().mkdirs();
            recordFile.createNewFile();
            return processedApps;
        }

        try (java.util.Scanner scanner = new java.util.Scanner(recordFile)) {
            while (scanner.hasNextLine()) {
                String appPackage = scanner.nextLine().trim();
                if (!appPackage.isEmpty()) {
                    processedApps.add(appPackage);
                }
            }
        }
        return processedApps;
    }

    private static void markAsProcessed(String appPackage) throws IOException {
        try (FileWriter writer = new FileWriter(ATGConfig.PROCESSED_RECORD_FILE, true)) {
            writer.write(appPackage + System.lineSeparator());
        }
    }

    private static List<File> filterToProcessFiles(File rawDir, List<String> targetFileNames, Set<String> processedApps) {
        List<File> toProcessFiles = new ArrayList<>();


        for (String fileName : targetFileNames) {
            if (!fileName.endsWith(".json")) {
                continue;
            }
            File jsonFile = new File(rawDir, fileName);
            String appPackage = fileName.replace(".json", "");

            if (jsonFile.exists() && !processedApps.contains(appPackage)) {
                toProcessFiles.add(jsonFile);
            }
        }
        return toProcessFiles;
    }
}