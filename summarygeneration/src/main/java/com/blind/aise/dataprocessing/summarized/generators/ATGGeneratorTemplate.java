package com.blind.aise.dataprocessing.summarized.generators;

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

public class ATGGeneratorTemplate {
    private static final Logger logger = LoggerFactory.getLogger(ATGGeneratorTemplate.class);

    public static final int SPLIT_COUNT = 10;

    public static void template(int index) {
        try {
            List<String> totalFileNames = loadTargetFileNames();

            List<String> targetFileNames = splitFileNames(index, totalFileNames);
            if (targetFileNames.isEmpty()) {
                return;
            }

            File rawDir = new File(ATGConfig.RAW_DATA_DIR);
            List<File> toProcessFiles = filterToProcessFiles(rawDir, targetFileNames);
            if (toProcessFiles.isEmpty()) {
                return;
            }

            int totalCount = toProcessFiles.size();
            for (int i = 0; i < totalCount; i++) {
                File jsonFile = toProcessFiles.get(i);
                String appPackage = jsonFile.getName().replace(".json", "");
                int currentIndex = i + 1;

                if (areAllOutputFilesExist(appPackage)) {
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


    private static boolean areAllOutputFilesExist(String appPackage) {
        File indexFile = new File(ATGConfig.ATG_INDEX_DIR + "/" + appPackage + ".txt");
        File activityFile = new File(ATGConfig.ATG_ACTIVITIES_DIR + "/" + appPackage + ".txt");
        File widgetFile = new File(ATGConfig.ATG_WIDGETS_DIR + "/" + appPackage + ".txt");

        boolean isIndexValid = indexFile.exists() && indexFile.length() > 0;
        boolean isActivityValid = activityFile.exists() && activityFile.length() > 0;
        if (!isIndexValid || !isActivityValid) {
            return false;
        }

        boolean isWidgetValid;
        if (widgetFile.exists() && widgetFile.length() > 0) {
            isWidgetValid = true;
        } else if (!widgetFile.exists() || widgetFile.length() == 0) {
            isWidgetValid = isIndexAllNoneWidget(indexFile);
        } else {
            isWidgetValid = false;
        }

        return isWidgetValid;
    }

    private static boolean isIndexAllNoneWidget(File indexFile) {
        try (Scanner scanner = new Scanner(indexFile)) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(";");
                if (parts.length != 4) {
                    return false;
                }
                if (!"NONE_WIDGET".equals(parts[2].trim())) {
                    return false;
                }
            }
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private static List<String> splitFileNames(int index, List<String> totalFileNames) {
        int totalSize = totalFileNames.size();
        int baseSize = totalSize / ATGGeneratorTemplate.SPLIT_COUNT;
        int remainder = totalSize % ATGGeneratorTemplate.SPLIT_COUNT;

        int startIndex;
        if (index < remainder) {
            startIndex = index * (baseSize + 1);
        } else {
            startIndex = remainder * (baseSize + 1) + (index - remainder) * baseSize;
        }

        int endIndex;
        if (index < remainder) {
            endIndex = startIndex + baseSize + 1;
        } else {
            endIndex = startIndex + baseSize;
        }
        endIndex = Math.min(endIndex, totalSize);

        return totalFileNames.subList(startIndex, endIndex);
    }

    private static String buildErrorMsg(int currentIndex, int totalCount, String appPackage, Throwable e) {
        return new StringJoiner(" ")
                .add("" + currentIndex + "/" + totalCount + "）")
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
            writer.println("[" + new Date() + "] " + errorMsg);
            e.printStackTrace(writer);
            writer.println("----------------------------------------");
        } catch (IOException ex) {
            logger.error(ex.getMessage(), ex);
        }
    }

    private static List<String> loadTargetFileNames() throws IOException {
        File fileListJson = new File(ATGConfig.TARGET_FILELIST);
        if (!fileListJson.exists()) {
            throw new IOException(ATGConfig.TARGET_FILELIST);
        }
        JSONArray jsonArray = JsonUtils.parseFileToJsonArray(fileListJson);
        List<String> targetFileNames = jsonArray.toJavaList(String.class);

        return targetFileNames.stream().sorted().toList();
    }

    private static List<File> filterToProcessFiles(File rawDir, List<String> targetFileNames) {
        List<File> toProcessFiles = new ArrayList<>();

        if (!rawDir.exists() || !rawDir.isDirectory()) {
            return toProcessFiles;
        }

        for (String fileName : targetFileNames) {
            if (!fileName.endsWith(".json")) {
                continue;
            }
            File jsonFile = new File(rawDir, fileName);

            if (jsonFile.exists()) {
                toProcessFiles.add(jsonFile);
            }
        }
        return toProcessFiles;
    }
}