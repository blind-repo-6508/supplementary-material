package com.blind.aise.dataprocessing.summarized.io;

import com.blind.aise.dataprocessing.summarized.config.ATGConfig;
import com.blind.aise.dataprocessing.summarized.data.Activity;
import com.blind.aise.dataprocessing.summarized.data.Edge;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class ATGIndexWriter {
    public static void writeIndexFile(String appPackage, List<Activity> activities, List<Edge> edges) throws IOException {
        File outputDir = new File(ATGConfig.ATG_INDEX_DIR);
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IOException("创建ATG索引目录失败: " + outputDir.getAbsolutePath());
        }

        Map<String, String> fullnameToId = new HashMap<>();
        for (Activity act : activities) {
            fullnameToId.put(act.getActivityFullname(), act.getNodeId());
        }

        Map<String, Set<String>> edgeWidgetMap = new HashMap<>();
        for (Edge edge : edges) {
            String sourceFullname = edge.getSourceActivity();
            String targetFullname = edge.getTargetActivity();

            if (!fullnameToId.containsKey(sourceFullname) || !fullnameToId.containsKey(targetFullname)) {
                continue;
            }


            String widgetId = edge.isHasWidget() && edge.getWidgetGuid() != null && !edge.getWidgetGuid().isEmpty()
                    ? edge.getWidgetGuid()
                    : "NONE_WIDGET";

            String edgeKey = sourceFullname + "#" + targetFullname;
            edgeWidgetMap.computeIfAbsent(edgeKey, k -> new HashSet<>()).add(widgetId);
        }

        try (FileWriter writer = new FileWriter(new File(outputDir, appPackage + ".txt"))) {
            for (Activity source : activities) {
                for (Activity target : activities) {
                    String sourceId = source.getNodeId();
                    String targetId = target.getNodeId();
                    String sourceFullname = source.getActivityFullname();
                    String targetFullname = target.getActivityFullname();

                    String edgeKey = sourceFullname + "#" + targetFullname;
                    Set<String> widgetIds = edgeWidgetMap.getOrDefault(edgeKey, Collections.emptySet());

                    if (widgetIds.isEmpty()) {
                        writer.write(String.format("%s;%s;%s;%d%n",
                                sourceId, targetId, "NONE_WIDGET", 0));
                    } else {
                        for (String singleWidgetId : widgetIds) {
                            writer.write(String.format("%s;%s;%s;%d%n", sourceId, targetId, singleWidgetId, 1));
                        }
                    }
                }
            }
        }
    }
}