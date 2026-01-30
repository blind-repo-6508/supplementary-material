package com.blind.aise.dataprocessing.summarized.io;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.blind.aise.dataprocessing.summarized.data.Activity;
import com.blind.aise.dataprocessing.summarized.data.Edge;
import com.blind.aise.dataprocessing.summarized.data.Widget;
import com.blind.aise.dataprocessing.summarized.util.JsonUtils;
import com.blind.aise.dataprocessing.summarized.util.StringUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class AppDataReader {
    public static JSONObject readAppRawData(File jsonFile) {
        return JsonUtils.parseFileToJsonObject(jsonFile);
    }

    public static List<Activity> parseActivities(JSONObject appData) {
        JSONArray nodes = appData.getJSONArray("nodes");
        return nodes.stream()
                .map(node -> JsonUtils.convert((JSONObject) node, Activity.class))
                .collect(Collectors.toList());
    }

    public static List<Edge> parseEdges(JSONObject appData) {
        JSONArray edges = appData.getJSONArray("edges");
        return edges.stream()
                .map(edge -> JsonUtils.convert((JSONObject) edge, Edge.class))
                .collect(Collectors.toList());
    }

    public static List<Widget> parseWidgets(List<Edge> edges) {
        Map<String, Widget> widgetMap = new LinkedHashMap<>();

        for (Edge edge : edges) {
            if (edge.isHasWidget() && edge.getWidgetGuid() != null && !edge.getWidgetGuid().isEmpty()) {
                String widgetGuid = edge.getWidgetGuid();
                if (!widgetMap.containsKey(widgetGuid)) {
                    Widget widget = new Widget();
                    widget.setWidgetGuid(widgetGuid);
                    widget.setWidgetViewClass(edge.getWidgetViewClass());
                    widget.setRawData(edge.getWidgetRawData());
                    widget.setListeners(StringUtils.formatListeners(edge.getListeners()).replace("void", ""));
                    widget.setApiCalls(StringUtils.formatApiCalls(edge.getApiCalls()));
                    widgetMap.put(widgetGuid, widget);
                }
            }
        }

        return new ArrayList<>(widgetMap.values());
    }
}