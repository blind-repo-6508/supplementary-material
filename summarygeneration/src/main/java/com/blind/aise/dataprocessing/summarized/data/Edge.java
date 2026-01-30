package com.blind.aise.dataprocessing.summarized.data;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import lombok.Data;

@Data
public class Edge {
    private String edgeId;
    private String sourceActivity;
    private String targetActivity;
    private boolean hasWidget;
    private String widgetGuid;
    private String widgetViewClass;
    private JSONObject widgetRawData;
    private JSONArray listeners;
    private JSONArray apiCalls;
}