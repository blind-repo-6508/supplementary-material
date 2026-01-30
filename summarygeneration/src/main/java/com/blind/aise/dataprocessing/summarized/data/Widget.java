package com.blind.aise.dataprocessing.summarized.data;

import com.alibaba.fastjson.JSONObject;
import lombok.Data;

@Data
public class Widget {
    private String widgetGuid;
    private String widgetViewClass;
    private String listeners;
    private String apiCalls;
    private JSONObject rawData;
}