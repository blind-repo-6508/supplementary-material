package com.blind.aise.dataprocessing.summarized.data;

import com.alibaba.fastjson.JSONObject;
import lombok.Data;

@Data
public class Activity {
    private String activityFullname;
    private String nodeId;
    private JSONObject viewInfo;
    private boolean inTransition;
}