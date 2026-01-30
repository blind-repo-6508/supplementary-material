package com.blind.aise.dataprocessing.summarized.gpt;

import com.alibaba.fastjson.JSONObject;
import com.blind.aise.dataprocessing.summarized.data.Activity;

public class ActivityPromptHandler extends BasePromptHandler<Activity> {

    @Override
    protected String buildInitialPrompt(Activity activity) {
        return PromptBuilder.buildActivityPrompt(
                activity.getActivityFullname(),
                activity.getNodeId(),
                activity.getViewInfo()
        );
    }

    @Override
    protected String buildRetryPrompt(Activity activity) {
        return PromptBuilder.buildRetryActivityPrompt(
                activity.getActivityFullname(),
                activity.getNodeId(),
                activity.getViewInfo()
        );
    }

    @Override
    protected boolean validateResult(Activity activity, JSONObject result) {
        return result.getString("activity_id").equals(activity.getNodeId())
                && result.getString("activity_name").equals(activity.getActivityFullname());
    }

    @Override
    protected String getEntityId(Activity activity) {
        return activity.getNodeId();
    }
}