package com.blind.aise.dataprocessing.summarized.gpt;

import com.alibaba.fastjson.JSONObject;
import com.blind.aise.dataprocessing.summarized.data.Widget;

public class WidgetPromptHandler extends BasePromptHandler<Widget> {

    @Override
    protected String buildInitialPrompt(Widget widget) {
        return PromptBuilder.buildWidgetPrompt(
                widget.getWidgetViewClass(),
                widget.getWidgetGuid(),
                widget.getRawData()
        );
    }

    @Override
    protected String buildRetryPrompt(Widget widget) {
        return PromptBuilder.buildRetryWidgetPrompt(
                widget.getWidgetViewClass(),
                widget.getWidgetGuid(),
                widget.getRawData()
        );
    }

    @Override
    protected boolean validateResult(Widget widget, JSONObject result) {
        return result.getString("widget_id").equals(widget.getWidgetGuid())
                && result.getString("widget_type").equals(widget.getWidgetViewClass());
    }

    @Override
    protected String getEntityId(Widget widget) {
        return widget.getWidgetGuid();
    }
}