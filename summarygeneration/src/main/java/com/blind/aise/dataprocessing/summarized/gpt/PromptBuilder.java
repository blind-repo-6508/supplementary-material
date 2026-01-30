package com.blind.aise.dataprocessing.summarized.gpt;

import com.alibaba.fastjson.JSONObject;
import org.yaml.snakeyaml.Yaml;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Map;
import java.util.Objects;


public class PromptBuilder {

    private static final String PROMPT_TEMPLATE_YAML = "";


    public static String buildActivityPrompt(String activityName, String activityId, JSONObject structure) {
        validateStructure(structure);
        return buildActivityPrompt(activityName, activityId, structure.toString());
    }


    public static String buildActivityPrompt(String activityName, String activityId, String structure) {
        return loadAndReplaceTemplate(
                "activity_prompt_template",
                activityId,
                activityName,
                structure
        );
    }


    public static String buildRetryActivityPrompt(String activityName, String activityId, JSONObject structure) {
        validateStructure(structure);
        return buildRetryActivityPrompt(activityName, activityId, structure.toString());
    }


    public static String buildRetryActivityPrompt(String activityName, String activityId, String structure) {
        return loadAndReplaceTemplate(
                "retry_prompt_template",
                activityId,
                activityName,
                structure
        );
    }


    public static String buildWidgetPrompt(String widgetType, String widgetId, JSONObject content) {
        validateStructure(content);
        return buildWidgetPrompt(widgetType, widgetId, content.toString());
    }

    public static String buildWidgetPrompt(String widgetType, String widgetId, String content) {
        return loadAndReplaceTemplate(
                "widget_prompt_template",
                widgetId,
                widgetType,
                content
        );
    }


    public static String buildRetryWidgetPrompt(String widgetType, String widgetId, JSONObject content) {
        validateStructure(content);
        return buildRetryWidgetPrompt(widgetType, widgetId, content.toString());
    }


    public static String buildRetryWidgetPrompt(String widgetType, String widgetId, String content) {
        return loadAndReplaceTemplate(
                "retry_widget_prompt_template",
                widgetId,
                widgetType,
                content
        );
    }


    private static String loadAndReplaceTemplate(String templateKey, String id, String name, String structure) {
        Objects.requireNonNull(id, "ID cannot be null");
        Objects.requireNonNull(name, "Name/Type cannot be null");
        Objects.requireNonNull(structure, "Structure/Content cannot be null");

        try (InputStream inputStream = new FileInputStream(PROMPT_TEMPLATE_YAML)) {
            Map<String, String> yamlData = new Yaml().load(inputStream);
            String template = yamlData.get(templateKey);

            if (template == null || template.trim().isEmpty()) {
                throw new IllegalArgumentException("YAML file missing template: " + templateKey);
            }

            return template.replace("[MASK_ID]", id)
                    .replace(
                            templateKey.contains("widget") ? "[MASK_TYPE]" : "[MASK_NAME]",
                            name
                    )
                    .replace(
                            templateKey.contains("widget") ? "[MASK_CONTENT]" : "[MASK_STRUCTURE]",
                            structure
                    );

        } catch (Exception e) {
            throw new RuntimeException("Failed to build prompt for template: " + templateKey, e);
        }
    }

    private static void validateStructure(JSONObject structure) {
        if (structure == null) {
            throw new IllegalArgumentException("Structure/Content JSON cannot be null");
        }
    }
}