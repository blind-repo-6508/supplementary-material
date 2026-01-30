package com.blind.aise.dataprocessing.summarized.io;

import com.blind.aise.dataprocessing.summarized.config.ATGConfig;
import com.blind.aise.dataprocessing.summarized.data.Widget;
import com.blind.aise.dataprocessing.summarized.gpt.WidgetPromptHandler;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class WidgetInfoWriter {
    private static final WidgetPromptHandler widgetPromptHandler = new WidgetPromptHandler();

    public static void writeWidgetInfoFile(String appPackage, List<Widget> widgets) throws IOException {
        File outputDir = new File(ATGConfig.ATG_WIDGETS_DIR);

        File outputFile = new File(outputDir, appPackage + ".txt");

        try (FileWriter writer = new FileWriter(outputFile)) {
            for (Widget widget : widgets) {
                String widgetId = widget.getWidgetGuid();
                String purpose = widgetPromptHandler.getPurpose(widget);
                writer.write(">" + widgetId + "\n");
                writer.write(buildWidgetInfoLine(widget, purpose) + "\n");
            }
        }
    }

    private static String buildWidgetInfoLine(Widget widget, String purpose) {
        String listeners = widget.getListeners() == null ? "" : widget.getListeners();
        String apiCalls = widget.getApiCalls() == null ? "" : widget.getApiCalls();
        return String.format("%s;%s;%s;%s", widget.getWidgetViewClass(), purpose, listeners, apiCalls);
    }
}