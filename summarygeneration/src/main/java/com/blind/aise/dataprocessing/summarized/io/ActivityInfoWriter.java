package com.blind.aise.dataprocessing.summarized.io;

import com.blind.aise.dataprocessing.summarized.config.ATGConfig;
import com.blind.aise.dataprocessing.summarized.data.Activity;
import com.blind.aise.dataprocessing.summarized.gpt.ActivityPromptHandler;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class ActivityInfoWriter {
    private static final ActivityPromptHandler activityPromptHandler = new ActivityPromptHandler();

    public static void writeActivityInfoFile(String appPackage, List<Activity> activities) throws IOException {
        File outputDir = new File(ATGConfig.ATG_ACTIVITIES_DIR);

        File outputFile = new File(outputDir, appPackage + ".txt");

        try (FileWriter writer = new FileWriter(outputFile)) {
            for (Activity activity : activities) {
                String activityId = activity.getNodeId();
                String purpose = activityPromptHandler.getPurpose(activity);
                writer.write(">" + activityId + "\n");
                writer.write(buildActivityInfoLine(activity, purpose) + "\n");
            }
        }
    }

    private static String buildActivityInfoLine(Activity activity, String purpose) {
        String fullname = activity.getActivityFullname() == null ? "UNKNOWN_ACTIVITY" : activity.getActivityFullname();
        String safePurpose = purpose == null ? "FAILED_TO_GET_PURPOSE" : purpose;
        return String.format("%s;%s", fullname, safePurpose);
    }
}