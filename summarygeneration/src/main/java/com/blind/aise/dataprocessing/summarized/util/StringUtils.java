package com.blind.aise.dataprocessing.summarized.util;

import com.alibaba.fastjson.JSONArray;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class StringUtils {
    public static String formatListeners(JSONArray listeners) {
        return formatFunctions(listeners);
    }

    public static String formatApiCalls(JSONArray apiCalls) {
        return formatFunctions(apiCalls);
    }

    private static String formatFunctions(JSONArray functionList) {
        if (functionList == null || functionList.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();

        for (Object apiObj : functionList) {
            String api = apiObj.toString().trim();
            String cleanStr = removeAngleBrackets(api);
            if (cleanStr.isEmpty()) {
                continue;
            }

            String[] classAndMethod = cleanStr.split(": ", 2);
            if (classAndMethod.length < 2) {
                continue;
            }
            String methodSignature = classAndMethod[1];

            // 步骤3：简化「返回值类型」和「参数类型」
            String simplifiedSignature = simplifyReturnTypeAndParams(methodSignature);

            sb.append(simplifiedSignature).append(",");
        }

        return !sb.isEmpty() ? sb.substring(0, sb.length() - 1) : "";
    }

    private static String simplifyReturnTypeAndParams(String methodSignature) {
        Pattern returnTypePattern = Pattern.compile("^(.*?)\\s+(\\w+\\(.*\\))$");
        Matcher returnMatcher = returnTypePattern.matcher(methodSignature);

        if (returnMatcher.find()) {
            String returnType = returnMatcher.group(1);
            String simplifiedReturnType = simplifySingleType(returnType);
            String methodNameWithParams = returnMatcher.group(2);
            String simplifiedMethodWithParams = simplifyMethodParams(methodNameWithParams);
            return simplifiedReturnType + " " + simplifiedMethodWithParams;
        }

        return simplifyMethodParams(methodSignature);
    }

    private static String simplifySingleType(String fullType) {
        if (fullType == null || fullType.isEmpty()) {
            return fullType;
        }
        String typeWithoutArray = fullType.replace("[]", "");
        String[] typeParts = typeWithoutArray.split("\\.");
        String simpleType = typeParts[typeParts.length - 1];
        return fullType.contains("[]") ? simpleType + "[]" : simpleType;
    }

    private static String simplifyMethodParams(String methodNameWithParams) {
        Pattern paramPattern = Pattern.compile("\\((.*?)\\)");
        Matcher matcher = paramPattern.matcher(methodNameWithParams);

        StringBuilder simplified = new StringBuilder();
        while (matcher.find()) {
            String params = matcher.group(1);
            String simplifiedParams = simplifyParamList(params);
            String escapedReplacement = simplifiedParams.replace("$", "\\$"); // 转义$
            matcher.appendReplacement(simplified, "(" + escapedReplacement + ")");
        }
        matcher.appendTail(simplified);
        return simplified.toString();
    }

    private static String simplifyParamList(String paramList) {
        if (paramList.isEmpty()) {
            return "";
        }

        StringBuilder simplifiedParams = new StringBuilder();
        String[] params = paramList.split(",");
        for (String param : params) {
            String trimmedParam = param.trim();
            String simpleParam = simplifySingleType(trimmedParam);
            simplifiedParams.append(simpleParam).append(",");
        }

        return !simplifiedParams.isEmpty() ? simplifiedParams.substring(0, simplifiedParams.length() - 1) : "";
    }

    private static String removeAngleBrackets(String str) {
        if (str.startsWith("<") && str.endsWith(">")) {
            return str.substring(1, str.length() - 1);
        }
        return str;
    }
}