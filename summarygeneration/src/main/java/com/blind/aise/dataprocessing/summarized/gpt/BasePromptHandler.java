package com.blind.aise.dataprocessing.summarized.gpt;

import com.alibaba.fastjson.JSONObject;
import com.blind.aise.dataprocessing.summarized.config.ATGConfig;

import java.io.IOException;

public abstract class BasePromptHandler<T> {
    public String getPurpose(T entity) throws IOException {
        String prompt = buildInitialPrompt(entity);
        String sessionId = "";

        for (int i = 0; i < ATGConfig.MAX_RETRY; i++) {
            JSONObject response = GPTClient.sendRequest(prompt, sessionId);
            if (!response.getBooleanValue("success")) {
                continue;
            }

            JSONObject data = response.getJSONObject("data");
            sessionId = data.getString("sessionId");
            String resultJson = data.getString("chat")
                    .replace("```json", "")
                    .replace("```", "")
                    .trim();
            JSONObject result = JSONObject.parseObject(resultJson);

            if (validateResult(entity, result)) {
                return result.getString("purpose");
            }

            prompt = buildRetryPrompt(entity);
        }

        throw new IOException( getEntityId(entity));
    }

    protected abstract String buildInitialPrompt(T entity);

    protected abstract String buildRetryPrompt(T entity);

    protected abstract boolean validateResult(T entity, JSONObject result);

    protected abstract String getEntityId(T entity);
}