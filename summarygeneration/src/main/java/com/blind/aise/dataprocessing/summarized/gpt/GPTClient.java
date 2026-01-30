package com.blind.aise.dataprocessing.summarized.gpt;

import com.alibaba.fastjson.JSONObject;
import com.blind.aise.dataprocessing.summarized.config.ATGConfig;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.io.IOException;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;
import java.util.concurrent.TimeUnit;

public class GPTClient {
    private static final Logger logger = LoggerFactory.getLogger(GPTClient.class);

    private static final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(3, TimeUnit.SECONDS)
            .sslSocketFactory(createSSLSocketFactory(), new X509TrustManager() {
                @Override
                public void checkClientTrusted(X509Certificate[] chain, String authType) {
                }

                @Override
                public void checkServerTrusted(X509Certificate[] chain, String authType) {
                }

                @Override
                public X509Certificate[] getAcceptedIssuers() {
                    return new X509Certificate[0];
                }
            })
            .hostnameVerifier((hostname, session) -> true)
            .build();

    private static SSLSocketFactory createSSLSocketFactory() {
        try {
            SSLContext sslContext = SSLContext.getInstance("TLS");
            sslContext.init(null, new TrustManager[]{new X509TrustManager() {
                @Override
                public void checkClientTrusted(X509Certificate[] chain, String authType) {
                }

                @Override
                public void checkServerTrusted(X509Certificate[] chain, String authType) {
                }

                @Override
                public X509Certificate[] getAcceptedIssuers() {
                    return new X509Certificate[0];
                }
            }}, new SecureRandom());
            return sslContext.getSocketFactory();
        } catch (Exception e) {
            throw new RuntimeException( e);
        }
    }

    public static JSONObject sendRequest(String prompt, String sessionId) throws IOException, TokenExhaustedException {
        int retryCount = 0;
        IOException lastException = null;

        while (retryCount < ATGConfig.MAX_RETRY) {
            try {
                JSONObject requestBody = new JSONObject();
                requestBody.put("modelType", ATGConfig.MODEL_TYPE);
                requestBody.put("sessionId", sessionId);
                requestBody.put("message", prompt);

                Request request = new Request.Builder()
                        .url(ATGConfig.GPT_API_URL)
                        .addHeader("Content-Type", "application/json")
                        .addHeader("Authorization", "Bearer " + ATGConfig.API_KEY)
                        .post(RequestBody.create(
                                requestBody.toString(),
                                MediaType.parse("application/json; charset=utf-8")
                        ))
                        .build();

                logger.info(String.valueOf(client.newCall(request).execute()));


                try (Response response = client.newCall(request).execute()) {


                    ResponseBody body = response.body();


                    String responseBody = body.string();
                    JSONObject result = JSONObject.parseObject(responseBody);

                    return result;
                } catch (IOException e) {
                    throw e;
                }

            } catch (TokenExhaustedException e) {
                throw e;
            } catch (IOException e) {
                lastException = e;

                retryCount++;

                if (retryCount < ATGConfig.MAX_RETRY) {
                    try {

                        long sleepMillis = (long) Math.pow(2, retryCount) * 1000;
                        Thread.sleep(sleepMillis);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        }

        throw new IOException(
                lastException
        );
    }


    public static class TokenExhaustedException extends IOException {
        public TokenExhaustedException(String message) {
            super(message);
        }
    }
}