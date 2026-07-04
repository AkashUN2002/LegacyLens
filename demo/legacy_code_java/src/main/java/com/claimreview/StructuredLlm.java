package com.claimreview;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatModel;

import java.util.List;

/**
 * Structured-output binding: mirrors LangChain's `llm.with_structured_output(Model)`.
 * Sends the messages, requests strict JSON, and parses the reply into the target type.
 */
public class StructuredLlm<T> {

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    private final ChatModel model;
    private final Class<T> type;

    public StructuredLlm(ChatModel model, Class<T> type) {
        this.model = model;
        this.type = type;
    }

    public T invoke(List<ChatMessage> messages) {
        AiMessage reply = model.chat(messages).aiMessage();
        String text = reply.text() == null ? "" : reply.text().trim();
        String json = extractJson(text);
        try {
            return MAPPER.readValue(json, type);
        } catch (Exception e) {
            throw new RuntimeException("structured parse failed: " + e.getMessage(), e);
        }
    }

    private static String extractJson(String text) {
        int start = text.indexOf('{');
        int end = text.lastIndexOf('}');
        if (start >= 0 && end > start) return text.substring(start, end + 1);
        return text;
    }
}
