package com.knowwhereyoulack.controller;

import com.knowwhereyoulack.model.ChatMessage;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/chat")
@CrossOrigin(origins = "*")
public class ChatController {
    
    // In-memory chat history (will persist during app runtime)
    private static List<ChatMessage> chatHistory = new ArrayList<>();

    @PostMapping("/message")
    public Map<String, String> sendMessage(@RequestBody Map<String, String> request) {
        String userMessage = request.get("message");
        
        // Simple response logic (replace with your actual chatbot logic)
        String botResponse = generateResponse(userMessage);
        
        // Save to history
        ChatMessage chat = new ChatMessage(userMessage, botResponse);
        chatHistory.add(chat);
        
        return Map.of("response", botResponse);
    }

    @GetMapping("/history")
    public List<Map<String, Object>> getHistory() {
        List<Map<String, Object>> history = new ArrayList<>();
        
        for (ChatMessage chat : chatHistory) {
            // User message
            history.add(Map.of(
                "text", chat.getMessage(),
                "sender", "user",
                "timestamp", chat.getTimestamp().toString()
            ));
            
            // Bot response
            history.add(Map.of(
                "text", chat.getResponse(),
                "sender", "ai",
                "timestamp", chat.getTimestamp().toString()
            ));
        }
        
        return history;
    }

    @DeleteMapping("/history")
    public Map<String, String> clearHistory() {
        chatHistory.clear();
        return Map.of("message", "Chat history cleared");
    }

    private String generateResponse(String message) {
        // Simple demo responses - replace with your actual Groq API call
        message = message.toLowerCase();
        
        if (message.contains("hello") || message.contains("hi")) {
            return "Hello! I'm Skilli, your AI learning assistant. How can I help you today?";
        } else if (message.contains("quiz")) {
            return "I can help you with quizzes! Here are some tips:\n\n- Start with easier difficulty\n- Review your mistakes\n- Practice regularly\n\nWhat subject would you like to focus on?";
        } else if (message.contains("study")) {
            return "Great question about studying! Here are some effective strategies:\n\n- Use active recall\n- Space your learning sessions\n- Teach concepts to others\n- Take regular breaks\n\nWould you like more details on any of these?";
        } else {
            return "I understand you're asking about: \"" + message + "\"\n\nLet me help you with that. Could you provide more details about what specific aspect you'd like to know?";
        }
    }
}
