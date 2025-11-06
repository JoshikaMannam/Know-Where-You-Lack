package com.knowwhereyoulack.model;

import java.time.LocalDateTime;

public class ChatMessage {
    private Long id;
    private String message;
    private String response;
    private LocalDateTime timestamp;
    private boolean isUser;

    // Constructors
    public ChatMessage() {}

    public ChatMessage(String message, String response) {
        this.message = message;
        this.response = response;
        this.timestamp = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }

    public String getResponse() { return response; }
    public void setResponse(String response) { this.response = response; }

    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }

    public boolean isUser() { return isUser; }
    public void setUser(boolean user) { isUser = user; }
}
