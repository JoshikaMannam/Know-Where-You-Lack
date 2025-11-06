package com.knowwhereyoulack.controller;

import com.knowwhereyoulack.dto.QuizResponseDto;
import com.knowwhereyoulack.dto.TopicWithQuestionCount;
import com.knowwhereyoulack.model.Question;
import com.knowwhereyoulack.model.Topic;
import com.knowwhereyoulack.repository.TopicRepository;
import com.knowwhereyoulack.service.QuizService;
import com.knowwhereyoulack.service.impl.QuizServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/quiz")
@CrossOrigin(origins = "http://localhost:5173")
public class QuizController {
    
    private final QuizService quizService;
    private final QuizServiceImpl quizServiceImpl;
    private final TopicRepository topicRepository;
    
    @Autowired
    public QuizController(QuizService quizService, QuizServiceImpl quizServiceImpl, TopicRepository topicRepository) {
        this.quizService = quizService;
        this.quizServiceImpl = quizServiceImpl;
        this.topicRepository = topicRepository;
    }
    
    /**
     * Get all topics with question counts
     */
    @GetMapping("/topics")
    public ResponseEntity<List<TopicWithQuestionCount>> getAllTopics() {
        try {
            List<TopicWithQuestionCount> topics = quizService.getAllTopicsWithQuestionCount();
            return ResponseEntity.ok(topics);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
    
    /**
     * SIMPLIFIED ENDPOINT: Get 10 quiz questions with options (NO DIFFICULTY)
     * GET /api/quiz/{topicId}?questionCount=10
     * Returns: Array of {id, questionText, options[]}
     */
    @GetMapping("/{topicId}")
    public ResponseEntity<List<Map<String, Object>>> getQuizQuestions(
            @PathVariable Long topicId,
            @RequestParam(required = false, defaultValue = "10") int questionCount) {
        
        try {
            System.out.println("🎯 Getting " + questionCount + " random questions for topic " + topicId);
            
            // Verify topic exists
            Optional<Topic> topicOptional = topicRepository.findById(topicId);
            if (!topicOptional.isPresent()) {
                System.err.println("❌ Topic " + topicId + " not found");
                return ResponseEntity.notFound().build();
            }
            
            // Get ALL questions for this topic (no difficulty filter)
            List<Question> allQuestions = quizService.getAllQuestionsByTopic(topicId);
            
            System.out.println("📊 Found " + allQuestions.size() + " total questions for topic " + topicId);
            
            // Shuffle and limit to requested questionCount
            Collections.shuffle(allQuestions);
            List<Question> selectedQuestions = allQuestions.stream()
                .limit(questionCount)
                .collect(Collectors.toList());
            
            // Format response with options for each question
            List<Map<String, Object>> response = selectedQuestions.stream()
                .map(question -> {
                    Map<String, Object> questionMap = new HashMap<>();
                    questionMap.put("id", question.getQuestionId());
                    questionMap.put("questionText", question.getQuestionText());
                    
                    // Get options from question_options table
                    List<String> options = quizServiceImpl.getQuestionOptions(question.getQuestionId());
                    questionMap.put("options", options);
                    
                    return questionMap;
                })
                .collect(Collectors.toList());
            
            System.out.println("✅ Returning " + response.size() + " questions with options for topic " + topicId);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            System.err.println("❌ ERROR in getQuizQuestions: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
    
    /**
     * NEW: Get 10 random questions by SUBJECT (not topic)
     */
    @GetMapping("/subject/{subjectId}")
    public ResponseEntity<List<Map<String, Object>>> getQuizBySubject(
            @PathVariable Long subjectId,
            @RequestParam(required = false, defaultValue = "10") int questionCount) {
        
        try {
            System.out.println("🎯 Getting " + questionCount + " questions for SUBJECT " + subjectId);
            
            List<Question> questions = quizServiceImpl.getQuestionsBySubject(subjectId, questionCount);
            
            System.out.println("📊 Found " + questions.size() + " questions for subject " + subjectId);
            
            // Format response with options
            List<Map<String, Object>> response = questions.stream()
                .map(question -> {
                    Map<String, Object> questionMap = new HashMap<>();
                    questionMap.put("id", question.getQuestionId());
                    questionMap.put("questionText", question.getQuestionText());
                    
                    List<String> options = quizServiceImpl.getQuestionOptions(question.getQuestionId());
                    questionMap.put("options", options);
                    
                    return questionMap;
                })
                .collect(Collectors.toList());
            
            System.out.println("✅ Returning " + response.size() + " questions for subject " + subjectId);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            System.err.println("❌ ERROR in getQuizBySubject: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
    
    /**
     * Get questions by topic and difficulty level (OLD ENDPOINT - KEPT FOR COMPATIBILITY)
     * Returns QuizResponseDto with topic info and 10 random questions
     */
    @GetMapping("/{topicId}/difficulty/{difficulty}")
    public ResponseEntity<QuizResponseDto> getQuestionsByDifficulty(
            @PathVariable Long topicId,
            @PathVariable String difficulty) {
        
        try {
            // Get the topic details
            Optional<Topic> topicOptional = topicRepository.findById(topicId);
            if (!topicOptional.isPresent()) {
                return ResponseEntity.notFound().build();
            }
            
            Topic topic = topicOptional.get();
            
            // Get 10 random questions for this topic and difficulty
            List<Question> questions = quizService.getQuestionsByTopicAndDifficulty(
                topicId, 
                difficulty.toUpperCase()
            );
            
            // Create response DTO
            QuizResponseDto response = new QuizResponseDto(
                topic.getTopicId(),
                topic.getTopicName(),
                questions
            );
            
            if (questions.isEmpty()) {
                System.out.println("⚠️ WARNING: No questions found for topic " + topicId + 
                                 " with difficulty " + difficulty);
            } else {
                System.out.println("✅ Returning " + questions.size() + " questions for topic " + 
                                 topicId + " with difficulty " + difficulty);
            }
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            System.err.println("❌ ERROR in getQuestionsByDifficulty: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
    
    /**
     * Get all questions for a topic
     */
    @GetMapping("/{topicId}/questions")
    public ResponseEntity<List<Question>> getAllQuestions(@PathVariable Long topicId) {
        try {
            List<Question> questions = quizService.getAllQuestionsByTopic(topicId);
            return ResponseEntity.ok(questions);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}
