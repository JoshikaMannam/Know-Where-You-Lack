package com.knowwhereyoulack.controller;

import com.knowwhereyoulack.model.Note;
import com.knowwhereyoulack.service.NoteService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

@RestController
@RequestMapping("/api/notes")
@CrossOrigin(origins = "http://localhost:5173")
public class NotesController {
    
    @Autowired
    private NoteService noteService;
    
    // Get all notes (hardcoded for Tier 2 demo)
    @GetMapping
    public List<Note> getAllNotes() {
        List<Note> notes = new ArrayList<>();
        
        Note note1 = new Note(1L, "Java Basics", 
            "Remember: Object-oriented programming has 4 pillars - Encapsulation, Inheritance, Polymorphism, and Abstraction.",
            "Programming");
        note1.setId(1L);
        note1.setCreatedAt(new Date());
        note1.setUpdatedAt(new Date());
        notes.add(note1);
        
        Note note2 = new Note(1L, "Data Structures",
            "Arrays vs LinkedList:\n- Arrays: O(1) access, O(n) insertion\n- LinkedList: O(n) access, O(1) insertion",
            "Computer Science");
        note2.setId(2L);
        note2.setCreatedAt(new Date());
        note2.setUpdatedAt(new Date());
        notes.add(note2);
        
        Note note3 = new Note(1L, "SQL Queries",
            "Common JOIN types:\n- INNER JOIN: Returns matching rows\n- LEFT JOIN: Returns all from left + matches\n- RIGHT JOIN: Returns all from right + matches\n- FULL OUTER JOIN: Returns all rows",
            "Database");
        note3.setId(3L);
        note3.setCreatedAt(new Date());
        note3.setUpdatedAt(new Date());
        notes.add(note3);
        
        Note note4 = new Note(1L, "Study Tips",
            "Active recall techniques:\n1. Flashcards\n2. Practice problems\n3. Teach concepts to others\n4. Write summaries without looking",
            "General");
        note4.setId(4L);
        note4.setCreatedAt(new Date());
        note4.setUpdatedAt(new Date());
        notes.add(note4);
        
        Note note5 = new Note(1L, "Algorithm Complexity",
            "Big O Notation quick reference:\n- O(1): Constant\n- O(log n): Logarithmic\n- O(n): Linear\n- O(n log n): Linearithmic\n- O(n²): Quadratic",
            "Algorithms");
        note5.setId(5L);
        note5.setCreatedAt(new Date());
        note5.setUpdatedAt(new Date());
        notes.add(note5);
        
        return notes;
    }
    
    // Get all notes for user
    @GetMapping("/user/{userId}")
    public ResponseEntity<List<Note>> getUserNotes(@PathVariable Long userId) {
        return ResponseEntity.ok(noteService.getUserNotes(userId));
    }
    
    // Get notes by subject
    @GetMapping("/user/{userId}/subject/{subject}")
    public ResponseEntity<List<Note>> getNotesBySubject(
            @PathVariable Long userId, 
            @PathVariable String subject) {
        return ResponseEntity.ok(noteService.getNotesBySubject(userId, subject));
    }
    
    // Create new note
    @PostMapping
    public ResponseEntity<Note> createNote(@RequestBody Note note) {
        Note created = noteService.createNote(note);
        return ResponseEntity.ok(created);
    }
    
    // Update note
    @PutMapping("/{id}")
    public ResponseEntity<Note> updateNote(@PathVariable Long id, @RequestBody Note note) {
        Note updated = noteService.updateNote(id, note);
        if (updated != null) {
            return ResponseEntity.ok(updated);
        }
        return ResponseEntity.notFound().build();
    }
    
    // Delete note
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteNote(@PathVariable Long id) {
        noteService.deleteNote(id);
        return ResponseEntity.ok().build();
    }
}
