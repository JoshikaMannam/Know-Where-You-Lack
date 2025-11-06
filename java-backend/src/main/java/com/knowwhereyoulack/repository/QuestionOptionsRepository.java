package com.knowwhereyoulack.repository;

import com.knowwhereyoulack.model.QuestionOption;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import java.util.List;

@org.springframework.stereotype.Repository
public interface QuestionOptionsRepository extends JpaRepository<QuestionOption, Long> {
    
    @Query(value = "SELECT option_text FROM question_options WHERE question_id = :questionId ORDER BY option_label", nativeQuery = true)
    List<String> findOptionsByQuestionId(@Param("questionId") Long questionId);
}
