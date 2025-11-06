-- =====================================================
-- POPULATE QUESTION_OPTIONS FROM QUESTIONS TABLE
-- Extracts real options from option_a, option_b, etc.
-- =====================================================

USE knowwhereyoulack;

-- Clean slate
DELETE FROM question_options;

-- Insert Option A
INSERT INTO question_options (question_id, option_text, option_label, is_correct)
SELECT 
    question_id,
    option_a as option_text,
    'A' as option_label,
    CASE WHEN correct_option = 'A' THEN 1 ELSE 0 END as is_correct
FROM questions
WHERE is_active = 1 AND option_a IS NOT NULL;

-- Insert Option B
INSERT INTO question_options (question_id, option_text, option_label, is_correct)
SELECT 
    question_id,
    option_b as option_text,
    'B' as option_label,
    CASE WHEN correct_option = 'B' THEN 1 ELSE 0 END as is_correct
FROM questions
WHERE is_active = 1 AND option_b IS NOT NULL;

-- Insert Option C
INSERT INTO question_options (question_id, option_text, option_label, is_correct)
SELECT 
    question_id,
    option_c as option_text,
    'C' as option_label,
    CASE WHEN correct_option = 'C' THEN 1 ELSE 0 END as is_correct
FROM questions
WHERE is_active = 1 AND option_c IS NOT NULL;

-- Insert Option D
INSERT INTO question_options (question_id, option_text, option_label, is_correct)
SELECT 
    question_id,
    option_d as option_text,
    'D' as option_label,
    CASE WHEN correct_option = 'D' THEN 1 ELSE 0 END as is_correct
FROM questions
WHERE is_active = 1 AND option_d IS NOT NULL;

-- Verification
SELECT 'Total questions' as metric, COUNT(*) as count FROM questions WHERE is_active = 1
UNION ALL
SELECT 'Total options created', COUNT(*) FROM question_options
UNION ALL
SELECT 'Average options per question', AVG(cnt) FROM (
    SELECT question_id, COUNT(*) as cnt 
    FROM question_options 
    GROUP BY question_id
) as subq;

-- Sample check
SELECT 
    q.question_id,
    q.question_text,
    qo.option_label,
    qo.option_text,
    qo.is_correct
FROM questions q
JOIN question_options qo ON q.question_id = qo.question_id
WHERE q.question_id IN (1, 2, 3)
ORDER BY q.question_id, qo.option_label;
