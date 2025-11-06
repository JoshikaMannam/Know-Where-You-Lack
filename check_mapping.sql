-- Check topic to subject mapping
SELECT 
    t.topic_id,
    t.topic_name,
    t.subject_id,
    s.subject_name,
    COUNT(q.question_id) as question_count
FROM topics t
LEFT JOIN subjects s ON t.subject_id = s.subject_id
LEFT JOIN questions q ON t.topic_id = q.topic_id
GROUP BY t.topic_id, t.topic_name, t.subject_id, s.subject_name
ORDER BY t.topic_id;
