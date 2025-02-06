SELECT climb_uuid, angle,v_scale.v_grade
FROM climb_stats 
	JOIN (
		SELECT 
			difficulty, 
			SUBSTR(boulder_name, INSTR(boulder_name, 'V') + 1, LENGTH(boulder_name)) AS v_grade
		FROM difficulty_grades
		) as 'v_scale'
	ON ROUND(benchmark_difficulty, 0) = v_scale.difficulty
WHERE benchmark_difficulty IS NOT NULL;
