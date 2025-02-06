SELECT 	
	uuid, 
	name,
	frames, 
	stats.angle, 
	INSTR(LOWER(description), 'no match') > 0 AS no_match, 
	v_scale.v_grade, 
	stats.ascensionist_count, 
	stats.quality_average
FROM climbs
JOIN (
	(SELECT 
		climb_uuid, 
		angle, 
		display_difficulty, 
		ascensionist_count, 
		quality_average
	FROM climb_stats
	) as 'stats'
	JOIN (
		SELECT 
			difficulty, 
			SUBSTR(boulder_name, INSTR(boulder_name, 'V') + 1, LENGTH(boulder_name)) AS v_grade
		FROM difficulty_grades
		) as 'v_scale'
	ON ROUND(stats.display_difficulty, 0) = v_scale.difficulty
	)
	ON climbs.uuid = stats.climb_uuid
	AND frames_count = 1
	AND layout_id = 9;
