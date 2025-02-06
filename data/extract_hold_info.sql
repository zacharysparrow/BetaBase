SELECT placements.id, x, y, xm, ym FROM placements
JOIN 
	(
	SELECT
		h1.id,
        	h1.x,
        	h1.y,
		h1.mirrored_hole_id,
        	h2.x as xm,
		h2.y as ym
	FROM holes h1
	JOIN holes h2
	ON h1.mirrored_hole_id = h2.id
	) as all_h
ON hole_id = all_h.id



