"""SQL queries for universities segmentation"""

UNIVERSITIES_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.university_id) AS num_universities,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.university_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS universities_per_10k_residents
FROM berlin_source_data.universities t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
