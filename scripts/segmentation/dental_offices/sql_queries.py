"""SQL queries for dental_offices segmentation"""

DENTAL_OFFICES_QUERY = """
SELECT
    r.district,
    COUNT(t.osm_id) AS num_dental_offices,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.osm_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS dental_offices_per_10k_residents
FROM berlin_source_data.dental_offices t
LEFT JOIN berlin_source_data.regional_statistics r
    ON r.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY r.district
"""
