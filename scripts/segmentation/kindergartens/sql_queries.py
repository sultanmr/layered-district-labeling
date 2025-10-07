"""SQL queries for kindergartens segmentation"""

KINDERGARTENS_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.kindergarten_id) AS num_kindergartens,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.kindergarten_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS kindergartens_per_10k_residents
FROM berlin_source_data.kindergartens t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
