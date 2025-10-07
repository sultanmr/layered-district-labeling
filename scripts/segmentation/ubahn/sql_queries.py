"""SQL queries for ubahn segmentation"""

UBAHN_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.station) AS num_ubahn,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.station) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS ubahn_per_10k_residents
FROM berlin_source_data.ubahn t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
