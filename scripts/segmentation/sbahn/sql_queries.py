"""SQL queries for sbahn segmentation"""

SBAHN_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.station_id) AS num_sbahn,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.station_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS sbahn_per_10k_residents
FROM berlin_source_data.sbahn t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
