"""SQL queries for districts_pop_stat segmentation"""

DISTRICTS_POP_STAT_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.district_id) AS num_districts_pop_stat,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.district_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS districts_pop_stat_per_10k_residents
FROM berlin_source_data.districts_pop_stat t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
