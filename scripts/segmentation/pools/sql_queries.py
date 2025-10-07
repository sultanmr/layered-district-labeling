"""SQL queries for pools segmentation"""

POOLS_QUERY = """
SELECT
    r.district,
    COUNT(t.pool_id) AS num_pools,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.pool_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS pools_per_10k_residents
FROM berlin_source_data.pools t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district_id = r.district_id AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY r.district
"""
