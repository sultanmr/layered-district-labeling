"""SQL queries for rent_stats_per_neighborhood segmentation"""

RENT_STATS_PER_NEIGHBORHOOD_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.district_id) AS num_rent_stats_per_neighborhood,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.district_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS rent_stats_per_neighborhood_per_10k_residents
FROM berlin_source_data.rent_stats_per_neighborhood t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
