"""SQL queries for venues segmentation"""

VENUES_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.venue_id) AS num_venues,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.venue_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS venues_per_10k_residents
FROM berlin_source_data.venues t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
