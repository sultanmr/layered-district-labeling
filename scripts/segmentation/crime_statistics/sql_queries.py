"""SQL queries for crime_statistics segmentation"""

CRIME_STATISTICS_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.id) AS num_crime_statistics,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS crime_statistics_per_10k_residents
FROM berlin_source_data.crime_statistics t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
