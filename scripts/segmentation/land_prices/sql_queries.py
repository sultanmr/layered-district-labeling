"""SQL queries for land_prices segmentation"""

LAND_PRICES_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.district_id) AS num_land_prices,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.district_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS land_prices_per_10k_residents
FROM berlin_source_data.land_prices t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
