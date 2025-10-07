"""SQL queries for green spaces segmentation"""

PARKS_QUERY = """
SELECT
    p.district,
    p.district_id,
    COUNT(p.park_id) AS num_green_spaces,
    SUM(p.area_sq_m) AS total_green_area,
    AVG(p.area_sq_m) AS avg_park_size,
    COUNT(DISTINCT p.district) AS districts_with_parks,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (SUM(p.area_sq_m) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS park_area_per_10k_residents,
    AVG(EXTRACT(YEAR FROM CURRENT_DATE) - 2020) AS avg_years_since_renovation
FROM berlin_source_data.parks p
LEFT JOIN berlin_source_data.regional_statistics r
    ON p.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY p.district, p.district_id
"""