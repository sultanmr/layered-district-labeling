"""SQL queries for bus and tram stops segmentation"""

BUS_TRAM_STOPS_QUERY = """
SELECT
    b.district,
    b.district_id,
    COUNT(b.stop_id) AS num_stops,
    COUNT(DISTINCT b.neighborhood) AS neighborhoods_covered,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(b.stop_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS stops_per_10k_residents,
    AVG(CASE WHEN b.latitude IS NOT NULL AND b.longitude IS NOT NULL THEN 1 ELSE 0 END) AS location_accuracy_score,
    COUNT(DISTINCT b.name) AS unique_stop_names,
    COUNT(CASE WHEN b.address IS NOT NULL THEN 1 END) AS stops_with_address
FROM berlin_source_data.bus_tram_stops b
LEFT JOIN berlin_source_data.regional_statistics r
    ON b.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY b.district, b.district_id
"""