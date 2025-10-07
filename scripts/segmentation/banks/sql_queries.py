"""SQL queries for banks segmentation"""

BANKS_QUERY = """
SELECT
    b.district,
    b.district_id,
    COUNT(b.bank_id) AS num_banks,
    COUNT(CASE WHEN b.atm = 'yes' THEN 1 END) AS num_atms,
    COUNT(CASE WHEN b.wheelchair = 'yes' THEN 1 END) AS num_wheelchair_accessible,
    AVG(CASE WHEN b.opening_hours IS NOT NULL THEN 1 ELSE 0 END) AS avg_opening_hours_availability,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(b.bank_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS banks_per_10k_residents,
    (COUNT(CASE WHEN b.atm = 'yes' THEN 1 END) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS atms_per_10k_residents
FROM berlin_source_data.banks b
LEFT JOIN berlin_source_data.regional_statistics r
    ON b.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY b.district, b.district_id
"""