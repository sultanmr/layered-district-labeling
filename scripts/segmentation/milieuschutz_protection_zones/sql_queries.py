"""SQL queries for milieuschutz_protection_zones segmentation"""

MILIEUSCHUTZ_PROTECTION_ZONES_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.protection_zone_id) AS num_milieuschutz_protection_zones,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.protection_zone_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS milieuschutz_protection_zones_per_10k_residents
FROM berlin_source_data.milieuschutz_protection_zones t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
