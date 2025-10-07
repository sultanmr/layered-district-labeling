"""SQL queries for long_term_listings segmentation"""

LONG_TERM_LISTINGS_QUERY = """
SELECT
    t.district,
    t.district_id,
    COUNT(t.listing_id) AS num_long_term_listings,
    COALESCE(SUM(r.inhabitants), 1) AS population,
    (COUNT(t.listing_id) * 10000.0) / NULLIF(SUM(r.inhabitants), 0) AS long_term_listings_per_10k_residents
FROM berlin_source_data.long_term_listings t
LEFT JOIN berlin_source_data.regional_statistics r
    ON t.district = r.district AND r.year = (SELECT MAX(year) FROM berlin_source_data.regional_statistics)
GROUP BY t.district, t.district_id
"""
