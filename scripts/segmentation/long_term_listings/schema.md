# 🗃️ Long Term Listings Schema Documentation

## Database Tables
### `berlin_source_data.long_term_listings` (Complete Schema)
listing_id     |character varying|                        |
detail_url     |text             |                        |
raw_info       |text             |                        |
type           |character varying|                        |
first_tenant   |character varying|                        |
price_euro     |integer          |                        |
number_of_rooms|double precision |                        |
surface_m2     |double precision |                        |
floor          |double precision |                        |
street         |character varying|                        |

## Data Relationships
- Long Term Listings → Districts: Direct mapping by district_id
- Long Term Listings → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Long Term Listings data at district level

## Field Descriptions
| character varying |  |  |
| text |  |  |
| text |  |  |
| character varying |  |  |
| character varying |  |  |

## Data Quality Notes
- Data from official Berlin Long Term Listings registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Long Term Listings in Berlin
- Updated annually with new additions and updates
