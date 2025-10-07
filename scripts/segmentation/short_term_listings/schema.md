# 🗃️ Short Term Listings Schema Documentation

## Database Tables
### `berlin_source_data.short_term_listings` (Complete Schema)
district_id                |character varying|                     100|
district                   |character varying|                      32|
id                         |bigint           |                        |
host_id                    |bigint           |                        |
neighborhood               |character varying|                      50|
latitude                   |numeric          |                        |
longitude                  |numeric          |                        |
property_type              |character varying|                      50|
room_type                  |character varying|                      50|
accommodates               |integer          |                        |

## Data Relationships
- Short Term Listings → Districts: Direct mapping by district_id
- Short Term Listings → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Short Term Listings data at district level

## Field Descriptions
| character varying | 100 |  |
| character varying | 32 |  |
| bigint |  |  |
| bigint |  |  |
| character varying | 50 |  |

## Data Quality Notes
- Data from official Berlin Short Term Listings registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Short Term Listings in Berlin
- Updated annually with new additions and updates
