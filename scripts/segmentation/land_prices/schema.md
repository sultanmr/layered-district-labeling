# 🗃️ Land Prices Schema Documentation

## Database Tables
### `berlin_source_data.land_prices` (Complete Schema)
district                   |character varying|                      32|
standard_land_value_per_sqm|double precision |                        |
typical_land_use_type      |character varying|                     100|
typical_floor_space_ratio  |numeric          |                        |
land_use_category          |character varying|                     100|
district_id                |character varying|                     100|
year                       |integer          |                        |
column_name    |data_type        |character_maximum_length|
listing_id     |character varying|                        |
detail_url     |text             |                        |

## Data Relationships
- Land Prices → Districts: Direct mapping by district_id
- Land Prices → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Land Prices data at district level

## Field Descriptions
| character varying | 32 |  |
| double precision |  |  |
| character varying | 100 |  |
| numeric |  |  |
| character varying | 100 |  |

## Data Quality Notes
- Data from official Berlin Land Prices registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Land Prices in Berlin
- Updated annually with new additions and updates
