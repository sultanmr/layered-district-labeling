# 🗃️ Ubahn Schema Documentation

## Database Tables
### `berlin_source_data.ubahn` (Complete Schema)
station     |character varying|                      50|
line        |character varying|                      10|
latitude    |numeric          |                        |
longitude   |numeric          |                        |
postcode    |character varying|                      10|
neighborhood|character varying|                      50|
district    |character varying|                      50|
district_id |character varying|                      50|
column_name       |data_type        |character_maximum_length|
id                |integer          |                        |

## Data Relationships
- Ubahn → Districts: Direct mapping by district_id
- Ubahn → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Ubahn data at district level

## Field Descriptions
| character varying | 50 |  |
| character varying | 10 |  |
| numeric |  |  |
| numeric |  |  |
| character varying | 10 |  |

## Data Quality Notes
- Data from official Berlin Ubahn registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Ubahn in Berlin
- Updated annually with new additions and updates
