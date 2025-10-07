# 🗃️ Sbahn Schema Documentation

## Database Tables
### `berlin_source_data.sbahn` (Complete Schema)
station_id |character varying|                        |
station    |character varying|                        |
line       |character varying|                        |
latitude   |real             |                        |
longitude  |real             |                        |
district   |character varying|                        |
district_id|character varying|                        |
column_name       |data_type        |character_maximum_length|
id                |integer          |                        |
bsn               |character varying|                        |

## Data Relationships
- Sbahn → Districts: Direct mapping by district_id
- Sbahn → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Sbahn data at district level

## Field Descriptions
| character varying |  |  |
| character varying |  |  |
| character varying |  |  |
| real |  |  |
| real |  |  |

## Data Quality Notes
- Data from official Berlin Sbahn registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Sbahn in Berlin
- Updated annually with new additions and updates
