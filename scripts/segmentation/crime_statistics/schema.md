# 🗃️ Crime Statistics Schema Documentation

## Database Tables
### `berlin_source_data.crime_statistics` (Complete Schema)
id                |integer                    |                        |
area_id           |character varying          |                      10|
locality          |character varying          |                     100|
district          |character varying          |                     100|
district_id       |character varying          |                      10|
year              |smallint                   |                        |
crime_type_german |character varying          |                     200|
crime_type_english|character varying          |                     200|
category          |character varying          |                     100|
total_number_cases|integer                    |                        |

## Data Relationships
- Crime Statistics → Districts: Direct mapping by district_id
- Crime Statistics → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Crime Statistics data at district level

## Field Descriptions
| integer |  |  |
| character varying | 10 |  |
| character varying | 100 |  |
| character varying | 100 |  |
| character varying | 10 |  |

## Data Quality Notes
- Data from official Berlin Crime Statistics registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Crime Statistics in Berlin
- Updated annually with new additions and updates
