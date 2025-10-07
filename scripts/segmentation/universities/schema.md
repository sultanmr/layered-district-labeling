# 🗃️ Universities Schema Documentation

## Database Tables
### `berlin_source_data.universities` (Complete Schema)
university_id             |integer          |                        |
university_name           |character varying|                     255|
rank_in_berlin_brandenburg|integer          |                        |
rank_in_germany           |integer          |                        |
enrollment                |integer          |                        |
founded                   |integer          |                        |
latitude                  |double precision |                        |
longitude                 |double precision |                        |
postcode                  |character varying|                      10|
district                  |character varying|                      50|

## Data Relationships
- Universities → Districts: Direct mapping by district_id
- Universities → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Universities data at district level

## Field Descriptions
| integer |  |  |
| character varying | 255 |  |
| integer |  |  |
| integer |  |  |
| integer |  |  |

## Data Quality Notes
- Data from official Berlin Universities registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Universities in Berlin
- Updated annually with new additions and updates
