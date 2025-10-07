# 🗃️ Banks Schema Documentation

## Database Tables
### `berlin_source_data.banks` (Complete Schema)
bank_id      |character varying|                      20|
name         |character varying|                     255|
brand        |character varying|                     255|
operator     |character varying|                     255|
street       |character varying|                     255|
housenumber  |character varying|                      50|
postcode     |character varying|                      10|
opening_hours|text             |                        |
atm          |character varying|                      10|
wheelchair   |character varying|                      10|

## Data Relationships
- Banks → Districts: Direct mapping by district_id
- Banks → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Banks data at district level

## Field Descriptions
| character varying | 20 |  |
| character varying | 255 |  |
| character varying | 255 |  |
| character varying | 255 |  |
| character varying | 255 |  |

## Data Quality Notes
- Data from official Berlin Banks registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Banks in Berlin
- Updated annually with new additions and updates
