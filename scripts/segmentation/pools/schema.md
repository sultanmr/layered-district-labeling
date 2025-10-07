# 🗃️ Pools Schema Documentation

## Database Tables
### `berlin_source_data.pools` (Complete Schema)
pool_id      |character varying|                      20|
district_id  |character varying|                      10|
name         |character varying|                     200|
pool_type    |character varying|                     200|
street       |character varying|                     200|
postal_code  |character varying|                      10|
latitude     |numeric          |                        |
longitude    |numeric          |                        |
open_all_year|boolean          |                        |
column_name                   |data_type        |character_maximum_length|

## Data Relationships
- Pools → Districts: Direct mapping by district_id
- Pools → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Pools data at district level

## Field Descriptions
| character varying | 20 |  |
| character varying | 10 |  |
| character varying | 200 |  |
| character varying | 200 |  |
| character varying | 200 |  |

## Data Quality Notes
- Data from official Berlin Pools registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Pools in Berlin
- Updated annually with new additions and updates
