# 🗃️ Venues Schema Documentation

## Database Tables
### `berlin_source_data.venues` (Complete Schema)
venue_id             |character varying|                      10|
district_id          |character varying|                      20|
name                 |character varying|                     200|
district             |character varying|                     200|
category             |character varying|                     200|
cuisine              |character varying|                     200|
phone                |character varying|                      50|
address              |character varying|                     200|
latitude             |numeric          |                        |
longitude            |numeric          |                        |

## Data Relationships
- Venues → Districts: Direct mapping by district_id
- Venues → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Venues data at district level

## Field Descriptions
| character varying | 10 |  |
| character varying | 20 |  |
| character varying | 200 |  |
| character varying | 200 |  |
| character varying | 200 |  |

## Data Quality Notes
- Data from official Berlin Venues registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Venues in Berlin
- Updated annually with new additions and updates
