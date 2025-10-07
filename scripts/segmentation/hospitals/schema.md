# 🗃️ Hospitals Schema Documentation

## Database Tables
### `berlin_source_data.hospitals` (Complete Schema)
district_id|character varying|                      10|
name       |character varying|                     200|
address    |character varying|                     200|
coordinates|character varying|                     200|
latitude   |numeric          |                        |
longitude  |numeric          |                        |
locality   |character varying|                     100|
district   |character varying|                     100|
distance   |numeric          |                        |
beds       |integer          |                        |

## Data Relationships
- Hospitals → Districts: Direct mapping by district_id
- Hospitals → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Hospitals data at district level

## Field Descriptions
| character varying | 10 |  |
| character varying | 200 |  |
| character varying | 200 |  |
| character varying | 200 |  |
| numeric |  |  |

## Data Quality Notes
- Data from official Berlin Hospitals registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Hospitals in Berlin
- Updated annually with new additions and updates
