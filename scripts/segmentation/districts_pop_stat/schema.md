# 🗃️ Districts Pop Stat Schema Documentation

## Database Tables
### `berlin_source_data.districts_pop_stat` (Complete Schema)
district_id               |character varying|                      10|
district                  |character varying|                      32|
male                      |integer          |                        |
female                    |integer          |                        |
germans                   |integer          |                        |
foreigners                |integer          |                        |
single                    |integer          |                        |
married                   |integer          |                        |
widowed                   |integer          |                        |
divorced                  |integer          |                        |

## Data Relationships
- Districts Pop Stat → Districts: Direct mapping by district_id
- Districts Pop Stat → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Districts Pop Stat data at district level

## Field Descriptions
| character varying | 10 |  |
| character varying | 32 |  |
| integer |  |  |
| integer |  |  |
| integer |  |  |

## Data Quality Notes
- Data from official Berlin Districts Pop Stat registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Districts Pop Stat in Berlin
- Updated annually with new additions and updates
