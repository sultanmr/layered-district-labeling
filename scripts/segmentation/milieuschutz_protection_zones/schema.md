# 🗃️ Milieuschutz Protection Zones Schema Documentation

## Database Tables
### `berlin_source_data.milieuschutz_protection_zones` (Complete Schema)
protection_zone_id  |character varying|                      50|
protection_zone_key |character varying|                      20|
protection_zone_name|character varying|                     100|
district            |character varying|                     100|
district_id         |character varying|                     100|
date_announced      |date             |                        |
date_effective      |date             |                        |
amendment_announced |date             |                        |
amendment_effective |date             |                        |
area_ha             |numeric          |                        |

## Data Relationships
- Milieuschutz Protection Zones → Districts: Direct mapping by district_id
- Milieuschutz Protection Zones → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Milieuschutz Protection Zones data at district level

## Field Descriptions
| character varying | 50 |  |
| character varying | 20 |  |
| character varying | 100 |  |
| character varying | 100 |  |
| character varying | 100 |  |

## Data Quality Notes
- Data from official Berlin Milieuschutz Protection Zones registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Milieuschutz Protection Zones in Berlin
- Updated annually with new additions and updates
