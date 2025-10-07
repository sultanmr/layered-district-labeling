# 🗃️ Playgrounds Schema Documentation

## Database Tables
### `berlin_source_data.playgrounds` (Complete Schema)
playground_id  |character varying|                        |
name           |character varying|                        |
latitude       |real             |                        |
longitude      |real             |                        |
district       |character varying|                        |
district_id    |character varying|                        |
neighborhood   |character varying|                        |
neighborhood_id|character varying|                        |
area_sq_m      |double precision |                        |
full_address   |character varying|                        |

## Data Relationships
- Playgrounds → Districts: Direct mapping by district_id
- Playgrounds → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Playgrounds data at district level

## Field Descriptions
| character varying |  |  |
| character varying |  |  |
| real |  |  |
| real |  |  |
| character varying |  |  |

## Data Quality Notes
- Data from official Berlin Playgrounds registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Playgrounds in Berlin
- Updated annually with new additions and updates
