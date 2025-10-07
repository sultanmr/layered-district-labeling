# 🗃️ Kindergartens Schema Documentation

## Database Tables
### `berlin_source_data.kindergartens` (Complete Schema)
kindergarten_id|character varying|                        |
name           |character varying|                        |
address        |character varying|                        |
latitude       |real             |                        |
longitude      |real             |                        |
district       |character varying|                        |
district_id    |character varying|                        |
neighborhood   |character varying|                        |
neighborhood_id|character varying|                        |
full_address   |character varying|                        |

## Data Relationships
- Kindergartens → Districts: Direct mapping by district_id
- Kindergartens → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Kindergartens data at district level

## Field Descriptions
| character varying |  |  |
| character varying |  |  |
| character varying |  |  |
| real |  |  |
| real |  |  |

## Data Quality Notes
- Data from official Berlin Kindergartens registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Kindergartens in Berlin
- Updated annually with new additions and updates
