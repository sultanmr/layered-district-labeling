# 🗃️ Dental Offices Schema Documentation

## Database Tables
### `berlin_source_data.dental_offices` (Complete Schema)
osm_id       |character varying|                        |
osm_type     |character varying|                        |
name         |character varying|                        |
street       |character varying|                        |
housenumber  |character varying|                        |
postcode     |character varying|                        |
city         |character varying|                        |
opening_hours|character varying|                        |
wheelchair   |character varying|                        |
phone        |character varying|                        |

## Data Relationships
- Dental Offices → Districts: Direct mapping by district_id
- Dental Offices → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Dental Offices data at district level

## Field Descriptions
| character varying |  |  |
| character varying |  |  |
| character varying |  |  |
| character varying |  |  |
| character varying |  |  |

## Data Quality Notes
- Data from official Berlin Dental Offices registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Dental Offices in Berlin
- Updated annually with new additions and updates
