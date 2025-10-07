# 🗃️ Rent Stats Per Neighborhood Schema Documentation

## Database Tables
### `berlin_source_data.rent_stats_per_neighborhood` (Complete Schema)
district_id           |character varying|                      32|
district              |character varying|                     100|
median_net_rent_per_m2|numeric          |                        |
number_of_cases       |integer          |                        |
mean_net_rent_per_m2  |numeric          |                        |
year                  |smallint         |                        |
column_name|data_type        |character_maximum_length|
station_id |character varying|                        |
station    |character varying|                        |
line       |character varying|                        |

## Data Relationships
- Rent Stats Per Neighborhood → Districts: Direct mapping by district_id
- Rent Stats Per Neighborhood → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Rent Stats Per Neighborhood data at district level

## Field Descriptions
| character varying | 32 |  |
| character varying | 100 |  |
| numeric |  |  |
| integer |  |  |
| numeric |  |  |

## Data Quality Notes
- Data from official Berlin Rent Stats Per Neighborhood registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Rent Stats Per Neighborhood in Berlin
- Updated annually with new additions and updates
