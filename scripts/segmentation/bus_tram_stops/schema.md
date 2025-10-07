# 🗃️ Bus Tram Stops Schema Documentation

## Database Tables
### `berlin_source_data.bus_tram_stops` (Complete Schema)
stop_id     |integer          |                        |
district_id |character varying|                     100|
name        |character varying|                     200|
address     |character varying|                     200|
latitude    |numeric          |                        |
longitude   |numeric          |                        |
neighborhood|character varying|                     100|
district    |character varying|                     100|
column_name       |data_type                  |character_maximum_length|
id                |integer                    |                        |

## Data Relationships
- Bus Tram Stops → Districts: Direct mapping by district_id
- Bus Tram Stops → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Bus Tram Stops data at district level

## Field Descriptions
| integer |  |  |
| character varying | 100 |  |
| character varying | 200 |  |
| character varying | 200 |  |
| numeric |  |  |

## Data Quality Notes
- Data from official Berlin Bus Tram Stops registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Bus Tram Stops in Berlin
- Updated annually with new additions and updates
