# 🗃️ Schools Schema Documentation

## Database Tables
### `berlin_source_data.schools` (Complete Schema)
id                |integer          |                        |
bsn               |character varying|                        |
school_name       |character varying|                        |
school_type_de    |character varying|                        |
ownership_en      |character varying|                        |
school_category_de|character varying|                        |
school_category_en|character varying|                        |
district_id       |character varying|                        |
district          |character varying|                        |
quarter           |character varying|                        |

## Data Relationships
- Schools → Districts: Direct mapping by district_id
- Schools → Neighborhoods: Spatial mapping by neighborhood_id
- Analysis aggregates Schools data at district level

## Field Descriptions
| integer |  |  |
| character varying |  |  |
| character varying |  |  |
| character varying |  |  |
| character varying |  |  |

## Data Quality Notes
- Data from official Berlin Schools registry
- Geographic coordinates precise to 6 decimal places
- Includes all public Schools in Berlin
- Updated annually with new additions and updates
