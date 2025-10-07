# CSV to Database Loader Script

This script (`save_to_db.py`) loads CSV data into the `district_labels` table in the database with conflict resolution.

## Features

- Validates CSV file structure
- Loads data from CSV into pandas DataFrame
- Connects to database using SQLAlchemy
- Inserts data with ON CONFLICT resolution (update existing records)
- Supports dry-run mode for testing
- Comprehensive logging

## Requirements

- Python 3.7+
- Dependencies (already in requirements.txt):
  - pandas
  - sqlalchemy
  - psycopg2-binary
  - python-dotenv

## Usage

### Basic Usage

```bash
python save_to_db.py <csv_file> <reference>
```

### Examples

1. **Load data with a specific reference:**
   ```bash
   python save_to_db.py labels.csv ubahn-rule-based
   ```

2. **Dry-run mode (validate without inserting):**
   ```bash
   python save_to_db.py labels.csv test-reference --dry-run
   ```

3. **Using existing CSV files from visualization output:**
   ```bash
   python save_to_db.py viz/rule_based/labels.csv rule-based-parks
   ```

### CSV File Format

The CSV file must have the following columns:
- `district` (string): The district name
- `hashtags` (string): Comma-separated hashtags

Example CSV:
```csv
district,hashtags
Mitte,"#citylife,#central"
Charlottenburg-Wilmersdorf,"#affluent,#shopping"
Friedrichshain-Kreuzberg,"#alternative,#nightlife"
```

### Reference Parameter

The `reference` parameter is used to identify the source of the data and is stored in the `source` column. This allows multiple data sources to coexist in the same table.

Examples of references:
- `ubahn-rule-based`
- `parks-llm` 
- `crime-statistics-react`
- `manual-entry`

## Database Schema

The script creates/uses the following table structure:

```sql
CREATE TABLE district_labels (
    district VARCHAR(100) NOT NULL,
    hashtags TEXT,
    source VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (district, source)
);
```

## Conflict Resolution

The script uses PostgreSQL's `ON CONFLICT` clause to handle duplicate entries:

```sql
INSERT INTO district_labels (district, hashtags, source)
VALUES (:district, :hashtags, :source)
ON CONFLICT (district, source) DO UPDATE
SET hashtags = EXCLUDED.hashtags,
    updated_at = CURRENT_TIMESTAMP;
```

This means:
- If a record with the same `district` and `source` exists, the `hashtags` will be updated
- The `updated_at` timestamp will be refreshed
- New records will be inserted normally

## Environment Configuration

The script uses the `DB_URL` environment variable from your `.env` file:

```bash
DB_URL="postgresql+psycopg2://username:password@localhost:5432/database_name"
```

## Error Handling

The script includes comprehensive error handling for:
- Missing CSV file
- Invalid CSV format
- Database connection issues
- SQL execution errors

All errors are logged with timestamps and appropriate error levels.

## Logging

The script uses Python's logging module with the following format:
```
2025-09-12 11:42:54,158 - INFO - CSV file validation passed: test_data.csv
```

Log levels:
- INFO: Normal operations
- ERROR: Error conditions
- WARNING: Warning conditions