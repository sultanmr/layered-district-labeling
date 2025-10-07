
#!/usr/bin/env python3
"""
Script to load CSV data into the district_labels table with conflict resolution.
Usage: python save_to_db.py <csv_file> <reference>
"""

import argparse
import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    """Create and return a database engine using the configured DB_URL"""
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise ValueError("DB_URL environment variable not set. Please check your .env file")
    
    try:
        engine = create_engine(db_url)
        logger.info(f"Database engine created successfully for {db_url}")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise

def validate_csv_file(csv_file):
    """Validate that the CSV file exists and has the correct format"""
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Check if file has required columns
    try:
        df = pd.read_csv(csv_file, nrows=1)
        required_columns = ['district', 'hashtags']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

def load_csv_data(csv_file, reference):
    """Load and prepare CSV data for insertion"""
    try:
        df = pd.read_csv(csv_file)
        
        # Add source/reference column
        df['source'] = reference
        
        # Ensure hashtags column is string type and handle NaN values
        df['hashtags'] = df['hashtags'].fillna('').astype(str)
        
        logger.info(f"Loaded {len(df)} rows from {csv_file}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

def load_csv_data(csv_file):
    """Load and prepare CSV data for insertion"""
    try:
        df = pd.read_csv(csv_file)
        
        # Ensure hashtags column is string type and handle NaN values
        df['hashtags'] = df['hashtags'].fillna('').astype(str)
        
        logger.info(f"Loaded {len(df)} rows from {csv_file}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

def insert_data_to_db(engine, df):
    """Insert data into district_labels table with conflict resolution"""
    try:
        with engine.connect() as conn:
            # Insert data with conflict resolution
            for _, row in df.iterrows():
                insert_sql = text("""
                INSERT INTO berlin_source_data.district_labels (district, hashtags, source)
                VALUES (:district, :hashtags, :source)
                ON CONFLICT (district, source) DO UPDATE
                SET hashtags = EXCLUDED.hashtags;
                """)
                
                conn.execute(insert_sql, {
                    'district': row['district'],
                    'hashtags': row['hashtags'],
                    'source': row['source']
                })
            
            conn.commit()
            logger.info(f"Successfully inserted/updated {len(df)} rows into district_labels table")
            
    except Exception as e:
        logger.error(f"Error inserting data into database: {e}")
        raise
'''
def main():
    root_folder = "viz"  # you just provide viz
    dataframes = []

    # Loop through all first-level folders inside viz (e.g., parks, gardens, ...)
    for first_level in os.listdir(root_folder):
        first_level_path = os.path.join(root_folder, first_level)
        
        if os.path.isdir(first_level_path):
            # Loop through all second-level folders inside (e.g., rule_based, llm, react)
            for subfolder in os.listdir(first_level_path):
                subfolder_path = os.path.join(first_level_path, subfolder)
                labels_file = os.path.join(subfolder_path, "labels.csv")
                
                if os.path.isdir(subfolder_path) and os.path.exists(labels_file):
                    try:
                        df = pd.read_csv(labels_file)
                        df["reference"] = f"sultan-{first_level}-{subfolder}"                       
                        dataframes.append(df)
                    except Exception as e:
                        print(f"Error reading {labels_file}: {e}")

    # Combine all DataFrames
    if dataframes:
        main_df = pd.concat(dataframes, ignore_index=True)
        main_df.to_csv("combined_labels.csv", index=False)
        print("✅ Combined DataFrame created successfully!")
        print(main_df.head())
    else:
        print("⚠️ No labels.csv files found in any subfolders.")



'''
def main():
    csv_file = './viz/combined_labels.csv' 
    #reference = 'sultan-parks-react' 
    
    #parser = argparse.ArgumentParser(description='Load CSV data into district_labels table')
    #parser.add_argument('csv', help='Path to the CSV file to load')
    #parser.add_argument('ref', help='Reference/source identifier for the data')
    
    #args = parser.parse_args()
    #csv_file = args.csv
    #reference = args.ref
    

    try:
        # Validate CSV file
        validate_csv_file(csv_file)
        logger.info(f"CSV file validation passed: {csv_file}")
        
        # Load data
        df = load_csv_data(csv_file)

        # Get database connection
        engine = get_db_engine()
        
        # Insert data
        insert_data_to_db(engine, df)
        
        logger.info("Data loading completed successfully")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())