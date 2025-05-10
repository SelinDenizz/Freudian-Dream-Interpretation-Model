import pandas as pd
import json
import re
import os
from datetime import datetime

def normalize_dreambank_data(input_file, output_dir="processed_data"):
    """
    Normalize DreamsBank data from TSV format to structured JSON and CSV formats.
    
    Args:
        input_file: Path to the TSV file containing dream data
        output_dir: Directory to save processed data
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the TSV file
    print(f"Reading data from {input_file}...")

    try:
        df = pd.read_csv(input_file, sep='\t')

        if len(df.columns) != 3 or not any('dream' in col.lower() for col in df.columns):
            # If not, try reading without header
            df = pd.read_csv(input_file, sep='\t', header=None)
            if len(df.columns) == 3:
                df.columns = ["n", "date", "dream"]
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # # Display the first few rows to check
    # print("\nFirst 5 rows of the original data:")
    # print(df.head())
    
    # Clean up the data
    # print("\nCleaning data...")
    
    # Remove quotes from strings if they exist
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('"', '', regex=False)
    
    # Extract year from date (assuming format like "(1985?)" or "(8/11/67)")
    def extract_year(date_str):
        if pd.isna(date_str):
            return None
        
        # Try to extract year
        year_match = re.search(r'\((\d{4})', date_str)
        if year_match:
            return year_match.group(1)
        
        # Try to extract from MM/DD/YY format
        date_match = re.search(r'\((\d{1,2})/(\d{1,2})/(\d{2})\)', date_str)
        if date_match:
            year = int(date_match.group(3))
            # Convert 2-digit year to 4-digit
            if year < 50:  # Assuming 00-49 means 2000-2049
                return f"20{year:02d}"
            else:  # Assuming 50-99 means 1950-1999
                return f"19{year:02d}"
        
        return None
    
    # Apply the extraction
    df['year'] = df['date'].apply(extract_year)
    
    # Clean dream text
    df['clean_dream'] = df['dream'].apply(lambda x: x.replace('""', '"') if isinstance(x, str) else x)
    
    # Add dreamer information from the JSON metadata
    dreamer_info = {
        "short_name": "alta",
        "long_name": "Alta: a detailed dreamer",
        "sex": "female",
        "description": "Adult woman who wrote down her dreams in the late 1980s and early 1990s, with a few from 1997"
    }
    
    for key, value in dreamer_info.items():
        df[key] = value
    
    # Add additional useful features
    df['dream_length'] = df['clean_dream'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    
    # Create standardized ID
    df['dream_id'] = df['short_name'] + '_' + df['n'].astype(str).str.zfill(3)
    
    # Save to CSV in a standardized format
    output_csv = os.path.join(output_dir, "alta_dreams_normalized.csv")
    print(f"Saving to CSV: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Create a version structured for NLP tasks
    nlp_df = df[['dream_id', 'year', 'clean_dream', 'dream_length', 'short_name', 'sex']].copy()
    nlp_df.rename(columns={'clean_dream': 'text'}, inplace=True)
    
    output_nlp_csv = os.path.join(output_dir, "alta_dreams_nlp.csv")
    print(f"Saving NLP version to CSV: {output_nlp_csv}")
    nlp_df.to_csv(output_nlp_csv, index=False)
    
    # Create JSON files for each dream for easier processing
    json_dir = os.path.join(output_dir, "json_dreams")
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    
    print(f"Creating individual JSON files in {json_dir}...")
    for i, row in df.iterrows():
        dream_data = {
            "dream_id": row['dream_id'],
            "dreamer": row['short_name'],
            "date": row['date'],
            "year": row['year'],
            "dream_text": row['clean_dream'],
            "word_count": row['dream_length']
        }
        
        with open(os.path.join(json_dir, f"{row['dream_id']}.json"), 'w') as f:
            json.dump(dream_data, f, indent=2)
    
    # Create metadata JSON
    metadata = {
        "short_name": dreamer_info["short_name"],
        "long_name": dreamer_info["long_name"],
        "n_dreams": len(df),
        "timeframe": f"{df['year'].min()}-{df['year'].max()}" if df['year'].min() and df['year'].max() else "1985-1997",
        "sex": dreamer_info["sex"],
        "description": dreamer_info["description"]
    }
    
    with open(os.path.join(output_dir, "alta_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nData normalization complete!")
    print(f"Total dreams processed: {len(df)}")
    print(f"All files saved to {output_dir}")
    
    return df

# Usage example
if __name__ == "__main__":
    # Replace with your actual file path
    input_file = "dreambank_alta.tsv"  # Rename this to your actual file
    normalized_data = normalize_dreambank_data(input_file)