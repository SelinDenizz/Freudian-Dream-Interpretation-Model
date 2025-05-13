import os
import pandas as pd
import re
import spacy
import nltk
from typing import Dict, List, Optional, Union
import json
import warnings

from script.dream_bank_normalizer import DreamNormalizer








def process_tsv_dream_file(file_path: str, output_dir: str = "processed_data") -> Optional[pd.DataFrame]:
    """
    Process a single TSV dream file using the DreamNormalizer.
    
    Args:
        file_path: Path to the TSV file to process
        output_dir: Directory to store the processed output
    
    Returns:
        DataFrame with normalized dream data or None if processing failed
    """
    try:
        # Extract dreamer ID from filename
        dreamer_id = os.path.basename(file_path).split('.')[0]
        print(f"Processing dreams for {dreamer_id}...")
        
        # Set up dreamer info
        dreamer_info = {
            'dreamer_id': dreamer_id,
            'gender': 'Unknown',
            'age': 'Unknown'
        }
        
        # Create normalizer
        normalizer = DreamNormalizer(output_dir=output_dir)
        
        # Process with normalizer
        result_df = normalizer.normalize_dream_data(file_path, dreamer_info)
        
        print(f"  Successfully processed {len(result_df)} dreams from {dreamer_id}")
        return result_df
        
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function to process all TSV files in the data directory.
    """
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "data")
    output_dir = os.path.join(parent_dir, "processed_data")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TSV files
    tsv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tsv')]
    
    if not tsv_files:
        print(f"No TSV files found in {data_dir}")
        return
    
    print(f"Found {len(tsv_files)} TSV files to process")
    
    # Process each file
    results = []
    successful_files = []
    
    for file_path in tsv_files:
        try:
            df = process_tsv_dream_file(file_path, output_dir)
            if df is not None and not df.empty:
                results.append(df)
                successful_files.append(os.path.basename(file_path))
        except Exception as e:
            print(f"  Failed to process {file_path}: {str(e)}")
            continue
    
    # Save summary report
    if successful_files:
        with open(os.path.join(output_dir, "processing_summary.txt"), 'w') as f:
            f.write(f"Successfully processed {len(successful_files)} out of {len(tsv_files)} files\n\n")
            f.write("Successful files:\n")
            for filename in successful_files:
                f.write(f"- {filename}\n")
    
    # Combine results if available
    if results:
        try:
            print(f"Successfully processed {sum(len(df) for df in results)} dreams from {len(successful_files)} files")
            
            # Combine in smaller batches to avoid memory issues
            if len(results) > 1:
                print("Combining results in batches...")
                batches = []
                batch_size = 5
                
                for i in range(0, len(results), batch_size):
                    batch = results[i:i+batch_size]
                    try:
                        batch_df = pd.concat(batch, ignore_index=True)
                        batches.append(batch_df)
                        print(f"  Batch {i//batch_size + 1} combined: {len(batch_df)} dreams")
                    except Exception as e:
                        print(f"  Warning: Could not combine batch {i//batch_size + 1}: {str(e)}")
                        # Save individual files instead
                        for j, df in enumerate(batch):
                            try:
                                idx = i + j
                                if idx < len(successful_files):
                                    filename = successful_files[idx]
                                    df.to_csv(os.path.join(output_dir, f"{filename}_processed.csv"), index=False)
                            except Exception as e2:
                                print(f"  Error saving individual file: {str(e2)}")
                
                if batches:
                    try:
                        print("Creating final combined dataset...")
                        all_dreams = pd.concat(batches, ignore_index=True)
                        all_dreams.to_csv(os.path.join(output_dir, "all_normalized_dreams.csv"), index=False)
                        print(f"Successfully created combined dataset with {len(all_dreams)} dreams")
                    except Exception as e:
                        print(f"Error creating final combined dataset: {str(e)}")
                        print("Saving individual batch files instead...")
                        for i, batch_df in enumerate(batches):
                            try:
                                batch_df.to_csv(os.path.join(output_dir, f"batch_{i+1}_dreams.csv"), index=False)
                                print(f"  Saved batch {i+1}: {len(batch_df)} dreams")
                            except Exception as e2:
                                print(f"  Error saving batch {i+1}: {str(e2)}")
            else:
                # Just one result to save
                results[0].to_csv(os.path.join(output_dir, "all_normalized_dreams.csv"), index=False)
                print(f"Saved combined dataset with {len(results[0])} dreams")
        except Exception as e:
            print(f"Error in final processing: {str(e)}")
            print(f"Individual results saved in the output directory")
    else:
        print("No dreams were successfully processed")


if __name__ == "_main_":
    main()