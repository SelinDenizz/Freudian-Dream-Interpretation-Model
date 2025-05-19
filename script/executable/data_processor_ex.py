import sys
import os
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from script.module.data_processor import DreamBankProcessor

def main():
    parser = argparse.ArgumentParser(description="Process DreamsBank data for LLM fine-tuning")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing DreamsBank files")
    parser.add_argument('--output_dir', type=str, default="processed_data", help="Directory to save processed data")
    args = parser.parse_args()
    
    processor = DreamBankProcessor(args.input_dir, args.output_dir)
    processor.run()

if __name__ == "__main__":
    main()