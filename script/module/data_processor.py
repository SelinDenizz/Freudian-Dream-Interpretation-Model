import os
import json
import re
import pandas as pd
from tqdm import tqdm

class DreamBankProcessor:
    def __init__(self, input_dir, output_dir="processed_data"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.formats_dir = os.path.join(self.output_dir, "fine_tuning_formats")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.formats_dir, exist_ok=True)

    def find_dream_files(self):
        data_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(('.tsv', '.txt', '.csv')) and "README" not in file:
                    data_files.append(os.path.join(root, file))
        return data_files

    def read_dream_file(self, file_path):
        try:
            df = pd.read_csv(file_path, sep='\t')
            if len(df.columns) != 3 or not any('dream' in col.lower() for col in df.columns):
                df = pd.read_csv(file_path, sep='\t', header=None)
                if len(df.columns) == 3:
                    df.columns = ["n", "date", "dream"]
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace('"', '', regex=False)
            return df
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def extract_dreams_from_df(self, df, dreamer_name):
        dreams = []
        for i, row in df.iterrows():
            dream_text = row.get('dream') if 'dream' in row else row.get(2)
            if not isinstance(dream_text, str) or pd.isna(dream_text):
                continue
            dream_text = dream_text.replace('""', '"')
            dream_id = f"{dreamer_name}_{str(row.get('n', i)).zfill(5)}"
            dream_data = {
                "dream_id": dream_id,
                "dreamer": dreamer_name,
                "text": dream_text
            }
            if 'date' in row and not pd.isna(row['date']):
                dream_data["date"] = row['date']
                year_match = re.search(r'\((\d{4})', row['date'])
                if year_match:
                    dream_data["year"] = year_match.group(1)
                else:
                    date_match = re.search(r'\((\d{1,2})/(\d{1,2})/(\d{2})\)', row['date'])
                    if date_match:
                        year = int(date_match.group(3))
                        dream_data["year"] = f"20{year:02d}" if year < 50 else f"19{year:02d}"
            dreams.append(dream_data)
        return dreams

    def save_json(self, data, filename):
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(data, f, indent=2)

    # Creates sample prompts for debugging and testing
    def save_sample_prompts(self, dreams):
        path = os.path.join(self.formats_dir, "sample_prompts.txt")
        with open(path, 'w') as f:
            for i, dream in enumerate(dreams[:5]):
                f.write(f"DREAM SAMPLE {i+1}:\n")
                f.write(f"Please analyze this dream from a Freudian perspective:\n\n{dream['text']}\n\n")
                f.write("-" * 80 + "\n\n")

    # Formats the data for different LLMs
    def format_llm_data(self, dreams, format_type):
        path = os.path.join(self.formats_dir, f"dreambank_finetune_{format_type}.jsonl")
        with open(path, 'w') as f:
            for dream in dreams:
                if format_type == 'llama':
                    text = f"<s>[INST] <<SYS>>\nYou are a Freudian dream analyst...<</SYS>>\n\nPlease analyze this dream:\n\n{dream['text']} [/INST]\n\nI'll analyze this dream...\n</s>"
                    f.write(json.dumps({"text": text}) + "\n")
                elif format_type == 'mistral':
                    text = f"<s>[INST] Please analyze this dream:\n\n{dream['text']} [/INST]\n\nI'll analyze this dream...\n</s>"
                    f.write(json.dumps({"text": text}) + "\n")
                elif format_type == 'chatml':
                    item = {
                        "messages": [
                            {"role": "system", "content": "You are a Freudian dream analyst..."},
                            {"role": "user", "content": f"Please analyze this dream:\n\n{dream['text']}"},
                            {"role": "assistant", "content": "I'll analyze this dream..."}
                        ]
                    }
                    f.write(json.dumps(item) + "\n")

    def run(self):
        data_files = self.find_dream_files()
        all_dreams = []
        for file_path in data_files:
            dreamer_name = os.path.basename(file_path).split('.')[0].lower()
            df = self.read_dream_file(file_path)
            if df is not None:
                dreams = self.extract_dreams_from_df(df, dreamer_name)
                all_dreams.extend(dreams)

        self.save_json(all_dreams, "all_dreams.json")
        self.format_llm_data(all_dreams, 'llama')
        self.format_llm_data(all_dreams, 'mistral')
        self.format_llm_data(all_dreams, 'chatml')
        self.save_sample_prompts(all_dreams)
        print("Dream processing completed.")
