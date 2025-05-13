import os
import pandas as pd
import re
import spacy
import nltk
from typing import Dict, List, Optional, Union
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="spacy")
warnings.filterwarnings("ignore", message=".*error_bad_lines.*")
warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy.*")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class DreamNormalizer:
    """
    A class to normalize dream text data using NLP techniques.
    Designed for processing DreamBank TSV files.
    """
    
    def __init__(self, output_dir: str = "processed_data"):
        """
        Initialize the DreamNormalizer with output directory.
        
        Args:
            output_dir: Directory to store the processed output
        """
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.json_dir = os.path.join(output_dir, "json_dreams")
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Load spaCy model for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        # Compile common patterns for dream analysis
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for dream content analysis."""
        
        # Dream structure patterns
        self.transition_pattern = re.compile(r'\b(?:suddenly|all of a sudden|the next thing I knew|then|switch to)\b', re.IGNORECASE)
        self.waking_pattern = re.compile(r'\b(?:I woke up|then I woke up|the dream ended|I awakened)\b', re.IGNORECASE)
        self.lucid_pattern = re.compile(r'\b(?:I realized I was dreaming|I knew I was in a dream|this must be a dream)\b', re.IGNORECASE)
        self.meta_comment_pattern = re.compile(r'\b(?:I think|I felt|I remember|it seemed like|I don\'t remember)\b', re.IGNORECASE)
        self.physical_pattern = re.compile(r'\b(?:feeling of|sensation of|feeling|I felt|painful|comfortable|floating|falling)\b', re.IGNORECASE)
        self.uncertainty_pattern = re.compile(r'\b(?:maybe|perhaps|not sure|unclear|I think)\b', re.IGNORECASE)
        
        # Character and identity patterns
        self.identity_pattern = re.compile(r'\b(?:I was now|I became|I turned into|half of the time I was)\b', re.IGNORECASE)
        
        # Common features of dream reports
        self.dream_title_pattern = re.compile(r'^\s*\[([^\]]+)\]\s*')
        self.date_pattern = re.compile(r'\((\d{4}[-/]?\d{1,2}[-/]?\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}\??|\?\?/\?\?/\?\?)\)')
        
        # Extra cleanup patterns
        self.whitespace_pattern = re.compile(r'\s+')
        self.multiple_newlines = re.compile(r'\n{3,}')
        self.quote_pattern = re.compile(r'"{2,}')
        self.apostrophe_pattern = re.compile(r"'{2,}")
    
    def normalize_dream_text(self, text: str) -> str:
        """
        Normalize dream text with custom text cleaning and content markers.
        
        Args:
            text: The raw dream text to normalize
            
        Returns:
            Normalized dream text with standardized formatting and markers
        """
        if not isinstance(text, str):
            return ""
        
        try:
            # Basic text cleaning
            # Handle unicode characters
            text = text.strip()
            
            # Normalize whitespace
            text = self.whitespace_pattern.sub(' ', text)  # Replace multiple spaces with single space
            text = self.multiple_newlines.sub('\n\n', text)  # Replace multiple newlines with double newline
            
            # Remove leading/trailing whitespace from lines
            lines = text.split('\n')
            lines = [line.strip() for line in lines]
            text = '\n'.join(lines)
            
            # Extract dream title if present
            title_match = self.dream_title_pattern.match(text)
            title = None
            if title_match:
                title = title_match.group(1).strip()
                text = self.dream_title_pattern.sub('', text)
            
            # Fix quotation marks
            text = self.quote_pattern.sub('"', text)  # Multiple quotes
            text = self.apostrophe_pattern.sub("'", text)  # Multiple apostrophes
            
            # Replace curly quotes with straight quotes
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace('\'', "'").replace('\'', "'")
            
            # Process with spaCy for sentence segmentation if possible
            try:
                doc = self.nlp(text)
                
                # Process sentences with consistent formatting
                processed_sentences = []
                for sent in doc.sents:
                    sent_text = sent.text.strip()
                    if not sent_text:
                        continue
                        
                    # Mark various dream elements
                    sent_text = self._mark_dream_elements(sent_text)
                    processed_sentences.append(sent_text)
                
                # Reconstruct text with consistent formatting
                processed_text = " ".join(processed_sentences)
            except Exception as e:
                print(f"  Warning: NLP processing error: {str(e)}")
                # Fall back to basic tokenization with NLTK
                try:
                    sentences = nltk.sent_tokenize(text)
                    processed_sentences = []
                    for sent in sentences:
                        sent = sent.strip()
                        if sent:
                            sent = self._mark_dream_elements(sent)
                            processed_sentences.append(sent)
                    processed_text = " ".join(processed_sentences)
                except Exception as e2:
                    print(f"  Warning: NLTK tokenization error: {str(e2)}")
                    # Last resort - just use the original text with markers
                    processed_text = self._mark_dream_elements(text)
            
            # Add title if found
            if title:
                processed_text = f"[DREAM_TITLE: {title}]\n{processed_text}"
            
            return processed_text
            
        except Exception as e:
            print(f"  Error in text normalization: {str(e)}")
            # Return original text if all else fails
            return text
    
    def _mark_dream_elements(self, text: str) -> str:
        """
        Mark specific dream elements in the text with standardized tags.
        
        Args:
            text: A single sentence from a dream
            
        Returns:
            Text with standardized markers added
        """
        try:
            # Mark scene transitions
            text = self.transition_pattern.sub(lambda m: f'[TRANSITION] {m.group(0)}', text)
            
            # Mark waking up
            text = self.waking_pattern.sub(lambda m: f'[WAKING] {m.group(0)}', text)
            
            # Mark lucid dreaming awareness
            text = self.lucid_pattern.sub(lambda m: f'[LUCID_MOMENT] {m.group(0)}', text)
            
            # Mark meta-commentary about the dream
            text = self.meta_comment_pattern.sub(lambda m: f'[META_COMMENT] {m.group(0)}', text)
            
            # Mark physical sensations
            text = self.physical_pattern.sub(lambda m: f'[PHYSICAL_SENSATION] {m.group(0)}', text)
            
            # Mark identity shifts
            text = self.identity_pattern.sub(lambda m: f'[IDENTITY_SHIFT] {m.group(0)}', text)
            
            # Mark uncertainty
            text = self.uncertainty_pattern.sub(lambda m: f'[UNCERTAINTY] {m.group(0)}', text)
            
            # Standardize uncertainty markers
            text = re.sub(r'\(\?\)', r'[UNCERTAIN]', text)
            
            return text
        except Exception as e:
            print(f"  Error marking dream elements: {str(e)}")
            return text  # Return original if marking fails
    
    def extract_date(self, text: str) -> Optional[str]:
        """
        Extract date information from dream text or date field.
        
        Args:
            text: Text that may contain date information
            
        Returns:
            Standardized date string or None if no date found
        """
        if not isinstance(text, str):
            return None
            
        try:
            date_match = self.date_pattern.search(text)
            if date_match:
                date_text = date_match.group(1)
                return date_text
        except Exception as e:
            print(f"  Error extracting date: {str(e)}")
            
        return None
    
    def normalize_dream_data(self, file_path: str, dreamer_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        Normalize dream data from a TSV file.
        
        Args:
            file_path: Path to the TSV file containing dream data
            dreamer_info: Optional metadata about the dreamer
            
        Returns:
            DataFrame containing the normalized data
        """
        # Set default dreamer info if none provided
        if dreamer_info is None:
            dreamer_id = os.path.basename(file_path).split('.')[0]
            dreamer_info = {
                'dreamer_id': dreamer_id,
                'gender': 'Unknown',
                'age': 'Unknown'
            }
            
        try:
            # Try to read the TSV file with pandas
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8', quotechar='"', 
                              engine='python', on_bad_lines='skip')
        except Exception as e:
            print(f"  Error reading file with pandas: {str(e)}")
            print(f"  Trying alternative approach...")
            
            # Manual parsing as fallback
            df = self._manual_parse_tsv(file_path)
            
        if df is None or df.empty:
            raise ValueError(f"Could not parse file: {file_path}")
            
        # Identify relevant columns
        dream_col = self._identify_dream_column(df)
        id_col = self._identify_id_column(df)
        date_col = self._identify_date_column(df)
        
        print(f"  Using column '{dream_col}' for dream text")
        
        # Create normalized dataframe
        normalized_data = []
        
        for idx, row in df.iterrows():
            try:
                # Skip if dream text is missing
                if dream_col not in row or pd.isna(row[dream_col]):
                    continue
                    
                # Get dream ID
                if id_col and id_col in row and not pd.isna(row[id_col]):
                    dream_id = str(row[id_col]).strip('"')
                else:
                    dream_id = f"{idx+1}"
                    
                # Get date if available
                date = None
                if date_col and date_col in row and not pd.isna(row[date_col]):
                    date = self.extract_date(str(row[date_col]))
                
                # Extract potential date from dream text if no specific date column
                if not date and dream_col in row:
                    date = self.extract_date(str(row[dream_col]))
                
                # Get raw dream text
                raw_dream = str(row[dream_col])
                
                # Normalize dream text
                clean_dream = self.normalize_dream_text(raw_dream)
                
                # Create entry
                entry = {
                    'dream_id': dream_id,
                    'date': date if date else "Unknown",
                    'raw_dream': raw_dream,
                    'clean_dream': clean_dream
                }
                
                # Add dreamer info
                for key, value in dreamer_info.items():
                    entry[key] = value
                    
                normalized_data.append(entry)
                
            except Exception as e:
                print(f"  Error processing row {idx}: {str(e)}")
                continue
        
        # Create dataframe
        result_df = pd.DataFrame(normalized_data)
        
        # Save individual dreams as JSON
        try:
            self._save_dreams_as_json(result_df, dreamer_info['dreamer_id'])
        except Exception as e:
            print(f"  Error saving JSON files: {str(e)}")
        
        # Save as CSV
        try:
            csv_path = os.path.join(self.output_dir, f"{dreamer_info['dreamer_id']}_normalized.csv")
            result_df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"  Error saving CSV file: {str(e)}")
        
        return result_df
    
    def _manual_parse_tsv(self, file_path: str) -> pd.DataFrame:
        """
        Manually parse a TSV file when pandas fails.
        
        Args:
            file_path: Path to the TSV file
            
        Returns:
            DataFrame with parsed content
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                
            if not lines:
                return pd.DataFrame()
                
            # Parse header
            header = lines[0].strip().split('\t')
            header = [h.strip('"') for h in header]
            
            # Parse content
            data = []
            for i, line in enumerate(lines[1:], 1):
                if not line.strip():
                    continue
                    
                parts = line.strip().split('\t')
                row = {}
                
                # Match parts to header
                for j, part in enumerate(parts):
                    if j < len(header):
                        row[header[j]] = part.strip('"')
                    else:
                        row[f'column_{j}'] = part.strip('"')
                        
                # Ensure all headers have values
                for h in header:
                    if h not in row:
                        row[h] = ""
                        
                data.append(row)
                
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"  Error in manual TSV parsing: {str(e)}")
            
            # Last resort - try to create a minimal DataFrame
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Create a minimal single-column DataFrame with the content
                return pd.DataFrame({'dream': [content]})
            
            except Exception as e2:
                print(f"  Fatal error reading file: {str(e2)}")
                return pd.DataFrame()
    
    def _identify_dream_column(self, df: pd.DataFrame) -> str:
        """
        Identify which column contains the dream text.
        
        Args:
            df: DataFrame with dream data
            
        Returns:
            Column name containing dream text
        """
        # Check for common dream column names
        for col in ['dream', 'dream_text', 'text', 'content', 'narrative']:
            if col in df.columns:
                return col
                
        # If no standard name, try to find column with longest text
        try:
            if len(df) > 0:
                avg_lengths = {}
                for col in df.columns:
                    if df[col].dtype == object:
                        # Safely compute average length
                        try:
                            avg_lengths[col] = df[col].astype(str).str.len().mean()
                        except:
                            avg_lengths[col] = 0
                            
                if avg_lengths:
                    return max(avg_lengths, key=avg_lengths.get)
        except Exception as e:
            print(f"  Error identifying dream column: {str(e)}")
        
        # Fallback to first column
        return df.columns[0]
    
    def _identify_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify which column contains dream IDs.
        
        Args:
            df: DataFrame with dream data
            
        Returns:
            Column name containing dream IDs or None
        """
        for col in ['id', 'n', 'dream_id', 'entry_id']:
            if col in df.columns:
                return col
        return None
    
    def _identify_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify which column contains dates.
        
        Args:
            df: DataFrame with dream data
            
        Returns:
            Column name containing dates or None
        """
        for col in ['date', 'dream_date']:
            if col in df.columns:
                return col
        return None
    
    def _save_dreams_as_json(self, df: pd.DataFrame, dreamer_id: str):
        """
        Save individual dreams as JSON files.
        
        Args:
            df: DataFrame with normalized dream data
            dreamer_id: ID of the dreamer
        """
        for idx, row in df.iterrows():
            try:
                json_data = {k: v for k, v in row.to_dict().items() if not pd.isna(v)}
                dream_id = json_data.get('dream_id', f"{idx+1}")
                json_path = os.path.join(self.json_dir, f"{dreamer_id}_{dream_id}.json")
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"  Error saving JSON for dream {dream_id}: {str(e)}")