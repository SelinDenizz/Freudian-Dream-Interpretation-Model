import os
import re
import pandas as pd

class DreamBankNormalizer:
    """
    Enhanced class to normalize DreamsBank data from different formats into structured data.
    Handles various text normalization cases for dream analysis.
    """
    
    def __init__(self, output_dir="processed_data"):
        """Initialize the normalizer with output directory."""
        # Original initialization code...
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create directory for JSON files
        self.json_dir = os.path.join(output_dir, "json_dreams")
        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir)
            
        # Compile regex patterns used throughout the normalizer
        self._compile_regex_patterns()
        
    def _compile_regex_patterns(self):
        """Compile regex patterns used for text normalization."""
        # Date patterns
        self.date_pattern = re.compile(r'\((\d{4}[-/]?\d{1,2}[-/]?\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}\??|\?\?/\?\?/\?\?)\s*(?:\(.*?\))?\)')
        self.partial_date_pattern = re.compile(r'\((\d{4}[-/]?\?\?[-/]?\?\?|\d{1,2}[-/]\?\?[-/]\d{2,4}|\d{4}-\d{1,2}-\?\?)\)')
        
        # ID patterns
        self.id_pattern = re.compile(r'^["\']?(\d{1,4})["\']?$')
        self.complex_id_pattern = re.compile(r'^["\']?([A-Za-z0-9_-]+\(?[A-Za-z]?\)?-\d{1,3})["\']?$')
        
        # Text patterns
        self.nested_quotes_pattern = re.compile(r'""([^"]*?)""')
        self.excess_whitespace_pattern = re.compile(r'\s{2,}')
        self.paragraph_break_pattern = re.compile(r'\n{2,}')
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        
        # Demographic patterns
        self.demographic_pattern = re.compile(r'\(([MF]),\s*age\s*(\d+)\)')
        
        # Common markers
        self.recurring_phrases = [
            "I woke up", "End of dream", "Then I woke up", "Dream ended", 
            "I don't remember any more", "That's all I remember"
        ]
        
        # Special character patterns
        self.special_char_pattern = re.compile(r'[©™®]')
        self.non_ascii_pattern = re.compile(r'[^\x00-\x7F]')
        
        # New patterns for enhanced normalization
        self.bracket_annotation_pattern = re.compile(r'\[([^\]]+)\]')
        self.editorial_comment_pattern = re.compile(r'\[\s*(?:NOTE|COMMENT):\s*([^\]]+)\]', re.IGNORECASE)
        self.meta_reference_pattern = re.compile(r'\[\s*(?:I woke up|This was a dream|In the dream|The dream)\s+([^\]]+)\]', re.IGNORECASE)
        self.indecipherable_pattern = re.compile(r'\[\s*(?:indecipherable|can\'t read|unclear|illegible)\s*(?:[^\]]+)?\]', re.IGNORECASE)
        self.dream_title_pattern = re.compile(r'^\s*\[([^\]]+)\]\s*')
        self.dream_sequence_pattern = re.compile(r'(?:Next scene|Then|Suddenly|All of a sudden|Switch to):', re.IGNORECASE)
        self.temporal_marker_pattern = re.compile(r'\b(?:Last night|Yesterday|Later|Soon|Then|Next|Afterwards|Eventually)\b', re.IGNORECASE)
        self.narrative_shift_pattern = re.compile(r'\b(?:I was now|I became|I turned into|Half of the time I was|Sometimes I was)\b', re.IGNORECASE)
        self.dream_awareness_pattern = re.compile(r'\b(?:I realized I was dreaming|I knew I was in a dream|This must be a dream|In my dream)\b', re.IGNORECASE)
        self.recurring_dream_pattern = re.compile(r'\b(?:I\'ve had this dream before|recurring dream|same dream again|this dream repeats)\b', re.IGNORECASE)
        
    def _normalize_entry_identifiers(self, text):
        """Normalize various dream entry ID formats for consistency."""
        if not isinstance(text, str):
            return text
            
        # Handle various ID formats (#101 vs "345" vs. 1)
        text = re.sub(r'^["\'](#?\d+)["\']', r'\1', text)
        
        # Standardize alphanumeric IDs with letter suffixes
        text = re.sub(r'^["\']([\w\d]+-\d+[a-z]?)["\']', r'\1', text)
        
        # Normalize IDs with explicit code indicators
        text = re.sub(r'^Code\s+(\d+)', r'Code-\1', text, flags=re.IGNORECASE)
        
        return text

    def _normalize_date_formats(self, text):
        """Normalize multiple date formats in dream entries."""
        if not isinstance(text, str):
            return text
            
        # Convert various date formats to standardized format
        # Match dates in parentheses with various formats and standardize them
        def standardize_date(match):
            date_text = match.group(1)
            
            # Handle completely uncertain dates
            if '??' in date_text:
                return "(DATE_UNCERTAIN)"
                
            # Try to parse the date format
            if re.match(r'\d{4}-\d{2}-\d{2}', date_text):
                return f"(DATE: {date_text})"  # Already in YYYY-MM-DD format
            elif re.match(r'\d{8}', date_text):  # YYYYMMDD format
                return f"(DATE: {date_text[:4]}-{date_text[4:6]}-{date_text[6:8]})"
            elif re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', date_text):  # MM/DD/YY or MM/DD/YYYY
                parts = date_text.split('/')
                year = parts[2].zfill(2)  # Ensure at least 2 digits
                if len(year) == 2:
                    year = '19' + year if int(year) > 50 else '20' + year  # Simple century heuristic
                return f"(DATE: {year}-{parts[0].zfill(2)}-{parts[1].zfill(2)})"
                
            # Return original if format not recognized
            return f"(DATE: {date_text})"
        
        # Apply the date standardization
        text = self.date_pattern.sub(standardize_date, text)
        
        # Handle partial dates with question marks
        text = self.partial_date_pattern.sub(r'(DATE_PARTIAL)', text)
        
        return text

    def _normalize_quotation_marks(self, text):
        """Normalize inconsistent quotation mark usage."""
        if not isinstance(text, str):
            return text
            
        # Replace curly quotes with straight quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('\'', "'").replace('\'', "'")
        
        # Fix triple or more consecutive quotation marks
        text = re.sub(r'"{3,}', '"', text)
        text = re.sub(r"'{3,}", "'", text)
        
        # Handle double nested quotes consistently
        text = re.sub(r'""([^"]*?)""', r'"\1"', text)
        
        # Fix mismatched opening/closing quotes - add missing closing quote
        # Count quotes in each line and fix if odd number
        lines = text.split('\n')
        for i in range(len(lines)):
            if lines[i].count('"') % 2 == 1:
                # Only add missing quote if there's an open one without a close
                if lines[i].find('"') < lines[i].rfind('"'):
                    pass  # Multiple quotes with one missing, more complex to fix
                else:
                    # Simple case of a single unclosed quote
                    lines[i] = lines[i] + '"'
        text = '\n'.join(lines)
        
        return text

    def _normalize_uncertain_references(self, text):
        """Normalize patterns showing uncertainty in dream recall."""
        if not isinstance(text, str):
            return text
            
        # Standardize common uncertainty markers
        text = re.sub(r'\(\?\)', r'[UNCERTAIN]', text)
        
        # Standardize uncertain name references like "My ex?"
        text = re.sub(r'([A-Z][a-z]+\s+[A-Za-z]+)\?', r'\1 [UNCERTAIN_IDENTITY]', text)
        
        # Mark unclear elements that appear in quotes
        text = re.sub(r'""([^"]+)""', r'"\1" [UNCLEAR_REFERENCE]', text)
        
        return text

    def _normalize_scene_transitions(self, text):
        """Normalize various scene transition markers in dreams."""
        if not isinstance(text, str):
            return text
            
        # Standardize common scene transition phrases
        transitions = [
            (r'(?:\-\-+|\*\*+)', '[MAJOR_SCENE_BREAK]'),
            (r'(?<!\w)All of a sudden(?!\w)', '[SUDDEN_TRANSITION]'),
            (r'(?<!\w)The next thing I knew(?!\w)', '[SCENE_TRANSITION]'),
            (r'(?<!\w)The scene changed(?!\w)', '[SCENE_TRANSITION]'),
            (r'(?<!\w)Before I knew it(?!\w)', '[SCENE_TRANSITION]'),
            (r'(?<!\w)Suddenly(?!\w)', '[SUDDEN_TRANSITION]'),
            (r'(?<!\w)Next scene(?!\w)', '[SCENE_TRANSITION]'),
            (r'(?<!\w)Later:(?!\w)', '[TIME_TRANSITION]'),
            (r'(?<!\w)Next:(?!\w)', '[SCENE_TRANSITION]'),
            (r'(?<!\w)Another place(?!\w)', '[SETTING_TRANSITION]'),
            (r'(?<!\w)Switch to(?!\w)', '[SCENE_TRANSITION]')
        ]
        
        for pattern, replacement in transitions:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Handle slash-separated scene transitions
        text = re.sub(r'(?<![a-zA-Z0-9])/(?![a-zA-Z0-9])', ' [SEGMENT_BREAK] ', text)
        
        return text

    def _normalize_meta_references(self, text):
        """Normalize meta-references about the dream documentation process."""
        if not isinstance(text, str):
            return text
            
        # Standardize meta-commentary patterns
        meta_patterns = [
            (r'\[Kids are so cruel to each other - a thought I just had as I\'m writing this\]', '[META_THOUGHT: Kids are so cruel to each other]'),
            (r'\[Woke up here, actually needing to pee\.\]', '[PHYSICAL_WAKING_CAUSE: needed to urinate]'),
            (r'The initials [A-Z]+ come after I am awake', '[POST_WAKE_INSIGHT]'),
            (r'I wake up very out of breath', '[PHYSICAL_WAKING_STATE: breathless]'),
            (r'My memory of this dream is vague', '[VAGUE_RECALL]'),
            (r'I don\'t remember', '[MEMORY_GAP]'),
            (r'That\'s all I remember', '[END_OF_RECALL]')
        ]
        
        for pattern, replacement in meta_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _normalize_dream_awareness(self, text):
        """Enhance normalization of lucid dream indicators and dream awareness."""
        if not isinstance(text, str):
            return text
            
        # Expand on lucid dreaming references
        lucidity_patterns = [
            (r'I know I am sleeping but I can\'t wake up', '[SLEEP_PARALYSIS]'),
            (r'I realized I was dreaming', '[LUCID_AWARENESS]'),
            (r'I knew I was in a dream', '[LUCID_AWARENESS]'),
            (r'I thought I woke up but was still dreaming', '[FALSE_AWAKENING]'),
            (r'I dreamed I woke up', '[FALSE_AWAKENING]'),
            (r'That was a dream too', '[NESTED_DREAM]')
        ]
        
        for pattern, replacement in lucidity_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _normalize_brackets_and_notes(self, text):
        """Normalize various types of bracketed comments and notes."""
        if not isinstance(text, str):
            return text
            
        # Handle notes with a consistent format
        text = re.sub(r'\[Note: ([^\]]+)\]', r'[NOTE: \1]', text, flags=re.IGNORECASE)
        
        # Handle measurements with a consistent format
        text = re.sub(r'\[abt\. (\d+)" x (\d+)"\]', r'[MEASUREMENT: \1" x \2"]', text)
        
        # Handle "see drawing" references 
        text = re.sub(r'\[see drawing\]', r'[DRAWING_REFERENCE]', text, flags=re.IGNORECASE)
        
        # Handle no continuity markers
        text = re.sub(r'\[no continuity\.\]', r'[DISCONTINUITY]', text, flags=re.IGNORECASE)
        
        # Handle either/or constructs for unclear memory
        text = re.sub(r'\[Either/or\]', r'[UNCERTAIN_RECALL]', text, flags=re.IGNORECASE)
        
        # Handle "lost" markers for forgotten segments
        text = re.sub(r'\[lost\]', r'[MEMORY_GAP]', text, flags=re.IGNORECASE)
        
        return text

    def _normalize_paragraph_formatting(self, text):
        """Normalize inconsistent paragraph formatting."""
        if not isinstance(text, str):
            return text
            
        # Standardize multiple blank lines to a single paragraph break
        text = self.paragraph_break_pattern.sub('\n\n', text)
        
        # Standardize inconsistent spacing
        text = self.excess_whitespace_pattern.sub(' ', text)
        
        # Remove excess whitespace at beginning/end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        return text

    def _normalize_continuity_markers(self, text):
        """Normalize references to dream continuity and connections."""
        if not isinstance(text, str):
            return text
            
        # Standardize references to previous dreams
        text = re.sub(r'This dream is a continuation of dream (\d+)', 
                    r'[DREAM_CONTINUATION: \1]', text)
        
        # Mark recurring dream references
        text = re.sub(r'I\'ve had this dream before', 
                    r'[RECURRING_DREAM]', text, flags=re.IGNORECASE)
        
        # Mark dream frequency indicators
        text = re.sub(r'(?<!\w)(often|frequently|recurring|repeatedly|again) (?:have|had|dream|dreamed) this',
                    r'[DREAM_FREQUENCY: \1]', text, flags=re.IGNORECASE)
        
        return text

    def _normalize_reference_phrases(self, text):
        """Normalize phrases used to break the fourth wall or address readers."""
        if not isinstance(text, str):
            return text
            
        # Standardize explanatory phrases directed at readers
        phrases = [
            r'you know the type',
            r'if you know what I mean',
            r'as you can imagine',
            r'you get the idea'
        ]
        
        for phrase in phrases:
            text = re.sub(fr'\b{phrase}\b', f'[READER_REFERENCE]', text, flags=re.IGNORECASE)
        
        return text

    def _normalize_special_characters(self, text):
        """Normalize special characters and symbols."""
        if not isinstance(text, str):
            return text
            
        # Convert ampersands to 'and'
        text = re.sub(r'(\s)&(\s)', r'\1and\2', text)
        
        # Standardize inconsistent time formats
        text = re.sub(r'(\d{1,2}):(\d{2}) (p\.m\.|a\.m\.)', r'\1:\2 \3', text)
        text = re.sub(r'(\d{1,2}) o\'clock', r'\1:00', text, flags=re.IGNORECASE)
        
        # Standardize money references
        text = re.sub(r'\$(\d+)\.(\d{2})', r'[MONEY: $\1.\2]', text)
        
        return text

    def _normalize_pronouns_and_references(self, text):
        """Normalize ambiguous pronoun references and identity markers."""
        if not isinstance(text, str):
            return text
            
        # Mark potentially unclear pronoun references
        # Note: This is difficult to do comprehensively without full NLP parsing
        # Here we just mark some common patterns
        
        # Mark shifts between first and third person
        text = re.sub(r'(?<=[.!?]\s)(?:I was|I am)(?=\s.*?(?:he|she) (?:was|is))', 
                    r'[POV_SHIFT] I was', text)
        
        # Mark pronoun indeterminacy
        text = re.sub(r'\((?:him|her)\)', r'[GENDER_INDETERMINATE]', text)
        text = re.sub(r'\((?:he|she)\)', r'[GENDER_INDETERMINATE]', text)
        text = re.sub(r'\(him/her\)', r'[GENDER_INDETERMINATE]', text)
        
        return text

    def _normalize_waking_markers(self, text):
        """Enhance normalization of waking transitions and physical sensations."""
        if not isinstance(text, str):
            return text
            
        # Expand waking transition markers
        waking_patterns = [
            (r'I wake up very out of breath', '[PHYSICAL_WAKING: breathless]'),
            (r'I can\'t breathe and my husband has to help get me awake', '[SLEEP_DISTRESS]'),
            (r'I am frightened in the dream and in actuality I have sleep paralysis', '[SLEEP_PARALYSIS]'),
            (r'I woke up with a dry throat', '[PHYSICAL_WAKING: dry throat]'),
            (r'When I awaken I tell myself', '[POST_WAKE_THOUGHT]')
        ]
        
        for pattern, replacement in waking_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _normalize_tabs_and_formatting(self, text):
        """Normalize tab characters and other formatting inconsistencies."""
        if not isinstance(text, str):
            return text
            
        # Replace tab characters with a standard amount of space
        text = text.replace('\t', '    ')
        
        # Standardize multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize start/end whitespace
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        return text

    def _normalize_verbatim_expressions(self, text):
        """Normalize verbal expressions and fillers."""
        if not isinstance(text, str):
            return text
            
        # Standardize non-standard expressions
        expressions = [
            (r'\bdah dah dah\b', '[VERBAL_FILLER]'),
            (r'\bHokay\b', 'Okay'),
            (r'\bO dear\b', 'Oh dear'),
            (r'\bGeez\b', 'Jeez'),
            (r'\bGolly\b', 'Golly')
        ]
        
        for pattern, replacement in expressions:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _normalize_dream_logic_transitions(self, text):
        """Normalize abrupt scene changes and dream logic transitions."""
        if not isinstance(text, str):
            return text
            
        # Standardize common dream transition phrases
        text = re.sub(r'\bAll of a sudden\b', 'Suddenly', text, flags=re.IGNORECASE)
        text = re.sub(r'\bThe next thing I knew\b', 'Then', text, flags=re.IGNORECASE)
        text = re.sub(r'\bBefore I knew it\b', 'Then', text, flags=re.IGNORECASE)
        
        # Identify and mark scene transitions
        text = self.dream_sequence_pattern.sub(r'[SCENE_TRANSITION] \1', text)
        
        return text
        
    def _normalize_bracket_annotations(self, text):
        """Normalize various types of bracketed annotations in dream text."""
        if not isinstance(text, str):
            return text
            
        # Handle editorial comments - preserve but standardize format
        text = self.editorial_comment_pattern.sub(r'[EDITORIAL: \1]', text)
        
        # Handle meta-references to the dream state
        text = self.meta_reference_pattern.sub(r'[DREAM_AWARENESS: \1]', text)
        
        # Handle indecipherable or unclear content
        text = self.indecipherable_pattern.sub('[UNCLEAR_CONTENT]', text)
        
        # Handle standard annotations that should be preserved
        # This preserves important annotations but in a standardized format
        text = self.bracket_annotation_pattern.sub(lambda m: f'[NOTE: {m.group(1)}]' 
                                                if not any(p in m.group(0) for p in ['EDITORIAL', 'DREAM_AWARENESS', 'UNCLEAR_CONTENT']) 
                                                else m.group(0), text)
        
        return text
        
    def _normalize_dream_titles(self, text):
        """Extract and normalize dream titles that appear in brackets at the beginning."""
        if not isinstance(text, str):
            return text
            
        # Check for dream title at the beginning
        title_match = self.dream_title_pattern.match(text)
        title = None
        
        if title_match:
            title = title_match.group(1).strip()
            # Remove the title from the text
            text = self.dream_title_pattern.sub('', text)
            # Add standardized title at the beginning
            text = f"[DREAM_TITLE: {title}]\n{text}"
            
        return text
        
    def _normalize_nested_quotes(self, text):
        """Handle and normalize nested quotation marks in dialog."""
        if not isinstance(text, str):
            return text
            
        # Replace double nested quotes with single quotes
        text = re.sub(r'""([^"]*?)""', r'"\1"', text)
        
        # Replace curly quotes with straight quotes consistently
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('\'', "'").replace('\'', "'")
        
        # Fix cases with multiple consecutive quotes
        text = re.sub(r'"{3,}', '"', text)
        text = re.sub(r"'{3,}", "'", text)
        
        return text
        
    def _normalize_narrative_perspective(self, text):
        """Normalize shifts in narrative perspective and dream identity."""
        if not isinstance(text, str):
            return text
            
        # Mark identity shifts while preserving the original content
        text = self.narrative_shift_pattern.sub(lambda m: f'[IDENTITY_SHIFT] {m.group(0)}', text)
        
        # Mark dream awareness moments
        text = self.dream_awareness_pattern.sub(lambda m: f'[LUCID_MOMENT] {m.group(0)}', text)
        
        return text
        
    def _normalize_temporal_markers(self, text):
        """Standardize temporal markers and time transitions in dreams."""
        if not isinstance(text, str):
            return text
            
        # Mark major temporal transitions
        text = self.temporal_marker_pattern.sub(lambda m: f'[TIME_MARKER] {m.group(0)}', text)
        
        # Standardize time format (10:30 p.m. vs 10:30 PM)
        text = re.sub(r'(\d{1,2}):(\d{2})\s*([ap])\.m\.', r'\1:\2 \3m', text, flags=re.IGNORECASE)
        
        return text
        
    def _normalize_recurring_dream_references(self, text):
        """Normalize references to recurring dreams or dream patterns."""
        if not isinstance(text, str):
            return text
            
        # Mark recurring dream references
        text = self.recurring_dream_pattern.sub(lambda m: f'[RECURRING_DREAM] {m.group(0)}', text)
        
        return text
        
    def _normalize_meta_commentary(self, text):
        """Normalize dreamer's reflections and commentary about the dream."""
        if not isinstance(text, str):
            return text
            
        # Pattern for identifying commentary phrases
        commentary_pattern = re.compile(r'\b(?:I think|I feel like|I believe|It seemed that|This probably means|This reminds me of)\b', re.IGNORECASE)
        
        # Mark meta-commentary while preserving original text
        text = commentary_pattern.sub(lambda m: f'[META_COMMENT] {m.group(0)}', text)
        
        return text
        
    def _normalize_physical_sensations(self, text):
        """Normalize descriptions of physical sensations in dreams."""
        if not isinstance(text, str):
            return text
            
        # Pattern for physical sensation descriptions
        sensation_pattern = re.compile(r'\b(?:I felt|feeling of|sensation of|feeling|I experienced|painful|comfortable|uncomfortable|heavy|light|floating|falling)\b', re.IGNORECASE)
        
        # Mark physical sensations for analysis
        text = sensation_pattern.sub(lambda m: f'[PHYSICAL_SENSATION] {m.group(0)}', text)
        
        return text
        
    def _normalize_dialogue(self, text):
        """Normalize and standardize dialogue formatting."""
        if not isinstance(text, str):
            return text
            
        # Handle different dialogue patterns
        # Direct speech with quotes
        speech_pattern = re.compile(r'"([^"]+)"(?:\s*(?:he|she|I|they|we|name)\s+(?:said|asked|replied|shouted|whispered|called|answered))')
        text = speech_pattern.sub(r'[DIALOGUE] "\1"', text)
        
        # Reported speech
        reported_pattern = re.compile(r'\b(?:he|she|I|they|we|name)\s+(?:said|told me|asked|mentioned) that\b', re.IGNORECASE)
        text = reported_pattern.sub(lambda m: f'[REPORTED_SPEECH] {m.group(0)}', text)
        
        return text
        
    def _normalize_waking_transitions(self, text):
        """Normalize references to waking up or transitions between dream states."""
        if not isinstance(text, str):
            return text
            
        # Pattern for waking transitions
        waking_pattern = re.compile(r'\b(?:I woke up|Then I woke up|I woke up from this dream|I awakened|I came to|The dream ended|I awoke)\b', re.IGNORECASE)
        
        # Mark waking transitions
        text = waking_pattern.sub(lambda m: f'[WAKING] {m.group(0)}', text)
        
        # Pattern for false awakenings
        false_pattern = re.compile(r'\b(?:I thought I woke up|I dreamed I woke up|I woke up but was still dreaming)\b', re.IGNORECASE)
        
        # Mark false awakenings
        text = false_pattern.sub(lambda m: f'[FALSE_AWAKENING] {m.group(0)}', text)
        
        return text

    def _standardize_dream_structure(self, text):
        """
        Apply a comprehensive sequence of normalization functions to standardize dream structure,
        including boundaries, transitions, and content markers.
        """
        if not isinstance(text, str):
            return text
            
        # First level - basic structural cleanup
        text = self._normalize_entry_identifiers(text)
        text = self._normalize_date_formats(text)
        text = self._normalize_tabs_and_formatting(text)
        text = self._normalize_paragraph_formatting(text)
        
        # Second level - text elements
        text = self._normalize_quotation_marks(text)
        text = self._normalize_brackets_and_notes(text)
        text = self._normalize_special_characters(text)
        text = self._normalize_verbatim_expressions(text)
        
        # Third level - content and meaning markers
        text = self._normalize_dream_titles(text)
        text = self._normalize_bracket_annotations(text)
        text = self._normalize_uncertain_references(text)
        text = self._normalize_scene_transitions(text)
        text = self._normalize_meta_references(text)
        text = self._normalize_continuity_markers(text)
        text = self._normalize_reference_phrases(text)
        
        # Fourth level - semantic content analysis markers
        text = self._normalize_narrative_perspective(text)
        text = self._normalize_pronouns_and_references(text)
        text = self._normalize_temporal_markers(text)
        text = self._normalize_dream_logic_transitions(text)
        # Note: The following methods have been removed as requested
        # text = self._normalize_characters(text)
        # text = self._normalize_sexual_content(text)
        # text = self._normalize_spatial_references(text)
        # text = self._normalize_dream_specific_elements(text)
        # text = self._normalize_emotional_content(text)
        # text = self._normalize_reality_checks(text)
        text = self._normalize_physical_sensations(text)
        text = self._normalize_dream_awareness(text)
        text = self._normalize_dialogue(text)
        text = self._normalize_waking_transitions(text)
        text = self._normalize_waking_markers(text)
        text = self._normalize_recurring_dream_references(text)
        text = self._normalize_meta_commentary(text)
        
        return text
    
    def normalize_dreambank_data(self, input_file, dreamer_info=None):
        """
        Normalize DreamsBank data from the input file format to structured JSON and CSV formats.
        
        Args:
            input_file: Path to the file containing dream data
            dreamer_info: Optional dictionary with metadata about the dreamer
                          Default provides basic info if none is provided
        
        Returns:
            DataFrame containing the normalized data
        """
        # Set default dreamer info if none provided
        if dreamer_info is None:
            dreamer_info = {
                'dreamer_id': os.path.basename(input_file).split('.')[0],
                'gender': 'Unknown',
                'age': 'Unknown'
            }
            
        # Read the input file
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Basic preprocessing
        # Remove non-ASCII characters that might cause issues
        content = re.sub(self.non_ascii_pattern, ' ', content)
        
        # Split content into dream entries based on patterns
        # This is a simplified approach - you may need to adapt this based on your specific file format
        entries = []
        
        # Use regex to find dream entries based on common patterns
        dream_patterns = [
            # Pattern for numbered entries with dates
            r'["\']?(\d+)["\']?\s*(?:\(([^)]+)\))?\s*(.+?)(?=\s*["\']?(?:\d+)["\']?|\Z)',
            # Pattern for code-based entries
            r'Code\s+(\d+)\s*(?:\(([^)]+)\))?\s*(.+?)(?=\s*Code\s+\d+|\Z)',
            # Simple pattern for entries separated by blank lines
            r'(?<=\n\n)(.+?)(?=\n\n|\Z)'
        ]
        
        for pattern in dream_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                if len(match.groups()) >= 3:  # For patterns with ID and date
                    dream_id = match.group(1)
                    date = match.group(2) if match.group(2) else "Unknown"
                    dream_text = match.group(3).strip()
                elif len(match.groups()) >= 1:  # For simple patterns
                    dream_id = f"entry_{len(entries) + 1}"
                    date = "Unknown"
                    dream_text = match.group(1).strip()
                
                if dream_text and not dream_text.isspace():
                    entries.append({
                        'dream_id': dream_id,
                        'date': date,
                        'raw_dream': dream_text
                    })
        
        # If no entries were found, try a simpler approach (line by line)
        if not entries:
            lines = content.split('\n')
            current_entry = ""
            for line in lines:
                line = line.strip()
                if not line:  # Empty line as separator
                    if current_entry:
                        entries.append({
                            'dream_id': f"entry_{len(entries) + 1}",
                            'date': "Unknown",
                            'raw_dream': current_entry.strip()
                        })
                        current_entry = ""
                else:
                    current_entry += line + " "
                    
            # Add the last entry if there is one
            if current_entry:
                entries.append({
                    'dream_id': f"entry_{len(entries) + 1}",
                    'date': "Unknown",
                    'raw_dream': current_entry.strip()
                })
                
        # Create a DataFrame
        df = pd.DataFrame(entries)
        
        # Add dreamer info
        for key, value in dreamer_info.items():
            df[key] = value
            
        # Clean and normalize the dreams
        df['clean_dream'] = df['raw_dream'].apply(self._standardize_dream_structure)
        
        # Extract demographics if available
        df['demographics'] = df['clean_dream'].apply(self._extract_demographics)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f"{dreamer_info['dreamer_id']}_normalized.csv")
        df.to_csv(csv_path, index=False)
        
        # Save each dream as a separate JSON file
        for idx, row in df.iterrows():
            json_data = row.to_dict()
            json_path = os.path.join(self.json_dir, f"{dreamer_info['dreamer_id']}_{row['dream_id']}.json")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(json_data, f, ensure_ascii=False, indent=4)
                
        return df
        
    def _extract_demographics(self, text):
        """
        Extract demographic information if present in the dream text.
        Looks for patterns like "(M, age 25)" or similar.
        
        Args:
            text: The dream text to analyze
            
        Returns:
            Dictionary with demographic info if found, empty dict otherwise
        """
        demographics = {}
        
        if not isinstance(text, str):
            return demographics
            
        # Look for gender and age patterns
        demo_match = self.demographic_pattern.search(text)
        if demo_match:
            demographics['gender'] = demo_match.group(1)
            demographics['age'] = demo_match.group(2)
            
        return demographics

