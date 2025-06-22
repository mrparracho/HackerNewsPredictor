import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import Dict, List, Set, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import html
import re
import string
from collections import Counter
import unicodedata

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet_ic', quiet=True)
except Exception as e:
    print(f"Warning: Some NLTK resources may not be available: {e}")

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.lock = threading.Lock()
        self.pos_tag_lock = threading.Lock()
        
        # Define POS tag mapping
        self.POS_TAG_MAP = {
            'J': 'a',  # adjective
            'N': 'n',  # noun
            'V': 'v',  # verb
            'R': 'r'   # adverb
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text by converting special characters to their closest ASCII equivalent."""
        if not isinstance(text, str):
            return text
        
        # First, normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to ASCII, ignoring non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Replace common special characters with their ASCII equivalents
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v',
            '±': '+/-',
            '×': 'x',
            '÷': '/',
            '©': '(c)',
            '®': '(r)',
            '™': '(tm)',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '°': ' degrees',
            '²': '2',
            '³': '3',
            'µ': 'u',
            '¶': 'P',
            '·': '.',
            '¹': '1',
            '¼': '1/4',
            '½': '1/2',
            '¾': '3/4',
            '×': 'x',
            '÷': '/',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '≈': '~=',
            '∞': 'inf',
            '±': '+/-',
            '∑': 'sum',
            '∏': 'prod',
            '√': 'sqrt',
            '∫': 'integral',
            '∆': 'delta',
            '∇': 'nabla',
            '∂': 'partial',
            '∝': 'propto',
            '∞': 'inf',
            '∅': 'empty',
            '∈': 'in',
            '∉': 'not in',
            '∋': 'contains',
            '∌': 'not contains',
            '⊂': 'subset',
            '⊃': 'superset',
            '⊆': 'subset or equal',
            '⊇': 'superset or equal',
            '⊕': 'xor',
            '⊗': 'tensor',
            '⊥': 'perp',
            '⊤': 'top',
            '⊢': 'proves',
            '⊨': 'models',
            '⊩': 'forces',
            '⊪': 'double turnstile',
            '⊫': 'double turnstile',
            '⊬': 'does not prove',
            '⊭': 'does not model',
            '⊮': 'does not force',
            '⊯': 'does not double turnstile',
            '⊰': 'precedes under relation',
            '⊱': 'succeeds under relation',
            '⊲': 'normal subgroup of',
            '⊳': 'contains as normal subgroup',
            '⊴': 'normal subgroup of or equal to',
            '⊵': 'contains as normal subgroup or equal to',
            '⊶': 'original of',
            '⊷': 'image of',
            '⊸': 'multimap',
            '⊹': 'hermitian conjugate matrix',
            '⊺': 'intercalate',
            '⊻': 'xor',
            '⊼': 'nand',
            '⊽': 'nor',
            '⊾': 'right angle with arc',
            '⊿': 'right triangle',
            '⋀': 'n-ary logical and',
            '⋁': 'n-ary logical or',
            '⋂': 'n-ary intersection',
            '⋃': 'n-ary union',
            '⋄': 'diamond operator',
            '⋅': 'dot operator',
            '⋆': 'star operator',
            '⋇': 'division times',
            '⋈': 'bowtie',
            '⋉': 'left normal factor semidirect product',
            '⋊': 'right normal factor semidirect product',
            '⋋': 'left semidirect product',
            '⋌': 'right semidirect product',
            '⋍': 'reversed tilde equals',
            '⋎': 'curly logical or',
            '⋏': 'curly logical and',
            '⋐': 'double subset',
            '⋑': 'double superset',
            '⋒': 'double intersection',
            '⋓': 'double union',
            '⋔': 'pitchfork',
            '⋕': 'equal and parallel to',
            '⋖': 'less than with dot',
            '⋗': 'greater than with dot',
            '⋘': 'very much less than',
            '⋙': 'very much greater than',
            '⋚': 'less than equal to or greater than',
            '⋛': 'greater than equal to or less than',
            '⋜': 'equal to or less than',
            '⋝': 'equal to or greater than',
            '⋞': 'equal to or precedes',
            '⋟': 'equal to or succeeds',
            '⋠': 'does not precede or equal',
            '⋡': 'does not succeed or equal',
            '⋢': 'not square image of or equal to',
            '⋣': 'not square original of or equal to',
            '⋤': 'square image of or not equal to',
            '⋥': 'square original of or not equal to',
            '⋦': 'less than but not equivalent to',
            '⋧': 'greater than but not equivalent to',
            '⋨': 'precedes but not equivalent to',
            '⋩': 'succeeds but not equivalent to',
            '⋪': 'not normal subgroup of',
            '⋫': 'does not contain as normal subgroup',
            '⋬': 'not normal subgroup of or equal to',
            '⋭': 'does not contain as normal subgroup or equal',
            '⋮': 'vertical ellipsis',
            '⋯': 'midline horizontal ellipsis',
            '⋰': 'up right diagonal ellipsis',
            '⋱': 'down right diagonal ellipsis',
            '⋲': 'element of with long horizontal stroke',
            '⋳': 'element of with vertical bar at end of horizontal stroke',
            '⋴': 'small element of with vertical bar at end of horizontal stroke',
            '⋵': 'element of with dot above',
            '⋶': 'element of with overbar',
            '⋷': 'small element of with overbar',
            '⋸': 'element of with underbar',
            '⋹': 'element of with two horizontal strokes',
            '⋺': 'contains with long horizontal stroke',
            '⋻': 'contains with vertical bar at end of horizontal stroke',
            '⋼': 'small contains with vertical bar at end of horizontal stroke',
            '⋽': 'contains with overbar',
            '⋾': 'small contains with overbar',
            '⋿': 'z notation bag membership',
        }
        
        for special, replacement in replacements.items():
            text = text.replace(special, replacement)
        
        return text

    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning:
        1. Convert HTML entities to Unicode
        2. Convert Unicode escape sequences
        3. Remove HTML tags and their content
        4. Convert to lowercase
        5. Remove URLs
        6. Remove punctuation and special characters
        7. Remove numbers
        8. Remove extra whitespace
        """
        if not text:
            return ""
        
        # First normalize the text to handle special characters
        text = self.normalize_text(text)
        
        # Convert HTML entities to Unicode
        text = html.unescape(text)
        
        # Convert Unicode escape sequences
        try:
            text = text.encode('utf-8').decode('unicode-escape')
        except UnicodeError:
            # If decoding fails, try to clean the text directly
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Remove HTML tags and their content (more aggressive)
        # First remove complete tags with content
        text = re.sub(r'<[^>]*>.*?</[^>]*>', ' ', text)  # Add space after removing tags
        # Then remove any remaining tags
        text = re.sub(r'<[^>]+>', ' ', text)  # Add space after removing tags
        # Remove any remaining HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # Add space after removing entities
        text = re.sub(r'&#[0-9]+;', ' ', text)  # Add space after removing entities
        text = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text)  # Add space after removing entities
        
        # Remove URLs (more aggressive)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # Add space after removing URLs
        
        # Remove any remaining HTML-like patterns
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)  # Add space after removing entities
        text = re.sub(r'<[^>]*>', ' ', text)  # Add space after removing tags
        text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', text)  # Add space after removing URLs
        
        # Remove any remaining HTML-like patterns
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)  # Add space after removing entities
        text = re.sub(r'<[^>]*>', ' ', text)  # Add space after removing tags
        text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', text)  # Add space after removing URLs
        
        # Remove any remaining HTML-like patterns
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)  # Add space after removing entities
        text = re.sub(r'<[^>]*>', ' ', text)  # Add space after removing tags
        text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', text)  # Add space after removing URLs
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions and possessives before removing punctuation
        # Replace common contractions with their expanded forms
        contractions = {
            "n't": " not",
            "'ll": " will", 
            "'re": " are",
            "'ve": " have",
            "'d": " would",
            "'m": " am",
            "'s": " is",  # Note: this will also affect possessives, but it's a trade-off
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove punctuation and special characters, but preserve spaces
        # Create a custom punctuation string without apostrophes
        custom_punctuation = string.punctuation.replace("'", "")
        text = text.translate(str.maketrans(custom_punctuation, ' ' * len(custom_punctuation)))
        
        # Remove numbers
        text = re.sub(r'\d+', ' ', text)  # Add space after removing numbers
        
        # Remove extra whitespace and normalize spaces
        text = ' '.join(text.split())
        
        # Final normalization to catch any remaining special characters
        text = self.normalize_text(text)
        
        # Ensure proper spacing between words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'([a-z])(\d)', r'\1 \2', text)  # Add space between letters and numbers
        text = re.sub(r'(\d)([a-z])', r'\1 \2', text)  # Add space between numbers and letters
        
        # Final cleanup of any remaining multiple spaces
        text = ' '.join(text.split())
        
        return text.strip()

    def get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to first character lemmatize() accepts."""
        with self.pos_tag_lock:
            tag = nltk.pos_tag([word])[0][1][0].upper()
        return self.POS_TAG_MAP.get(tag, 'n')  # Default to noun if tag not found

    def lemmatize_word(self, word: str) -> str:
        """Lemmatize a single word using its POS tag."""
        cleaned_word = self.clean_text(word)
        if not cleaned_word:
            return word
        
        try:
            pos = self.get_wordnet_pos(cleaned_word)
            return self.lemmatizer.lemmatize(cleaned_word, pos)
        except Exception:
            return cleaned_word  # Return cleaned word if lemmatization fails

    def process_batch(self, words: List[str]) -> Dict[str, str]:
        """Process a batch of words and return word to lemma mapping."""
        return {word: self.lemmatize_word(word) for word in words}

    def lemmatize_word_index_dict(self, word_to_index: Dict[str, int], num_threads: int = 4, default_pos: str = 'v') -> Dict[str, int]:
        """Lemmatize words in parallel and create word to lemma index mapping using a default POS (e.g., 'v' for verb)."""
        print(f"Lemmatizing words with default POS '{default_pos}'...")
        words = list(word_to_index.keys())
        batch_size = len(words) // num_threads + 1
        word_to_lemma_index = {}

        def lemmatize_batch(batch):
            batch_results = {}
            for word in batch:
                try:
                    cleaned_word = self.clean_text(word)
                    if cleaned_word:
                        lemma = self.lemmatizer.lemmatize(cleaned_word, default_pos)
                        batch_results[word] = lemma
                    else:
                        batch_results[word] = word  # Keep original if cleaning results in empty string
                except Exception as e:
                    # If lemmatization fails, use the cleaned word or original word as fallback
                    cleaned_word = self.clean_text(word)
                    batch_results[word] = cleaned_word if cleaned_word else word
            return batch_results

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(0, len(words), batch_size):
                batch = words[i:i + batch_size]
                futures.append(executor.submit(lemmatize_batch, batch))
            for future in tqdm(as_completed(futures), total=len(futures)):
                batch_results = future.result()
                with self.lock:
                    word_to_lemma_index.update(batch_results)

        # Create lemma to index mapping
        lemma_to_index = {}
        for word, lemma in word_to_lemma_index.items():
            if lemma not in lemma_to_index:
                lemma_to_index[lemma] = len(lemma_to_index)
        # Update word_to_lemma_index to use lemma indices
        return {word: lemma_to_index[lemma] for word, lemma in word_to_lemma_index.items()}

    def process_chunk(self, chunk: str) -> Tuple[int, Counter]:
        """Process a chunk of text and return word count and frequency."""
        cleaned_text = self.clean_text(chunk)
        words = cleaned_text.split()
        return len(words), Counter(words) 