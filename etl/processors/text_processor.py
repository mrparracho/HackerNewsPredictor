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

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

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

    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning:
        1. Convert HTML entities to Unicode
        2. Convert Unicode escape sequences
        3. Remove HTML tags
        4. Convert to lowercase
        5. Remove URLs
        6. Remove punctuation and special characters
        7. Remove numbers
        8. Remove extra whitespace
        """
        if not text:
            return ""
        
        # Convert HTML entities to Unicode
        text = html.unescape(text)
        
        # Convert Unicode escape sequences
        text = text.encode('utf-8').decode('unicode-escape')
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

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

    def lemmatize_word_index_dict(self, word_to_index: Dict[str, int], num_threads: int = 4) -> Dict[str, int]:
        """Lemmatize words in parallel and create word to lemma index mapping."""
        print("Lemmatizing words...")
        words = list(word_to_index.keys())
        batch_size = len(words) // num_threads + 1
        word_to_lemma_index = {}
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(0, len(words), batch_size):
                batch = words[i:i + batch_size]
                futures.append(executor.submit(self.process_batch, batch))
            
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