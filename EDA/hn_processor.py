import threading
from collections import Counter
from typing import Dict, List, Tuple
import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

# Download required NLTK data
nltk.download('wordnet')  # for lemmatization
nltk.download('averaged_perceptron_tagger')  # for POS tagging
nltk.download('punkt')  # for tokenization
nltk.download('averaged_perceptron_tagger_eng')  # specific English model

# Pre-load WordNet to avoid lazy loading issues
wordnet.ensure_loaded()

# Initialize NLTK resources globally
lemmatizer = WordNetLemmatizer()
pos_tag_lock = threading.Lock()

# Define POS tag mapping directly
POS_TAG_MAP = {
    'J': 'a',  # adjective
    'N': 'n',  # noun
    'V': 'v',  # verb
    'R': 'r'   # adverb
}

def clean_text(text: str) -> str:
    """
    Clean text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing special characters
    4. Removing numbers
    5. Removing extra whitespace
    """
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

def get_wordnet_pos(word: str) -> str:
    """
    Map POS tag to first character lemmatize() accepts.
    Uses thread-safe POS tagging and direct POS mapping.
    """
    with pos_tag_lock:
        tag = nltk.pos_tag([word])[0][1][0].upper()
    
    return POS_TAG_MAP.get(tag, 'n')  # Default to noun if tag not found

def lemmatize_word(word: str) -> str:
    """
    Lemmatize a single word using a thread-safe approach.
    """
    cleaned_word = clean_text(word)
    if not cleaned_word:
        return word
    
    try:
        pos = get_wordnet_pos(cleaned_word)
        return lemmatizer.lemmatize(cleaned_word, pos)
    except Exception:
        return cleaned_word  # Return cleaned word if lemmatization fails

def process_word_batch(batch: List[str]) -> Dict[str, str]:
    """
    Process a batch of words to get their lemmas.
    Uses thread-safe lemmatization.
    """
    result = {}
    
    for word in batch:
        lemma = lemmatize_word(word)
        if lemma:  # Only add if we got a valid lemma
            result[word] = lemma
    
    return result

def lemmatize_word_index_dict(word_to_index: Dict[str, int], num_threads: int = 4) -> Dict[str, int]:
    """
    Transform a word-to-index dictionary by mapping words to their lemma indices.
    Uses parallel processing for lemmatization.
    """
    # Split words into batches for parallel processing
    words = list(word_to_index.keys())
    batch_size = len(words) // num_threads + 1
    batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
    
    # Process batches in parallel
    lemma_to_index = {}
    word_to_lemma = {}
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all batches for processing
        future_to_batch = {executor.submit(process_word_batch, batch): batch for batch in batches}
        
        # Process results as they complete
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            for word, lemma in batch_results.items():
                if lemma not in lemma_to_index:
                    lemma_to_index[lemma] = len(lemma_to_index)
                word_to_lemma[word] = lemma
    
    # Create final word to lemma index mapping
    word_to_lemma_index = {word: lemma_to_index[lemma] for word, lemma in word_to_lemma.items()}
    
    return word_to_lemma_index

class HNProcessor:
    def __init__(self, file_path: str, chunk_size: int = 1024 * 1024):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.word_count = 0
        self.word_freq = Counter()
        self.lock = threading.Lock()

    def process_chunk(self, chunk: str) -> Tuple[int, Counter]:
        """Process a chunk of text and return word count and frequency."""
        # Clean the text before splitting
        cleaned_chunk = clean_text(chunk)
        words = cleaned_chunk.split()
        return len(words), Counter(words)

    def process_file_chunk(self, start_pos: int, end_pos: int):
        """Process a specific portion of the file."""
        with open(self.file_path, 'r', encoding='utf-8') as file:
            file.seek(start_pos)
            chunk = file.read(end_pos - start_pos)
            word_count, word_freq = self.process_chunk(chunk)
            
            with self.lock:
                self.word_count += word_count
                self.word_freq.update(word_freq)

    def get_file_size(self) -> int:
        """Get the total size of the file."""
        with open(self.file_path, 'r', encoding='utf-8') as file:
            file.seek(0, 2)  # Seek to end of file
            return file.tell()

    def process_file(self, num_threads: int = 4) -> Tuple[int, Dict[str, int]]:
        """Process the file using multiple threads."""
        file_size = self.get_file_size()
        chunk_size = file_size // num_threads
        
        threads = []
        for i in range(num_threads):
            start_pos = i * chunk_size
            end_pos = start_pos + chunk_size if i < num_threads - 1 else file_size
            
            thread = threading.Thread(
                target=self.process_file_chunk,
                args=(start_pos, end_pos)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return self.word_count, dict(self.word_freq)

def run_stats(total_words: int, word_freq: Dict[str, int]):
    """Run statistical analysis and create visualizations for word frequency data."""
    # Calculate probabilities
    word_probs = {word: freq/total_words for word, freq in word_freq.items()}

    # Sort words by probability
    sorted_probs = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)

    # Print some statistics
    print(f"\nTop 25 words and their probabilities:")
    for word, prob in sorted_probs[:25]:
        print(f"{word}: {prob:.6f}")

    print(f"\nBottom 25 words and their probabilities:")
    for word, prob in sorted_probs[-25:]:
        print(f"{word}: {prob:.6f}")

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the probability distribution
    probabilities = [prob for _, prob in sorted_probs]
    plt.plot(range(len(probabilities)), probabilities, 'b-', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('Word Rank (log scale)')
    plt.ylabel('Probability (log scale)')
    plt.title('Word Probability Distribution (Zip\'s Law)')

    # Add grid
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Show the plot
    plt.show()

def main():
    # Get the absolute path to the HN data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hn_data_path = os.path.join(current_dir, '..', 'data', 'hn_data_massive.txt')
    
    # Initialize the processor
    hn_processor = HNProcessor(hn_data_path)
    
    # Process the file and measure time
    start_time = time.time()
    
    print("Processing HN data file...")
    hn_words, hn_freq = hn_processor.process_file()
    
    end_time = time.time()
    
    # Print results
    print(f"\nResults:")
    print(f"HN data - Total words: {hn_words}, Unique words: {len(hn_freq)}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    # Create word to index mapping
    word_to_index = {word: idx for idx, (word, _) in enumerate(sorted(hn_freq.items(), key=lambda x: x[1], reverse=True))}
    
    # Create lemmatized version of the dictionary using parallel processing
    print("\nStarting parallel lemmatization...")
    lemmatization_start = time.time()
    word_to_lemma_index = lemmatize_word_index_dict(word_to_index, num_threads=8)
    lemmatization_end = time.time()
    print(f"Lemmatization completed in {lemmatization_end - lemmatization_start:.2f} seconds")
    
    # Save the word_to_lemma_index dictionary as JSON file
    root_dir = os.path.join(current_dir, '..')
    json_file_path = os.path.join(root_dir, 'hn_word_to_lemma_index.json')
    
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(word_to_lemma_index, json_file, indent=2, ensure_ascii=False)
    
    # Calculate and print statistics about the lemma index
    unique_lemmas = set(word_to_lemma_index.values())
    print(f"\nLemma Index Statistics:")
    print(f"Total number of words in dictionary: {len(word_to_lemma_index)}")
    print(f"Number of unique lemma indices: {len(unique_lemmas)}")
    print(f"Compression ratio (words/lemmas): {len(word_to_lemma_index)/len(unique_lemmas):.2f}")
    
    # Print some examples of lemmatization
    print("\nExamples of lemmatization:")
    example_words = ['running', 'runs', 'ran', 'better', 'best', 'good', 'programming', 'programs', 'programmed']
    for word in example_words:
        if word in word_to_lemma_index:
            print(f"Word: {word:10} -> Lemma index: {word_to_lemma_index[word]}")

    # Run statistical analysis and visualization
    run_stats(hn_words, hn_freq)

if __name__ == "__main__":
    main() 