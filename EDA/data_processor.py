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

# Download required NLTK data
nltk.download('wordnet')  # for lemmatization
nltk.download('averaged_perceptron_tagger')  # for POS tagging
nltk.download('punkt')  # for tokenization
nltk.download('averaged_perceptron_tagger_eng')  # specific English model

def get_wordnet_pos(word: str) -> str:
    """
    Map POS tag to first character lemmatize() accepts.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_word_index_dict(word_to_index: Dict[str, int]) -> Dict[str, int]:
    """
    Transform a word-to-index dictionary by mapping words to their lemma indices.
    
    Args:
        word_to_index (Dict[str, int]): Original dictionary mapping words to indices
        
    Returns:
        Dict[str, int]: New dictionary where words are mapped to their lemma indices
    """
    lemmatizer = WordNetLemmatizer()
    lemma_to_index = {}
    word_to_lemma_index = {}
    
    # First pass: collect all unique lemmas and assign them indices
    for word in word_to_index.keys():
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
        if lemma not in lemma_to_index:
            lemma_to_index[lemma] = len(lemma_to_index)
    
    # Second pass: map each word to its lemma's index
    for word in word_to_index.keys():
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
        word_to_lemma_index[word] = lemma_to_index[lemma]
    
    return word_to_lemma_index

class TextProcessor:
    def __init__(self, file_path: str, chunk_size: int = 1024 * 1024):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.word_count = 0
        self.word_freq = Counter()
        self.lock = threading.Lock()

    def process_chunk(self, chunk: str) -> Tuple[int, Counter]:
        """Process a chunk of text and return word count and frequency."""
        words = chunk.split()
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

    def create_word_index_dict(self) -> Dict[str, int]:
        """
        Create a dictionary mapping unique words to their indices.
        Words are sorted by frequency (most frequent first).
        
        Returns:
            Dict[str, int]: Dictionary mapping words to their indices
        """
        # Sort words by frequency (most frequent first)
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Create word to index mapping
        word_to_index = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        
        return word_to_index

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
    # Get the absolute path to the text8 file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'text8')
    
    # Initialize the processor
    processor = TextProcessor(file_path)
    
    # Process the file and measure time
    start_time = time.time()
    total_words, word_freq = processor.process_file()
    end_time = time.time()
    
    # Print results
    print(f"Total number of words: {total_words}")
    print(f"Number of unique words: {len(word_freq)}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    # Create word to index mapping
    word_to_index = processor.create_word_index_dict()
    
    # Create lemmatized version of the dictionary
    word_to_lemma_index = lemmatize_word_index_dict(word_to_index)
    
    # Save the word_to_lemma_index dictionary as JSON file in root folder
    root_dir = os.path.join(current_dir, '..')
    json_file_path = os.path.join(root_dir, 'word_to_lemma_index.json')
    
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(word_to_lemma_index, json_file, indent=2, ensure_ascii=False)
    
    print(f"\nSaved word_to_lemma_index dictionary to: {json_file_path}")
    print(f"Dictionary contains {len(word_to_lemma_index)} words mapped to {len(set(word_to_lemma_index.values()))} unique lemmas")
    
    # Print some examples of lemmatization
    print("\nExamples of lemmatization:")
    example_words = ['running', 'runs', 'ran', 'better', 'best', 'good']
    for word in example_words:
        if word in word_to_lemma_index:
            print(f"Word: {word:10} -> Lemma index: {word_to_lemma_index[word]}")

    # # Run statistical analysis and visualization
    # run_stats(total_words, word_freq)

if __name__ == "__main__":
    main() 