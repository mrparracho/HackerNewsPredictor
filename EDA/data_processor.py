import threading
from collections import Counter
from typing import Dict, List, Tuple
import time
import os
import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == "__main__":
    main() 