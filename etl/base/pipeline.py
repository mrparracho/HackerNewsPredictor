from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import threading
from collections import Counter
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import os

class BasePipeline(ABC):
    def __init__(self, output_dir: str = "data", chunk_size: int = 1024 * 1024):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.word_count = 0
        self.word_freq = Counter()
        self.lock = threading.Lock()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def get_data(self) -> str:
        """Get data from source (file path or database)"""
        pass

    def process_chunk(self, chunk: str) -> Tuple[int, Counter]:
        """Process a chunk of text and return word count and frequency."""
        words = chunk.split()
        return len(words), Counter(words)

    def process_file_chunk(self, start_pos: int, end_pos: int, file_path: str):
        """Process a specific portion of the file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            file.seek(start_pos)
            chunk = file.read(end_pos - start_pos)
            word_count, word_freq = self.process_chunk(chunk)
            
            with self.lock:
                self.word_count += word_count
                self.word_freq.update(word_freq)

    def tokenize(self, file_path: str, num_threads: int = 4) -> Tuple[int, Dict[str, int]]:
        """Tokenize the file using multiple threads."""
        print("Tokenizing text...")
        start_time = time.time()
        
        file_size = os.path.getsize(file_path)
        chunk_size = file_size // num_threads
        
        threads = []
        for i in range(num_threads):
            start_pos = i * chunk_size
            end_pos = start_pos + chunk_size if i < num_threads - 1 else file_size
            
            thread = threading.Thread(
                target=self.process_file_chunk,
                args=(start_pos, end_pos, file_path)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
            
        end_time = time.time()
        print(f"Tokenization completed in {end_time - start_time:.2f} seconds")
        
        return self.word_count, dict(self.word_freq)

    def save_results(self, word_to_index: Dict[str, int], word_to_lemma_index: Dict[str, int], prefix: str):
        """Save processing results to files."""
        # Save word to index mapping
        index_path = os.path.join(self.output_dir, f"{prefix}_word_to_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(word_to_index, f, indent=2, ensure_ascii=False)
        
        # Save lemma index mapping
        lemma_path = os.path.join(self.output_dir, f"{prefix}_word_to_lemma_index.json")
        with open(lemma_path, 'w', encoding='utf-8') as f:
            json.dump(word_to_lemma_index, f, indent=2, ensure_ascii=False)

    @abstractmethod
    def run(self, num_threads: int = 4) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Run the pipeline and return word_to_index and word_to_lemma_index mappings."""
        pass 