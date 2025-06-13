import os
import json
from typing import Dict, Tuple
from ..base.pipeline import BasePipeline
from ..processors.text_processor import TextProcessor
from ..processors.db_processor import DatabaseProcessor
from collections import Counter

class HNPipeline(BasePipeline):
    def __init__(self, output_dir: str = "data", limit: int = None):
        super().__init__(output_dir)
        self.text_processor = TextProcessor()
        self.db_processor = DatabaseProcessor()
        self.hn_data_path = os.path.join(output_dir, "hn_data.txt")
        self.limit = limit

    def get_data(self) -> str:
        """Get HN data from database and save to file if not exists."""
        if not os.path.exists(self.hn_data_path):
            print("Fetching HN data from database...")
            # Get raw data
            results = self.db_processor.get_hn_data(limit=self.limit)
            
            # Process the raw data
            processed_posts = self.db_processor.process_hn_data(results)
            
            # Save processed posts with scores to JSON
            json_path = os.path.join(os.path.dirname(self.hn_data_path), "hn_data_cleaned.json")
            print(f"[DEBUG] Writing processed posts with scores to: {json_path}")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(processed_posts, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] Finished writing JSON file: {json_path}")
            
            # Create massive string
            massive_string = []
            for post in processed_posts:
                # Add title
                massive_string.append(post['Title'])
                
                # Add content if not URL
                if post['Content'] and not post['Content'].startswith('http'):
                    massive_string.append(post['Content'])
                
                # Add comments
                if post['Comments']:
                    for comment in post['Comments']:
                        massive_string.append(comment)
                
                # Add separator
                massive_string.append("\n" + "="*80 + "\n")
            
            # Save to file
            os.makedirs(os.path.dirname(self.hn_data_path), exist_ok=True)
            with open(self.hn_data_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(massive_string))
            
            print("Data saved to file.")
        
        return self.hn_data_path

    def process_chunk(self, chunk: str) -> Tuple[int, Counter]:
        """Process a chunk of text with cleaning."""
        return self.text_processor.process_chunk(chunk)

    def run(self, num_threads: int = 4) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Run the HN pipeline."""
        # Get data
        file_path = self.get_data()
        
        # Tokenize
        word_count, word_freq = self.tokenize(file_path, num_threads)
        print(f"Total words: {word_count}")
        print(f"Unique words: {len(word_freq)}")
        
        # Create word to index mapping
        word_to_index = {word: idx for idx, word in enumerate(word_freq.keys())}
        
        # Lemmatize
        word_to_lemma_index = self.text_processor.lemmatize_word_index_dict(
            word_to_index, num_threads
        )
        
        # Save results
        self.save_results(word_to_index, word_to_lemma_index, "hn")
        
        return word_to_index, word_to_lemma_index 