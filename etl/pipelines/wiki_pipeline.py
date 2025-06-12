import os
import requests
from typing import Dict, Tuple
from ..base.pipeline import BasePipeline
from ..processors.text_processor import TextProcessor

class WikiPipeline(BasePipeline):
    def __init__(self, output_dir: str = "data"):
        super().__init__(output_dir)
        self.text_processor = TextProcessor()
        self.text8_url = "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8"
        self.text8_path = os.path.join(output_dir, "text8")

    def get_data(self) -> str:
        """Download text8 dataset if not exists."""
        if not os.path.exists(self.text8_path):
            print("Downloading text8 dataset...")
            response = requests.get(self.text8_url, stream=True)
            response.raise_for_status()
            
            with open(self.text8_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download completed.")
        
        return self.text8_path

    def run(self, num_threads: int = 4) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Run the Wiki pipeline."""
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
        self.save_results(word_to_index, word_to_lemma_index, "wiki")
        
        return word_to_index, word_to_lemma_index 