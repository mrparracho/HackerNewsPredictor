import os
from typing import Dict, Tuple
from .pipelines.wiki_pipeline import WikiPipeline
from .pipelines.hn_pipeline import HNPipeline
from .processors.text_processor import TextProcessor
import json
import argparse

class Word2VecDataProcessor:
    """Process data for word2vec model training."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def merge_vocabularies(
        self,
        wiki_word_to_index: Dict[str, int],
        wiki_word_to_lemma_index: Dict[str, int],
        hn_word_to_index: Dict[str, int],
        hn_word_to_lemma_index: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Merge vocabularies from both sources."""
        print("Merging vocabularies from Wiki and HN sources...")
        
        # Create combined word to index mapping
        combined_word_to_index = {}
        current_idx = 0
        
        # Add Wiki words
        print(f"Processing {len(wiki_word_to_index)} Wiki words...")
        for word in wiki_word_to_index:
            if word not in combined_word_to_index:
                combined_word_to_index[word] = current_idx
                current_idx += 1
        
        # Add HN words
        print(f"Processing {len(hn_word_to_index)} HN words...")
        for word in hn_word_to_index:
            if word not in combined_word_to_index:
                combined_word_to_index[word] = current_idx
                current_idx += 1
        
        print(f"Combined vocabulary size: {len(combined_word_to_index)} unique words")
        
        # Create combined lemma to index mapping
        lemma_to_index = {}
        current_lemma_idx = 0
        
        # Process all words
        print("Creating lemma mappings...")
        for word in combined_word_to_index:
            # Try to get lemma from either source
            lemma = wiki_word_to_lemma_index.get(word) or hn_word_to_lemma_index.get(word)
            if lemma is not None and lemma not in lemma_to_index:
                lemma_to_index[lemma] = current_lemma_idx
                current_lemma_idx += 1
        
        # Create final word to lemma index mapping
        combined_word_to_lemma_index = {}
        for word in combined_word_to_index:
            lemma = wiki_word_to_lemma_index.get(word) or hn_word_to_lemma_index.get(word)
            if lemma is not None:
                combined_word_to_lemma_index[word] = lemma_to_index[lemma]
        
        print(f"Created {len(lemma_to_index)} unique lemmas")
        print(f"Mapped {len(combined_word_to_lemma_index)} words to lemmas")
        
        return combined_word_to_index, combined_word_to_lemma_index

    def merge_text_data(self, combined_word_to_index: Dict[str, int]) -> None:
        """Merge text data from both sources into a combined file."""
        print("\nMerging text data from Wiki and HN sources...")
        
        # Read text8 data
        text8_path = os.path.join(self.output_dir, 'text8')
        print(f"Reading Wiki text from: {text8_path}")
        with open(text8_path, 'r', encoding='utf-8') as f:
            wiki_text = f.read()
        
        # Read HN data
        hn_path = os.path.join(self.output_dir, 'hn_data.txt')
        print(f"Reading HN text from: {hn_path}")
        with open(hn_path, 'r', encoding='utf-8') as f:
            hn_text = f.read()
        
        # Combine texts with a space
        print("Combining text data...")
        combined_text = f"{wiki_text} {hn_text}"
        
        # Save combined text
        combined_path = os.path.join(self.output_dir, 'combined_data.txt')
        print(f"Saving combined text to: {combined_path}")
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Count words in each source
        wiki_words = len(wiki_text.split())
        hn_words = len(hn_text.split())
        total_words = wiki_words + hn_words
        
        print(f"Text data merged successfully!")
        print(f"Wiki words: {wiki_words:,}")
        print(f"HN words: {hn_words:,}")
        print(f"Total words: {total_words:,}")
        print(f"Combined text size: {len(combined_text):,} characters")

    def process_word2vec_data(self, wiki_limit: int = 0, hn_limit: int = 100000, lemmatise: bool = False) -> None:
        """Process data for word2vec model training."""
        print("Processing data for word2vec model training...")
        
        # Run Wiki pipeline
        print("\n" + "="*50)
        print("Running Wiki pipeline...")
        print("="*50)
        wiki_pipeline = WikiPipeline(self.output_dir)
        wiki_word_to_index, wiki_word_to_lemma_index = wiki_pipeline.run(num_threads=4, lemmatise=lemmatise)
        print(f"Wiki pipeline completed:")
        print(f"  - Unique words: {len(wiki_word_to_index):,}")
        if lemmatise:
            print(f"  - Unique lemmas: {len(set(wiki_word_to_lemma_index.values())):,}")
        
        # Run HN pipeline
        print("\n" + "="*50)
        print(f"Running HN pipeline (limit: {hn_limit:,} rows)...")
        print("="*50)
        hn_pipeline = HNPipeline(self.output_dir, limit=hn_limit)
        hn_word_to_index, hn_word_to_lemma_index = hn_pipeline.run(num_threads=4, lemmatise=lemmatise)
        print(f"HN pipeline completed:")
        print(f"  - Unique words: {len(hn_word_to_index):,}")
        if lemmatise:
            print(f"  - Unique lemmas: {len(set(hn_word_to_lemma_index.values())):,}")
        
        # Merge vocabularies
        print("\n" + "="*50)
        print("Merging vocabularies...")
        print("="*50)
        if lemmatise:
            combined_word_to_index, combined_word_to_lemma_index = self.merge_vocabularies(
                wiki_word_to_index,
                wiki_word_to_lemma_index,
                hn_word_to_index,
                hn_word_to_lemma_index
            )
        else:
            combined_word_to_index = {**wiki_word_to_index, **hn_word_to_index}
            combined_word_to_lemma_index = {}
        
        # Save combined results
        print("\nSaving combined results...")
        combined_word_path = os.path.join(self.output_dir, "combined_word_to_index.json")
        print(f"Saving combined word-to-index mapping to: {combined_word_path}")
        with open(combined_word_path, 'w', encoding='utf-8') as f:
            json.dump(combined_word_to_index, f, indent=2, ensure_ascii=False)
        
        if lemmatise:
            combined_lemma_path = os.path.join(self.output_dir, "combined_word_to_lemma_index.json")
            print(f"Saving combined word-to-lemma mapping to: {combined_lemma_path}")
            with open(combined_lemma_path, 'w', encoding='utf-8') as f:
                json.dump(combined_word_to_lemma_index, f, indent=2, ensure_ascii=False)
        
        # Merge text data
        print("\n" + "="*50)
        print("Merging text data...")
        print("="*50)
        self.merge_text_data(combined_word_to_index)
        
        # Final summary
        print("\n" + "="*50)
        print("Word2Vec data processing complete!")
        print("="*50)
        print(f"Total unique words: {len(combined_word_to_index):,}")
        if lemmatise:
            print(f"Total unique lemmas: {len(set(combined_word_to_lemma_index.values())):,}")
        print(f"Files created:")
        print(f"  - {combined_word_path}")
        if lemmatise:
            print(f"  - {combined_lemma_path}")
        print(f"  - {os.path.join(self.output_dir, 'combined_data.txt')}")

def main():
    """Main function to run the word2vec data processing."""
    parser = argparse.ArgumentParser(description="Process data for word2vec model training.")
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--wiki_limit', type=int, default=0, help='Limit for Wiki data (0 = no limit)')
    parser.add_argument('--hn_limit', type=int, default=100000, help='Limit for HN data')
    parser.add_argument('--lemmatise', type=str, default='false', choices=['true', 'false'], help='Whether to lemmatise words (true/false, default: false)')
    args = parser.parse_args()

    lemmatise = args.lemmatise.lower() == 'true'
    processor = Word2VecDataProcessor(args.output_dir)
    processor.process_word2vec_data(
        wiki_limit=args.wiki_limit,
        hn_limit=args.hn_limit,
        lemmatise=lemmatise
    )

if __name__ == "__main__":
    main() 