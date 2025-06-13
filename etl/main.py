import os
from typing import Dict, Tuple
from .pipelines.wiki_pipeline import WikiPipeline
from .pipelines.hn_pipeline import HNPipeline
from .processors.text_processor import TextProcessor
import json

def merge_vocabularies(
    wiki_word_to_index: Dict[str, int],
    wiki_word_to_lemma_index: Dict[str, int],
    hn_word_to_index: Dict[str, int],
    hn_word_to_lemma_index: Dict[str, int]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Merge vocabularies from both sources."""
    # Create combined word to index mapping
    combined_word_to_index = {}
    current_idx = 0
    
    # Add Wiki words
    for word in wiki_word_to_index:
        if word not in combined_word_to_index:
            combined_word_to_index[word] = current_idx
            current_idx += 1
    
    # Add HN words
    for word in hn_word_to_index:
        if word not in combined_word_to_index:
            combined_word_to_index[word] = current_idx
            current_idx += 1
    
    # Create combined lemma to index mapping
    lemma_to_index = {}
    current_lemma_idx = 0
    
    # Process all words
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
    
    return combined_word_to_index, combined_word_to_lemma_index

def merge_text_data(output_dir: str, combined_word_to_index: Dict[str, int]) -> None:
    """Merge text data from both sources into a combined file."""
    print("\nMerging text data...")
    
    # Read text8 data
    with open(os.path.join(output_dir, 'text8'), 'r', encoding='utf-8') as f:
        wiki_text = f.read()
    
    # Read HN data
    with open(os.path.join(output_dir, 'hn_data.txt'), 'r', encoding='utf-8') as f:
        hn_text = f.read()
    
    # Combine texts with a space
    combined_text = f"{wiki_text} {hn_text}"
    
    # Save combined text
    with open(os.path.join(output_dir, 'combined_data.txt'), 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    # Count words in each source
    wiki_words = len(wiki_text.split())
    hn_words = len(hn_text.split())
    total_words = wiki_words + hn_words
    
    print(f"Text data merged successfully!")
    print(f"Wiki words: {wiki_words:,}")
    print(f"HN words: {hn_words:,}")
    print(f"Total words: {total_words:,}")

def main():
    # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Wiki pipeline
    print("Running Wiki pipeline...")
    wiki_pipeline = WikiPipeline(output_dir)
    wiki_word_to_index, wiki_word_to_lemma_index = wiki_pipeline.run(num_threads=4)
    
    # Run HN pipeline with 10000 row limit
    print("\nRunning HN pipeline (100000 row limit)...")
    hn_pipeline = HNPipeline(output_dir, limit=100000)
    hn_word_to_index, hn_word_to_lemma_index = hn_pipeline.run(num_threads=4)
    
    # Merge vocabularies
    print("\nMerging vocabularies...")
    combined_word_to_index, combined_word_to_lemma_index = merge_vocabularies(
        wiki_word_to_index,
        wiki_word_to_lemma_index,
        hn_word_to_index,
        hn_word_to_lemma_index
    )
    
    # Save combined results
    print("Saving combined results...")
    with open(os.path.join(output_dir, "combined_word_to_index.json"), 'w', encoding='utf-8') as f:
        json.dump(combined_word_to_index, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, "combined_word_to_lemma_index.json"), 'w', encoding='utf-8') as f:
        json.dump(combined_word_to_lemma_index, f, indent=2, ensure_ascii=False)
    
    # Merge text data
    merge_text_data(output_dir, combined_word_to_index)
    
    print("\nProcessing complete!")
    print(f"Total unique words: {len(combined_word_to_index)}")
    print(f"Total unique lemmas: {len(set(combined_word_to_lemma_index.values()))}")

if __name__ == "__main__":
    main() 