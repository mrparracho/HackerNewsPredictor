from .pipelines.wiki_pipeline import WikiPipeline
import os

def main():
    # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and run Wiki pipeline
    print("Running Wiki pipeline...")
    wiki_pipeline = WikiPipeline(output_dir)
    word_to_index, word_to_lemma_index = wiki_pipeline.run(num_threads=4)
    
    print("\nProcessing complete!")
    print(f"Total unique words: {len(word_to_index)}")
    print(f"Total unique lemmas: {len(set(word_to_lemma_index.values()))}")
    
    # Print some example mappings
    print("\nExample word mappings:")
    example_words = list(word_to_index.keys())[:5]
    for word in example_words:
        print(f"Word: {word}")
        print(f"  Index: {word_to_index[word]}")
        print(f"  Lemma Index: {word_to_lemma_index.get(word, 'N/A')}")

if __name__ == "__main__":
    main() 