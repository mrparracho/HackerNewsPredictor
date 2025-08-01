import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import argparse

def visualize_embeddings(model_type, max_embeddings=1000):
    # Load embeddings
    checkpoint_dir = f'models/word2vec/{model_type}/checkpoints'
    embeddings = torch.load(os.path.join(checkpoint_dir, f'{model_type}_input_embeddings.pt'))
    
    # Load vocabulary
    words = []
    with open(os.path.join(checkpoint_dir, f'{model_type}_input_embeddings.txt'), 'r') as f:
        words = [line.split()[0] for line in f]
    
    # Limit number of embeddings if needed
    if len(words) > max_embeddings:
        print(f"\nLimiting visualization to {max_embeddings} embeddings (out of {len(words)} total)")
        words = words[:max_embeddings]
        embeddings = embeddings[:max_embeddings]
    
    # Reduce dimension to 2D
    reduced = TSNE(n_components=2, perplexity=min(4, len(words)-1)).fit_transform(embeddings.numpy())
    
    # Plot
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y))
    plt.title(f"{model_type.upper()} Embeddings (top {len(words)})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cbow', 'skipgram'], required=True,
                      help='Type of model to visualize (cbow or skipgram)')
    parser.add_argument('--max-embeddings', type=int, default=1000,
                      help='Maximum number of embeddings to visualize (default: 1000)')
    args = parser.parse_args()
    visualize_embeddings(args.model, args.max_embeddings)
