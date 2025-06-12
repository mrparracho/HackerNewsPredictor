import torch
import yaml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from model import SkipGramNS

def load_model():
    checkpoint_dir = 'models/word2vec/skipgram/checkpoints'
    
    # Load the model checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'skipgram_model.pt'))
    
    # Get dimensions from saved embeddings
    vocab_size, embed_size = checkpoint['model_state_dict']['in_embed.weight'].size()
    model = SkipGramNS(vocab_size, embed_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the embeddings
    model.in_embed.weight.data = torch.load(os.path.join(checkpoint_dir, 'skipgram_input_embeddings.pt'))
    model.out_embed.weight.data = torch.load(os.path.join(checkpoint_dir, 'skipgram_output_embeddings.pt'))
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    return model

def predict_context_words(center_word, word_to_ix, model, top_k):
    model.eval()
    with torch.no_grad():
        center_idx = torch.tensor([word_to_ix[center_word]])
        v_c = model.in_embed(center_idx)              # (1, embed_size)

        # Score for all words in the vocabulary
        scores = torch.matmul(model.out_embed.weight, v_c.squeeze())  # (vocab_size,)
        topk = torch.topk(scores, top_k)

        predicted_words = [list(word_to_ix.keys())[i] for i in topk.indices.tolist()]
        print(f"\n🎯 Center word: {center_word}")
        print(f"📈 Predicted context (2 before and 2 after): {predicted_words}")
        return predicted_words

def visualize_embeddings(model, word_to_ix):
    # Reduce dimension to 2D
    embeds = model.in_embed.weight.detach().numpy()
    words = list(word_to_ix.keys())
    
    # Use a lower perplexity value (must be less than n_samples)
    perplexity = min(4, len(words) - 1)  # Use 4 or n_samples-1, whichever is smaller
    reduced = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeds)

    # Plot
    plt.figure(figsize=(6, 6))
    for i, label in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(label, (x, y))
    plt.title("Visualization of Skip-gram embeddings")
    plt.grid(True)
    plt.show()

def main():
    # Load hyperparameters
    with open('models/word2vec/skipgram/skipgram_ns.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Prepare vocabulary
    vocab = [
        'hi', 'my', 'name', 'is', 'kernel',
        'and', 'i', 'like', 'to', 'code',
        'python', 'machine', 'learning', 'is', 'fun'
    ]
    word_to_ix = {word: i for i, word in enumerate(set(vocab))}

    # Load model
    model = load_model()

    # Make predictions for center word "learning"
    # Given "learning", predict what words should be before and after it
    predict_context_words(
        center_word="learning",  # Try to predict 4 context words around "learning"
        word_to_ix=word_to_ix,
        model=model,
        top_k=config['TOP_K_PREDICTIONS']  # We want 4 words: 2 before and 2 after
    )

    # Visualize embeddings
    visualize_embeddings(model, word_to_ix)

if __name__ == "__main__":
    main() 