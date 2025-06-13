import torch
import yaml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import json
from model import CBOWNS

def load_model():
    checkpoint_dir = 'models/word2vec/cbow/checkpoints'
    
    # Load the model checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'cbow_model.pt'))
    
    # Get dimensions from saved embeddings
    vocab_size, embed_size = checkpoint['model_state_dict']['in_embed.weight'].size()
    model = CBOWNS(vocab_size, embed_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the embeddings
    model.in_embed.weight.data = torch.load(os.path.join(checkpoint_dir, 'cbow_input_embeddings.pt'))
    model.out_embed.weight.data = torch.load(os.path.join(checkpoint_dir, 'cbow_output_embeddings.pt'))
    
    # Load the vocabulary
    word_to_index = checkpoint['word_to_index']
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    return model, word_to_index

def predict_center_word(context_words, word_to_index, model, top_k):
    model.eval()
    with torch.no_grad():
        # Convert words to indices, using UNK for unknown words
        context_idxs = []
        for word in context_words:
            if word in word_to_index:
                context_idxs.append(word_to_index[word])
            else:
                context_idxs.append(word_to_index['<UNK>'])
        context_idxs = torch.tensor([context_idxs])
        
        v_c = model.in_embed(context_idxs)           # (1, context_len, embed_size)
        v_c = torch.mean(v_c, dim=1)                 # (1, embed_size)

        # Score for all words in the vocabulary
        scores = torch.matmul(model.out_embed.weight, v_c.squeeze())  # (vocab_size,)
        topk = torch.topk(scores, top_k)

        # Create reverse mapping from index to word
        index_to_word = {idx: word for word, idx in word_to_index.items()}
        predicted_words = [index_to_word[i.item()] for i in topk.indices]
        
        print(f"\n🎯 Context (2 before and 2 after): {context_words}")
        print(f"📈 Predicted center word (top-{top_k}): {predicted_words}")
        return predicted_words

def visualize_embeddings(model, word_to_index):
    # Reduce dimension to 2D
    embeds = model.in_embed.weight.detach().numpy()
    words = list(word_to_index.keys())
    
    # Use a lower perplexity value (must be less than n_samples)
    perplexity = min(4, len(words) - 1)  # Use 4 or n_samples-1, whichever is smaller
    reduced = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeds)

    # Plot
    plt.figure(figsize=(6, 6))
    for i, label in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(label, (x, y))
    plt.title("Visualization of CBOW embeddings")
    plt.grid(True)
    plt.show()

def main():
    # Load hyperparameters
    with open('models/word2vec/cbow/cbow_ns.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Load model and vocabulary
    model, word_to_index = load_model()

    # Make predictions with 4 context words (2 before and 2 after)
    predict_center_word(
        context_words=["python", "machine", "is", "fun"],  # Context: [before1, before2, after1, after2]
        word_to_index=word_to_index,
        model=model,
        top_k=config['TOP_K_PREDICTIONS']
    )

    # Visualize embeddings
    visualize_embeddings(model, word_to_index)

if __name__ == "__main__":
    main() 