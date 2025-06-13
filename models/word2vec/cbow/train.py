import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import yaml
import os
import json
import argparse
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from model import CBOWNS, CBOWNegativeSamplingDataset

def load_vocabulary():
    """Load the combined vocabulary from our ETL pipeline."""
    with open('data/combined_word_to_index.json', 'r', encoding='utf-8') as f:
        word_to_index = json.load(f)
    
    with open('data/combined_word_to_lemma_index.json', 'r', encoding='utf-8') as f:
        word_to_lemma_index = json.load(f)
    
    return word_to_index, word_to_lemma_index

def get_dummy_vocabulary():
    """Return the dummy vocabulary for testing."""
    vocab = [
        'hi', 'my', 'name', 'is', 'kernel',
        'and', 'i', 'like', 'to', 'code',
        'python', 'machine', 'learning', 'is', 'fun',
        'king', 'queen', 'man', 'woman', 'dog', 'cat'  # Add analogy words
    ]
    # Create unique vocabulary
    unique_vocab = list(set(vocab))
    # Create mappings
    word_to_index = {word: i for i, word in enumerate(unique_vocab)}
    word_to_lemma_index = {word: i for i, word in enumerate(unique_vocab)}  # Dummy lemma mapping
    # Add UNK token with last index
    unk_idx = len(unique_vocab)
    word_to_index['<UNK>'] = unk_idx
    word_to_lemma_index['<UNK>'] = unk_idx
    return word_to_index, word_to_lemma_index

def get_training_data(word_to_index):
    """Get training data from the combined data file."""
    print("\nLoading combined training data...")
    with open(os.path.join('data', 'combined_data.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.lower().split()
    # Use UNK token for unknown words
    encoded = [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in words]
    print(f"Loaded {len(encoded):,} words")
    return encoded

import torch.nn.functional as F
from scipy.stats import spearmanr
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

def evaluate_model(model, step, word_to_ix, embedding_layer=None):
    """
    Evaluate the model on the analogy task and log the results to wandb.
    """
    # 1. Get embeddings
    if embedding_layer is None:
        embedding_matrix = model.in_embed.weight.detach()
    else:
        embedding_matrix = embedding_layer.weight.detach()

    ix_to_word = {ix: w for w, ix in word_to_ix.items()}
    words = list(word_to_ix.keys())

    # 2. Similarity (Spearman)
    similarity_pairs = [("king", "queen"), ("dog", "cat"), ("king", "dog")]
    human_scores = [0.95, 0.85, 0.2]
    model_scores = []
    for (w1, w2) in similarity_pairs:
        v1, v2 = embedding_matrix[word_to_ix[w1]], embedding_matrix[word_to_ix[w2]]
        sim = F.cosine_similarity(v1, v2, dim=0).item()
        model_scores.append(sim)
    spearman_corr, _ = spearmanr(model_scores, human_scores)

    # 3. Vector analogy
    def predict_analogy(a, b, c):
        va, vb, vc = embedding_matrix[word_to_ix[a]], embedding_matrix[word_to_ix[b]], embedding_matrix[word_to_ix[c]]
        pred_vec = vb - va + vc
        sims = F.cosine_similarity(embedding_matrix, pred_vec.unsqueeze(0)).numpy()
        best_ix = sims.argmax()
        return ix_to_word[best_ix]

    predicted = predict_analogy("king", "man", "woman")
    analogy_correct = int(predicted == "queen")

    # 4. t-SNE visualization
    # def log_tsne():
    #     vectors = embedding_matrix.cpu().numpy()
    #     reduced = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(vectors)
    #     plt.figure(figsize=(6, 6))
    #     for i, word in enumerate(words):
    #         x, y = reduced[i]
    #         plt.scatter(x, y)
    #         plt.annotate(word, (x, y))
    #     plt.title(f"t-SNE dos Embeddings - Step {step}")
    #     plt.grid(True)
    #     wandb.log({f"tsne_step_{step}": wandb.Image(plt)})
    #     plt.close()

    # 5. Logging to wandb
    wandb.log({
        "eval/similarity_spearman": spearman_corr,
        "eval/analogy_correct": analogy_correct,
        "eval/analogy_accuracy": float(analogy_correct),  # Convert to float for metric logging
        "eval/analogy_prediction": predicted,
        "eval/analogy_expected": "queen"  # Log the expected word for reference
    })

    # log_tsne()

def train(dummy=False):
    """
    Train the CBOW model.
    """
    # Load hyperparameters from YAML file
    with open('models/word2vec/cbow/cbow_ns.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = 'models/word2vec/cbow/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(project=config['PROJECT_NAME'], name=config['RUN_NAME'])
    
    if dummy:
        print("Using dummy vocabulary for testing...")
        word_to_index, word_to_lemma_index = get_dummy_vocabulary()
        # Use the dummy example
        example_text = "python machine learning is fun and i like to code python is a great language for machine learning and coding is fun with python king queen man woman king queen man woman"
        words = example_text.lower().split()
        encoded = [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in words]
        print(encoded)
    else:
        print("Loading vocabulary and training data from ETL pipeline...")
        word_to_index, word_to_lemma_index = load_vocabulary()
        # Add UNK token with last index if not present
        if '<UNK>' not in word_to_lemma_index:
            unk_idx = len(word_to_lemma_index)
            word_to_lemma_index['<UNK>'] = unk_idx
            word_to_index['<UNK>'] = unk_idx
        encoded = get_training_data(word_to_index)
    
    vocab_size = len(word_to_lemma_index)
    print(f"Vocabulary size: {vocab_size}")

    # Create dataset and dataloader
    dataset = CBOWNegativeSamplingDataset(
        encoded, 
        window_size=config['WINDOW_SIZE'], 
        num_negatives=config['NUM_NEGATIVES']
    )
    loader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True)

    # Initialize model and optimizer
    model = CBOWNS(vocab_size, embed_size=config['EMBEDDING_SIZE'])
    opt = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    # Training loop
    best_loss = float('inf')
    for epoch in range(config['NUM_EPOCHS']):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, (context, target, negatives) in enumerate(pbar):
            loss = model(context, target, negatives)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            # Log batch loss to wandb
            wandb.log({
                "epoch": epoch + 1, 
                "batch": batch_idx,
                "batch_loss": loss.item()
            })
            
            # Evaluate model periodically
            if batch_idx % 100 == 0:
                evaluate_model(model, batch_idx + epoch * len(loader), word_to_index)
            
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(loader)
        wandb.log({
            "epoch": epoch + 1,
            "epoch_loss": avg_loss
        })

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': best_loss,
                'word_to_index': word_to_index,
                'word_to_lemma_index': word_to_lemma_index
            }, os.path.join(checkpoint_dir, 'cbow_model.pt'))
            
            # Save embeddings
            torch.save(model.in_embed.weight.data, os.path.join(checkpoint_dir, 'cbow_input_embeddings.pt'))
            torch.save(model.out_embed.weight.data, os.path.join(checkpoint_dir, 'cbow_output_embeddings.pt'))

            # Save as readable text file
            with open(os.path.join(checkpoint_dir, 'cbow_input_embeddings.txt'), "w", encoding='utf-8') as f:
                for word, idx in word_to_index.items():
                    vector = model.in_embed.weight[idx].tolist()
                    vector_str = " ".join(f"{v:.4f}" for v in vector)
                    f.write(f"{word} {vector_str}\n")

    wandb.finish()
    return model, word_to_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CBOW model')
    parser.add_argument('--dummy', action='store_true', help='Use dummy vocabulary for testing', default=False)
    args = parser.parse_args()
    
    train(dummy=args.dummy)