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
        'python', 'machine', 'learning', 'is', 'fun'
    ]
    word_to_index = {word: i for i, word in enumerate(set(vocab))}
    word_to_lemma_index = {word: i for i, word in enumerate(set(vocab))}  # Dummy lemma mapping
    return word_to_index, word_to_lemma_index

def get_training_data(word_to_index):
    """Get training data from the combined data file."""
    print("\nLoading combined training data...")
    with open(os.path.join('data', 'combined_data.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.lower().split()
    encoded = [word_to_index[word] for word in words if word in word_to_index]
    print(f"Loaded {len(encoded):,} words")
    return encoded

def train(dummy=False):
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
        example_text = "python machine learning is fun"
        words = example_text.lower().split()
        encoded = [word_to_index[word] for word in words if word in word_to_index]
    else:
        print("Loading vocabulary and training data from ETL pipeline...")
        word_to_index, word_to_lemma_index = load_vocabulary()
        encoded = get_training_data(word_to_index)
    
    vocab_size = len(word_to_index)
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
        pbar = tqdm(loader, desc=f"Época {epoch+1}", unit="batch")
        for context, target, negatives in pbar:
            loss = model(context, target, negatives)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(loader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

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