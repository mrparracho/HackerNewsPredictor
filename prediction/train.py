import os
import sys
import json
import torch
from torch.utils.data import DataLoader, random_split

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predictor import SimplePredictor
from data.dataset import SimpleHNDataset
from utils.data_processing import load_cbow_embeddings, create_title_embeddings
from utils.training import train_model

def main():
    print("\n=== Starting Main Function ===")
    
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load word to index mapping
    print("\n[Main] Loading word to index mapping...")
    word_to_ix_path = os.path.join(parent_dir, 'data/combined_word_to_lemma_index.json')
    with open(word_to_ix_path, 'r') as f:
        word_to_ix = json.load(f)
    print(f"[Main] Loaded {len(word_to_ix)} word mappings")
    
    # Load CBOW embeddings
    print("\n[Main] Loading CBOW embeddings...")
    try:
        cbow_model_path = os.path.join(parent_dir, 'models/word2vec/cbow/checkpoints/cbow_model.pt')
        embeddings = load_cbow_embeddings(cbow_model_path)
        embedding_dim = embeddings.shape[1]  # Get the dimension of the embeddings
        print("[Main] CBOW embeddings loaded successfully")
    except Exception as e:
        print(f"[Main] Error loading CBOW embeddings: {e}")
        return
    
    # Load the data for titles
    print("\n[Main] Loading data for titles...")
    data_path = os.path.join(parent_dir, 'data/hn_data_cleaned.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Pre-compute title embeddings
    print("\n[Main] Starting title embedding creation...")
    try:
        title_embeddings, scores = create_title_embeddings(data_path, word_to_ix, embeddings)
        print(f"[Main] Created embeddings for {len(title_embeddings)} titles")
    except Exception as e:
        print(f"[Main] Error creating embeddings: {e}")
        return
    
    # Create dataset
    print("\n[Main] Creating dataset...")
    full_dataset = SimpleHNDataset(title_embeddings, scores, data)
    print(f"[Main] Dataset created with {len(full_dataset)} samples")
    
    # Split into train and test sets (80/20 split)
    print("\n[Main] Splitting dataset...")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    print(f"[Main] Split into {train_size} training and {test_size} test samples")
    
    # Create data loaders
    print("\n[Main] Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("[Main] Data loaders created")
    
    # Initialize and train model
    print("\n[Main] Initializing model...")
    model = SimplePredictor(
             input_dim=embedding_dim,
        # hidden_dim=cfg.HIDDEN_DIM,    # if you swept hidden_dim
         #dropout=cfg.DROPOUT           # if you swept dropout
         )
    print("[Main] Starting training process...")
    train_model(model, train_loader, test_loader, full_dataset, num_epochs=50)
    
    print("\n=== Main Function Completed ===")

if __name__ == "__main__":
    main() 