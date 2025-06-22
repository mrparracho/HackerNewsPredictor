import torch
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
import yaml
import os
import json
import argparse
import sys
import numpy as np

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from model import EnhancedHNPredictor, EnhancedPredictorConfig, HNFeatureEngineer, EnhancedHNDataset

# load YAML config defaults
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_vocabulary():
    """Load the combined vocabulary from our ETL pipeline."""
    with open('data/combined_word_to_index.json', 'r', encoding='utf-8') as f:
        word_to_index = json.load(f)
    
    with open('data/combined_word_to_lemma_index.json', 'r', encoding='utf-8') as f:
        word_to_lemma_index = json.load(f)
    
    return word_to_index, word_to_lemma_index

def load_cbow_embeddings():
    """Load pre-trained CBOW embeddings."""
    checkpoint_dir = 'models/word2vec/cbow/checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'cbow_model.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CBOW model not found at {checkpoint_path}. Please train CBOW first.")
    
    checkpoint = torch.load(checkpoint_path)
    embeddings = checkpoint['model_state_dict']['in_embed.weight']
    print(f"Loaded CBOW embeddings with shape: {embeddings.shape}")
    return embeddings

def load_hn_data(limit=None):
    """Load HN data from database or JSON file."""
    # Try to load from JSON first
    json_path = 'data/hn_data_raw.json'
    if os.path.exists(json_path):
        print(f"Loading HN data from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if limit:
            data = data[:limit]
        return data
    
    # If JSON doesn't exist, try to load from database
    try:
        from etl.processors.db_processor import DatabaseProcessor
        db_processor = DatabaseProcessor()
        print("Loading HN data from database...")
        data = db_processor.get_hn_data(limit=limit)
        return data
    except ImportError:
        raise FileNotFoundError("No HN data found. Please run the ETL pipeline first.")

def evaluate_model(model, test_loader, device, criterion):
    """Evaluate the model on the test set."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for title_emb, content_emb, categorical_features, scores in test_loader:
            title_emb = title_emb.to(device)
            content_emb = content_emb.to(device)
            categorical_features = categorical_features.to(device)
            scores = scores.to(device)
            
            predictions = model(title_emb, content_emb, categorical_features)
            loss = criterion(predictions, scores)
            
            total_loss += loss.item() * len(scores)
            total_samples += len(scores)
    
    avg_loss = total_loss / total_samples
    return avg_loss

def train_run(dummy=False):
    """
    Train the Enhanced HN Predictor model.
    """
    # Create checkpoints directory if it doesn't exist
    base_ckpt_dir = "models/predictor/checkpoints"
    run_ckpt_dir = os.path.join(base_ckpt_dir, "cur_run")
    os.makedirs(run_ckpt_dir, exist_ok=True)
    
    # Clear out any leftover files from previous run
    for fname in os.listdir(run_ckpt_dir):
        path = os.path.join(run_ckpt_dir, fname)
        if os.path.isfile(path):
            os.remove(path)
    
    best_loss_path = os.path.join(base_ckpt_dir, "best_loss.txt")
    
    # Load configuration
    config = load_yaml_config("models/predictor/predictor.yml")
    # Ensure numeric config values are correct types
    config['LEARNING_RATE'] = float(config['LEARNING_RATE'])
    config['WEIGHT_DECAY'] = float(config['WEIGHT_DECAY'])
    config['BATCH_SIZE'] = int(config['BATCH_SIZE'])
    config['NUM_EPOCHS'] = int(config['NUM_EPOCHS'])
    config['PATIENCE'] = int(config['PATIENCE'])
    config['DATA_LIMIT'] = int(config['DATA_LIMIT'])
    
    print("Loading vocabulary and embeddings...")
    word_to_index, word_to_lemma_index = load_vocabulary()
    embeddings = load_cbow_embeddings()
    
    print("Loading HN data...")
    if dummy:
        # Create dummy data for testing
        dummy_posts = [
            {
                'id': 1, 'type': 'story', 'by': 'test_user', 'time': 1640995200,
                'title': 'Show HN: My amazing AI project', 'text': 'This is a test post',
                'url': 'https://github.com/test/project', 'score': 10, 'descendants': 5,
                'dead': False
            },
            {
                'id': 2, 'type': 'story', 'by': 'test_user2', 'time': 1640995200,
                'title': 'Ask HN: How to learn machine learning?', 'text': 'I want to learn ML',
                'url': None, 'score': 15, 'descendants': 8, 'dead': False
            }
        ]
        posts_data = dummy_posts
    else:
        posts_data = load_hn_data(limit=config.get('DATA_LIMIT', 10000))
    
    print(f"Loaded {len(posts_data)} posts")
    
    # Initialize feature engineer
    print("Initializing feature engineer...")
    embedding_dim = embeddings.shape[1]  # Use actual CBOW embedding size
    hidden_dim = config['HIDDEN_DIM']
    dropout = config['DROPOUT']
    feature_engineer = HNFeatureEngineer(
        word_to_ix=word_to_index,
        embeddings=embeddings,
        embedding_dim=embedding_dim
    )
    
    # Create feature matrix
    feature_matrices, feature_names = feature_engineer.create_feature_matrix(posts_data)
    
    # Create dataset
    dataset = EnhancedHNDataset(
        title_embeddings=feature_matrices['title_embeddings'],
        content_embeddings=feature_matrices['content_embeddings'],
        categorical_features=feature_matrices['categorical_features'],
        scores=feature_matrices['scores']
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    
    # Initialize model
    model_config = EnhancedPredictorConfig(
        embedding_dim=embedding_dim,
        num_categorical_features=len(feature_names),
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    model = EnhancedHNPredictor(model_config)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training for {config['NUM_EPOCHS']} epochs...")
    
    for epoch in range(config['NUM_EPOCHS']):
        print(f"\nEpoch {epoch + 1}/{config['NUM_EPOCHS']}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for title_emb, content_emb, categorical_features, scores in train_pbar:
            title_emb = title_emb.to(device)
            content_emb = content_emb.to(device)
            categorical_features = categorical_features.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            predictions = model(title_emb, content_emb, categorical_features)
            loss = criterion(predictions, scores)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * len(scores)
            train_samples += len(scores)
            
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / train_samples
        
        # Evaluation phase
        avg_test_loss = evaluate_model(model, test_loader, device, criterion)
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'learning_rate': current_lr
        })
        
        print(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': model_config.to_dict(),
                'feature_names': feature_names,
                'word_to_index': word_to_index
            }, os.path.join(run_ckpt_dir, 'predictor_model.pt'))
            
            # Save feature engineer for later use
            torch.save({
                'author_stats': feature_engineer.author_stats,
                'domain_stats': feature_engineer.domain_stats
            }, os.path.join(run_ckpt_dir, 'feature_engineer_stats.pt'))
            
            print(f"New best model saved with test loss: {best_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['PATIENCE']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model info
    model_info = {
        'epoch': epoch + 1,
        'best_loss': best_loss,
        'embedding_dim': embedding_dim,
        'num_categorical_features': len(feature_names),
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'feature_names': feature_names
    }
    
    with open(os.path.join(run_ckpt_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best test loss: {best_loss:.4f}")
    print(f"Model saved to: {run_ckpt_dir}")
    
    return model, feature_engineer

def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train Enhanced HN Predictor')
    parser.add_argument('--dummy', action='store_true', help='Use dummy data for testing')
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="enhanced-hn-predictor", name="enhanced-predictor-run")
    
    # Train the model
    model, feature_engineer = train_run(dummy=args.dummy)
    
    wandb.finish()

if __name__ == "__main__":
    main() 