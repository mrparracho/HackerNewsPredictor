import torch
import torch.nn as nn
import json
import os
from torch.utils.data import Dataset, DataLoader, random_split

class SimpleHNDataset(Dataset):
    def __init__(self, data_path: str, word_to_ix: dict, max_length=50):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.word_to_ix = word_to_ix
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        title = item['Title'].lower().split()
        score = float(item['score'])
        
        # Convert words to indices and pad/truncate to max_length
        word_indices = []
        for word in title[:self.max_length]:  # truncate if too long
            if word in self.word_to_ix:
                word_indices.append(self.word_to_ix[word])
            else:
                word_indices.append(1)  # unknown word index
        
        # Pad with zeros if too short
        if len(word_indices) < self.max_length:
            word_indices.extend([0] * (self.max_length - len(word_indices)))
                
        return torch.tensor(word_indices), torch.tensor(score, dtype=torch.float32)

class SimplePredictor(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super(SimplePredictor, self).__init__()
        
        # Load pre-trained embeddings from CBOW model
        cbow_checkpoint = torch.load('best_cbow_model.pth')
        pretrained_embeddings = cbow_checkpoint['model_state_dict']['embeddings.weight']
        
        # Embedding layer with pre-trained weights
        self.embeddings = nn.Embedding.from_pretrained(
            pretrained_embeddings,
            freeze=True  # Freeze the embeddings
        )
        
        # Simple network: embedding -> hidden -> output
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, max_length)
        # Get embeddings for each word
        embedded = self.embeddings(x)  # shape: (batch_size, max_length, embedding_dim)
        
        # Average pooling over sequence length
        pooled = torch.mean(embedded, dim=1)  # shape: (batch_size, embedding_dim)
        
        # Predict score
        return self.network(pooled).squeeze()

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the model on the test set and return the average loss."""
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for titles, scores in test_loader:
            titles, scores = titles.to(device), scores.to(device)
            predictions = model(titles)
            loss = criterion(predictions, scores)
            total_loss += loss.item() * len(titles)
    
    return total_loss / len(test_loader.dataset)

def train_simple_model(model, train_loader, test_loader, word_to_ix, full_dataset, num_epochs=5):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Best model tracking
    best_loss = float('inf')
    best_model_path = 'best_predictor.pth'
    model_info_path = 'predictor_info.json'
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_samples = len(train_loader.dataset)
        
        for titles, scores in train_loader:
            titles, scores = titles.to(device), scores.to(device)
            
            optimizer.zero_grad()
            predictions = model(titles)
            loss = criterion(predictions, scores)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(titles)
        
        # Calculate training metrics
        avg_train_loss = total_loss / num_samples
        
        # Evaluation phase
        avg_test_loss = evaluate_model(model, test_loader, device)
        
        # Check if this is the best model so far
        is_best = avg_test_loss < best_loss
        if is_best:
            best_loss = avg_test_loss
            # Save the best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'vocab_size': model.embeddings.num_embeddings,
                'embedding_dim': model.embeddings.embedding_dim
            }, best_model_path)
            
            # Save model info as JSON
            model_info = {
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'test_loss': float(avg_test_loss),
                'vocab_size': model.embeddings.num_embeddings,
                'embedding_dim': model.embeddings.embedding_dim,
                'total_epochs': num_epochs,
                'model_file': best_model_path,
                'max_epochs': num_epochs
            }
            
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
        
        # Print epoch metrics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Samples Processed: {num_samples}")
        if is_best:
            print(f"  *** NEW BEST MODEL SAVED! ***")
        
        # Show some test predictions
        print("\n  Sample Test Predictions:")
        print("  " + "-" * 50)
        model.eval()
        with torch.no_grad():
            # Get a batch from test set
            test_batch = next(iter(test_loader))
            titles, scores = test_batch
            titles, scores = titles.to(device), scores.to(device)
            predictions = model(titles)
            
            # Show first 3 examples
            for i in range(min(3, len(titles))):
                # Get the original index in the full dataset
                test_idx = test_loader.dataset.indices[i]
                title = full_dataset.data[test_idx]['Title']
                actual_score = scores[i].item()
                predicted_score = predictions[i].item()
                print(f"  Title: {title}")
                print(f"  Predicted Score: {predicted_score:.2f}")
                print(f"  Actual Score: {actual_score:.2f}")
                print("  " + "-" * 50)
        
        print("-" * 40)
    
    # Training completed - print best model summary
    print("="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    if os.path.exists(best_model_path):
        print(f"Best model saved to: {best_model_path}")
        print(f"Best test loss achieved: {best_loss:.4f}")
        print(f"Model info saved to: {model_info_path}")
    else:
        print("No model was saved (this shouldn't happen)")
    
    print("="*50)

def main():
    # Load word to index mapping
    with open('word_to_lemma_index.json', 'r') as f:
        word_to_ix = json.load(f)
    
    # Create dataset
    full_dataset = SimpleHNDataset('hn_data_cleaned.json', word_to_ix)
    
    # Split into train and test sets (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train model
    model = SimplePredictor(vocab_size=len(word_to_ix))
    train_simple_model(model, train_loader, test_loader, word_to_ix, full_dataset)

if __name__ == "__main__":
    main() 