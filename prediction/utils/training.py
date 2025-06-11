import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        
    Returns:
        float: Average loss on the test set
    """
    print("\n[Evaluation] Starting model evaluation...")
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for i, (embeddings, scores) in enumerate(test_loader):
            if i % 2 == 0:  # Print progress every 2 batches
                print(f"[Evaluation] Processing batch {i+1}")
            embeddings, scores = embeddings.to(device), scores.to(device)
            predictions = model(embeddings)
            loss = criterion(predictions, scores)
            total_loss += loss.item() * len(embeddings)
    
    avg_loss = total_loss / len(test_loader.dataset)
    print(f"[Evaluation] Completed. Average loss: {avg_loss:.4f}")
    return avg_loss

def train_model(model, train_loader, test_loader, full_dataset, num_epochs=50):
    """
    Train the model with early stopping and learning rate scheduling.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        full_dataset (Dataset): Full dataset for title display
        num_epochs (int): Maximum number of epochs to train
    """
    print("\n[Training] Starting training process...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Training] Using device: {device}")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Best model tracking
    best_loss = float('inf')
    best_model_path = 'best_predictor.pth'
    model_info_path = 'predictor_info.json'
    patience = 10  # Early stopping patience
    no_improve_epochs = 0
    current_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(num_epochs):
        print(f"\n[Training] Starting epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        total_loss = 0
        num_samples = len(train_loader.dataset)
        print(f"[Training] Training on {num_samples} samples")
        
        for batch_idx, (embeddings, scores) in enumerate(train_loader):
            if batch_idx % 100 == 0:  # Print progress every 100 batches
                print(f"[Training] Processing batch {batch_idx+1}/{len(train_loader)}")
                
            embeddings, scores = embeddings.to(device), scores.to(device)
            
            optimizer.zero_grad()
            predictions = model(embeddings)
            loss = criterion(predictions, scores)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(embeddings)
            
            if batch_idx % 100 == 0:  # Print batch metrics every 100 batches
                print(f"[Training] Batch {batch_idx+1} loss: {loss.item():.4f}")
        
        # Calculate training metrics
        avg_train_loss = total_loss / num_samples
        print(f"\n[Training] Epoch {epoch + 1} training completed")
        print(f"[Training] Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation phase
        print("\n[Training] Starting evaluation phase...")
        avg_test_loss = evaluate_model(model, test_loader, device)
        
        # Update learning rate and check if it changed
        old_lr = current_lr
        scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"\n[Training] Learning rate decreased from {old_lr:.6f} to {current_lr:.6f}")
        
        # Check if this is the best model so far
        is_best = avg_test_loss < best_loss
        if is_best:
            print("\n[Training] New best model found!")
            best_loss = avg_test_loss
            no_improve_epochs = 0
            # Save the best model
            print("[Training] Saving best model...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'input_dim': model.network[0].in_features
            }, best_model_path)
            
            # Save model info as JSON
            model_info = {
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'test_loss': float(avg_test_loss),
                'input_dim': model.network[0].in_features,
                'total_epochs': num_epochs,
                'model_file': best_model_path,
                'max_epochs': num_epochs,
                'learning_rate': float(current_lr)
            }
            
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            print("[Training] Model info saved")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\n[Training] Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Print epoch metrics
        print(f"\n[Training] Epoch {epoch + 1} Summary:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Samples Processed: {num_samples}")
        if is_best:
            print(f"  *** NEW BEST MODEL SAVED! ***")
        
        # Show some test predictions
        print("\n[Training] Sample Test Predictions:")
        print("  " + "-" * 50)
        model.eval()
        with torch.no_grad():
            # Get a batch from test set
            test_batch = next(iter(test_loader))
            embeddings, scores = test_batch
            embeddings, scores = embeddings.to(device), scores.to(device)
            predictions = model(embeddings)
            
            # Show first 3 examples
            for i in range(min(3, len(embeddings))):
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
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    if os.path.exists(best_model_path):
        print(f"Best model saved to: {best_model_path}")
        print(f"Best test loss achieved: {best_loss:.4f}")
        print(f"Model info saved to: {model_info_path}")
    else:
        print("No model was saved (this shouldn't happen)")
    
    print("="*50) 