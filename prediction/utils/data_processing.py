import torch
import json
import numpy as np

def load_cbow_embeddings(model_path: str):
    """
    Load pre-trained CBOW embeddings from the model file.
    
    Args:
        model_path (str): Path to the CBOW model file
        
    Returns:
        torch.Tensor: Loaded embeddings
    """
    print(f"\n[Embedding Load] Loading embeddings from {model_path}")
    checkpoint = torch.load(model_path)
    embeddings = checkpoint['model_state_dict']['in_embed.weight']
    # Move to CPU before converting to numpy
    embeddings = embeddings.cpu()
    print(f"[Embedding Load] Loaded embeddings with shape: {embeddings.shape}")
    return embeddings

def create_title_embeddings(data_path: str, word_to_ix: dict, embeddings, embedding_dim=128, max_samples=None):
    """
    Pre-compute embeddings for all titles using pre-trained embeddings.
    
    Args:
        data_path (str): Path to the data file
        word_to_ix (dict): Word to index mapping
        embeddings (torch.Tensor): Pre-trained embeddings
        embedding_dim (int): Dimension of embeddings
        max_samples (int, optional): Maximum number of samples to process
        
    Returns:
        tuple: (title_embeddings, scores)
    """
    print(f"\n[Embedding Creation] Starting with max_samples={max_samples if max_samples else 'all'}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"[Embedding Creation] Loaded {len(data)} total samples")
    
    # Use all samples if max_samples is None
    if max_samples:
        data = data[:max_samples]
    
    title_embeddings = []
    scores = []
    
    for i, item in enumerate(data):
        if i % 1000 == 0:  # Print progress every 1000 items
            print(f"[Embedding Creation] Processing item {i+1}/{len(data)}")
            
        title = item['Title'].lower().split()
        score = float(item['score'])
        
        # Get embeddings for each word
        word_embeddings = []
        for word in title:
            if word in word_to_ix:
                word_ix = word_to_ix[word]
                # Get embedding directly from the embeddings tensor and ensure it's on CPU
                embedding = embeddings[word_ix].cpu().numpy()
                word_embeddings.append(embedding)
        
        # If no valid words found, use zero vector
        if not word_embeddings:
            if i % 1000 == 0:  # Only print warnings occasionally
                print(f"[Embedding Creation] Warning: No valid words found in title: {title}")
            title_embedding = np.zeros(embedding_dim)
        else:
            # Average the word embeddings
            title_embedding = np.mean(word_embeddings, axis=0)
        
        title_embeddings.append(title_embedding)
        scores.append(score)
    
    print(f"[Embedding Creation] Completed. Created {len(title_embeddings)} embeddings")
    # Print shape of first embedding for debugging
    if title_embeddings:
        print(f"[Embedding Creation] First embedding shape: {title_embeddings[0].shape}")
    return np.array(title_embeddings), np.array(scores) 