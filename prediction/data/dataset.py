import torch
from torch.utils.data import Dataset

class SimpleHNDataset(Dataset):
    """Dataset class for Hacker News title embeddings and scores."""
    
    def __init__(self, embeddings, scores, data=None):
        """
        Initialize the dataset.
        
        Args:
            embeddings (numpy.ndarray): Array of title embeddings
            scores (numpy.ndarray): Array of corresponding scores
            data (list, optional): Original data for title display
        """
        # Ensure embeddings are the right shape (n_samples, embedding_dim)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        if len(self.embeddings.shape) == 1:
            self.embeddings = self.embeddings.unsqueeze(0)
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.data = data  # Store the original data for title display
        print(f"[Dataset] Created dataset with embeddings shape: {self.embeddings.shape}")
        
    def __len__(self):
        """Return the total number of samples."""
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        """Get a single sample by index."""
        return self.embeddings[idx], self.scores[idx] 