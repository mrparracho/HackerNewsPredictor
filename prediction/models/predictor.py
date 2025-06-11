import torch
import torch.nn as nn

class SimplePredictor(nn.Module):
    """A simple neural network for predicting Hacker News scores from title embeddings."""
    
    def __init__(self, input_dim: int = 32):
        """
        Initialize the predictor model.
        
        Args:
            input_dim (int): Dimension of input embeddings (default: 32)
        """
        super(SimplePredictor, self).__init__()
        
        # Deeper network with dropout for better learning
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim)
            
        Returns:
            torch.Tensor: Predicted scores
        """
        # Ensure input is the right shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        elif len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # Flatten if more than 2 dimensions
        return self.network(x).squeeze() 