import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
import json

class EnhancedPredictorConfig:
    def __init__(
        self,
        embedding_dim=32,
        num_categorical_features=38,
        hidden_dim=128,
        dropout=0.2
    ):
        self.embedding_dim = embedding_dim
        self.num_categorical_features = num_categorical_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
    
    def to_dict(self):
        return {
            'embedding_dim': self.embedding_dim,
            'num_categorical_features': self.num_categorical_features,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class EnhancedHNDataset(Dataset):
    """Dataset class for enhanced HN features."""
    
    def __init__(self, title_embeddings, content_embeddings, categorical_features, scores):
        """
        Initialize the dataset.
        
        Args:
            title_embeddings (numpy.ndarray): Array of title embeddings
            content_embeddings (numpy.ndarray): Array of content embeddings
            categorical_features (numpy.ndarray): Array of categorical features
            scores (numpy.ndarray): Array of corresponding scores
        """
        self.title_embeddings = torch.tensor(title_embeddings, dtype=torch.float32)
        self.content_embeddings = torch.tensor(content_embeddings, dtype=torch.float32)
        self.categorical_features = torch.tensor(categorical_features, dtype=torch.float32)
        self.scores = torch.tensor(scores, dtype=torch.float32)
        
        print(f"[Dataset] Created dataset with:")
        print(f"  Title embeddings shape: {self.title_embeddings.shape}")
        print(f"  Content embeddings shape: {self.content_embeddings.shape}")
        print(f"  Categorical features shape: {self.categorical_features.shape}")
        print(f"  Scores shape: {self.scores.shape}")
        
    def __len__(self):
        """Return the total number of samples."""
        return len(self.title_embeddings)
    
    def __getitem__(self, idx):
        """Get a single sample by index."""
        return (
            self.title_embeddings[idx], 
            self.content_embeddings[idx], 
            self.categorical_features[idx], 
            self.scores[idx]
        )

class EnhancedHNPredictor(nn.Module):
    """Enhanced neural network for predicting Hacker News scores with comprehensive features."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text embedding layers
        self.title_encoder = nn.Linear(config.embedding_dim, 64)
        self.content_encoder = nn.Linear(config.embedding_dim, 64)
        
        # Categorical features encoder
        self.categorical_encoder = nn.Linear(config.num_categorical_features, 32)
        
        # Combined network
        self.network = nn.Sequential(
            nn.Linear(64 + 64 + 32, config.hidden_dim),  # title + content + categorical
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(self, title_emb, content_emb, categorical_features):
        """
        Forward pass through the network.
        
        Args:
            title_emb (torch.Tensor): Title embeddings
            content_emb (torch.Tensor): Content embeddings
            categorical_features (torch.Tensor): Categorical features
            
        Returns:
            torch.Tensor: Predicted scores
        """
        title_encoded = self.title_encoder(title_emb)
        content_encoded = self.content_encoder(content_emb)
        categorical_encoded = self.categorical_encoder(categorical_features)
        
        combined = torch.cat([title_encoded, content_encoded, categorical_encoded], dim=1)
        return self.network(combined).squeeze()
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save the model to directory."""
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save config
        config_dict = self.config.to_dict()
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load the model from directory."""
        config_dict = kwargs.pop("config", None)
        if config_dict is None:
            with open(os.path.join(pretrained_model_name_or_path, "config.json"), "r") as f:
                config_dict = json.load(f)
            config = EnhancedPredictorConfig.from_dict(config_dict)
        
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")))
        return model 