#!/usr/bin/env python3
"""
Prediction script for HN Predictor using shared feature engineer from training.
"""

import torch
import yaml
import os
import json
import numpy as np
from datetime import datetime
import sys
from typing import Dict, Any, List, Optional
from model import EnhancedHNPredictor, EnhancedPredictorConfig

# Add etl to path for feature engineer
sys.path.append('./etl')
from feature_engineer import HNFeatureEngineer

def load_model(embedding_dim=None):
    """Load the trained predictor model, optionally checking embedding_dim."""
    checkpoint_dir = 'models/predictor/checkpoints/cur_run'
    checkpoint_path = os.path.join(checkpoint_dir, 'predictor_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
    checkpoint = torch.load(checkpoint_path)
    config = EnhancedPredictorConfig(**checkpoint['config'])
    # Check embedding_dim if provided
    if embedding_dim is not None and config.embedding_dim != embedding_dim:
        print(f"❌ Model was trained with embedding_dim={config.embedding_dim}, but CBOW embeddings have dimension {embedding_dim}.")
        print("Please retrain the predictor model with the correct embedding_dim, or retrain CBOW to match the model.")
        exit(1)
    model = EnhancedHNPredictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Load feature engineer stats
    stats_path = os.path.join(checkpoint_dir, 'feature_engineer_stats.pt')
    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
        author_stats = stats['author_stats']
        domain_stats = stats['domain_stats']
    else:
        author_stats = {}
        domain_stats = {}
    word_to_index = checkpoint['word_to_index']
    feature_names = checkpoint['feature_names']
    loss = checkpoint.get('loss', None)
    epoch = checkpoint.get('epoch', 0)
    
    if loss is not None:
        print(f"Loaded model from epoch {epoch} with loss {loss:.4f}")
    else:
        print(f"Loaded model from epoch {epoch} (loss not available)")
    
    return model, word_to_index, feature_names, author_stats, domain_stats, config.embedding_dim

def load_cbow_embeddings():
    """Load pre-trained CBOW embeddings."""
    checkpoint_dir = 'models/word2vec/cbow/checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'cbow_model.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CBOW model not found at {checkpoint_path}. Please train CBOW first.")
    
    checkpoint = torch.load(checkpoint_path)
    embeddings = checkpoint['model_state_dict']['in_embed.weight']
    return embeddings

def predict_score(title, content="", url="", author="", timestamp=None, model=None, 
                 word_to_index=None, feature_names=None, author_stats=None, domain_stats=None, embeddings=None, embedding_dim=None):
    """
    Predict the score for a single HN post.
    
    Args:
        title (str): Post title
        content (str): Post content/text
        url (str): Post URL
        author (str): Author username
        timestamp (int): Unix timestamp
        model: Trained model
        word_to_index: Word to index mapping
        feature_names: List of feature names
        author_stats: Author statistics
        domain_stats: Domain statistics
        embeddings: Pre-trained embeddings
        embedding_dim: Dimension of the embeddings
    
    Returns:
        float: Predicted score (converted from log scale)
    """
    if model is None or word_to_index is None or feature_names is None or author_stats is None or domain_stats is None or embeddings is None or embedding_dim is None:
        embeddings = load_cbow_embeddings()
        embedding_dim = embeddings.shape[1]
        model, word_to_index, feature_names, author_stats, domain_stats, model_embedding_dim = load_model(embedding_dim=embedding_dim)
    
    # Create post data structure
    post_data = {
        'title': title,
        'text': content,
        'url': url,
        'by': author,
        'time': timestamp if timestamp else int(datetime.now().timestamp()),
        'descendants': 0,  # Default value for prediction
        'score': 0,  # Placeholder
        'dead': False,
        'type': 'story'
    }
    
    # Initialize feature engineer with correct embedding_dim
    feature_engineer = HNFeatureEngineer(
        word_to_ix=word_to_index,
        embeddings=embeddings,
        embedding_dim=embedding_dim
    )
    
    # Load pre-computed stats
    feature_engineer.author_stats = author_stats
    feature_engineer.domain_stats = domain_stats
    
    # Create features using the same method as training
    features = feature_engineer.create_enhanced_features(post_data)

    # Ensure we have all expected features by using the same feature list as training
    categorical_feature_names = [
        'title_length', 'title_char_length', 'title_has_question', 'title_has_exclamation',
        'title_has_numbers', 'title_has_brackets', 'title_starts_with_show_hn',
        'title_starts_with_ask_hn', 'title_starts_with_tell_hn', 'title_has_technical_terms',
        'title_has_buzzwords', 'content_is_url', 'content_is_text', 'content_has_video',
        'content_has_pdf', 'is_tech_domain', 'is_news_domain', 'is_blog_domain',
        'domain_post_count', 'hour_of_day', 'day_of_week', 'month_of_year', 'week_of_year',
        'is_weekend', 'is_work_hours', 'is_late_night', 'is_peak_hours', 'is_holiday_season',
        'author_total_posts', 'author_avg_score', 'author_max_score', 'author_is_regular',
        'author_score_variance', 'comment_count', 'has_comments', 'comment_engagement_ratio',
        'is_dead'
    ]

    # Extract features in the correct order, using the same list as training
    categorical_features = torch.tensor([features.get(name, 0) for name in categorical_feature_names], dtype=torch.float32).unsqueeze(0)

    # Extract embeddings from features
    title_embedding = torch.tensor(features['title_embedding'], dtype=torch.float32).unsqueeze(0)
    content_embedding = torch.tensor(features['content_embedding'], dtype=torch.float32).unsqueeze(0)

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(title_embedding, content_embedding, categorical_features)
        predicted_log_score = prediction.item()

    # Convert from log scale back to original scale
    predicted_score = np.expm1(predicted_log_score)

    # Round to nearest integer since HN scores are always integers
    predicted_score = max(0, round(predicted_score))

    return predicted_score

def predict_batch_scores(posts_data, model=None, word_to_index=None, feature_names=None, 
                        author_stats=None, domain_stats=None):
    """
    Predict scores for multiple posts.
    
    Args:
        posts_data (list): List of post dictionaries
        model: Trained model
        word_to_index: Word to index mapping
        feature_names: List of feature names
        author_stats: Author statistics
        domain_stats: Domain statistics
    
    Returns:
        list: List of predicted scores
    """
    if model is None:
        model, word_to_index, feature_names, author_stats, domain_stats = load_model()
        embeddings = load_cbow_embeddings()
    else:
        embeddings = load_cbow_embeddings()
    
    # Initialize feature engineer
    feature_engineer = HNFeatureEngineer(
        word_to_ix=word_to_index,
        embeddings=embeddings,
        embedding_dim=32
    )
    
    # Load pre-computed stats
    feature_engineer.author_stats = author_stats
    feature_engineer.domain_stats = domain_stats
    
    # Create feature matrices
    feature_matrices, _ = feature_engineer.create_feature_matrix(posts_data)
    
    # Convert to tensors
    title_embeddings = torch.tensor(feature_matrices['title_embeddings'], dtype=torch.float32)
    content_embeddings = torch.tensor(feature_matrices['content_embeddings'], dtype=torch.float32)
    categorical_features = torch.tensor(feature_matrices['categorical_features'], dtype=torch.float32)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(title_embeddings, content_embeddings, categorical_features)
        predicted_log_scores = predictions.numpy()
    
    # Convert from log scale back to original scale
    predicted_scores = np.expm1(predicted_log_scores)
    
    # Ensure non-negative scores and round to integers since HN scores are always integers
    predicted_scores = np.maximum(0, predicted_scores)
    predicted_scores = np.round(predicted_scores).astype(int)
    
    return predicted_scores.tolist()

def analyze_feature_importance(model, feature_names, sample_size=1000):
    """
    Analyze feature importance by computing gradients.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        sample_size: Number of samples to use for analysis
    
    Returns:
        dict: Feature importance scores
    """
    # This is a simplified feature importance analysis
    # In practice, you might want to use more sophisticated methods
    pass

class HNPredictor:
    """Main predictor class for easy usage."""
    
    def __init__(self, model_path: str = "models/predictor/checkpoints/cur_run/predictor_model.pt"):
        """Initialize the predictor."""
        self.model_path = model_path
        self.model = None
        self.word_to_index = None
        self.feature_names = None
        self.author_stats = None
        self.domain_stats = None
        self.feature_engineer = None
        self.embedding_dim = 32
        
        # Load model and dependencies
        self._load_model()
        self._load_feature_engineer()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            checkpoint = torch.load(self.model_path)
            
            # Load configuration
            config = EnhancedPredictorConfig(**checkpoint['config'])
            self.model = EnhancedHNPredictor(config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load feature engineer stats
            stats_path = self.model_path.replace('predictor_model.pt', 'feature_engineer_stats.pt')
            if os.path.exists(stats_path):
                stats = torch.load(stats_path)
                self.author_stats = stats['author_stats']
                self.domain_stats = stats['domain_stats']
            
            # Load vocabulary and feature names
            self.word_to_index = checkpoint['word_to_index']
            self.feature_names = checkpoint['feature_names']
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_feature_engineer(self):
        """Load the feature engineer with embeddings."""
        try:
            embeddings = load_cbow_embeddings()
            embedding_dim = embeddings.shape[1]
            self.feature_engineer = HNFeatureEngineer(
                word_to_ix=self.word_to_index,
                embeddings=embeddings,
                embedding_dim=embedding_dim
            )
            if self.author_stats is not None:
                self.feature_engineer.author_stats = self.author_stats
            if self.domain_stats is not None:
                self.feature_engineer.domain_stats = self.domain_stats
            print("✅ Feature engineer loaded successfully!")
        except ImportError as e:
            print(f"⚠️  Warning: Could not load word2vec models: {e}")
            print("Using dummy embeddings for feature extraction...")
            embedding_dim = self.embedding_dim or 256
            dummy_embeddings = np.random.randn(1000, embedding_dim)
            self.feature_engineer = HNFeatureEngineer(
                word_to_ix=self.word_to_index or {},
                embeddings=dummy_embeddings,
                embedding_dim=embedding_dim
            )
            if self.author_stats is not None:
                self.feature_engineer.author_stats = self.author_stats
            if self.domain_stats is not None:
                self.feature_engineer.domain_stats = self.domain_stats
    
    def predict_score(self, post_data: Dict[str, Any]) -> int:
        """
        Predict the score for a single HN post.
        
        Args:
            post_data: Dictionary containing post data with keys:
                - title: str (required)
                - text: str (optional)
                - url: str (optional)
                - by: str (optional)
                - time: int or str (optional)
                - descendants: int (optional)
                - score: int (optional, for validation)
                - type: str (optional, defaults to 'story')
                - dead: bool (optional, defaults to False)
        
        Returns:
            int: Predicted score (rounded to integer)
        """
        if not self.feature_engineer:
            self._load_feature_engineer()
        
        # Ensure required fields
        if not post_data.get('title'):
            raise ValueError("Post title is required for prediction")
        
        # Set defaults
        post_data.setdefault('type', 'story')
        post_data.setdefault('dead', False)
        post_data.setdefault('text', '')
        post_data.setdefault('url', '')
        post_data.setdefault('by', '')
        post_data.setdefault('time', int(datetime.now().timestamp()))
        post_data.setdefault('descendants', 0)
        
        # Extract features using shared feature engineer
        features = self.feature_engineer.create_enhanced_features(post_data)
        
        # Prepare input tensors
        title_embedding = torch.tensor(features['title_embedding'], dtype=torch.float32).unsqueeze(0)
        content_embedding = torch.tensor(features['content_embedding'], dtype=torch.float32).unsqueeze(0)
        
        # Create categorical features vector
        categorical_features = []
        for name in self.feature_names:
            value = features.get(name, 0)
            # Handle string categorical features
            if isinstance(value, str):
                if name == 'post_type':
                    # Convert post type to numeric
                    if value == 'story':
                        value = 1.0
                    elif value == 'ask_hn':
                        value = 2.0
                    elif value == 'show_hn':
                        value = 3.0
                    else:
                        value = 0.0
                else:
                    # For other string features, use 0 as default
                    value = 0.0
            categorical_features.append(float(value))
        categorical_tensor = torch.tensor(categorical_features, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(title_embedding, content_embedding, categorical_tensor)
            predicted_log_score = prediction.item()
        
        # Convert from log scale back to original scale
        predicted_score = np.expm1(predicted_log_score)
        predicted_score = max(0, round(predicted_score))
        
        return predicted_score
    
    def predict_batch(self, posts_data: List[Dict[str, Any]]) -> List[int]:
        """
        Predict scores for multiple HN posts.
        
        Args:
            posts_data: List of post data dictionaries
            
        Returns:
            List[int]: List of predicted scores
        """
        if not self.feature_engineer:
            self._load_feature_engineer()
        
        predictions = []
        
        for i, post_data in enumerate(posts_data):
            try:
                score = self.predict_score(post_data)
                predictions.append(score)
            except Exception as e:
                print(f"Error predicting post {i}: {e}")
                predictions.append(0)  # Default score
        
        return predictions
    
    def get_feature_importance(self, post_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Get feature importance for a post (basic implementation).
        
        Args:
            post_data: Post data dictionary
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if not self.feature_engineer:
            self._load_feature_engineer()
        
        # Extract features
        features = self.feature_engineer.create_enhanced_features(post_data)
        
        # Calculate basic importance based on feature values
        importance = {}
        for name in self.feature_names:
            value = features.get(name, 0)
            # Handle string categorical features
            if isinstance(value, str):
                if name == 'post_type':
                    # Convert post type to numeric
                    if value == 'story':
                        value = 1.0
                    elif value == 'ask_hn':
                        value = 2.0
                    elif value == 'show_hn':
                        value = 3.0
                    else:
                        value = 0.0
                else:
                    # For other string features, use 0 as default
                    value = 0.0
            # Simple importance: absolute value normalized
            importance[name] = abs(float(value))
        
        # Normalize importance scores
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance

def main():
    """Main function for testing predictions."""
    try:
        # Initialize predictor
        predictor = HNPredictor()
        
        # Test predictions
        test_posts = [
            {
                'title': 'Show HN: My amazing AI project',
                'text': 'I built this cool AI project that predicts HN scores',
                'url': 'https://github.com/test/project',
                'by': 'test_user',
                'time': int(datetime.now().timestamp())
            },
            {
                'title': 'Ask HN: How to learn machine learning?',
                'text': 'I want to learn ML but don\'t know where to start',
                'by': 'ml_learner',
                'time': int(datetime.now().timestamp())
            },
            {
                'title': 'The future of artificial intelligence',
                'url': 'https://medium.com/ai-future',
                'by': 'ai_expert',
                'time': int(datetime.now().timestamp())
            }
        ]
        
        print("\n🧪 Testing predictions...")
        for i, post in enumerate(test_posts):
            predicted_score = predictor.predict_score(post)
            print(f"Post {i+1}: '{post['title']}' → Predicted score: {predicted_score}")
            
            # Get feature importance
            importance = predictor.get_feature_importance(post)
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top features: {dict(top_features)}")
        
        print("\n✅ Prediction testing completed!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nTo fix this:")
        print("1. Run ETL: python etl/pipelines/hn_pipeline.py")
        print("2. Train word2vec: python models/word2vec/cbow/train.py")
        print("3. Train model: python models/predictor/train.py")
        print("4. Test predictions: python models/predictor/predict.py")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 