import torch
import yaml
import os
import json
import numpy as np
from datetime import datetime
from model import EnhancedHNPredictor, EnhancedPredictorConfig, HNFeatureEngineer

def load_model():
    """Load the trained enhanced predictor model."""
    checkpoint_dir = 'models/predictor/checkpoints/cur_run'
    
    # Load the model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'predictor_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Load configuration
    config = EnhancedPredictorConfig(**checkpoint['config'])
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
    
    # Load vocabulary
    word_to_index = checkpoint['word_to_index']
    feature_names = checkpoint['feature_names']
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    return model, word_to_index, feature_names, author_stats, domain_stats

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
                 word_to_index=None, feature_names=None, author_stats=None, domain_stats=None):
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
    
    Returns:
        float: Predicted score
    """
    if model is None:
        model, word_to_index, feature_names, author_stats, domain_stats = load_model()
        embeddings = load_cbow_embeddings()
    else:
        embeddings = load_cbow_embeddings()
    
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
    
    # Initialize feature engineer
    feature_engineer = HNFeatureEngineer(
        word_to_ix=word_to_index,
        embeddings=embeddings,
        embedding_dim=32
    )
    
    # Load pre-computed stats
    feature_engineer.author_stats = author_stats
    feature_engineer.domain_stats = domain_stats
    
    # Create features
    features = feature_engineer.create_enhanced_features(post_data)
    
    # Extract features in the correct order
    title_embedding = torch.tensor(features['title_embedding'], dtype=torch.float32).unsqueeze(0)
    content_embedding = torch.tensor(features['content_embedding'], dtype=torch.float32).unsqueeze(0)
    categorical_features = torch.tensor([features.get(name, 0) for name in feature_names], dtype=torch.float32).unsqueeze(0)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(title_embedding, content_embedding, categorical_features)
        predicted_score = prediction.item()
    
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
        predicted_scores = predictions.numpy()
    
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
    
    print("Feature importance analysis (simplified):")
    print("This shows the order of features as they appear in the model:")
    
    importance_scores = {}
    for i, feature_name in enumerate(feature_names):
        importance_scores[feature_name] = i  # Placeholder importance
    
    # Sort by importance (in this case, just by order)
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1])
    
    print("\nFeature order in model:")
    for feature_name, importance in sorted_features:
        print(f"  {feature_name}: {importance}")
    
    return importance_scores

def main():
    """Main function to demonstrate prediction."""
    # Load hyperparameters
    config_path = 'models/predictor/predictor.yml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = {
            'EMBEDDING_DIM': 32,
            'HIDDEN_DIM': 128,
            'DROPOUT': 0.2,
            'LEARNING_RATE': 0.001,
            'BATCH_SIZE': 32,
            'NUM_EPOCHS': 50,
            'PATIENCE': 10,
            'WEIGHT_DECAY': 1e-5
        }
    
    try:
        # Load model
        model, word_to_index, feature_names, author_stats, domain_stats = load_model()
        
        # Example predictions
        print("\n🎯 Example Predictions:")
        print("=" * 50)
        
        # Example 1: Show HN post
        title1 = "Show HN: I built an AI-powered code review tool"
        predicted_score1 = predict_score(
            title=title1,
            url="https://github.com/user/code-review-ai",
            author="ai_developer",
            model=model,
            word_to_index=word_to_index,
            feature_names=feature_names,
            author_stats=author_stats,
            domain_stats=domain_stats
        )
        print(f"Title: {title1}")
        print(f"Predicted Score: {predicted_score1:.1f}")
        print("-" * 50)
        
        # Example 2: Ask HN post
        title2 = "Ask HN: How do you stay productive while working remotely?"
        predicted_score2 = predict_score(
            title=title2,
            content="I've been working remotely for 2 years and I'm struggling with productivity...",
            author="remote_worker",
            model=model,
            word_to_index=word_to_index,
            feature_names=feature_names,
            author_stats=author_stats,
            domain_stats=domain_stats
        )
        print(f"Title: {title2}")
        print(f"Predicted Score: {predicted_score2:.1f}")
        print("-" * 50)
        
        # Example 3: News article
        title3 = "OpenAI releases GPT-5 with revolutionary new capabilities"
        predicted_score3 = predict_score(
            title=title3,
            url="https://techcrunch.com/2024/openai-gpt5-release",
            author="tech_news",
            model=model,
            word_to_index=word_to_index,
            feature_names=feature_names,
            author_stats=author_stats,
            domain_stats=domain_stats
        )
        print(f"Title: {title3}")
        print(f"Predicted Score: {predicted_score3:.1f}")
        print("-" * 50)
        
        # Feature importance analysis
        print("\n📊 Feature Importance Analysis:")
        print("=" * 50)
        analyze_feature_importance(model, feature_names)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python train.py")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main() 