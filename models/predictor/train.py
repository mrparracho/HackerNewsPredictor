#!/usr/bin/env python3
"""
Training script for HN Predictor using shared feature engineer.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import sys
import json

# Add models/predictor to path
sys.path.append('./models/predictor')

from model import EnhancedHNPredictor, EnhancedPredictorConfig, EnhancedHNDataset

# Add the project root directory to Python path for utils
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.optional_deps import WandbLogger, HuggingFaceHub, check_optional_deps

def load_yaml_config(path):
    """Load YAML configuration file."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_vocabulary():
    """Load the combined vocabulary from our ETL pipeline."""
    try:
        with open('data/combined_word_to_index.json', 'r', encoding='utf-8') as f:
            word_to_index = json.load(f)
        
        with open('data/combined_word_to_lemma_index.json', 'r', encoding='utf-8') as f:
            word_to_lemma_index = json.load(f)
        
        return word_to_index, word_to_lemma_index
    except FileNotFoundError:
        print("Warning: Vocabulary files not found. Using empty vocabulary.")
        return {}, {}

def load_cbow_embeddings():
    """Load pre-trained CBOW embeddings."""
    checkpoint_dir = 'models/word2vec/cbow/checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'cbow_model.pt')
    
    if not os.path.exists(checkpoint_path):
        print("Warning: CBOW model not found. Using random embeddings.")
        return np.random.randn(1000, 256)  # Dummy embeddings
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
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
        results = db_processor.get_hn_data(limit=limit)
        processed_posts = db_processor.process_hn_data(results)
        
        # Convert to expected format
        converted_posts = []
        for post in processed_posts:
            converted_post = {
                'id': post.get('id', 0),
                'type': 'story',
                'title': post.get('Title', ''),
                'text': post.get('Content', ''),
                'url': post.get('url', ''),
                'by': post.get('Author', ''),
                'score': post.get('Score', 0),
                'descendants': len(post.get('Comments', [])),
                'time': post.get('time', 0),
                'dead': False,
                'Comments': post.get('Comments', [])
            }
            converted_posts.append(converted_post)
        
        return converted_posts
        
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

def log1p_transform(scores):
    """Apply log1p transformation to scores."""
    return np.log1p(scores)

def expm1_transform(log_scores):
    """Convert log-transformed scores back to original scale."""
    return np.expm1(log_scores)

def load_precomputed_features(limit=None):
    """Load pre-computed features from predictor_data.json."""
    json_path = 'data/predictor_data.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Pre-computed features not found at {json_path}. Please run the ETL pipeline first.")
    
    print(f"Loading pre-computed features from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    print(f"Loaded {len(data)} pre-computed feature records")
    
    # Extract raw post data and categorical features
    posts_data = []
    categorical_features = []
    scores = []
    
    # Load stats to get feature names
    stats_path = 'data/predictor_stats.json'
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        feature_names = stats.get('feature_names', [])
    else:
        # Fallback feature names if stats not available
        feature_names = [
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
    
    for record in data:
        # Extract raw post data
        post_data = {
            'id': record.get('id', 0),
            'type': 'story',  # Default to story
            'title': record.get('title', ''),
            'text': record.get('text', ''),
            'url': record.get('url', ''),
            'by': record.get('by', ''),
            'score': record.get('score', 0),
            'descendants': record.get('descendants', 0),
            'time': record.get('time', 0),
            'dead': record.get('dead', False)
        }
        posts_data.append(post_data)
        
        # Extract categorical features from the features dict
        features = record.get('features', {})
        categorical_feature_values = []
        for name in feature_names:
            value = features.get(name, 0)
            # Handle different data types
            if isinstance(value, (int, float)):
                categorical_feature_values.append(float(value))
            elif isinstance(value, bool):
                categorical_feature_values.append(float(value))
            elif isinstance(value, str):
                # Convert string values to numeric (e.g., 'story' -> 1.0)
                if value.lower() in ['story', 'true', 'yes']:
                    categorical_feature_values.append(1.0)
                elif value.lower() in ['comment', 'false', 'no']:
                    categorical_feature_values.append(0.0)
                else:
                    categorical_feature_values.append(0.0)  # Default for unknown strings
            else:
                categorical_feature_values.append(0.0)  # Default for other types
        
        categorical_features.append(categorical_feature_values)
        
        # Apply log1p transformation to scores to handle skewed distribution
        raw_score = float(record.get('score', 0))
        log_score = np.log1p(raw_score)  # log(1 + score)
        scores.append(log_score)
    
    # Convert to numpy arrays
    categorical_features = np.array(categorical_features)
    scores = np.array(scores)
    
    # Log score distribution info
    print(f"Score distribution after log1p transform:")
    print(f"  Min: {scores.min():.2f}")
    print(f"  Max: {scores.max():.2f}")
    print(f"  Mean: {scores.mean():.2f}")
    print(f"  Std: {scores.std():.2f}")
    
    print(f"Pre-computed features shape:")
    print(f"  Posts data: {len(posts_data)} records")
    print(f"  Categorical features: {categorical_features.shape}")
    print(f"  Scores: {scores.shape}")
    
    return {
        'posts_data': posts_data,
        'categorical_features': categorical_features,
        'scores': scores
    }, feature_names

def train_run(dummy=False, use_precomputed=True, use_wandb=True, use_hf=True):
    """
    Train the HN Predictor model using pre-computed features or shared feature engineer.
    
    Args:
        dummy (bool): If True, use dummy data for testing
        use_precomputed (bool): If True, use pre-computed features from predictor_data.json
        use_wandb (bool): If True, enable Weights & Biases logging
        use_hf (bool): If True, enable Hugging Face Hub integration
        
    Returns:
        tuple: (trained_model, metadata)
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
    
    # Initialize variables
    feature_engineer = None
    word_to_index = {}
    embeddings = None
    embedding_dim = None
    
    if dummy:
        # Create dummy data for testing
        print("Using dummy data for testing...")
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
        use_precomputed = False  # Force feature engineering for dummy data
    else:
        posts_data = None  # Not needed when using pre-computed features
    
    if use_precomputed:
        # Use pre-computed features from predictor_data.json
        print("Using pre-computed features from predictor_data.json...")
        try:
            feature_data, feature_names = load_precomputed_features(limit=config.get('DATA_LIMIT', 10000))
            posts_data = feature_data['posts_data']
            precomputed_categorical_features = feature_data['categorical_features']
            scores = feature_data['scores']
            print(f"✅ Successfully loaded pre-computed features for {len(scores)} samples")
            
            # Load vocabulary and embeddings for feature engineering
            print("Loading vocabulary and embeddings...")
            word_to_index, word_to_lemma_index = load_vocabulary()
            embeddings = load_cbow_embeddings()
            embedding_dim = embeddings.shape[1]
            
            # Import feature engineer
            sys.path.append('./etl')
            from feature_engineer import HNFeatureEngineer
            
            # Initialize feature engineer
            print("Initializing feature engineer...")
            feature_engineer = HNFeatureEngineer(
                word_to_ix=word_to_index,
                embeddings=embeddings,
                embedding_dim=embedding_dim
            )
            
            # Generate embeddings on-the-fly using the feature engineer
            print("Generating embeddings on-the-fly...")
            feature_matrices, _ = feature_engineer.create_feature_matrix(posts_data)
            
            # Use pre-computed categorical features instead of regenerating them
            feature_matrices['categorical_features'] = precomputed_categorical_features
            feature_matrices['scores'] = scores
            
            # Ensure we have the full feature list (37 features)
            expected_features = [
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
            
            if len(feature_names) != len(expected_features):
                print(f"⚠️  Warning: Expected {len(expected_features)} features, got {len(feature_names)}")
                print("Using expected feature list...")
                feature_names = expected_features
            else:
                print(f"✅ Using full feature list: {len(feature_names)} features")
            
            print(f"📊 Author stats computed: {len(feature_engineer.author_stats)} authors")
            print(f"🌐 Domain stats computed: {len(feature_engineer.domain_stats)} domains")
            
        except FileNotFoundError as e:
            print(f"⚠️  Pre-computed features not found: {e}")
            print("Falling back to on-the-fly feature engineering...")
            use_precomputed = False
    
    if not use_precomputed:
        # Load HN data and do feature engineering on-the-fly
        print("Loading HN data and performing feature engineering...")
        posts_data = load_hn_data(limit=config.get('DATA_LIMIT', 10000))
        print(f"Loaded {len(posts_data)} posts")
        
        # Filter for story posts with titles and scores
        valid_posts = []
        for post in posts_data:
            if (post.get('type') == 'story' and 
                post.get('title') and 
                post.get('score') is not None):
                valid_posts.append(post)
        
        print(f"Valid posts for training: {len(valid_posts)}")
        
        # Load vocabulary and embeddings for feature engineering
        print("Loading vocabulary and embeddings...")
        word_to_index, word_to_lemma_index = load_vocabulary()
        embeddings = load_cbow_embeddings()
        
        # Initialize feature engineer
        print("Initializing feature engineer...")
        embedding_dim = embeddings.shape[1]  # Use actual CBOW embedding size
        
        # Import feature engineer only when needed
        sys.path.append('./etl')
        from feature_engineer import HNFeatureEngineer
        
        feature_engineer = HNFeatureEngineer(
            word_to_ix=word_to_index,
            embeddings=embeddings,
            embedding_dim=embedding_dim
        )
        
        # Create feature matrix using shared feature engineer
        print("Creating feature matrix...")
        feature_matrices, feature_names = feature_engineer.create_feature_matrix(valid_posts)
        
        # Apply log1p transformation to scores to handle skewed distribution
        print("Applying log1p transformation to scores...")
        raw_scores = feature_matrices['scores']
        log_scores = np.log1p(raw_scores)
        feature_matrices['scores'] = log_scores
        
        # Log score distribution info
        print(f"Score distribution after log1p transform:")
        print(f"  Min: {log_scores.min():.2f}")
        print(f"  Max: {log_scores.max():.2f}")
        print(f"  Mean: {log_scores.mean():.2f}")
        print(f"  Std: {log_scores.std():.2f}")
        
        # Ensure we have the full feature list (37 features)
        expected_features = [
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
        
        if len(feature_names) != len(expected_features):
            print(f"⚠️  Warning: Expected {len(expected_features)} features, got {len(feature_names)}")
            print("Using expected feature list...")
            feature_names = expected_features
        else:
            print(f"✅ Using full feature list: {len(feature_names)} features")
        
        print(f"📊 Author stats computed: {len(feature_engineer.author_stats)} authors")
        print(f"🌐 Domain stats computed: {len(feature_engineer.domain_stats)} domains")
    
    # Create dataset
    dataset = EnhancedHNDataset(
        title_embeddings=feature_matrices['title_embeddings'],
        content_embeddings=feature_matrices['content_embeddings'],
        categorical_features=feature_matrices['categorical_features'],
        scores=feature_matrices['scores']  # These are log-transformed for training
    )
    
    # Store original scores for final evaluation
    original_scores = []
    for post in posts_data:
        if post.get('type') == 'story' and post.get('title'):
            original_scores.append(int(post.get('score', 0)))
    original_scores = np.array(original_scores)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Further split test_dataset into validation and final test sets (50/50)
    val_size = test_size // 2
    final_test_size = test_size - val_size
    val_dataset, final_test_dataset = random_split(test_dataset, [val_size, final_test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    final_test_loader = DataLoader(final_test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    
    # Split original scores to match the dataset splits
    train_scores = original_scores[:train_size]
    val_scores = original_scores[train_size:train_size + val_size]
    final_test_scores = original_scores[train_size + val_size:]
    
    # Initialize model
    model_config = EnhancedPredictorConfig(
        embedding_dim=embedding_dim,
        num_categorical_features=len(feature_names),
        hidden_dim=config['HIDDEN_DIM'],
        dropout=config['DROPOUT']
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
    
    # Check optional dependencies
    deps_status = check_optional_deps()
    
    # Initialize logging
    logger = WandbLogger(
        project_name="hn-predictor",
        run_name="predictor-run",
        config=config,
        enabled=use_wandb and deps_status['wandb']
    )
    
    # Initialize Hugging Face Hub
    hf_hub = HuggingFaceHub() if use_hf else None
    
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
        
        # Validation phase
        avg_val_loss = evaluate_model(model, val_loader, device, criterion)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr
        })
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config.to_dict(),
                'feature_names': feature_names,
                'author_stats': feature_engineer.author_stats if feature_engineer else {},
                'domain_stats': feature_engineer.domain_stats if feature_engineer else {},
                'word_to_index': word_to_index,
                'loss': avg_val_loss  # Save the actual validation loss
            }, os.path.join(run_ckpt_dir, 'predictor_model.pt'))
            
            # Save best loss
            with open(best_loss_path, 'w') as f:
                f.write(f"{best_loss:.6f}")
            
            print(f"New best model saved with val loss: {best_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['PATIENCE']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Push to Hugging Face Hub if enabled
    if hf_hub and hf_hub.available and hf_hub.token:
        try:
            print("\nPushing model to Hugging Face Hub...")
            repo_name = f"{os.environ.get('HF_REPO_PREFIX', 'roshbeed')}/hn-predictor-best"
            
            model_config = {
                "model_type": "hn_predictor",
                "embedding_dim": embedding_dim,
                "num_categorical_features": len(feature_names),
                "hidden_dim": config['HIDDEN_DIM'],
                "dropout": config['DROPOUT'],
                "training_config": config
            }
            
            success = hf_hub.push_model(model, repo_name, model_config)
            if success:
                print(f"Model pushed to {repo_name}")
            else:
                print("Failed to push model to Hugging Face Hub")
        except Exception as e:
            print(f"Warning: Failed to push model to Hugging Face Hub: {e}")

    logger.finish()
    print("\nTraining completed!")
    print(f"Best val loss: {best_loss:.4f}")
    print(f"Model saved to: {run_ckpt_dir}")

    # Final test set evaluation
    final_test_loss = evaluate_model(model, final_test_loader, device, criterion)
    print(f"\nFinal test set loss: {final_test_loss:.4f}")

    # Sample 3 posts from final test set and show predictions vs true scores
    print("\nSample predictions from final test set:")
    model.eval()
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (title_emb, content_emb, categorical_features, log_scores) in enumerate(final_test_loader):
            if sample_count >= 3:
                break
            title_emb = title_emb.to(device)
            content_emb = content_emb.to(device)
            categorical_features = categorical_features.to(device)
            log_scores = log_scores.to(device)
            
            predictions = model(title_emb, content_emb, categorical_features)
            
            for i in range(min(3 - sample_count, len(predictions))):
                predicted_log_score = predictions[i].item()
                
                # Convert from log scale back to original scale
                predicted_score = np.expm1(predicted_log_score)
                
                # Get the corresponding original score
                final_test_idx = batch_idx * config['BATCH_SIZE'] + i
                if final_test_idx < len(final_test_scores):
                    true_score = final_test_scores[final_test_idx]
                else:
                    # Fallback to converting log score back
                    true_score = int(round(np.expm1(log_scores[i].item())))
                
                print(f"Post {sample_count + 1}: Predicted: {int(round(predicted_score))}, True: {true_score}")
                sample_count += 1

    # After training, save feature_engineer_stats.pt
    stats_path = os.path.join(run_ckpt_dir, 'feature_engineer_stats.pt')
    torch.save({
        'author_stats': feature_engineer.author_stats if feature_engineer else {},
        'domain_stats': feature_engineer.domain_stats if feature_engineer else {}
    }, stats_path)
    print(f"Saved feature engineer stats to: {stats_path}")

    return model, {
        'feature_names': feature_names,
        'author_stats': feature_engineer.author_stats if feature_engineer else {},
        'domain_stats': feature_engineer.domain_stats if feature_engineer else {},
        'embedding_dim': embedding_dim,
        'word_to_index': word_to_index
    }

def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train HN Predictor')
    parser.add_argument('--dummy', action='store_true', help='Use dummy data for testing')
    parser.add_argument('--no-precomputed', action='store_true', help='Force on-the-fly feature engineering instead of using pre-computed features')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--no-hf', action='store_true', help='Disable Hugging Face Hub integration')
    args = parser.parse_args()
    
    # Check optional dependencies
    deps_status = check_optional_deps()
    
    # Set flags based on arguments and availability
    use_wandb = not args.no_wandb and deps_status['wandb']
    use_hf = not args.no_hf and deps_status['huggingface']
    
    if not deps_status['wandb'] and not args.no_wandb:
        print("\nWarning: Weights & Biases (wandb) not available.")
        print("Install with: pip install wandb")
        print("Or use --no-wandb to disable wandb logging.\n")
    
    if not deps_status['huggingface'] and not args.no_hf:
        print("\nWarning: Hugging Face libraries not available.")
        print("Install with: pip install transformers huggingface_hub")
        print("Or use --no-hf to disable Hugging Face integration.\n")
    
    try:
        use_precomputed = not args.no_precomputed
        model, metadata = train_run(
            dummy=args.dummy, 
            use_precomputed=use_precomputed,
            use_wandb=use_wandb,
            use_hf=use_hf
        )
        print("✅ Training completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nTo fix this:")
        if use_precomputed:
            print("1. Run the ETL pipeline: python etl/pipelines/hn_pipeline.py")
            print("2. Generate predictor data: python etl/predictor.py")
            print("3. Train word2vec models: python models/word2vec/cbow/train.py")
            print("4. Then run training: python models/predictor/train.py")
        else:
            print("1. Run the ETL pipeline: python etl/pipelines/hn_pipeline.py")
            print("2. Train word2vec models: python models/word2vec/cbow/train.py")
            print("3. Then run training: python models/predictor/train.py")
        
    except Exception as e:
        print(f"❌ Unexpected error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 