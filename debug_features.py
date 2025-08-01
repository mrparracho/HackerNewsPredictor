#!/usr/bin/env python3
"""
Debug script to see what features are being created vs expected.
"""

import sys
import os

# Add the project root to path
sys.path.append('.')

def debug_feature_creation():
    """Debug feature creation to see what's happening."""
    try:
        from models.predictor.predict import load_model, load_cbow_embeddings
        from etl.feature_engineer import HNFeatureEngineer
        from datetime import datetime
        
        print("🔍 Debugging feature creation...")
        print("=" * 50)
        
        # Load model and dependencies
        print("📚 Loading model and dependencies...")
        model, word_to_index, feature_names, author_stats, domain_stats, embedding_dim = load_model()
        embeddings = load_cbow_embeddings()
        
        print(f"✅ Model loaded successfully")
        print(f"🔧 Expected feature count: {len(feature_names)}")
        print(f"📋 Expected features: {feature_names}")
        
        # Test case
        test_case = {
            "title": "Show HN: My amazing AI project",
            "content": "This is a test post about machine learning",
            "url": "https://github.com/test/project",
            "author": "test_user"
        }
        
        print(f"\n🎯 Testing feature creation for: '{test_case['title']}'")
        print("-" * 50)
        
        # Create post data structure
        post_data = {
            'title': test_case['title'],
            'text': test_case['content'],
            'url': test_case['url'],
            'by': test_case['author'],
            'time': int(datetime.now().timestamp()),
            'descendants': 0,
            'score': 0,
            'dead': False,
            'type': 'story'
        }
        
        # Initialize feature engineer
        feature_engineer = HNFeatureEngineer(
            word_to_ix=word_to_index,
            embeddings=embeddings,
            embedding_dim=embedding_dim
        )
        
        # Load pre-computed stats
        feature_engineer.author_stats = author_stats
        feature_engineer.domain_stats = domain_stats
        
        # Create features
        print("🔧 Creating features...")
        features = feature_engineer.create_enhanced_features(post_data)
        
        print(f"✅ Features created: {len(features)} total features")
        print(f"📋 Created features: {list(features.keys())}")
        
        # Check which expected features are missing
        missing_features = []
        for expected_feature in feature_names:
            if expected_feature not in features:
                missing_features.append(expected_feature)
        
        print(f"\n❌ Missing features ({len(missing_features)}): {missing_features}")
        
        # Check which unexpected features are present
        unexpected_features = []
        for created_feature in features.keys():
            if created_feature not in feature_names and created_feature not in ['title_embedding', 'content_embedding']:
                unexpected_features.append(created_feature)
        
        print(f"⚠️  Unexpected features ({len(unexpected_features)}): {unexpected_features}")
        
        # Try to create the categorical features tensor
        print(f"\n🔧 Creating categorical features tensor...")
        categorical_features = [features.get(name, 0) for name in feature_names]
        print(f"✅ Categorical features created: {len(categorical_features)} features")
        print(f"📊 Feature values: {categorical_features}")
        
        # Check if we can create the tensor
        import torch
        try:
            categorical_tensor = torch.tensor(categorical_features, dtype=torch.float32).unsqueeze(0)
            print(f"✅ Tensor created successfully: {categorical_tensor.shape}")
        except Exception as e:
            print(f"❌ Failed to create tensor: {e}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_creation() 