#!/usr/bin/env python3
"""
Test script for local inference to verify the API works correctly.
"""

import sys
import os

# Add the project root to path
sys.path.append('.')

def test_prediction():
    """Test the prediction function directly."""
    try:
        from models.predictor.predict import predict_score, load_model, load_cbow_embeddings
        
        print("🔍 Testing local inference...")
        print("=" * 50)
        
        # Load model and dependencies
        print("📚 Loading model and dependencies...")
        model, word_to_index, feature_names, author_stats, domain_stats, embedding_dim = load_model()
        embeddings = load_cbow_embeddings()
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Vocabulary size: {len(word_to_index)}")
        print(f"🔧 Feature count: {len(feature_names)}")
        print(f"👥 Author stats: {len(author_stats)}")
        print(f"🌐 Domain stats: {len(domain_stats)}")
        print(f"📐 Embedding dimension: {embedding_dim}")
        
        # Test cases
        test_cases = [
            {
                "title": "Show HN: My amazing AI project",
                "content": "This is a test post about machine learning",
                "url": "https://github.com/test/project",
                "author": "test_user"
            },
            {
                "title": "Ask HN: How to learn machine learning?",
                "content": "I want to learn ML and AI",
                "url": "",
                "author": "newbie"
            },
            {
                "title": "Tesla announces new electric vehicle",
                "content": "Breaking news about Tesla",
                "url": "https://techcrunch.com/tesla-news",
                "author": "tech_news"
            }
        ]
        
        print("\n🎯 Running predictions...")
        print("-" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: '{test_case['title']}'")
            print(f"URL: {test_case['url'] if test_case['url'] else 'None'}")
            print(f"Author: {test_case['author']}")
            
            try:
                predicted_score = predict_score(
                    title=test_case['title'],
                    content=test_case['content'],
                    url=test_case['url'],
                    author=test_case['author'],
                    model=model,
                    word_to_index=word_to_index,
                    feature_names=feature_names,
                    author_stats=author_stats,
                    domain_stats=domain_stats,
                    embeddings=embeddings,
                    embedding_dim=embedding_dim
                )
                
                print(f"✅ Predicted score: {predicted_score}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("✅ Local inference test completed!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()

def test_api_endpoint():
    """Test the API endpoint locally."""
    try:
        import requests
        import json
        
        print("\n🌐 Testing API endpoint...")
        print("=" * 50)
        
        # Test data
        test_data = {
            "title": "Show HN: My amazing AI project",
            "content": "This is a test post about machine learning",
            "url": "https://github.com/test/project",
            "author": "test_user"
        }
        
        print(f"📤 Sending request: {json.dumps(test_data, indent=2)}")
        
        # Send request to local API
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  API not running. Start it with: python api_service.py")
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    print("🚀 MLX-Week1 Local Inference Test")
    print("=" * 60)
    
    # Test direct prediction
    test_prediction()
    
    # Test API endpoint (if running)
    test_api_endpoint() 