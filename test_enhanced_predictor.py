#!/usr/bin/env python3
"""
Test script for the HN Predictor
"""

import sys
import os

# Add models/predictor to path
sys.path.append('./models/predictor')

def test_predictor():
    """Test the predictor functionality."""
    
    print("🧪 Testing HN Predictor...")
    
    try:
        from models.predictor.predict import predict_score, load_model, load_cbow_embeddings
        
        print("✅ Successfully imported predictor modules")
        
        # Test loading models
        print("\n🔄 Loading models...")
        model, word_to_index, feature_names, author_stats, domain_stats, embedding_dim = load_model()
        embeddings = load_cbow_embeddings()
        
        print(f"✅ Models loaded successfully")
        print(f"📚 Vocabulary size: {len(word_to_index)}")
        print(f"🔧 Features count: {len(feature_names)}")
        print(f"👥 Authors with stats: {len(author_stats)}")
        print(f"🌐 Domains with stats: {len(domain_stats)}")
        
        # Test predictions
        test_cases = [
            {
                "title": "Show HN: I built a machine learning platform",
                "content": "This platform helps developers build ML models easily",
                "url": "https://github.com/user/ml-platform",
                "author": "ml_developer"
            },
            {
                "title": "Ask HN: How to learn machine learning?",
                "content": "I want to get started with ML",
                "url": "",
                "author": "newbie"
            },
            {
                "title": "Tell HN: My experience with PyTorch",
                "content": "After using PyTorch for 2 years...",
                "url": "https://blog.example.com/pytorch-experience",
                "author": "experienced_dev"
            }
        ]
        
        print("\n🎯 Testing predictions...")
        for i, test_case in enumerate(test_cases, 1):
            score = predict_score(
                title=test_case["title"],
                content=test_case["content"],
                url=test_case["url"],
                author=test_case["author"],
                model=model,
                word_to_index=word_to_index,
                feature_names=feature_names,
                author_stats=author_stats,
                domain_stats=domain_stats
            )
            
            print(f"Test {i}: '{test_case['title']}' → {score} points")
        
        print("\n✅ All tests passed! Predictor is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install -r models/predictor/requirements.txt")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure you have trained the model:")
        print("python models/predictor/train.py")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_api_integration():
    """Test API integration."""
    
    print("\n🌐 Testing API integration...")
    
    try:
        import requests
        import json
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API health check passed: {health_data}")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
        
        # Test prediction endpoint
        test_data = {
            "title": "Show HN: Test API integration",
            "content": "Testing the API",
            "url": "https://example.com",
            "author": "tester"
        }
        
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API prediction test passed: {result['predicted_score']} points")
            return True
        else:
            print(f"❌ API prediction test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running:")
        print("python api_service.py")
        return False
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 HN Predictor Test Suite")
    print("=" * 50)
    
    # Test the predictor directly
    predictor_ok = test_predictor()
    
    # Test API integration (if API is running)
    api_ok = test_api_integration()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Predictor: {'✅ PASS' if predictor_ok else '❌ FAIL'}")
    print(f"API Integration: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if predictor_ok and api_ok:
        print("\n🎉 All tests passed! Your predictor is ready to use!")
    elif predictor_ok:
        print("\n⚠️ Predictor works but API is not running. Start it with:")
        print("python api_service.py")
    else:
        print("\n❌ Some tests failed. Check the error messages above.") 