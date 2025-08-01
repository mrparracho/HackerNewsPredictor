#!/usr/bin/env python3
"""
Test script with realistic HN post data to verify predictions make sense.
"""

import sys
import os

# Add the project root to path
sys.path.append('.')

def test_realistic_predictions():
    """Test with realistic HN post data."""
    try:
        from models.predictor.predict import predict_score, load_model, load_cbow_embeddings
        
        print("🔍 Testing realistic HN predictions...")
        print("=" * 60)
        
        # Load model and dependencies
        print("📚 Loading model and dependencies...")
        model, word_to_index, feature_names, author_stats, domain_stats, embedding_dim = load_model()
        embeddings = load_cbow_embeddings()
        
        print(f"✅ Model loaded successfully")
        
        # Realistic test cases based on actual HN posts
        test_cases = [
            {
                "title": "Show HN: I built a machine learning model to predict HN scores",
                "content": "I trained a neural network on historical HN data to predict post scores. The model uses features like title length, content type, and posting time.",
                "url": "https://github.com/user/hn-predictor",
                "author": "ml_enthusiast",
                "expected_high": True
            },
            {
                "title": "Ask HN: What's the best way to learn Python?",
                "content": "I'm a beginner and want to learn Python. What resources would you recommend?",
                "url": "",
                "author": "python_newbie",
                "expected_high": True
            },
            {
                "title": "Tesla stock price reaches new high",
                "content": "Tesla shares hit record levels today...",
                "url": "https://finance.yahoo.com/tesla-news",
                "author": "finance_news",
                "expected_high": False
            },
            {
                "title": "The future of artificial intelligence in healthcare",
                "content": "AI is revolutionizing healthcare with new diagnostic tools...",
                "url": "https://medium.com/ai-healthcare",
                "author": "health_tech",
                "expected_high": True
            },
            {
                "title": "Breaking: Major tech company acquires startup",
                "content": "A major acquisition was announced today...",
                "url": "https://techcrunch.com/acquisition",
                "author": "tech_news",
                "expected_high": False
            }
        ]
        
        print("\n🎯 Running realistic predictions...")
        print("-" * 60)
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: '{test_case['title']}'")
            print(f"Type: {'Show HN' if 'Show HN' in test_case['title'] else 'Ask HN' if 'Ask HN' in test_case['title'] else 'Regular post'}")
            print(f"URL: {'Yes' if test_case['url'] else 'No'}")
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
                
                # Analyze the prediction
                if predicted_score > 100:
                    quality = "🔥 Very High"
                elif predicted_score > 50:
                    quality = "⭐ High"
                elif predicted_score > 20:
                    quality = "👍 Medium"
                elif predicted_score > 5:
                    quality = "📉 Low"
                else:
                    quality = "💤 Very Low"
                
                print(f"📊 Quality: {quality}")
                
                # Check if prediction matches expectation
                if test_case['expected_high'] and predicted_score > 20:
                    print("✅ Prediction matches expectation (expected high, got high)")
                elif not test_case['expected_high'] and predicted_score <= 20:
                    print("✅ Prediction matches expectation (expected low, got low)")
                else:
                    print("⚠️  Prediction doesn't match expectation")
                
                results.append({
                    'test': i,
                    'title': test_case['title'],
                    'predicted_score': predicted_score,
                    'expected_high': test_case['expected_high'],
                    'quality': quality
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'test': i,
                    'title': test_case['title'],
                    'predicted_score': -1,
                    'error': str(e)
                })
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 PREDICTION SUMMARY")
        print("=" * 60)
        
        successful_predictions = [r for r in results if r['predicted_score'] >= 0]
        if successful_predictions:
            avg_score = sum(r['predicted_score'] for r in successful_predictions) / len(successful_predictions)
            print(f"📈 Average predicted score: {avg_score:.1f}")
            print(f"🎯 Successful predictions: {len(successful_predictions)}/{len(results)}")
            
            print("\n📋 Detailed Results:")
            for result in results:
                if result['predicted_score'] >= 0:
                    print(f"  Test {result['test']}: {result['predicted_score']} points - {result['quality']}")
                else:
                    print(f"  Test {result['test']}: ERROR - {result.get('error', 'Unknown error')}")
        
        print("\n✅ Realistic inference test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_realistic_predictions() 