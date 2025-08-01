import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from datetime import datetime
from urllib.parse import urlparse
from collections import defaultdict
import re
import sys

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.pipelines.hn_pipeline import HNPipeline
from etl.processors.db_processor import DatabaseProcessor
from etl.feature_engineer import HNFeatureEngineer

class PredictorDataProcessor:
    """Process HN data for predictor model training."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.db_processor = DatabaseProcessor()
        os.makedirs(output_dir, exist_ok=True)
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def process_predictor_data(self, limit: int = 50000) -> None:
        """Process HN data for predictor model training."""
        print("Processing HN data for predictor model...")
        
        # Get raw HN data
        print("Fetching HN data from database...")
        results = self.db_processor.get_hn_data(limit=limit)
        print(f"Total rows fetched: {len(results)}")
        
        # Process the raw data
        processed_posts = self.db_processor.process_hn_data(results)
        print(f"Processed {len(processed_posts)} posts")
        
        # Convert processed posts to the format expected by the predictor
        # The processed data has capitalized field names, convert to lowercase
        converted_posts = []
        for post in processed_posts:
            converted_post = {
                'id': post.get('id', 0),
                'type': 'story',  # All processed posts are stories
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
        
        # Filter for story posts with titles and scores
        valid_posts = []
        for post in converted_posts:
            if (post.get('type') == 'story' and 
                post.get('title') and 
                post.get('score') is not None):
                valid_posts.append(post)
        
        print(f"Valid posts for training: {len(valid_posts)}")
        
        # Save raw data for model training
        print("Saving raw data for model training...")
        
        # Save raw posts
        posts_path = os.path.join(self.output_dir, "hn_data_raw.json")
        posts_serializable = self.convert_numpy_types(valid_posts)
        with open(posts_path, 'w', encoding='utf-8') as f:
            json.dump(posts_serializable, f, indent=2, ensure_ascii=False)
        
        # Generate features using feature engineer
        print("Generating features using feature engineer...")
        self.generate_features(valid_posts)
        
        # Save summary statistics
        summary_path = os.path.join(self.output_dir, "hn_data_summary.json")
        summary = {
            'total_posts': len(valid_posts),
            'processed_at': datetime.now().isoformat(),
            'data_schema': {
                'id': 'int',
                'type': 'str',
                'title': 'str', 
                'text': 'str',
                'url': 'str',
                'by': 'str',
                'score': 'int',
                'descendants': 'int',
                'time': 'int',
                'dead': 'bool'
            }
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Raw data saved to: {posts_path}")
        print(f"✅ Summary saved to: {summary_path}")
        print(f"📊 Total posts processed: {len(valid_posts)}")
        print(f"📝 Data ready for model training with feature engineering")
    
    def generate_features(self, posts_data):
        """Generate features using the feature engineer and save to predictor_data.json."""
        try:
            # Load vocabulary and embeddings
            print("Loading vocabulary and embeddings...")
            word_to_index = {}
            embeddings = None
            
            # Load vocabulary
            vocab_path = os.path.join(self.output_dir, 'combined_word_to_index.json')
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    word_to_index = json.load(f)
                print(f"Loaded vocabulary with {len(word_to_index)} words")
            
            # Load embeddings
            embeddings_path = 'models/word2vec/cbow/checkpoints/cbow_model.pt'
            if os.path.exists(embeddings_path):
                checkpoint = torch.load(embeddings_path, map_location='cpu')
                # The embeddings are stored in the model state dict
                embeddings = checkpoint['model_state_dict']['in_embed.weight'].numpy()
                print(f"Loaded embeddings with shape {embeddings.shape}")
            else:
                print("⚠️  Embeddings not found, using dummy embeddings")
                embeddings = np.random.randn(1000, 32)
            
            # Initialize feature engineer
            embedding_dim = embeddings.shape[1] if embeddings is not None else 32
            feature_engineer = HNFeatureEngineer(
                word_to_ix=word_to_index,
                embeddings=embeddings,
                embedding_dim=embedding_dim
            )
            
            # Create feature matrix
            print("Creating feature matrix...")
            feature_matrices, feature_names = feature_engineer.create_feature_matrix(posts_data)
            
            # Create predictor data with features and embeddings
            predictor_data = []
            for i, post in enumerate(posts_data):
                if i < len(feature_matrices['scores']):
                    predictor_record = {
                        'id': post.get('id', i),
                        'title': post.get('title', ''),
                        'text': post.get('text', ''),
                        'url': post.get('url', ''),
                        'score': int(feature_matrices['scores'][i]),
                        'title_embedding': feature_matrices['title_embeddings'][i].tolist(),
                        'content_embedding': feature_matrices['content_embeddings'][i].tolist(),
                        'features': {}
                    }
                    
                    # Add categorical features
                    categorical_features = feature_matrices['categorical_features'][i]
                    for j, feature_name in enumerate(feature_names):
                        if j < len(categorical_features):
                            predictor_record['features'][feature_name] = float(categorical_features[j])
                    
                    predictor_data.append(predictor_record)
            
            # Save predictor data
            predictor_path = os.path.join(self.output_dir, "predictor_data.json")
            with open(predictor_path, 'w', encoding='utf-8') as f:
                json.dump(predictor_data, f, indent=2, ensure_ascii=False)
            
            # Save feature engineer stats
            stats_path = os.path.join(self.output_dir, "predictor_stats.json")
            stats = {
                'author_stats': feature_engineer.author_stats,
                'domain_stats': feature_engineer.domain_stats,
                'feature_names': feature_names,
                'embedding_dim': embedding_dim,
                'total_samples': len(predictor_data)
            }
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Predictor data saved to: {predictor_path}")
            print(f"✅ Predictor stats saved to: {stats_path}")
            print(f"📊 Generated features for {len(predictor_data)} samples")
            
        except Exception as e:
            print(f"❌ Error generating features: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the predictor data processing."""
    output_dir = "data"
    
    # Create processor
    processor = PredictorDataProcessor(output_dir)
    
    # Process data with limit (adjust as needed)
    processor.process_predictor_data(limit=100000)
    
    print("\n🎉 Predictor data processing complete!")
    print("\nNext steps:")
    print("1. Train word2vec models: python models/word2vec/cbow/train.py")
    print("2. Run model training: python models/predictor/train.py")
    print("3. Test predictions: python models/predictor/predict.py")

if __name__ == "__main__":
    main()