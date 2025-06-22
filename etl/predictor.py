import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from datetime import datetime
from urllib.parse import urlparse
from collections import defaultdict
import re

from .pipelines.hn_pipeline import HNPipeline
from .processors.db_processor import DatabaseProcessor

class PredictorDataProcessor:
    """Process HN data for predictor model training."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.db_processor = DatabaseProcessor()
        os.makedirs(output_dir, exist_ok=True)
        
        # Technical terms and buzzwords for detection
        self.technical_terms = {
            'ai', 'ml', 'machine learning', 'deep learning', 'neural network',
            'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'web3',
            'api', 'rest', 'graphql', 'microservices', 'kubernetes', 'docker',
            'python', 'javascript', 'react', 'vue', 'angular', 'node.js',
            'aws', 'azure', 'gcp', 'cloud', 'serverless', 'lambda'
        }
        
        self.buzzwords = {
            'revolutionary', 'game-changing', 'disruptive', 'innovative',
            'cutting-edge', 'next-generation', 'breakthrough', 'groundbreaking',
            'state-of-the-art', 'world-class', 'enterprise-grade', 'scalable'
        }
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url or not url.startswith('http'):
            return ''
        try:
            return urlparse(url).netloc.lower()
        except:
            return ''
    
    def extract_title_features(self, title: str) -> Dict[str, Any]:
        """Extract features from title."""
        if not title:
            return {}
        
        features = {}
        
        # Basic text features
        features['title_length'] = len(title.split())
        features['title_char_length'] = len(title)
        features['title_has_question'] = '?' in title
        features['title_has_exclamation'] = '!' in title
        features['title_has_numbers'] = bool(re.search(r'\d+', title))
        features['title_has_brackets'] = '[' in title and ']' in title
        
        # Post type detection
        title_lower = title.lower()
        features['title_starts_with_show_hn'] = title_lower.startswith('show hn')
        features['title_starts_with_ask_hn'] = title_lower.startswith('ask hn')
        features['title_starts_with_tell_hn'] = title_lower.startswith('tell hn')
        
        # Technical content detection
        title_words = set(title_lower.split())
        features['title_has_technical_terms'] = len(title_words.intersection(self.technical_terms))
        features['title_has_buzzwords'] = len(title_words.intersection(self.buzzwords))
        
        return features
    
    def extract_content_features(self, content: str, url: str) -> Dict[str, Any]:
        """Extract features from content/URL."""
        features = {}
        
        # Content type detection
        features['content_is_url'] = bool(url and url.startswith('http'))
        features['content_is_text'] = bool(content and not url)
        features['content_has_video'] = False
        features['content_has_pdf'] = False
        
        if url:
            url_lower = url.lower()
            features['content_has_video'] = 'youtube' in url_lower or 'video' in url_lower
            features['content_has_pdf'] = '.pdf' in url_lower
            
            # Domain features
            domain = self.extract_domain(url)
            if domain:
                features['domain'] = domain
                features['is_tech_domain'] = domain in ['github.com', 'stackoverflow.com', 'arxiv.org', 'medium.com']
                features['is_news_domain'] = domain in ['nytimes.com', 'bbc.com', 'reuters.com', 'cnn.com']
                features['is_blog_domain'] = 'blog' in domain or 'medium.com' in domain
            else:
                features['domain'] = None
                features['is_tech_domain'] = False
                features['is_news_domain'] = False
                features['is_blog_domain'] = False
        
        return features
    
    def extract_time_features(self, timestamp: int) -> Dict[str, Any]:
        """Extract time-based features."""
        if not timestamp or not isinstance(timestamp, int):
            return {}
        
        dt = datetime.fromtimestamp(timestamp)
        
        features = {}
        features['hour_of_day'] = dt.hour
        features['day_of_week'] = dt.weekday()  # 0=Monday, 6=Sunday
        features['month_of_year'] = dt.month
        features['week_of_year'] = dt.isocalendar()[1]
        features['is_weekend'] = dt.weekday() >= 5
        features['is_work_hours'] = 9 <= dt.hour <= 17
        features['is_late_night'] = 0 <= dt.hour <= 6
        features['is_peak_hours'] = 8 <= dt.hour <= 10 or 17 <= dt.hour <= 19
        features['is_holiday_season'] = dt.month in [11, 12]  # November/December
        
        return features
    
    def extract_author_features(self, author: str, author_stats: Dict) -> Dict[str, Any]:
        """Extract author-based features."""
        features = {}
        
        if author and author in author_stats:
            stats = author_stats[author]
            features['author_total_posts'] = stats['total_posts']
            features['author_avg_score'] = stats['avg_score']
            features['author_max_score'] = stats['max_score']
            features['author_is_regular'] = stats['is_regular']
            features['author_score_variance'] = stats['score_variance']
        else:
            features['author_total_posts'] = 0
            features['author_avg_score'] = 0
            features['author_max_score'] = 0
            features['author_is_regular'] = False
            features['author_score_variance'] = 0
        
        return features
    
    def extract_engagement_features(self, descendants: int, score: int) -> Dict[str, Any]:
        """Extract engagement features."""
        features = {}
        
        features['comment_count'] = descendants or 0
        features['has_comments'] = bool(descendants and descendants > 0)
        features['comment_engagement_ratio'] = (descendants or 0) / max(score, 1) if score else 0
        
        return features
    
    def calculate_author_stats(self, posts: List[Dict]) -> Dict[str, Dict]:
        """Calculate author statistics from all posts."""
        author_posts = defaultdict(list)
        author_scores = defaultdict(list)
        
        for post in posts:
            if post.get('type') == 'story' and post.get('by') and post.get('score') is not None:
                author_posts[post['by']].append(post)
                author_scores[post['by']].append(post['score'])
        
        author_stats = {}
        for author in author_scores:
            scores = author_scores[author]
            author_stats[author] = {
                'total_posts': len(scores),
                'avg_score': np.mean(scores),
                'max_score': np.max(scores),
                'score_variance': np.var(scores),
                'is_regular': len(scores) > 10
            }
        
        return author_stats
    
    def calculate_domain_stats(self, posts: List[Dict]) -> Dict[str, Dict]:
        """Calculate domain statistics from all posts."""
        domain_posts = defaultdict(list)
        domain_scores = defaultdict(list)
        
        for post in posts:
            if post.get('type') == 'story' and post.get('url'):
                domain = self.extract_domain(post['url'])
                if domain and post.get('score') is not None:
                    domain_posts[domain].append(post)
                    domain_scores[domain].append(post['score'])
        
        domain_stats = {}
        for domain in domain_scores:
            scores = domain_scores[domain]
            domain_stats[domain] = {
                'post_count': len(scores),
                'avg_score': np.mean(scores),
                'max_score': np.max(scores)
            }
        
        return domain_stats
    
    def create_enhanced_features(self, post: Dict, author_stats: Dict, domain_stats: Dict) -> Dict[str, Any]:
        """Create enhanced features for a single post."""
        features = {}
        
        # Title features
        title_features = self.extract_title_features(post.get('title', ''))
        features.update(title_features)
        
        # Content features
        content_features = self.extract_content_features(
            post.get('text', ''), 
            post.get('url', '')
        )
        features.update(content_features)
        
        # Time features
        time_features = self.extract_time_features(post.get('time', 0))
        features.update(time_features)
        
        # Author features
        author_features = self.extract_author_features(post.get('by', ''), author_stats)
        features.update(author_features)
        
        # Engagement features
        engagement_features = self.extract_engagement_features(
            post.get('descendants', 0), 
            post.get('score', 0)
        )
        features.update(engagement_features)
        
        # Post status features
        features['is_dead'] = post.get('dead', False)
        features['post_type'] = post.get('type', 'unknown')
        
        return features
    
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
        
        # Calculate author and domain statistics
        print("Calculating author and domain statistics...")
        author_stats = self.calculate_author_stats(converted_posts)
        domain_stats = self.calculate_domain_stats(converted_posts)
        
        # Create enhanced features for each post
        print("Creating enhanced features...")
        enhanced_posts = []
        
        for i, post in enumerate(valid_posts):
            if i % 1000 == 0:
                print(f"Processing post {i+1}/{len(valid_posts)}")
            
            features = self.create_enhanced_features(post, author_stats, domain_stats)
            
            enhanced_post = {
                'id': post.get('id'),
                'title': post.get('title', ''),
                'text': post.get('text', ''),
                'url': post.get('url', ''),
                'score': post.get('score', 0),
                'features': features
            }
            enhanced_posts.append(enhanced_post)
        
        # Save enhanced data
        print("Saving enhanced predictor data...")
        enhanced_data_path = os.path.join(self.output_dir, "predictor_data.json")
        # Convert numpy types for JSON serialization
        enhanced_posts_serializable = self.convert_numpy_types(enhanced_posts)
        with open(enhanced_data_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_posts_serializable, f, indent=2, ensure_ascii=False)
        
        # Save author and domain statistics
        stats_path = os.path.join(self.output_dir, "predictor_stats.json")
        stats_data = {
            'author_stats': author_stats,
            'domain_stats': domain_stats,
            'total_posts': len(enhanced_posts),
            'feature_names': list(enhanced_posts[0]['features'].keys()) if enhanced_posts else []
        }
        # Convert numpy types for JSON serialization
        stats_data_serializable = self.convert_numpy_types(stats_data)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Enhanced predictor data saved to: {enhanced_data_path}")
        print(f"Statistics saved to: {stats_path}")
        print(f"Total posts processed: {len(enhanced_posts)}")
        print(f"Number of features: {len(stats_data['feature_names'])}")

def main():
    """Main function to run the predictor data processing."""
    output_dir = "data"
    
    # Create processor
    processor = PredictorDataProcessor(output_dir)
    
    # Process data with limit (adjust as needed)
    processor.process_predictor_data(limit=100000)
    
    print("\nPredictor data processing complete!")

if __name__ == "__main__":
    main()