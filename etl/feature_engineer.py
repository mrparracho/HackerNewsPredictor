import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import re
from datetime import datetime
from urllib.parse import urlparse
from collections import defaultdict
import os
import json

class HNFeatureEngineer:
    """Feature engineering class for Hacker News posts using the actual schema."""
    
    def __init__(self, word_to_ix, embeddings, embedding_dim=32):
        self.word_to_ix = word_to_ix
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        
        # Cache for author and domain statistics
        self.author_stats = {}
        self.domain_stats = {}
        
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
    
    def extract_domain(self, url):
        """Extract domain from URL."""
        if not url or not url.startswith('http'):
            return None
        try:
            return urlparse(url).netloc.lower()
        except:
            return None
    
    def create_text_embedding(self, text):
        """Create embedding for text (title or content)."""
        if not text:
            return np.zeros(self.embeddings.shape[1])
        words = text.lower().split()
        word_embeddings = []
        for word in words:
            if word in self.word_to_ix:
                word_ix = self.word_to_ix[word]
                if word_ix < self.embeddings.shape[0]:
                    emb = self.embeddings[word_ix]
                    # Handle both torch.Tensor and np.ndarray
                    if hasattr(emb, 'cpu'):
                        embedding = emb.cpu().numpy()
                    else:
                        embedding = np.array(emb)
                    word_embeddings.append(embedding)
        if not word_embeddings:
            return np.zeros(self.embeddings.shape[1])
        return np.mean(word_embeddings, axis=0)
    
    def calculate_author_stats(self, all_posts):
        """Calculate author statistics from all posts."""
        author_posts = defaultdict(list)
        author_scores = defaultdict(list)
        
        for post in all_posts:
            if post['type'] == 'story' and post['by'] and post['score'] is not None:
                author_posts[post['by']].append(post)
                author_scores[post['by']].append(post['score'])
        
        for author in author_scores:
            scores = author_scores[author]
            self.author_stats[author] = {
                'total_posts': len(scores),
                'avg_score': np.mean(scores),
                'max_score': np.max(scores),
                'score_variance': np.var(scores),
                'is_regular': len(scores) > 10
            }
    
    def calculate_domain_stats(self, all_posts):
        """Calculate domain statistics from all posts."""
        domain_posts = defaultdict(list)
        domain_scores = defaultdict(list)
        
        for post in all_posts:
            if post['type'] == 'story' and post['url']:
                domain = self.extract_domain(post['url'])
                if domain and post['score'] is not None:
                    domain_posts[domain].append(post)
                    domain_scores[domain].append(post['score'])
        
        for domain in domain_scores:
            scores = domain_scores[domain]
            self.domain_stats[domain] = {
                'post_count': len(scores),
                'avg_score': np.mean(scores),
                'max_score': np.max(scores)
            }
    
    def extract_title_features(self, title):
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
    
    def extract_content_features(self, content, url):
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
                
                # Domain statistics
                if domain in self.domain_stats:
                    features['domain_post_count'] = self.domain_stats[domain]['post_count']
                    features['domain_avg_score'] = self.domain_stats[domain]['avg_score']
                else:
                    features['domain_post_count'] = 0
                    features['domain_avg_score'] = 0
            else:
                features['domain'] = None
                features['is_tech_domain'] = False
                features['is_news_domain'] = False
                features['is_blog_domain'] = False
                features['domain_post_count'] = 0
                features['domain_avg_score'] = 0
        
        return features
    
    def extract_time_features(self, timestamp):
        """Extract time-based features."""
        if not timestamp:
            return {}
        
        # Handle ISO format timestamps (e.g., "2024-10-07T04:43:00")
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                # Fallback to current time if parsing fails
                dt = datetime.now()
        else:
            # Handle Unix timestamp
            try:
                dt = datetime.fromtimestamp(timestamp)
            except (TypeError, ValueError):
                # Fallback to current time if parsing fails
                dt = datetime.now()
        
        features = {}
        features['hour_of_day'] = dt.hour
        features['day_of_week'] = dt.weekday()  # 0=Monday, 6=Sunday
        features['month_of_year'] = dt.month
        features['week_of_year'] = dt.isocalendar()[1]
        
        # Derived time features
        features['is_weekend'] = dt.weekday() >= 5
        features['is_work_hours'] = 9 <= dt.hour <= 17
        features['is_late_night'] = dt.hour >= 22 or dt.hour <= 4
        features['is_peak_hours'] = 8 <= dt.hour <= 20
        features['is_holiday_season'] = dt.month in [11, 12]  # Thanksgiving/Christmas
        
        return features
    
    def extract_author_features(self, author):
        """Extract author-based features."""
        features = {}
        
        if not author:
            features['author_total_posts'] = 0
            features['author_avg_score'] = 0
            features['author_max_score'] = 0
            features['author_is_regular'] = False
            features['author_score_variance'] = 0
        else:
            if author in self.author_stats:
                stats = self.author_stats[author]
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
    
    def extract_engagement_features(self, descendants, score):
        """Extract engagement-based features."""
        features = {}
        
        features['comment_count'] = descendants if descendants else 0
        features['has_comments'] = (descendants or 0) > 0
        features['comment_engagement_ratio'] = (descendants or 0) / max(score or 1, 1)
        
        return features
    
    def create_enhanced_features(self, post_data, all_posts=None):
        """Create comprehensive feature set for a single post."""
        
        # Pre-calculate statistics if not done
        if all_posts and not self.author_stats:
            self.calculate_author_stats(all_posts)
            self.calculate_domain_stats(all_posts)
        
        features = {}
        
        # 1. Text embeddings
        features['title_embedding'] = self.create_text_embedding(post_data.get('title', ''))
        features['content_embedding'] = self.create_text_embedding(post_data.get('text', ''))
        
        # 2. Title features
        title_features = self.extract_title_features(post_data.get('title', ''))
        features.update(title_features)
        
        # 3. Content features
        content_features = self.extract_content_features(
            post_data.get('text', ''), 
            post_data.get('url', '')
        )
        features.update(content_features)
        
        # 4. Time features
        time_features = self.extract_time_features(post_data.get('time'))
        features.update(time_features)
        
        # 5. Author features
        author_features = self.extract_author_features(post_data.get('by'))
        features.update(author_features)
        
        # 6. Engagement features
        engagement_features = self.extract_engagement_features(
            post_data.get('descendants'), 
            post_data.get('score')
        )
        features.update(engagement_features)
        
        # 7. Post status features
        features['is_dead'] = post_data.get('dead', False)
        features['post_type'] = post_data.get('type', 'unknown')
        
        return features
    
    def create_feature_matrix(self, posts_data):
        """Create feature matrix for all posts."""
        print("Calculating author and domain statistics...")
        self.calculate_author_stats(posts_data)
        self.calculate_domain_stats(posts_data)
        
        print("Creating feature matrix...")
        feature_matrices = {
            'title_embeddings': [],
            'content_embeddings': [],
            'categorical_features': [],
            'scores': []
        }
        
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
        
        for i, post in enumerate(posts_data):
            if i % 1000 == 0:
                print(f"Processing post {i+1}/{len(posts_data)}")
            if post['type'] != 'story' or not post.get('title'):
                continue
            features = self.create_enhanced_features(post, posts_data)
            # Store embeddings (ensure 1D numpy arrays of correct length)
            title_emb = np.array(features['title_embedding']).flatten()
            content_emb = np.array(features['content_embedding']).flatten()
            feature_matrices['title_embeddings'].append(title_emb)
            feature_matrices['content_embeddings'].append(content_emb)
            # Store categorical features (ensure all are scalars)
            categorical_features = [float(features.get(name, 0)) for name in categorical_feature_names]
            feature_matrices['categorical_features'].append(categorical_features)
            # Store score
            feature_matrices['scores'].append(float(post.get('score', 0)))
        
        # Convert to numpy arrays
        for key in feature_matrices:
            feature_matrices[key] = np.array(feature_matrices[key])
        
        print(f"Created feature matrix with {feature_matrices['scores'].shape[0]} samples")
        print(f"Title embedding shape: {feature_matrices['title_embeddings'].shape}")
        print(f"Content embedding shape: {feature_matrices['content_embeddings'].shape}")
        print(f"Categorical features shape: {feature_matrices['categorical_features'].shape}")
        
        return feature_matrices, categorical_feature_names 