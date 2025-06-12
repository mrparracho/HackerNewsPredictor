import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict
import os

# Database connection string
CONNECTION_STRING = "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"

class DatabaseProcessor:
    def __init__(self):
        self.connection_string = CONNECTION_STRING

    def get_connection(self):
        """Create and return a database connection."""
        return psycopg2.connect(self.connection_string)

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a query and return results as a list of dictionaries."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def get_hn_data(self, limit: int = None) -> List[Dict]:
        """Get Hacker News data from the database."""
        query = """
            SELECT 
                id, type, by, title, text, parent, kids, 
                url, score, descendants, time
            FROM hacker_news.items_by_month_2024_10 
        """
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)

    def process_hn_data(self, results: List[Dict]) -> List[Dict]:
        """Process raw Hacker News data from database."""
        processed_posts = []
        
        # First, identify all stories (posts)
        stories = [row for row in results if row['type'] == 'story' and row['title'] is not None]
        
        for story in stories:
            # For stories, use URL if available, otherwise use text (for Ask HN posts)
            post_content = ""
            if story.get('url'):
                post_content = story['url']
            elif story.get('text'):
                post_content = story['text']
            
            post = {
                'Title': story['title'],
                'Author': story['by'],
                'Content': post_content,
                'Comments': []
            }
            
            # Find all comments for this story
            def get_comments(parent_id):
                comments = []
                for row in results:
                    if row['type'] == 'comment' and row['parent'] == parent_id and row['text'] is not None:
                        comments.append(row['text'])
                        # Recursively get nested comments
                        if row['kids']:
                            comments.extend(get_comments(row['id']))
                return comments
            
            # Get all comments for this story
            post['Comments'] = get_comments(story['id'])
            processed_posts.append(post)
        
        return processed_posts

    def clean_processed_data(self, data: List[Dict]) -> List[Dict]:
        """Clean the processed Hacker News data."""
        cleaned_posts = []
        
        for post in data:
            cleaned_post = {
                'Title': self.clean_text(post['Title']),
                'Author': self.clean_text(post['Author']),
                'Content': self.clean_text(post['Content']),
                'Comments': [self.clean_text(comment) for comment in post['Comments']]
            }
            cleaned_posts.append(cleaned_post)
        
        return cleaned_posts

    def create_massive_string(self, data: List[Dict]) -> str:
        """
        Creates a massive string from the cleaned data by concatenating all text content.
        Each post and its comments are separated by newlines.
        Excludes author names and URLs.
        """
        massive_string = []
        
        for post in data:
            # Add title only (no author)
            massive_string.append(post['Title'])
            
            # Add content only if it's not a URL
            if post['Content'] and not post['Content'].startswith('http'):
                massive_string.append(post['Content'])
            
            # Add comments
            if post['Comments']:
                for comment in post['Comments']:
                    massive_string.append(comment)
            
            # Add separator between posts
            massive_string.append("\n" + "="*80 + "\n")
        
        return "\n".join(massive_string)

    def save_hn_data_to_file(self, output_path: str, limit: int = None):
        """Save Hacker News data to a text file."""
        # Get raw data
        results = self.get_hn_data(limit)
        
        # Process the raw data
        processed_posts = self.process_hn_data(results)
        
        # Clean the processed data
        cleaned_posts = self.clean_processed_data(processed_posts)
        
        # Create and save massive string
        massive_string = self.create_massive_string(cleaned_posts)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(massive_string) 