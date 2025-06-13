import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict
import os
import re
from html import unescape
from .text_processor import TextProcessor
import json
from datetime import datetime
import unicodedata

# Database connection string
CONNECTION_STRING = "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"

class DatabaseProcessor:
    def __init__(self):
        self.connection_string = CONNECTION_STRING
        self.text_processor = TextProcessor()

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
        """Get Hacker News data from multiple tables."""
        all_results = []
        
        # List of tables to query
        tables = [
            'hacker_news.items_by_month_2024_10',
            'hacker_news.items_by_month_2024_09',
            'hacker_news.items_by_month_2024_08',
            'hacker_news.items_by_month_2024_07',
            'hacker_news.items_by_month_2024_06',
            'hacker_news.items_by_month_2024_05',
            'hacker_news.items_by_month_2024_04',
            'hacker_news.items_by_month_2024_03',
            'hacker_news.items_by_month_2024_02',
            'hacker_news.items_by_month_2024_01',
        ]
        
        # Calculate limit per table
        per_table_limit = limit // len(tables) if limit else None
        
        for table in tables:
            query = f"""
                SELECT 
                    id, type, by, title, text, parent, kids, 
                    url, score, descendants, time
                FROM {table}
            """
            if per_table_limit:
                query += f" LIMIT {per_table_limit}"
            
            results = self.execute_query(query)
            all_results.extend(results)
            print(f"Fetched {len(results)} rows from {table}")
        
        return all_results

    def normalize_text(self, text: str) -> str:
        """Normalize text by converting special characters to their closest ASCII equivalent."""
        if not isinstance(text, str):
            return text
        
        # First, normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to ASCII, ignoring non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Replace common special characters with their ASCII equivalents
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v',
            '±': '+/-',
            '×': 'x',
            '÷': '/',
            '©': '(c)',
            '®': '(r)',
            '™': '(tm)',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '°': ' degrees',
            '²': '2',
            '³': '3',
            'µ': 'u',
            '¶': 'P',
            '·': '.',
            '¹': '1',
            '¼': '1/4',
            '½': '1/2',
            '¾': '3/4',
            '×': 'x',
            '÷': '/',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '≈': '~=',
            '∞': 'inf',
            '±': '+/-',
            '∑': 'sum',
            '∏': 'prod',
            '√': 'sqrt',
            '∫': 'integral',
            '∆': 'delta',
            '∇': 'nabla',
            '∂': 'partial',
            '∝': 'propto',
            '∞': 'inf',
            '∅': 'empty',
            '∈': 'in',
            '∉': 'not in',
            '∋': 'contains',
            '∌': 'not contains',
            '⊂': 'subset',
            '⊃': 'superset',
            '⊆': 'subset or equal',
            '⊇': 'superset or equal',
            '⊕': 'xor',
            '⊗': 'tensor',
            '⊥': 'perp',
            '⊤': 'top',
            '⊢': 'proves',
            '⊨': 'models',
            '⊩': 'forces',
            '⊪': 'double turnstile',
            '⊫': 'double turnstile',
            '⊬': 'does not prove',
            '⊭': 'does not model',
            '⊮': 'does not force',
            '⊯': 'does not double turnstile',
            '⊰': 'precedes under relation',
            '⊱': 'succeeds under relation',
            '⊲': 'normal subgroup of',
            '⊳': 'contains as normal subgroup',
            '⊴': 'normal subgroup of or equal to',
            '⊵': 'contains as normal subgroup or equal to',
            '⊶': 'original of',
            '⊷': 'image of',
            '⊸': 'multimap',
            '⊹': 'hermitian conjugate matrix',
            '⊺': 'intercalate',
            '⊻': 'xor',
            '⊼': 'nand',
            '⊽': 'nor',
            '⊾': 'right angle with arc',
            '⊿': 'right triangle',
            '⋀': 'n-ary logical and',
            '⋁': 'n-ary logical or',
            '⋂': 'n-ary intersection',
            '⋃': 'n-ary union',
            '⋄': 'diamond operator',
            '⋅': 'dot operator',
            '⋆': 'star operator',
            '⋇': 'division times',
            '⋈': 'bowtie',
            '⋉': 'left normal factor semidirect product',
            '⋊': 'right normal factor semidirect product',
            '⋋': 'left semidirect product',
            '⋌': 'right semidirect product',
            '⋍': 'reversed tilde equals',
            '⋎': 'curly logical or',
            '⋏': 'curly logical and',
            '⋐': 'double subset',
            '⋑': 'double superset',
            '⋒': 'double intersection',
            '⋓': 'double union',
            '⋔': 'pitchfork',
            '⋕': 'equal and parallel to',
            '⋖': 'less than with dot',
            '⋗': 'greater than with dot',
            '⋘': 'very much less than',
            '⋙': 'very much greater than',
            '⋚': 'less than equal to or greater than',
            '⋛': 'greater than equal to or less than',
            '⋜': 'equal to or less than',
            '⋝': 'equal to or greater than',
            '⋞': 'equal to or precedes',
            '⋟': 'equal to or succeeds',
            '⋠': 'does not precede or equal',
            '⋡': 'does not succeed or equal',
            '⋢': 'not square image of or equal to',
            '⋣': 'not square original of or equal to',
            '⋤': 'square image of or not equal to',
            '⋥': 'square original of or not equal to',
            '⋦': 'less than but not equivalent to',
            '⋧': 'greater than but not equivalent to',
            '⋨': 'precedes but not equivalent to',
            '⋩': 'succeeds but not equivalent to',
            '⋪': 'not normal subgroup of',
            '⋫': 'does not contain as normal subgroup',
            '⋬': 'not normal subgroup of or equal to',
            '⋭': 'does not contain as normal subgroup or equal',
            '⋮': 'vertical ellipsis',
            '⋯': 'midline horizontal ellipsis',
            '⋰': 'up right diagonal ellipsis',
            '⋱': 'down right diagonal ellipsis',
            '⋲': 'element of with long horizontal stroke',
            '⋳': 'element of with vertical bar at end of horizontal stroke',
            '⋴': 'small element of with vertical bar at end of horizontal stroke',
            '⋵': 'element of with dot above',
            '⋶': 'element of with overbar',
            '⋷': 'small element of with overbar',
            '⋸': 'element of with underbar',
            '⋹': 'element of with two horizontal strokes',
            '⋺': 'contains with long horizontal stroke',
            '⋻': 'contains with vertical bar at end of horizontal stroke',
            '⋼': 'small contains with vertical bar at end of horizontal stroke',
            '⋽': 'contains with overbar',
            '⋾': 'small contains with overbar',
            '⋿': 'z notation bag membership',
        }
        
        for special, replacement in replacements.items():
            text = text.replace(special, replacement)
        
        return text

    def save_raw_data_to_json(self, data: List[Dict], output_path: str):
        """Save raw data to JSON file."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert datetime objects to ISO format strings and normalize text
        def json_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, str):
                return self.normalize_text(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # Save raw data to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=json_handler)

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
                'Title': self.clean_text(story['title']),
                'Author': story['by'],
                'Content': self.clean_text(post_content),
                'Score': story['score'],
                'Comments': []
            }
            
            # Find all comments for this story
            def get_comments(parent_id):
                comments = []
                for row in results:
                    if row['type'] == 'comment' and row['parent'] == parent_id and row['text'] is not None:
                        comments.append(self.clean_text(row['text']))
                        # Recursively get nested comments
                        if row['kids']:
                            comments.extend(get_comments(row['id']))
                return comments
            
            # Get all comments for this story
            post['Comments'] = get_comments(story['id'])
            processed_posts.append(post)
        
        return processed_posts

    def create_massive_string(self, data: List[Dict]) -> str:
        """
        Creates a massive string from the cleaned data by concatenating all text content.
        Each post and its comments are separated by spaces.
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
        
        return " ".join(massive_string)

    def save_hn_data_to_file(self, output_path: str, limit: int = None):
        """Save Hacker News data to a text file."""
        # Get raw data
        results = self.get_hn_data(limit)
        
        # Save raw data to JSON first
        raw_json_path = os.path.join(os.path.dirname(output_path), "hn_data_raw.json")
        self.save_raw_data_to_json(results, raw_json_path)
        print(f"Raw data saved to: {raw_json_path}")
        
        # Process and clean the raw data
        processed_posts = self.process_hn_data(results)
        
        # Create and save massive string
        massive_string = self.create_massive_string(processed_posts)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(massive_string)

    def clean_text(self, text: str) -> str:
        """Clean text using the TextProcessor."""
        return self.text_processor.clean_text(text) 