import psycopg2
import psycopg2.extras
import json
import html
import re

# Database connection string
CONNECTION_STRING = "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"

def connect_db():
    """Connect to PostgreSQL database"""
    try:
        connection = psycopg2.connect(CONNECTION_STRING)
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        print("Connected to database successfully")
        return connection, cursor
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None, None

def execute_query(cursor, query, params=None):
    """Execute SQL query and return results"""
    try:
        cursor.execute(query, params)
        if query.strip().upper().startswith('SELECT'):
            return cursor.fetchall()
        else:
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"
    except psycopg2.Error as e:
        print(f"Error executing query: {e}")
        return None

def clean_text(text):
    """
    Clean text by:
    1. Converting HTML entities to Unicode
    2. Converting Unicode escape sequences to actual characters
    3. Removing HTML tags
    """
    if not text:
        return ""
    
    # Convert HTML entities to Unicode
    text = html.unescape(text)
    
    # Convert Unicode escape sequences
    text = text.encode('utf-8').decode('unicode-escape')
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text

def process_hn_data(results):
    """Process raw Hacker News data from database"""
    processed_posts = []
    
    # First, identify all stories (posts)
    stories = [row for row in results if row['type'] == 'story' and row['title'] is not None]
    
    for story in stories:
        # For stories, use URL if available, otherwise use text (for Ask HN posts)
        post_content = ""
        if story.get('url'):
            post_content = story['url']  # Store just the URL without the "URL: " prefix
        elif story.get('text'):
            post_content = story['text']
        
        post = {
            'Title': story['title'],
            'Author': story['by'],
            'Content': post_content,  # Renamed from 'Post' to 'Content' to be more generic
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

def clean_processed_data(data):
    """Clean the processed Hacker News data"""
    cleaned_posts = []
    
    for post in data:
        cleaned_post = {
            'Title': clean_text(post['Title']),
            'Author': clean_text(post['Author']),
            'Content': clean_text(post['Content']),  # Updated from 'Post' to 'Content'
            'Comments': [clean_text(comment) for comment in post['Comments']]
        }
        cleaned_posts.append(cleaned_post)
    
    return cleaned_posts

def create_massive_string(data):
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

def main():
    connection, cursor = connect_db()
    
    if not connection:
        return
    
    try:
        # Query to get HN data
        custom_query = """
            SELECT 
                id, type, by, title, text, parent, kids, 
                url, score, descendants, time
            FROM hacker_news.items_by_month_2024_10 
            LIMIT 20000;
        """
        results = execute_query(cursor, custom_query)
        
        # Process the raw data
        processed_posts = process_hn_data(results)
        
        # Clean the processed data
        cleaned_posts = clean_processed_data(processed_posts)
        
        # Save raw processed data
        with open('hn_data.json', 'w', encoding='utf-8') as f:
            json.dump(processed_posts, f, ensure_ascii=False, indent=2)
        
        # Save cleaned data
        with open('hn_data_cleaned.json', 'w', encoding='utf-8') as f:
            json.dump(cleaned_posts, f, ensure_ascii=False, indent=2)
        
        # Create and save massive string to data directory
        massive_string = create_massive_string(cleaned_posts)
        with open('data/hn_data_massive.txt', 'w', encoding='utf-8') as f:
            f.write(massive_string)
        
        # Print some examples to verify the cleaning
        print("\nExample of cleaned data (first 3 posts):")
        for post in cleaned_posts[:3]:
            print("\nTitle:", post['Title'])
            print("Content:", post['Content'])
            print("Number of Comments:", len(post['Comments']))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    main() 