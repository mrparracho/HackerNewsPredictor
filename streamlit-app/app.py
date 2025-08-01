import streamlit as st
import requests
import os

# Set page config
st.set_page_config(
    page_title="HN Score Predictor",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #1E88E5;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .api-status.online {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .api-status.offline {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .feature-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration - Use environment variable in Docker, fallback to localhost
API_URL = os.getenv('API_URL', 'http://localhost:8000')

def check_api_status():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, {}

def predict_score_api(title, content="", url="", author=""):
    """Make prediction using the enhanced API service."""
    try:
        payload = {
            "title": title,
            "content": content,
            "url": url,
            "author": author
        }
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error: {str(e)}")

def main():
    # Title and description
    st.markdown('<div class="title">HN Score Predictor 🚀</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict upvotes using advanced features and ML!</div>', unsafe_allow_html=True)
    
    # Show API URL for debugging
    st.sidebar.markdown(f"**API URL:** {API_URL}")
    
    # Check API status
    api_online, health_info = check_api_status()
    if api_online:
        st.markdown('<div class="api-status online">✅ Enhanced API Service Online</div>', unsafe_allow_html=True)
        
        # Show model info
        if health_info.get('model_loaded'):
            st.markdown('<div class="feature-info">', unsafe_allow_html=True)
            st.markdown(f"**Model:** HN Predictor")
            st.markdown(f"**Vocabulary:** {health_info.get('vocabulary_size', 0)} words")
            st.markdown(f"**Features:** {health_info.get('features_count', 0)} enhanced features")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status offline">❌ Enhanced API Service Offline</div>', unsafe_allow_html=True)
        st.error(f"""
        **Enhanced API Service is not running!**
        
        Expected API URL: {API_URL}
        
        To use this app, please start the enhanced API service first:
        ```bash
        python api_service.py
        ```
        """)
        return
    
    # Input section
    st.markdown("### Enter your post details")
    
    # Title input
    title = st.text_input("Title", placeholder="Type your Hacker News title here...")
    
    # Optional fields
    with st.expander("Advanced Options (Optional)"):
        content = st.text_area("Content/Text", placeholder="Post content or description...")
        url = st.text_input("URL", placeholder="https://example.com")
        author = st.text_input("Author", placeholder="Your username")
    
    if title:
        # Make prediction
        try:
            with st.spinner("Making enhanced prediction..."):
                result = predict_score_api(title, content, url, author)
            
            predicted_score = result["predicted_score"]
            features_used = result.get("features_used", 0)
            
            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Enhanced Prediction")
            st.markdown(f"**Predicted Score:** {predicted_score} upvotes")
            st.markdown(f"**Features Used:** {features_used} enhanced features")
            
            # Show API source information
            st.markdown(f"**API Source:** {result.get('api_source', 'Unknown')}")
            st.markdown(f"**Model:** {result.get('model_info', 'Unknown')}")
            
            # Add some context
            if predicted_score > 100:
                st.success("🚀 This title looks very promising! It might get significant attention.")
            elif predicted_score > 50:
                st.info("👍 This title could do well, but might not be viral.")
            elif predicted_score > 20:
                st.warning("⚠️ This title might get moderate attention. Consider improving it!")
            else:
                st.error("📉 This title might not get much attention. Consider making it more engaging!")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show API response details
            with st.expander("🔍 Full API Response Details"):
                st.json(result)
                st.markdown("**This proves the Streamlit app is using your Enhanced FastAPI backend!** 🎉")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Add some tips
    with st.expander("💡 Tips for Better Titles"):
        st.markdown("""
        **Enhanced Features Used:**
        - Title length and character count
        - Question/exclamation detection
        - Technical terms recognition
        - Post type detection (Show HN, Ask HN)
        - Domain reputation analysis
        - Author history (if available)
        - Time-based features
        
        **General Tips:**
        - Be clear and specific
        - Use technical terms when appropriate
        - Keep it concise
        - Avoid clickbait
        - Include relevant keywords
        - Consider posting time
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Enhanced FastAPI")

if __name__ == "__main__":
    main() 