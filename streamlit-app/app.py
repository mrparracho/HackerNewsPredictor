import streamlit as st
import requests
import json
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
    </style>
""", unsafe_allow_html=True)

# API Configuration - Use environment variable in Docker, fallback to localhost
API_URL = os.getenv('API_URL', 'http://localhost:8000')

def check_api_status():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_score_api(title):
    """Make prediction using the API service."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"title": title},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error: {str(e)}")

def main():
    # Title and description
    st.markdown('<div class="title">Hacker News Score Predictor 🚀</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict how many upvotes your post might get!</div>', unsafe_allow_html=True)
    
    # Show API URL for debugging
    st.sidebar.markdown(f"**API URL:** {API_URL}")
    
    # Check API status
    api_online = check_api_status()
    if api_online:
        st.markdown('<div class="api-status online">✅ API Service Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status offline">❌ API Service Offline</div>', unsafe_allow_html=True)
        st.error(f"""
        **API Service is not running!**
        
        Expected API URL: {API_URL}
        
        To use this app, please start the API service first:
        ```bash
        python api_service.py
        ```
        """)
        return
    
    # Input section
    st.markdown("### Enter your title")
    title = st.text_input("", placeholder="Type your Hacker News title here...")
    
    if title:
        # Make prediction
        try:
            with st.spinner("Making prediction..."):
                result = predict_score_api(title)
            
            predicted_score = result["predicted_score"]
            
            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Prediction")
            st.markdown(f"**Predicted Score:** {predicted_score:.1f} upvotes")
            
            # Show API source information
            st.markdown(f"**API Source:** {result.get('api_source', 'Unknown')}")
            st.markdown(f"**Model:** {result.get('model_info', 'Unknown')}")
            
            # Add some context
            if predicted_score > 100:
                st.success("This title looks promising! It might get significant attention.")
            elif predicted_score > 50:
                st.info("This title could do well, but might not be viral.")
            else:
                st.warning("This title might not get much attention. Consider making it more engaging!")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show API response details
            with st.expander("🔍 Full API Response Details"):
                st.json(result)
                st.markdown("**This proves the Streamlit app is using your FastAPI backend!** 🎉")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Add some tips
    with st.expander("Tips for Better Titles"):
        st.markdown("""
        - Be clear and specific
        - Use technical terms when appropriate
        - Keep it concise
        - Avoid clickbait
        - Include relevant keywords
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and FastAPI")

if __name__ == "__main__":
    main() 