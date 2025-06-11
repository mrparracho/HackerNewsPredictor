import streamlit as st
import torch
import json
import os
import sys
import numpy as np

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.models.predictor import SimplePredictor
from prediction.utils.data_processing import load_cbow_embeddings

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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_embeddings():
    """Load the model and embeddings (cached to avoid reloading)."""
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load word to index mapping
    word_to_ix_path = os.path.join(parent_dir, 'word_to_lemma_index.json')
    with open(word_to_ix_path, 'r') as f:
        word_to_ix = json.load(f)
    
    # Load CBOW embeddings
    cbow_model_path = os.path.join(parent_dir, 'best_cbow_model.pth')
    embeddings = load_cbow_embeddings(cbow_model_path)
    
    # Load the predictor model
    model = SimplePredictor(input_dim=32)
    predictor_path = os.path.join(parent_dir, 'best_predictor.pth')
    checkpoint = torch.load(predictor_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, embeddings, word_to_ix

def create_title_embedding(title, word_to_ix, embeddings, embedding_dim=32):
    """Create embedding for a single title."""
    words = title.lower().split()
    word_embeddings = []
    
    for word in words:
        if word in word_to_ix:
            word_ix = word_to_ix[word]
            embedding = embeddings[word_ix].cpu().numpy()
            word_embeddings.append(embedding)
    
    if not word_embeddings:
        return np.zeros(embedding_dim)
    
    return np.mean(word_embeddings, axis=0)

def predict_score(title, model, embeddings, word_to_ix):
    """Predict score for a given title."""
    # Create embedding
    title_embedding = create_title_embedding(title, word_to_ix, embeddings)
    
    # Convert to tensor and add batch dimension
    title_tensor = torch.tensor(title_embedding, dtype=torch.float32).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(title_tensor)
    
    return prediction.item()

def main():
    # Title and description
    st.markdown('<div class="title">Hacker News Score Predictor 🚀</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict how many upvotes your post might get!</div>', unsafe_allow_html=True)
    
    # Load model and embeddings
    try:
        model, embeddings, word_to_ix = load_model_and_embeddings()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you have trained the model first!")
        return
    
    # Input section
    st.markdown("### Enter your title")
    title = st.text_input("", placeholder="Type your Hacker News title here...")
    
    if title:
        # Make prediction
        try:
            predicted_score = predict_score(title, model, embeddings, word_to_ix)
            
            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Prediction")
            st.markdown(f"**Predicted Score:** {predicted_score:.1f} upvotes")
            
            # Add some context
            if predicted_score > 100:
                st.success("This title looks promising! It might get significant attention.")
            elif predicted_score > 50:
                st.info("This title could do well, but might not be viral.")
            else:
                st.warning("This title might not get much attention. Consider making it more engaging!")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show word analysis
            st.markdown("### Word Analysis")
            words = title.lower().split()
            known_words = [word for word in words if word in word_to_ix]
            unknown_words = [word for word in words if word not in word_to_ix]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Known Words:**")
                for word in known_words:
                    st.markdown(f"- {word}")
            
            with col2:
                if unknown_words:
                    st.markdown("**Unknown Words:**")
                    for word in unknown_words:
                        st.markdown(f"- {word}")
                    st.warning("Some words weren't in our training data. This might affect the prediction.")
            
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
    st.markdown("Built with ❤️ using Streamlit and PyTorch")

if __name__ == "__main__":
    main() 