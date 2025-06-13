"""
FastAPI service for MLX Hacker News Score Predictor
Simplified for single title predictions only
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import sys
import numpy as np

# Add prediction module to path
sys.path.append('./prediction')

from prediction.models.predictor import SimplePredictor
from prediction.utils.data_processing import load_cbow_embeddings

app = FastAPI(
    title="HN Score Predictor",
    description="Predict Hacker News upvote scores from titles",
    version="1.0.0"
)

# Global model variables
predictor = None
embeddings = None
word_to_ix = None
model_loaded = False

class PredictionRequest(BaseModel):
    title: str

class PredictionResponse(BaseModel):
    title: str
    predicted_score: float
    api_source: str = "FastAPI Service"
    model_info: str = "CBOW + SimplePredictor"

def create_title_embedding(title, word_to_ix, embeddings, embedding_dim=32):
    """Create a title embedding using the same method as training."""
    words = title.lower().split()
    word_embeddings = []
    
    for word in words:
        if word in word_to_ix:
            word_ix = word_to_ix[word]
            # Check if word index is within bounds of the CBOW model
            if word_ix < embeddings.shape[0]:
                embedding = embeddings[word_ix].cpu().numpy()
                word_embeddings.append(embedding)
            else:
                # Word index out of bounds, skip this word
                print(f"⚠️ Word '{word}' (index {word_ix}) not in CBOW model (size {embeddings.shape[0]})")
        else:
            print(f"⚠️ Word '{word}' not in vocabulary")
    
    if not word_embeddings:
        print("⚠️ No valid words found, using zero vector")
        return torch.zeros(embedding_dim, dtype=torch.float32)
    
    # Average the word embeddings (same as training)
    title_embedding = np.mean(word_embeddings, axis=0)
    
    # Ensure we have the right dimension (32) for the predictor
    if len(title_embedding) != embedding_dim:
        if len(title_embedding) < embedding_dim:
            # Pad with zeros if CBOW embeddings are smaller
            padding = np.zeros(embedding_dim - len(title_embedding))
            title_embedding = np.concatenate([title_embedding, padding])
        else:
            # Truncate if CBOW embeddings are larger
            title_embedding = title_embedding[:embedding_dim]
    
    return torch.tensor(title_embedding, dtype=torch.float32)

@app.on_event("startup")
async def load_models():
    """Load the trained models on startup"""
    global predictor, embeddings, word_to_ix, model_loaded
    
    try:
        # Load CBOW model and its word mapping
        cbow_checkpoint = torch.load('models/word2vec/cbow/checkpoints/cbow_model.pt')
        word_to_ix = cbow_checkpoint.get('word_to_lemma_index', {})
        
        # Load CBOW embeddings (needed to create title embeddings)
        embeddings = load_cbow_embeddings('models/word2vec/cbow/checkpoints/cbow_model.pt')
        
        # Load the trained predictor model
        predictor = SimplePredictor(input_dim=32)
        checkpoint = torch.load('prediction/best_predictor.pth', map_location='cpu')
        predictor.load_state_dict(checkpoint['model_state_dict'])
        predictor.eval()
        model_loaded = True
        print("✅ Models loaded successfully")
        print(f"📚 Using CBOW vocabulary with {len(word_to_ix)} words")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        model_loaded = False

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HN Score Predictor API",
        "endpoint": "/predict",
        "usage": "POST /predict with {\"title\": \"your title here\"}"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_score(request: PredictionRequest):
    """Predict upvote score for a single title"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    print(f"🔍 API received request for title: '{request.title}'")
    
    try:
        # Create title embedding using the same method as training
        title_vector = create_title_embedding(request.title, word_to_ix, embeddings)
        print(f"📊 Created title vector shape: {title_vector.shape}")
        
        # Use the trained predictor model
        with torch.no_grad():
            predicted_score = predictor(title_vector)
            score_value = predicted_score.item()
        
        print(f"🎯 Predicted score: {score_value:.2f}")
        
        result = PredictionResponse(
            title=request.title,
            predicted_score=round(score_value, 2)
        )
        
        print(f"✅ Returning result: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 