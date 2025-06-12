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

def process_title(title, word_to_ix, embeddings, embedding_dim=32):
    """Process a title and create its embedding."""
    words = title.lower().split()
    word_embeddings = []
    
    for word in words:
        if word in word_to_ix:
            word_ix = word_to_ix[word]
            embedding = embeddings[word_ix].cpu().numpy()
            word_embeddings.append(embedding)
    
    if not word_embeddings:
        return torch.zeros(embedding_dim, dtype=torch.float32)
    
    title_embedding = np.mean(word_embeddings, axis=0)
    
    # Ensure correct dimension (32)
    if len(title_embedding) != embedding_dim:
        if len(title_embedding) < embedding_dim:
            padding = np.zeros(embedding_dim - len(title_embedding))
            title_embedding = np.concatenate([title_embedding, padding])
        else:
            title_embedding = title_embedding[:embedding_dim]
    
    return torch.tensor(title_embedding, dtype=torch.float32)

@app.on_event("startup")
async def load_models():
    """Load the trained models on startup"""
    global predictor, embeddings, word_to_ix, model_loaded
    
    try:
        # Load word to index mapping
        with open('word_to_lemma_index.json', 'r') as f:
            word_to_ix = json.load(f)
        
        # Load CBOW embeddings
        embeddings = load_cbow_embeddings('best_cbow_model.pth')
        
        # Load the predictor model
        predictor = SimplePredictor(input_dim=32)
        checkpoint = torch.load('prediction/best_predictor.pth', map_location='cpu')
        predictor.load_state_dict(checkpoint['model_state_dict'])
        predictor.eval()
        model_loaded = True
        print("✅ Models loaded successfully")
            
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
        # Process the title
        title_vector = process_title(request.title, word_to_ix, embeddings)
        print(f"📊 Processed title vector shape: {title_vector.shape}")
        
        # Make prediction
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