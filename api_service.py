"""
FastAPI service for MLX Hacker News Score Predictor
Using Predictor from models/predictor
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sys
import numpy as np
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add models/predictor to path
sys.path.append('./models/predictor')

from models.predictor.predict import predict_score, load_model, load_cbow_embeddings

app = FastAPI(
    title="HN Score Predictor",
    description="Predict Hacker News upvote scores from titles using Predictor",
    version="2.0.0"
)

# Global model variables
model: Optional[Any] = None
word_to_index: Optional[Dict[str, int]] = None
feature_names: Optional[List[str]] = None
author_stats: Optional[Dict[str, Any]] = None
domain_stats: Optional[Dict[str, Any]] = None
embeddings: Optional[torch.Tensor] = None
model_loaded: bool = False

class PredictionRequest(BaseModel):
    title: str
    content: str = ""
    url: str = ""
    author: str = ""
    timestamp: Optional[int] = None

class PredictionResponse(BaseModel):
    title: str
    predicted_score: int
    api_source: str = "FastAPI Service"
    model_info: str = "HN Predictor"
    features_used: int = 0

@app.on_event("startup")
async def load_models():
    """Load the trained models on startup"""
    global model, word_to_index, feature_names, author_stats, domain_stats, embeddings, model_loaded
    
    try:
        print("🔄 Loading HN Predictor models...")
        
        # Load the predictor model
        model, word_to_index, feature_names, author_stats, domain_stats, embedding_dim = load_model()
        
        # Load CBOW embeddings
        embeddings = load_cbow_embeddings()
        
        # Load feature engineer stats if available
        stats_path = os.path.join('models/predictor/checkpoints/cur_run', 'feature_engineer_stats.pt')
        if os.path.exists(stats_path):
            stats = torch.load(stats_path)
            author_stats = stats.get('author_stats')
            if author_stats is None:
                author_stats = {}
            domain_stats = stats.get('domain_stats')
            if domain_stats is None:
                domain_stats = {}
            print(f"✅ Loaded feature engineer stats: {len(author_stats)} authors, {len(domain_stats)} domains")
        else:
            print("⚠️  feature_engineer_stats.pt not found, using empty stats.")
        
        model_loaded = True
        print("✅ Models loaded successfully")
        print(f"📚 Using CBOW vocabulary with {len(word_to_index) if word_to_index else 0} words")
        print(f"🔧 Using {len(feature_names) if feature_names else 0} enhanced features")
        print(f"👥 Loaded stats for {len(author_stats) if author_stats else 0} authors and {len(domain_stats) if domain_stats else 0} domains")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        model_loaded = False

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HN Score Predictor API",
        "version": "2.0.0",
        "model": "HN Predictor",
        "endpoint": "/predict",
        "usage": "POST /predict with {\"title\": \"your title here\", \"content\": \"optional content\", \"url\": \"optional url\", \"author\": \"optional author\"}"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "vocabulary_size": len(word_to_index) if word_to_index else 0,
        "features_count": len(feature_names) if feature_names else 0
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "HN Predictor",
        "vocabulary_size": len(word_to_index) if word_to_index else 0,
        "features_count": len(feature_names) if feature_names else 0,
        "authors_with_stats": len(author_stats) if author_stats else 0,
        "domains_with_stats": len(domain_stats) if domain_stats else 0,
        "embedding_dim": embeddings.shape[1] if embeddings is not None else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_score_endpoint(request: PredictionRequest):
    """Predict upvote score for a single title using enhanced features"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    print(f"🔍 API received request for title: '{request.title}'")
    
    try:
        # Use the predictor
        predicted_score = predict_score(
            title=request.title,
            content=request.content,
            url=request.url,
            author=request.author,
            timestamp=request.timestamp,
            model=model,
            word_to_index=word_to_index,
            feature_names=feature_names,
            author_stats=author_stats,
            domain_stats=domain_stats
        )
        
        print(f"🎯 Prediction score: {predicted_score}")
        
        result = PredictionResponse(
            title=request.title,
            predicted_score=round(predicted_score),
            features_used=len(feature_names) if feature_names else 0
        )
        
        print(f"✅ Returning result: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch_scores(request: List[str]):
    """Predict scores for multiple titles"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        from models.predictor.predict import predict_batch_scores
        
        # Convert simple titles to post data format
        posts_data = []
        for title in request:
            posts_data.append({
                'title': title,
                'text': '',
                'url': '',
                'by': '',
                'time': int(datetime.now().timestamp()),
                'descendants': 0,
                'score': 0,
                'dead': False,
                'type': 'story'
            })
        
        # Get batch predictions
        predicted_scores = predict_batch_scores(
            posts_data=posts_data,
            model=model,
            word_to_index=word_to_index,
            feature_names=feature_names,
            author_stats=author_stats,
            domain_stats=domain_stats
        )
        
        return {
            "titles": request,
            "predicted_scores": predicted_scores,
            "model_info": "HN Predictor"
        }
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 