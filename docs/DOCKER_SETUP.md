# 🐳 Docker Setup Guide for MLX Hacker News Predictor

## 📋 **Architecture Overview**

We use **3 independent containers** for production deployment:

### **Container 1: ML Training Service** (`Dockerfile.ml`)
- **Purpose**: Train CBOW, SkipGram, and Predictor models independently
- **Base Image**: PyTorch with CUDA support
- **Independence**: Runs independently, saves models to shared volumes

### **Container 2: API Service** (`Dockerfile.api`)
- **Purpose**: Serve predictions via REST API
- **Base Image**: Lightweight Python slim
- **Independence**: Loads trained models, serves independently

### **Container 3: Streamlit Service** (`Dockerfile.streamlit`)
- **Purpose**: Web UI for predictions
- **Base Image**: Lightweight Python slim
- **Independence**: Connects to API service independently

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Install Docker and Docker Compose
# Install NVIDIA Docker (for GPU support)
```

### **1. Environment Setup**
```bash
# Create .env file
echo "WANDB_API_KEY=your_wandb_key_here" > .env
```

### **2. Build and Run**
```bash
# Build all services
docker-compose build

# Run all services
docker-compose up
```

### **3. Access Services**
- **Streamlit UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 **Data Flow**

```
Raw Data → ETL Pipeline → Processed Data → ML Training → Trained Models → API Service → Predictions
```

## 🔧 **Why This Architecture?**

### **Why 3 Independent Containers?**
- ✅ **Service Independence**: Each service runs independently
- ✅ **Resource Optimization**: Heavy training vs lightweight serving
- ✅ **Scalability**: Can scale services independently
- ✅ **Security**: Training environment isolated from production
- ✅ **Maintenance**: Can update services independently

### **Why No Database Container?**
- ✅ **Static Data**: Datasets are static files
- ✅ **No Runtime DB**: Models are trained once, then served from files
- ✅ **Simpler**: Fewer moving parts, easier to manage
- ✅ **Faster**: No database overhead during inference

## 📊 **Usage Examples**

### **Train Models**
```bash
# Train all models
docker-compose up ml_training
```

### **Make Predictions**
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"title": "Show HN: I built a machine learning platform"}'

# Batch predictions
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '["Title 1", "Title 2", "Title 3"]'
```

### **Check Model Status**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
```

## 🛠️ **Development Workflow**

### **1. Data Preparation**
```bash
# Your data files should be in:
./data/combined_data.txt          # Training corpus
./data/hn_data_raw.json          # Hacker News data
```

### **2. Model Training**
```bash
# Train all models
docker-compose up ml_training
```

### **3. Model Serving**
```bash
# Start API service
docker-compose up api
```

### **4. Web Interface**
```bash
# Start Streamlit service
docker-compose up streamlit
```

## 🔍 **File Structure**
```
MLX-Week1/
├── Dockerfile.ml              # ML training container
├── Dockerfile.api             # API service container
├── Dockerfile.streamlit       # Streamlit service container
├── docker-compose.yml         # Orchestration
├── api_service.py             # FastAPI service
├── data/                      # Static datasets
│   ├── combined_data.txt
│   └── hn_data_raw.json
├── models/                    # Trained models
│   ├── word2vec/cbow/checkpoints/
│   ├── word2vec/skipgram/checkpoints/
│   └── predictor/checkpoints/
├── streamlit-app/             # Streamlit application
└── .env                      # Environment variables
```

## 🚨 **Troubleshooting**

### **GPU Issues**
```bash
# Check NVIDIA Docker installation
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### **Memory Issues**
```bash
# Increase Docker memory limit in Docker Desktop
# Or use CPU-only training by removing GPU sections
```

### **Model Loading Issues**
```bash
# Check model files exist
ls -la models/*/checkpoints/
```

## 📈 **Production Considerations**

### **For Production**
- Use Docker Swarm or Kubernetes
- Add load balancer for API
- Implement model versioning
- Add monitoring and logging
- Use production-grade base images

### **For Development**
- Use volume mounts for live code editing
- Use smaller datasets for faster iteration
- Enable debug logging

## 🎯 **Next Steps**

1. **Set up environment variables**
2. **Prepare your data files**
3. **Run training pipeline**
4. **Test API endpoints**
5. **Deploy to production**

This architecture gives you a clean, scalable ML platform that's easy to develop with and deploy! 🚀 