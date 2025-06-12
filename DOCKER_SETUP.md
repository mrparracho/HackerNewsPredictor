# 🐳 Docker Setup Guide for MLX Hacker News Predictor

## 📋 **Architecture Overview**

We use **2 main containers** for a clean, efficient setup:

### **Container 1: ML Training Service** (`Dockerfile.ml`)
- **Purpose**: Train CBOW embeddings and prediction models
- **Base Image**: PyTorch with CUDA support
- **Why**: Heavy computation, GPU access needed
- **Data**: Uses static files (`text8`, `hn_data_cleaned.json`)

### **Container 2: API Service** (`Dockerfile.api`)
- **Purpose**: Serve predictions via REST API
- **Base Image**: Lightweight Python slim
- **Why**: Fast inference, minimal dependencies
- **Data**: Uses trained models (`best_cbow_model.pth`, `best_predictor.pth`)

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Install Docker and Docker Compose
# Install NVIDIA Docker (for GPU support)
# Set up your environment
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

# Run ML training (this will train your models)
docker-compose up ml_training

# After training completes, run the API
docker-compose up api
```

### **3. Access Services**
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Jupyter** (optional): http://localhost:8888

## 📁 **Data Flow**

```
Static Data Files → ML Training → Trained Models → API Service
     ↓                    ↓              ↓              ↓
  text8              CBOW Model    best_cbow_model   Predictions
  hn_data.json       Predictor     best_predictor    REST API
```

## 🔧 **Why This Architecture?**

### **Why No Database Container?**
- ✅ **Static Data**: Your datasets (`text8`, `hn_data_cleaned.json`) are static files
- ✅ **No Runtime DB**: Models are trained once, then served from files
- ✅ **Simpler**: Fewer moving parts, easier to manage
- ✅ **Faster**: No database overhead during inference

### **Why 2 Containers Instead of 1?**
- ✅ **Resource Optimization**: Heavy training vs lightweight inference
- ✅ **Scalability**: Can scale API independently
- ✅ **Security**: Training environment isolated from production API
- ✅ **Development**: Can run training separately from serving

## 📊 **Usage Examples**

### **Train Models**
```bash
# Train CBOW embeddings
docker-compose run ml_training python cbow.py

# Train prediction model
docker-compose run ml_training python prediction/train.py
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
./data/text8                    # Wikipedia corpus
./data/hn_data_cleaned.json     # Hacker News data
```

### **2. Model Training**
```bash
# Train CBOW embeddings
docker-compose run ml_training python cbow.py

# Train prediction model
docker-compose run ml_training python prediction/train.py
```

### **3. Model Serving**
```bash
# Start API service
docker-compose up api
```

### **4. Development with Jupyter**
```bash
# Start Jupyter notebook
docker-compose --profile development up jupyter
```

## 🔍 **File Structure**
```
MLX-Week1/
├── Dockerfile.ml              # ML training container
├── Dockerfile.api             # API service container
├── docker-compose.yml         # Orchestration
├── api_service.py             # FastAPI service
├── data/                      # Static datasets
│   ├── text8
│   └── hn_data_cleaned.json
├── models/                    # Trained models
│   ├── best_cbow_model.pth
│   └── best_predictor.pth
├── prediction/                # Prediction code
├── EDA/                      # Data exploration
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
ls -la best_cbow_model.pth best_predictor.pth
```

## 📈 **Scaling Considerations**

### **For Production**
- Use Docker Swarm or Kubernetes
- Add load balancer for API
- Implement model versioning
- Add monitoring and logging

### **For Development**
- Use volume mounts for live code editing
- Enable Jupyter notebook for exploration
- Use smaller datasets for faster iteration

## 🎯 **Next Steps**

1. **Set up environment variables**
2. **Prepare your data files**
3. **Run training pipeline**
4. **Test API endpoints**
5. **Deploy to production**

This architecture gives you a clean, scalable ML platform that's easy to develop with and deploy! 🚀 