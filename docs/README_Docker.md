# Docker Setup for HN Score Predictor

## 🐳 Production-Ready Containerized Architecture

This project uses **3 independent containers** for production deployment:

### Services Overview

1. **ML Training Service** - Independent training of CBOW, SkipGram, and Predictor models
2. **API Service** - Serves predictions via FastAPI (independent of training)
3. **Streamlit Service** - Web UI for predictions (independent of training)

## 🚀 Quick Start

### 1. Build and Run All Services
```bash
# Build and start all services independently
docker-compose up --build
```

### 2. Access Services
- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/

## 📁 Service Independence

```
ML Training Service (Independent)
API Service (Independent)
Streamlit Service (Independent)
```

## 🔧 Individual Service Commands

### Train Models Only
```bash
docker-compose up ml_training
```

### Run API Only (independent)
```bash
docker-compose up api
```

### Run Streamlit Only (independent)
```bash
docker-compose up streamlit
```

### Run Full Stack (all independent)
```bash
docker-compose up
```

## 🛠️ Development Workflow

1. **Train Models**: `docker-compose up ml_training`
2. **Start API**: `docker-compose up api`
3. **Start Streamlit**: `docker-compose up streamlit`
4. **Access UI**: http://localhost:8501

## 📊 Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   ML Training   │
│   (Port 8501)   │◄──►│   (Port 8000)   │◄──►│   (GPU)         │
│   Frontend      │    │   Backend       │    │   Models        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
   User Interface         Prediction API           Model Training
```

## 🔍 Troubleshooting

### Check Service Status
```bash
docker-compose ps
```

### View Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs streamlit
docker-compose logs api
docker-compose logs ml_training
```

### Rebuild Services
```bash
# Rebuild specific service
docker-compose build streamlit

# Rebuild all services
docker-compose build --no-cache
```

### Clean Up
```bash
# Stop and remove containers
docker-compose down

# Remove volumes too
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## 📝 Environment Variables

Set these in a `.env` file:
```bash
WANDB_API_KEY=your_wandb_key_here
```

## 🎯 What's Dockerized

- ✅ **ML Training Pipeline** (CBOW + SkipGram + Predictor)
- ✅ **FastAPI Backend** (Model serving)
- ✅ **Streamlit Frontend** (User interface)
- ✅ **GPU Support** (CUDA for training)
- ✅ **Service Communication** (Internal networking)
- ✅ **Volume Mounting** (Data persistence)

## 🔄 Service Communication

- **Streamlit** → **API**: HTTP requests to `http://api:8000`
- **API** → **Models**: Local file access to trained models
- **Training** → **Models**: Saves models to shared volume

All services communicate through the `mlx_network` Docker network. 