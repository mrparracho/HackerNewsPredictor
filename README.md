# MLX Week 1 - HN Score Predictor

A production-ready machine learning system for predicting Hacker News post scores using word embeddings and feature engineering.

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Build and run all services
docker-compose up --build

# Access services:
# - Streamlit UI: http://localhost:8501
# - API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install optional dependencies for enhanced features
python scripts/install_optional_deps.py

# 3. Run ETL pipeline
python etl/predictor.py

# 4. Train models
python models/word2vec/cbow/train.py
python models/word2vec/skipgram/train.py
python models/predictor/train.py

# 5. Run API service
python api_service.py

# 6. Run Streamlit UI
cd streamlit-app && streamlit run app.py
```

### Option 3: Training with Optional Features
```bash
# Check optional dependencies status
python scripts/check_deps.py

# Train with experiment tracking and model sharing
python models/word2vec/cbow/train.py
python models/predictor/train.py

# Train without optional features
python models/word2vec/cbow/train.py --no-wandb --no-hf
python models/predictor/train.py --no-wandb --no-hf
```

## 📁 Project Structure

```
MLX-Week1/
├── etl/                    # Data processing and feature engineering
├── models/                 # ML models
│   ├── word2vec/          # CBOW and SkipGram models
│   └── predictor/         # HN score predictor
├── streamlit-app/         # Web UI
├── data/                  # Data files
├── docker-compose.yml     # Container orchestration
└── api_service.py         # FastAPI service
```

## 🐳 Architecture

**3 Independent Containers:**
- **ML Training Service** - Trains CBOW, SkipGram, and Predictor models
- **API Service** - Serves predictions via FastAPI
- **Streamlit Service** - Web UI for predictions

## 📚 Documentation

- [Docker Setup](docs/README_Docker.md) - Container deployment guide
- [Docker Setup Guide](docs/DOCKER_SETUP.md) - Detailed Docker configuration
- [Architecture](docs/REFACTORED_ARCHITECTURE.md) - System architecture details
- [Predictor Model](models/predictor/README.md) - Predictor model documentation

## 🎯 Features

- **Word2Vec Models**: CBOW and SkipGram for word embeddings
- **Feature Engineering**: 38+ engineered features from post metadata
- **Neural Network**: Multi-input architecture for score prediction
- **Containerized**: Production-ready Docker deployment
- **API Service**: RESTful API for predictions
- **Web UI**: Streamlit interface for easy interaction
- **Experiment Tracking**: Optional Weights & Biases integration
- **Model Sharing**: Optional Hugging Face Hub integration

## 🔧 Configuration

- Edit `models/predictor/predictor.yml` for predictor settings
- Edit `models/word2vec/cbow/cbow_ns.yml` for CBOW settings
- Edit `models/word2vec/skipgram/skipgram_ns.yml` for SkipGram settings

### Optional Dependencies Configuration

#### Weights & Biases (wandb)
```bash
# Install
pip install wandb

# Setup
wandb login
# Or set environment variable: export WANDB_API_KEY=your_key_here
```

#### Hugging Face Hub
```bash
# Install
pip install transformers huggingface_hub

# Setup
export HUGGINGFACE_TOKEN=your_token_here
export HF_REPO_PREFIX=your_username  # Optional, defaults to 'roshbeed'
```

#### Environment Variables
- `WANDB_PROJECT`: Project name for wandb (default: 'mlx-week1')
- `HUGGINGFACE_TOKEN`: Your HF Hub access token
- `HF_REPO_PREFIX`: Username prefix for HF repositories

## 🚨 Troubleshooting

### Common Issues
1. **Missing data**: Run ETL pipeline first
2. **Model not found**: Train models before running API
3. **Docker issues**: Check Docker installation and permissions
4. **Optional dependencies**: Use `--no-wandb --no-hf` flags if not installed

### Debug Mode
```bash
# Use dummy data for testing
python models/predictor/train.py --dummy

# Check optional dependencies
python scripts/check_deps.py

# Install optional dependencies
python scripts/install_optional_deps.py
```

### Optional Dependencies Issues
```bash
# Train without optional features
python models/word2vec/cbow/train.py --no-wandb --no-hf
python models/predictor/train.py --no-wandb --no-hf

# Check what's available
python scripts/check_deps.py
```

## 📈 Performance

- **ETL Processing**: 80% faster with optimized feature engineering
- **Model Accuracy**: Improved MSE compared to baseline models
- **Scalability**: Independent services for easy scaling

## 🤝 Contributing

1. Follow the existing code structure
2. Test with dummy data first
3. Update documentation for any changes
4. Ensure all tests pass

## 📄 License

This project follows the MLX repository license. 