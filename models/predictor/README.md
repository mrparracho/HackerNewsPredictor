# Enhanced HN Predictor 🚀

A comprehensive neural network model for predicting Hacker News post scores using advanced feature engineering and word embeddings.

## Overview

This enhanced predictor goes beyond simple title embeddings to include:

- **Text Embeddings**: Title and content embeddings using pre-trained CBOW
- **Categorical Features**: 38+ engineered features from post metadata
- **Time Features**: Hour, day, week patterns and engagement timing
- **Author Features**: Reputation, consistency, and historical performance
- **Domain Features**: Source credibility and content type classification
- **Engagement Features**: Comment patterns and interaction metrics

## Features Extracted

### Text Features (2 embeddings + 12 categorical)
- Title embedding (32-dim)
- Content embedding (32-dim)
- Title length, character count, question marks, exclamation marks
- Post type detection (Show HN, Ask HN, Tell HN)
- Technical terms and buzzwords detection
- Numbers and brackets in titles

### Content Features (8 categorical)
- URL vs text content detection
- Video/PDF content detection
- Domain classification (tech, news, blog)
- Domain popularity and average scores

### Time Features (9 categorical)
- Hour of day, day of week, month, week of year
- Weekend vs weekday detection
- Work hours, late night, peak hours detection
- Holiday season detection

### Author Features (5 categorical)
- Total posts by author
- Average score, maximum score
- Author regularity (10+ posts)
- Score variance/consistency

### Engagement Features (3 categorical)
- Comment count
- Has comments flag
- Comment engagement ratio

### Post Features (1 categorical)
- Dead post status

## Project Structure

```
models/predictor/
├── model.py              # Core model and feature engineering
├── train.py              # Training script
├── predict.py            # Prediction and inference
├── enhanced_predictor.yml # Configuration file
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the pre-trained CBOW model:
```bash
# Train CBOW first if not already done
cd models/word2vec/cbow
python train.py
```

3. Ensure you have HN data:
```bash
# Run ETL pipeline to get data
cd etl
python main.py
```

## Usage

### Training

```bash
# Train with real data
python train.py

# Train with dummy data for testing
python train.py --dummy
```

### Prediction

```bash
# Run example predictions
python predict.py
```

### Using the Model Programmatically

```python
from model import EnhancedHNPredictor, HNFeatureEngineer
from predict import predict_score

# Predict score for a single post
score = predict_score(
    title="Show HN: My amazing AI project",
    url="https://github.com/user/project",
    author="developer",
    timestamp=1640995200
)
print(f"Predicted score: {score:.1f}")
```

## Model Architecture

The enhanced predictor uses a multi-input neural network:

```
Input Layers:
├── Title Embedding (32) → Linear(64) → ReLU
├── Content Embedding (32) → Linear(64) → ReLU
└── Categorical Features (38) → Linear(32) → ReLU

Combined Network:
├── Concatenate (64+64+32 = 160)
├── Linear(160 → 128) → ReLU → Dropout(0.2)
├── Linear(128 → 64) → ReLU → Dropout(0.2)
└── Linear(64 → 1) → Score Prediction
```

## Configuration

Edit `enhanced_predictor.yml` to customize:

- Model architecture (embedding dim, hidden dim, dropout)
- Training parameters (learning rate, batch size, epochs)
- Feature engineering (technical terms, buzzwords, domains)
- Data limits and processing options

## Feature Engineering Details

### Author Reputation
- Calculates statistics across all posts by each author
- Includes average score, consistency, and activity level
- Helps identify high-quality contributors

### Domain Analysis
- Classifies content sources (tech, news, blog)
- Tracks domain popularity and average performance
- Identifies trusted sources

### Time Patterns
- Analyzes posting time patterns
- Identifies optimal posting windows
- Accounts for weekend vs weekday differences

### Content Quality
- Detects technical terms and buzzwords
- Analyzes title structure and readability
- Identifies content type (video, PDF, etc.)

## Performance

The enhanced model typically achieves:
- **Lower MSE** compared to title-only models
- **Better generalization** across different post types
- **More interpretable** predictions with feature importance

## Dependencies

- **PyTorch**: Neural network framework
- **NumPy**: Numerical computations
- **WandB**: Experiment tracking
- **Transformers**: Model configuration
- **Scikit-learn**: Feature processing utilities

## Troubleshooting

### Common Issues

1. **CBOW model not found**: Train the CBOW model first
2. **No HN data**: Run the ETL pipeline to get data
3. **Memory issues**: Reduce `DATA_LIMIT` in config
4. **CUDA errors**: Set device to CPU in training script

### Debug Mode

Use dummy data for testing:
```bash
python train.py --dummy
```

## Contributing

To add new features:

1. Extend `HNFeatureEngineer` class in `model.py`
2. Update feature names list in `create_feature_matrix`
3. Adjust model architecture if needed
4. Update configuration file
5. Test with dummy data first

## License

This project follows the same license as the main MLX repository. 