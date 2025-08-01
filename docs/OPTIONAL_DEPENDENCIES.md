# Optional Dependencies Guide

This document explains how to use the optional dependencies for enhanced training features in the MLX-Week1 project.

## Overview

The project now supports optional integration with:
- **Weights & Biases (wandb)**: For experiment tracking and visualization
- **Hugging Face Hub**: For model sharing and versioning

These dependencies are completely optional - the project works without them, but they provide enhanced functionality when available.

## Quick Setup

### 1. Install Optional Dependencies

```bash
# Automatic installation
python scripts/install_optional_deps.py

# Manual installation
pip install wandb transformers huggingface_hub datasets
```

### 2. Set Up Accounts and Tokens

#### Weights & Biases
1. Create account at [wandb.ai](https://wandb.ai)
2. Get your API key from profile settings
3. Login: `wandb login` or set `export WANDB_API_KEY=your_key_here`

#### Hugging Face Hub
1. Create account at [huggingface.co](https://huggingface.co)
2. Get access token from [settings/tokens](https://huggingface.co/settings/tokens)
3. Set environment variable: `export HUGGINGFACE_TOKEN=your_token_here`

### 3. Check Status

```bash
python scripts/check_deps.py
```

## Usage

### Training with All Features

```bash
# CBOW training with experiment tracking and model sharing
python models/word2vec/cbow/train.py

# SkipGram training with experiment tracking and model sharing
python models/word2vec/skipgram/train.py

# Predictor training with experiment tracking and model sharing
python models/predictor/train.py
```

### Training with Specific Features Disabled

```bash
# Disable wandb logging
python models/word2vec/cbow/train.py --no-wandb

# Disable Hugging Face Hub
python models/word2vec/cbow/train.py --no-hf

# Disable both
python models/word2vec/cbow/train.py --no-wandb --no-hf
```

## Features

### Weights & Biases Integration

When enabled, wandb provides:

- **Experiment Tracking**: Log training metrics, hyperparameters, and model artifacts
- **Visualization**: Real-time training curves, model comparisons
- **Hyperparameter Sweeps**: Automated hyperparameter optimization
- **Model Versioning**: Track model checkpoints and configurations

#### Logged Metrics
- Training/validation loss
- Learning rate schedules
- Model evaluation metrics (similarity, analogy accuracy)
- Training time and resource usage

#### Example wandb Dashboard
```
Project: mlx-week1
├── Runs
│   ├── cbow-sweep-run_lr0.001_emb256_win4
│   ├── cbow-sweep-run_lr0.0001_emb128_win8
│   └── predictor-run
└── Sweeps
    └── cbow-hyperparameter-sweep
```

### Hugging Face Hub Integration

When enabled, HF Hub provides:

- **Model Sharing**: Push trained models to HF Hub for sharing
- **Model Versioning**: Track model versions and configurations
- **Easy Deployment**: Download models for inference
- **Community Sharing**: Share models with the ML community

#### Model Repository Structure
```
roshbeed/cbow-model-best/
├── pytorch_model.bin          # Model weights
├── config.json               # Model configuration
├── README.md                 # Model documentation
└── training_config.json      # Training parameters
```

#### Environment Variables
- `HUGGINGFACE_TOKEN`: Your HF Hub access token
- `HF_REPO_PREFIX`: Username prefix for repositories (default: 'roshbeed')

## Configuration

### Environment Variables

```bash
# Weights & Biases
export WANDB_PROJECT="mlx-week1"
export WANDB_API_KEY="your_wandb_api_key"

# Hugging Face Hub
export HUGGINGFACE_TOKEN="your_hf_token"
export HF_REPO_PREFIX="your_username"
```

### Training Configuration Files

The optional dependencies respect existing configuration files:

- `models/word2vec/cbow/cbow_ns.yml`
- `models/word2vec/skipgram/skipgram_ns.yml`
- `models/predictor/predictor.yml`

Additional wandb-specific configurations are automatically added when wandb is enabled.

## Troubleshooting

### Common Issues

#### wandb Not Available
```bash
# Error: No module named 'wandb'
pip install wandb

# Or disable wandb
python models/word2vec/cbow/train.py --no-wandb
```

#### Hugging Face Libraries Not Available
```bash
# Error: No module named 'transformers'
pip install transformers huggingface_hub

# Or disable HF integration
python models/word2vec/cbow/train.py --no-hf
```

#### Missing API Keys
```bash
# wandb: Not logged in
wandb login

# HF: Token not set
export HUGGINGFACE_TOKEN=your_token_here
```

### Debug Mode

```bash
# Check dependency status
python scripts/check_deps.py

# Train with verbose logging
python models/word2vec/cbow/train.py --dummy --no-wandb --no-hf
```

## Advanced Usage

### Custom wandb Configuration

```python
# In your training script
import wandb

wandb.init(
    project="my-custom-project",
    name="experiment-001",
    config={
        "learning_rate": 0.001,
        "embedding_size": 256,
        "custom_param": "value"
    }
)
```

### Custom HF Hub Repository Names

```bash
# Set custom repository prefix
export HF_REPO_PREFIX="my-username"

# Models will be pushed to:
# my-username/cbow-model-best
# my-username/skipgram-model-best
# my-username/hn-predictor-best
```

### Integration with Existing Workflows

The optional dependencies are designed to be non-intrusive:

- **Backward Compatible**: Existing scripts work without changes
- **Graceful Degradation**: Features are disabled when dependencies unavailable
- **Local Fallback**: Models are always saved locally regardless of HF Hub status

## Performance Impact

### Minimal Overhead
- **wandb**: ~5-10% training time overhead for logging
- **HF Hub**: Only affects model saving (negligible during training)
- **Memory**: Minimal additional memory usage

### Network Usage
- **wandb**: Continuous logging to wandb servers
- **HF Hub**: Only during model upload (typically once per training run)

## Best Practices

### For Development
```bash
# Use dummy data and disable optional features for quick testing
python models/word2vec/cbow/train.py --dummy --no-wandb --no-hf
```

### For Production Training
```bash
# Enable all features for comprehensive tracking
python models/word2vec/cbow/train.py
```

### For CI/CD
```bash
# Disable optional features in automated environments
python models/word2vec/cbow/train.py --no-wandb --no-hf
```

## Migration Guide

### From Previous Versions

If you're upgrading from a version without optional dependencies:

1. **No Breaking Changes**: Existing scripts continue to work
2. **Enhanced Features**: Enable optional dependencies for new features
3. **Gradual Adoption**: Enable features one at a time

### Example Migration
```bash
# Old way (still works)
python models/word2vec/cbow/train.py

# New way with enhanced features
python models/word2vec/cbow/train.py

# Explicitly disable (same as old behavior)
python models/word2vec/cbow/train.py --no-wandb --no-hf
```

## Support

For issues with optional dependencies:

1. Check the troubleshooting section above
2. Run `python scripts/check_deps.py` for diagnostics
3. Try training with `--no-wandb --no-hf` to isolate issues
4. Check the respective documentation:
   - [wandb docs](https://docs.wandb.ai/)
   - [Hugging Face docs](https://huggingface.co/docs) 