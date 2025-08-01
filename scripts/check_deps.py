#!/usr/bin/env python3
"""
Script to check the status of optional dependencies for Hugging Face and Weights & Biases.
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.optional_deps import check_optional_deps, get_training_config

def main():
    """Main function to check dependencies."""
    print("🔍 Checking optional dependencies for MLX-Week1")
    print("=" * 60)
    
    # Check dependencies
    deps_status = check_optional_deps()
    
    # Get training config
    config = get_training_config()
    
    print("\n📋 Configuration:")
    print(f"  WANDB_PROJECT: {config['wandb_project']}")
    print(f"  HF_REPO_PREFIX: {config['hf_repo_prefix']}")
    
    print("\n🔑 Environment Variables:")
    print(f"  HUGGINGFACE_TOKEN: {'✅ Set' if config['huggingface_token'] else '❌ Not set'}")
    print(f"  WANDB_API_KEY: {'✅ Set' if os.environ.get('WANDB_API_KEY') else '❌ Not set'}")
    
    print("\n🚀 Usage Examples:")
    print("\nWith all features enabled:")
    print("  python models/word2vec/cbow/train.py")
    print("  python models/word2vec/skipgram/train.py")
    print("  python models/predictor/train.py")
    
    print("\nWith specific features disabled:")
    print("  python models/word2vec/cbow/train.py --no-wandb")
    print("  python models/word2vec/cbow/train.py --no-hf")
    print("  python models/word2vec/cbow/train.py --no-wandb --no-hf")
    
    print("\n📚 Setup Instructions:")
    if not deps_status['wandb']:
        print("\nTo enable Weights & Biases:")
        print("  1. Install: pip install wandb")
        print("  2. Create account: https://wandb.ai")
        print("  3. Login: wandb login")
    
    if not deps_status['huggingface']:
        print("\nTo enable Hugging Face:")
        print("  1. Install: pip install transformers huggingface_hub")
        print("  2. Create account: https://huggingface.co")
        print("  3. Set token: export HUGGINGFACE_TOKEN=your_token_here")
    
    if not deps_status['huggingface_token'] and deps_status['huggingface']:
        print("\nTo enable Hugging Face Hub model sharing:")
        print("  1. Get token from: https://huggingface.co/settings/tokens")
        print("  2. Set environment variable: export HUGGINGFACE_TOKEN=your_token_here")
    
    print("\n" + "=" * 60)
    print("✅ Dependency check completed!")

if __name__ == "__main__":
    main() 