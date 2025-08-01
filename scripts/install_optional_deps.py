#!/usr/bin/env python3
"""
Script to install optional dependencies for Hugging Face and Weights & Biases.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main function to install optional dependencies."""
    print("🚀 Installing optional dependencies for MLX-Week1")
    print("=" * 60)
    
    # Check if pip is available
    if not check_pip():
        print("❌ pip is not available. Please install pip first.")
        return 1
    
    # Install Weights & Biases
    print("\n📊 Weights & Biases (wandb) for experiment tracking")
    print("-" * 50)
    wandb_success = run_command(
        f"{sys.executable} -m pip install wandb",
        "Installing Weights & Biases"
    )
    
    if wandb_success:
        print("\nTo use wandb:")
        print("1. Create an account at https://wandb.ai")
        print("2. Get your API key from your profile settings")
        print("3. Run: wandb login")
        print("4. Or set environment variable: export WANDB_API_KEY=your_key_here")
    
    # Install Hugging Face libraries
    print("\n🤗 Hugging Face libraries for model sharing")
    print("-" * 50)
    hf_success = run_command(
        f"{sys.executable} -m pip install transformers huggingface_hub datasets",
        "Installing Hugging Face libraries"
    )
    
    if hf_success:
        print("\nTo use Hugging Face Hub:")
        print("1. Create an account at https://huggingface.co")
        print("2. Get your access token from your profile settings")
        print("3. Set environment variable: export HUGGINGFACE_TOKEN=your_token_here")
    
    # Install visualization libraries (optional)
    print("\n📈 Visualization libraries (optional)")
    print("-" * 50)
    viz_success = run_command(
        f"{sys.executable} -m pip install matplotlib seaborn scikit-learn",
        "Installing visualization libraries"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Installation Summary:")
    print(f"  Weights & Biases: {'✅' if wandb_success else '❌'}")
    print(f"  Hugging Face: {'✅' if hf_success else '❌'}")
    print(f"  Visualization: {'✅' if viz_success else '❌'}")
    
    if wandb_success and hf_success:
        print("\n🎉 All optional dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Set up your API keys (see instructions above)")
        print("2. Run training with optional features:")
        print("   python models/word2vec/cbow/train.py")
        print("   python models/predictor/train.py")
        print("\nTo disable optional features:")
        print("   python models/word2vec/cbow/train.py --no-wandb --no-hf")
        return 0
    else:
        print("\n⚠️  Some dependencies failed to install.")
        print("You can still run training without optional features using --no-wandb --no-hf flags.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 