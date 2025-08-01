"""
Optional dependencies utility module for Hugging Face and Weights & Biases integration.
This module provides safe imports and fallback functionality when these dependencies are not installed.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import wandb
    WANDB_AVAILABLE = True
    logger.info("Weights & Biases (wandb) is available")
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases (wandb) not available. Install with: pip install wandb")

try:
    from transformers import PreTrainedModel, PretrainedConfig
    from huggingface_hub import HfApi, create_repo, upload_folder
    HF_AVAILABLE = True
    logger.info("Hugging Face libraries are available")
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face libraries not available. Install with: pip install transformers huggingface_hub")


class WandbLogger:
    """Safe wrapper for wandb logging with fallback to local logging."""
    
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any], enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.run = None
        
        if self.enabled:
            try:
                self.run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config
                )
                logger.info(f"Initialized wandb run: {project_name}/{run_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.enabled = False
        else:
            logger.info(f"wandb logging disabled. Run: {project_name}/{run_name}")
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log data to wandb or print to console."""
        if self.enabled and self.run:
            try:
                if step is not None:
                    wandb.log(data, step=step)
                else:
                    wandb.log(data)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
        
        # Always log to console for debugging
        if step:
            logger.info(f"Step {step}: {data}")
        else:
            logger.info(f"Log: {data}")
    
    def finish(self):
        """Finish the wandb run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
                logger.info("wandb run finished")
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")


class HuggingFaceHub:
    """Safe wrapper for Hugging Face Hub operations."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get('HUGGINGFACE_TOKEN')
        self.available = HF_AVAILABLE
        
        if not self.available:
            logger.warning("Hugging Face Hub not available")
        elif not self.token:
            logger.warning("HUGGINGFACE_TOKEN not set. Set it using: export HUGGINGFACE_TOKEN=your_token")
    
    def push_model(self, model, repo_name: str, model_config: Dict[str, Any]) -> bool:
        """Push model to Hugging Face Hub."""
        if not self.available:
            logger.warning("Cannot push model: Hugging Face libraries not available")
            return False
        
        if not self.token:
            logger.warning("Cannot push model: HUGGINGFACE_TOKEN not set")
            return False
        
        try:
            # Create temporary directory
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model and config
                model_path = os.path.join(temp_dir, "model")
                os.makedirs(model_path, exist_ok=True)
                
                # Save model weights
                if hasattr(model, 'save_pretrained'):
                    model.save_pretrained(model_path)
                else:
                    import torch
                    torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
                
                # Save config
                config_path = os.path.join(model_path, "config.json")
                with open(config_path, 'w') as f:
                    json.dump(model_config, f, indent=2)
                
                # Create README
                readme_path = os.path.join(model_path, "README.md")
                with open(readme_path, 'w') as f:
                    f.write(f"# {repo_name}\n\n")
                    f.write("This model was trained using the MLX-Week1 project.\n\n")
                    f.write("## Model Configuration\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(model_config, indent=2))
                    f.write("\n```\n")
                
                # Create repo if it doesn't exist
                try:
                    create_repo(repo_name, token=self.token, exist_ok=True)
                except Exception as e:
                    if "already exists" not in str(e):
                        raise
                
                # Upload to hub
                api = HfApi()
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=repo_name,
                    repo_type="model",
                    token=self.token
                )
                
                logger.info(f"Successfully pushed model to {repo_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to push model to Hugging Face Hub: {e}")
            return False
    
    def download_model(self, repo_name: str, local_path: str) -> bool:
        """Download model from Hugging Face Hub."""
        if not self.available:
            logger.warning("Cannot download model: Hugging Face libraries not available")
            return False
        
        try:
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id=repo_name,
                local_dir=local_path,
                token=self.token
            )
            
            logger.info(f"Successfully downloaded model from {repo_name} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model from Hugging Face Hub: {e}")
            return False


def check_optional_deps():
    """Check which optional dependencies are available."""
    deps_status = {
        'wandb': WANDB_AVAILABLE,
        'huggingface': HF_AVAILABLE,
        'huggingface_token': bool(os.environ.get('HUGGINGFACE_TOKEN'))
    }
    
    logger.info("Optional dependencies status:")
    for dep, available in deps_status.items():
        status = "✅ Available" if available else "❌ Not available"
        logger.info(f"  {dep}: {status}")
    
    return deps_status


def get_training_config(use_wandb: bool = True, use_hf: bool = True) -> Dict[str, Any]:
    """Get training configuration with optional features."""
    config = {
        'use_wandb': use_wandb and WANDB_AVAILABLE,
        'use_huggingface': use_hf and HF_AVAILABLE,
        'huggingface_token': os.environ.get('HUGGINGFACE_TOKEN'),
        'wandb_project': os.environ.get('WANDB_PROJECT', 'mlx-week1'),
        'hf_repo_prefix': os.environ.get('HF_REPO_PREFIX', 'roshbeed')
    }
    
    return config 