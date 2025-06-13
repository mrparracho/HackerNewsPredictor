import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from transformers import PreTrainedModel, PretrainedConfig
import os
from huggingface_hub import HfApi, create_repo

class CBOWConfig(PretrainedConfig):
    model_type = "cbow"
    
    def __init__(
        self,
        vocab_size=10000,
        embed_size=100,
        window_size=4,
        num_negatives=5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.window_size = window_size
        self.num_negatives = num_negatives

class CBOWNegativeSamplingDataset(Dataset):
    def __init__(self, data, window_size, num_negatives):
        self.samples = []
        self.vocab_size = len(set(data))
        self.num_negatives = num_negatives
        for i in range(window_size, len(data) - window_size):
            context = data[i - window_size:i] + data[i + 1:i + window_size + 1]
            target = data[i]
            self.samples.append((context, target))

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = random.randint(0, self.vocab_size - 1)
            if neg != target:
                negatives.append(neg)
        return torch.tensor(context), torch.tensor(target), torch.tensor(negatives)

    def __len__(self):
        return len(self.samples)

class CBOWNS(PreTrainedModel):
    config_class = CBOWConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.in_embed = nn.Embedding(config.vocab_size, config.embed_size)
        self.out_embed = nn.Embedding(config.vocab_size, config.embed_size)

    def forward(self, context_idxs, target_idx, negative_idxs):
        v_c = self.in_embed(context_idxs)              # (B, C, E)
        v_c = torch.mean(v_c, dim=1)                   # (B, E)
        u_o = self.out_embed(target_idx)               # (B, E)
        u_k = self.out_embed(negative_idxs)            # (B, N, E)

        # Positive loss
        pos = torch.sum(v_c * u_o, dim=1)
        pos_loss = -F.logsigmoid(pos)

        # Negative loss
        neg = torch.bmm(u_k, v_c.unsqueeze(2)).squeeze()  # (B, N)
        neg_loss = -torch.sum(F.logsigmoid(-neg), dim=1)   # Sum over negative samples

        return (pos_loss + neg_loss).mean()

    def save_pretrained(self, save_directory, **kwargs):
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save config
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")))
        return model

def push_to_hub(model, repo_name):
    """Push the model to Hugging Face Hub"""
    api = HfApi()
    
    # Check for token
    if 'HUGGINGFACE_TOKEN' not in os.environ:
        raise EnvironmentError(
            "HUGGINGFACE_TOKEN environment variable not set. "
            "Please set it using: export HUGGINGFACE_TOKEN=your_token_here"
        )
    
    token = os.environ['HUGGINGFACE_TOKEN']
    
    # Create temp directory
    temp_dir = "temp_model"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Create repo if it doesn't exist
        try:
            create_repo(repo_name, token=token)
        except Exception as e:
            if "already exists" not in str(e):
                raise Exception(f"Failed to create repo: {e}")
            print(f"Repo {repo_name} already exists, proceeding with upload...")
        
        # Save model locally first
        model.save_pretrained(temp_dir)
        
        # Upload to hub
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            repo_type="model",
            token=token
        )
        
        print(f"Successfully pushed model to {repo_name}")
        
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        raise
    finally:
        # Clean up
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir) 