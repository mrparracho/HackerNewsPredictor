import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from model import SkipGramNS, SkipGramNegativeSamplingDataset
from utils.optional_deps import WandbLogger, HuggingFaceHub, check_optional_deps

def train(use_wandb=True, use_hf=True):
    # Load hyperparameters from YAML file
    with open('models/word2vec/skipgram/skipgram_ns.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = 'models/word2vec/skipgram/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check optional dependencies
    deps_status = check_optional_deps()
    
    # Initialize logging
    logger = WandbLogger(
        project_name=config.get('PROJECT_NAME', 'word2vec-skipgram-ns'),
        run_name=config.get('RUN_NAME', 'skipgram-run'),
        config=config,
        enabled=use_wandb and deps_status['wandb']
    )
    
    # Initialize Hugging Face Hub
    hf_hub = HuggingFaceHub() if use_hf else None

    # Prepare data
    vocab = [
        'hi', 'my', 'name', 'is', 'kernel',
        'and', 'i', 'like', 'to', 'code',
        'python', 'machine', 'learning', 'is', 'fun'
    ]
    word_to_ix = {word: i for i, word in enumerate(set(vocab))}
    encoded = [word_to_ix[w] for w in vocab]
    vocab_size = len(word_to_ix)

    # Create dataset and dataloader
    dataset = SkipGramNegativeSamplingDataset(
        encoded, 
        window_size=config['WINDOW_SIZE'], 
        num_negatives=config['NUM_NEGATIVES']
    )
    loader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True)

    # Initialize model and optimizer
    model = SkipGramNS(vocab_size, embed_size=config['EMBEDDING_SIZE'])
    opt = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    # Training loop
    best_loss = float('inf')
    for epoch in range(config['NUM_EPOCHS']):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Época {epoch+1}", unit="batch")
        for center, context, negatives in pbar:
            loss = model(center, context, negatives)
            opt.zero_grad()
            loss.backward()
            opt.step()
                    total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(loader)
    logger.log({"epoch": epoch + 1, "loss": avg_loss})

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': best_loss,
            }, os.path.join(checkpoint_dir, 'skipgram_model.pt'))
            
            # Save embeddings
            torch.save(model.in_embed.weight.data, os.path.join(checkpoint_dir, 'skipgram_input_embeddings.pt'))
            torch.save(model.out_embed.weight.data, os.path.join(checkpoint_dir, 'skipgram_output_embeddings.pt'))

            # Save as readable text file
            with open(os.path.join(checkpoint_dir, 'skipgram_input_embeddings.txt'), "w") as f:
                for idx, word in enumerate(word_to_ix):
                    vector = model.in_embed.weight[idx].tolist()
                    vector_str = " ".join(f"{v:.4f}" for v in vector)
                    f.write(f"{word} {vector_str}\n")

    # Push to Hugging Face Hub if enabled
    if hf_hub and hf_hub.available and hf_hub.token:
        try:
            print("\nPushing model to Hugging Face Hub...")
            repo_name = f"{os.environ.get('HF_REPO_PREFIX', 'roshbeed')}/skipgram-model-best"
            
            model_config = {
                "model_type": "skipgram",
                "vocab_size": vocab_size,
                "embed_size": config.get("EMBEDDING_SIZE", 256),
                "window_size": config.get("WINDOW_SIZE", 4),
                "num_negatives": config.get("NUM_NEGATIVES", 15),
                "training_config": config
            }
            
            success = hf_hub.push_model(model, repo_name, model_config)
            if success:
                print(f"Model pushed to {repo_name}")
            else:
                print("Failed to push model to Hugging Face Hub")
        except Exception as e:
            print(f"Warning: Failed to push model to Hugging Face Hub: {e}")

    logger.finish()
    return model, word_to_ix

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SkipGram model")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
        default=False,
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Disable Hugging Face Hub integration",
        default=False,
    )
    args = parser.parse_args()

    # Check optional dependencies
    deps_status = check_optional_deps()
    
    # Set flags based on arguments and availability
    use_wandb = not args.no_wandb and deps_status['wandb']
    use_hf = not args.no_hf and deps_status['huggingface']
    
    if not deps_status['wandb'] and not args.no_wandb:
        print("\nWarning: Weights & Biases (wandb) not available.")
        print("Install with: pip install wandb")
        print("Or use --no-wandb to disable wandb logging.\n")
    
    if not deps_status['huggingface'] and not args.no_hf:
        print("\nWarning: Hugging Face libraries not available.")
        print("Install with: pip install transformers huggingface_hub")
        print("Or use --no-hf to disable Hugging Face integration.\n")
    
    train(use_wandb=use_wandb, use_hf=use_hf) 