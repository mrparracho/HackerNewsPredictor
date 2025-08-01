import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import json
import argparse
import sys
import itertools

import torch.nn.functional as F
from scipy.stats import spearmanr
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from model import CBOWNS, CBOWNegativeSamplingDataset, CBOWConfig
from utils.optional_deps import WandbLogger, HuggingFaceHub, check_optional_deps

# load YAML config defaults (some keys are lists for sweep)
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# find which keys are lists → sweep axes
def get_sweep_keys(cfg):
    return [k for k, v in cfg.items() if isinstance(v, list)]

# from defaults+axes build each run's flat config
def generate_configs(defaults, sweep_keys):
    if not sweep_keys:
        return [defaults]
    values = [defaults[k] for k in sweep_keys]
    combos = itertools.product(*values)
    runs = []
    for combo in combos:
        c = defaults.copy()
        for k, v in zip(sweep_keys, combo):
            c[k] = v
        runs.append(c)
    return runs

def load_vocabulary():
    """Load the combined vocabulary from our ETL pipeline."""
    with open('data/combined_word_to_index.json', 'r', encoding='utf-8') as f:
        word_to_index = json.load(f)
    
    with open('data/combined_word_to_lemma_index.json', 'r', encoding='utf-8') as f:
        word_to_lemma_index = json.load(f)
    
    return word_to_index, word_to_lemma_index

def get_dummy_vocabulary():
    """Return the dummy vocabulary for testing."""
    vocab = [
        'hi', 'my', 'name', 'is', 'kernel',
        'and', 'i', 'like', 'to', 'code',
        'python', 'machine', 'learning', 'is', 'fun',
        'king', 'queen', 'man', 'woman', 'dog', 'cat'  # Add analogy words
    ]
    # Create unique vocabulary
    unique_vocab = list(set(vocab))
    # Create mappings
    word_to_index = {word: i for i, word in enumerate(unique_vocab)}
    word_to_lemma_index = {word: i for i, word in enumerate(unique_vocab)}  # Dummy lemma mapping
    # Add UNK token with last index
    unk_idx = len(unique_vocab)
    word_to_index['<UNK>'] = unk_idx
    word_to_lemma_index['<UNK>'] = unk_idx
    return word_to_index, word_to_lemma_index

def get_training_data(word_to_index):
    """Get training data from the combined data file."""
    print("\nLoading combined training data...")
    with open(os.path.join('data', 'combined_data.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.lower().split()
    # Use UNK token for unknown words
    encoded = [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in words]
    print(f"Loaded {len(encoded):,} words")
    return encoded


def evaluate_model(model, step, word_to_ix, logger, embedding_layer=None):
    """
    Evaluate the model on the analogy task and log the results.
    """
    # 1. Get embeddings
    if embedding_layer is None:
        embedding_matrix = model.in_embed.weight.detach()
    else:
        embedding_matrix = embedding_layer.weight.detach()

    ix_to_word = {ix: w for w, ix in word_to_ix.items()}
    words = list(word_to_ix.keys())

    # 2. Similarity (Spearman)
    similarity_pairs = [("king", "queen"), ("dog", "cat"), ("king", "dog")]
    human_scores = [0.95, 0.85, 0.2]
    model_scores = []
    for (w1, w2) in similarity_pairs:
        v1, v2 = embedding_matrix[word_to_ix[w1]], embedding_matrix[word_to_ix[w2]]
        sim = F.cosine_similarity(v1, v2, dim=0).item()
        model_scores.append(sim)
    spearman_corr, _ = spearmanr(model_scores, human_scores)

    # 3. Vector analogy
    def predict_analogy(a, b, c):
        va, vb, vc = embedding_matrix[word_to_ix[a]], embedding_matrix[word_to_ix[b]], embedding_matrix[word_to_ix[c]]
        pred_vec = vb - va + vc
        sims = F.cosine_similarity(embedding_matrix, pred_vec.unsqueeze(0)).numpy()
        best_ix = sims.argmax()
        return ix_to_word[best_ix]

    predicted = predict_analogy("king", "man", "woman")
    analogy_correct = int(predicted == "queen")

    # 4. t-SNE visualization (commented out for now)
    # def log_tsne():
    #     vectors = embedding_matrix.cpu().numpy()
    #     reduced = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(vectors)
    #     plt.figure(figsize=(6, 6))
    #     for i, word in enumerate(words):
    #         x, y = reduced[i]
    #         plt.scatter(x, y)
    #         plt.annotate(word, (x, y))
    #     plt.title(f"t-SNE dos Embeddings - Step {step}")
    #     plt.grid(True)
    #     logger.log({f"tsne_step_{step}": wandb.Image(plt)})
    #     plt.close()

    # 5. Logging results
    eval_data = {
        "eval/similarity_spearman": spearman_corr,
        "eval/analogy_correct": analogy_correct,
        "eval/analogy_accuracy": float(analogy_correct),
        "eval/analogy_prediction": predicted,
        "eval/analogy_expected": "queen"
    }
    
    logger.log(eval_data, step=step)
    
    # log_tsne()

def train_run(dummy=False, use_wandb=True, use_hf=True):
    """
    Train the CBOW model.
    """
    # Create checkpoints directory if it doesn't exist
    base_ckpt_dir = "models/word2vec/cbow/checkpoints"
    run_ckpt_dir  = os.path.join(base_ckpt_dir, "cur_sweep")

    # ensure both checkpoints/ and checkpoints/cur_sweep/ exist
    os.makedirs(run_ckpt_dir, exist_ok=True)

    # now clear out any leftover files from previous run
    for fname in os.listdir(run_ckpt_dir):
        path = os.path.join(run_ckpt_dir, fname)
        if os.path.isfile(path):
            os.remove(path)

    best_loss_path = os.path.join(base_ckpt_dir, "best_loss.txt")

    # Check optional dependencies
    deps_status = check_optional_deps()
    
    # Initialize logging
    if use_wandb and deps_status['wandb']:
        # Use wandb config if available
        try:
            import wandb
            cfg = wandb.config
            logger = WandbLogger(
                project_name=cfg.get("PROJECT_NAME", "word2vec-cbow-ns"),
                run_name=cfg.get("RUN_NAME", "cbow-run"),
                config=cfg,
                enabled=use_wandb
            )
        except (ImportError, AttributeError):
            # Fallback to default config
            cfg = load_yaml_config("models/word2vec/cbow/cbow_ns.yml")
            logger = WandbLogger(
                project_name=cfg.get("PROJECT_NAME", "word2vec-cbow-ns"),
                run_name=cfg.get("RUN_NAME", "cbow-run"),
                config=cfg,
                enabled=False
            )
    else:
        # Use default config without wandb
        cfg = load_yaml_config("models/word2vec/cbow/cbow_ns.yml")
        logger = WandbLogger(
            project_name=cfg.get("PROJECT_NAME", "word2vec-cbow-ns"),
            run_name=cfg.get("RUN_NAME", "cbow-run"),
            config=cfg,
            enabled=False
        )
    
    # Initialize Hugging Face Hub
    hf_hub = HuggingFaceHub() if use_hf else None

    if dummy:
        print("Using dummy vocabulary for testing...")
        word_to_index, word_to_lemma_index = get_dummy_vocabulary()
        # Use the dummy example
        example_text = "python machine learning is fun and i like to code python is a great language for machine learning and coding is fun with python king queen man woman king queen man woman"
        words = example_text.lower().split()
        encoded = [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in words]
        print(encoded)
    else:
        print("Loading vocabulary and training data from ETL pipeline...")
        word_to_index, word_to_lemma_index = load_vocabulary()
        # Add UNK token with last index if not present
        if '<UNK>' not in word_to_lemma_index:
            unk_idx = len(word_to_lemma_index)
            word_to_lemma_index['<UNK>'] = unk_idx
            word_to_index['<UNK>'] = unk_idx
        encoded = get_training_data(word_to_index)
    
    vocab_size = len(word_to_lemma_index)
    print(f"Vocabulary size: {vocab_size}")

    # Create dataset and dataloader
    dataset = CBOWNegativeSamplingDataset(
        encoded, 
        window_size=cfg.WINDOW_SIZE, 
        num_negatives=cfg.NUM_NEGATIVES,
    )
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # Initialize model and optimizer
    model_config = CBOWConfig(
        vocab_size=vocab_size,
        embed_size=cfg.EMBEDDING_SIZE,
        window_size=cfg.WINDOW_SIZE,
        num_negatives=cfg.NUM_NEGATIVES
    )
    model = CBOWNS(model_config)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # Training loop
    best_loss: float = float("inf") # track best loss across epochs
    best_model = None
    for epoch in range(cfg.NUM_EPOCHS):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, (context, target, negatives) in enumerate(pbar):
            loss = model(context, target, negatives)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

            # Log batch loss
            logger.log({
                "epoch": epoch + 1,
                "batch": batch_idx,
                "batch_loss": loss.item(),
            })

            # Evaluate periodically
            if batch_idx % 100 == 0:
                evaluate_model(model, batch_idx + epoch * len(loader), word_to_index, logger)

            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        logger.log({
            "epoch": epoch + 1,
            "epoch_loss": avg_loss,
        })

        # if this epoch is the best so far, save into the run's subdir
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model.state_dict()

            # model checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": best_loss,
                    "word_to_index": word_to_index,
                    "word_to_lemma_index": word_to_lemma_index,
                    "config": model_config.to_dict(),
                },
                os.path.join(run_ckpt_dir, "cbow_model.pt"),
            )
            # embeddings
            torch.save(
                model.in_embed.weight.data,
                os.path.join(run_ckpt_dir, "cbow_input_embeddings.pt"),
            )
            torch.save(
                model.out_embed.weight.data,
                os.path.join(run_ckpt_dir, "cbow_output_embeddings.pt"),
            )
            # readable embeddings
            with open(os.path.join(run_ckpt_dir, "cbow_input_embeddings.txt"), "w", encoding="utf-8") as f:
                for word, idx in word_to_index.items():
                    vector = model.in_embed.weight[idx].tolist()
                    vector_str = " ".join(f"{v:.4f}" for v in vector)
                    f.write(f"{word} {vector_str}\n")

    # after all epochs: record this run's best loss
    run_loss_file = os.path.join(run_ckpt_dir, "run_best_loss.txt")
    with open(run_loss_file, "w", encoding="utf-8") as f:
        f.write(f"{best_loss}")

    # compare against the global best and promote if better
    try:
        with open(best_loss_path, "r", encoding="utf-8") as f:
            global_best = float(f.read())
    except FileNotFoundError:
        global_best = float("inf")

    if best_loss < global_best:
        import shutil

        for fname in [
            "cbow_model.pt",
            "cbow_input_embeddings.pt",
            "cbow_output_embeddings.pt",
            "cbow_input_embeddings.txt",
        ]:
            shutil.copy(
                os.path.join(run_ckpt_dir, fname),
                os.path.join(base_ckpt_dir, fname),
            )
        # overwrite the global best‐loss record
        with open(best_loss_path, "w", encoding="utf-8") as f:
            f.write(f"{best_loss}")

        # Push the best model to Hugging Face Hub
        if hf_hub and hf_hub.available and hf_hub.token:
            try:
                print("\nPushing best model to Hugging Face Hub...")
                model.load_state_dict(best_model)
                
                # Create repo name
                if logger.enabled and hasattr(logger, 'run') and logger.run:
                    repo_name = f"{os.environ.get('HF_REPO_PREFIX', 'roshbeed')}/cbow-model-{logger.run.name}"
                else:
                    repo_name = f"{os.environ.get('HF_REPO_PREFIX', 'roshbeed')}/cbow-model-best"
                
                # Create model config for HF Hub
                model_config = {
                    "model_type": "cbow",
                    "vocab_size": vocab_size,
                    "embed_size": cfg.get("EMBEDDING_SIZE", 256),
                    "window_size": cfg.get("WINDOW_SIZE", 4),
                    "num_negatives": cfg.get("NUM_NEGATIVES", 15),
                    "training_config": cfg
                }
                
                success = hf_hub.push_model(model, repo_name, model_config)
                if success:
                    print(f"Model pushed to {repo_name}")
                else:
                    print("Failed to push model to Hugging Face Hub")
            except Exception as e:
                print(f"\nWarning: Failed to push model to Hugging Face Hub: {e}")
                print("The model was still saved locally in the checkpoints directory.")

    logger.finish()
    return model, word_to_index

def main():
    parser = argparse.ArgumentParser(
        description="Train CBOW or run embedded hyper-parameter sweep"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy vocabulary for testing",
        default=False,
    )
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
    
    if not deps_status['huggingface_token'] and use_hf:
        print("\nWarning: HUGGINGFACE_TOKEN environment variable not set.")
        print("The model will be saved locally but won't be pushed to Hugging Face Hub.")
        print("To enable pushing to Hugging Face Hub, set your token using:")
        print("export HUGGINGFACE_TOKEN=your_token_here\n")

    defaults = load_yaml_config("models/word2vec/cbow/cbow_ns.yml")
    sweep_keys = get_sweep_keys(defaults)
    all_cfgs = generate_configs(defaults, sweep_keys)

    for cfg in all_cfgs:
        if sweep_keys:
            suffix = "_".join(f"{k}{cfg[k]}" for k in sweep_keys)
            run_name = f"{defaults['RUN_NAME']}_{suffix}"
        else:
            run_name = defaults["RUN_NAME"]

        # Initialize wandb if enabled
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg["PROJECT_NAME"],
                    name=run_name,
                    config=cfg,
                )
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
                use_wandb = False
        
        train_run(dummy=args.dummy, use_wandb=use_wandb, use_hf=use_hf)

if __name__ == "__main__":
    main()
