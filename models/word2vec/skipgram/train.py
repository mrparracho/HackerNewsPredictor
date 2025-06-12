import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import yaml
import os
from model import SkipGramNS, SkipGramNegativeSamplingDataset

def train():
    # Load hyperparameters from YAML file
    with open('models/word2vec/skipgram/skipgram_ns.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = 'models/word2vec/skipgram/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(project=config['PROJECT_NAME'], name=config['RUN_NAME'])

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
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

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

    wandb.finish()
    return model, word_to_ix

if __name__ == "__main__":
    train() 