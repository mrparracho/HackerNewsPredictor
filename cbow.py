# TODO:
# handle unknown words
# - add a way to save the model
# - add a way to load the model
# - add a way to save the word_to_ix dictionary
# - add a way to load the word_to_ix dictionary
# - add a way to save the raw_text
# - add a way to load the raw_text


import torch
import torch.nn as nn
import os
import json
import gc  # For garbage collection
import wandb
from datetime import datetime

# Check we have the wandb key in our env
if 'WANDB_API_KEY' not in os.environ:
    raise EnvironmentError("Please set the WANDB_API_KEY environment variable in your .env file before running this script.")
# Initialize wandb
wandb.login()  # Ensure to run source.env in terminal before running this script

CONTEXT_SIZE = 4  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 8  # Reduced from 128
EPOCHS = 1
BATCH_SIZE = 1024  # Process data in smaller batches
ACCUMULATION_STEPS = 4  # Accumulate gradients over multiple batches
LEARNING_RATE = 0.001




def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

# --- Configuration for file loading ---
TEXT_FILE_PATH = os.path.join('data/', 'text8') # Path to your text file
print(TEXT_FILE_PATH)
# -----------------------------------

# --- Read raw text from file ---
raw_text = []
try:
    with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
        file_content = f.read()
        raw_text = file_content.lower().split()
except FileNotFoundError:
    print(f"Error: The file '{TEXT_FILE_PATH}' was not found.")
    print("Please make sure you have created the 'data' folder and 'my_corpus.txt' inside it.")
    exit()

# load vocab
with open('text8_word_to_lemma_index.json', 'r') as f:
    word_to_ix = json.load(f)
ix_to_word = {ix:word for word, ix in word_to_ix.items()}

vocab = set(word_to_ix.keys())
vocab_size = len(set(word_to_ix.values()))

data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.activation_function = nn.LogSoftmax(dim = -1)

    def forward(self, inputs):
        embeds = torch.mean(self.embeddings(inputs), dim=0)
        out = self.linear(embeds)
        out = self.activation_function(out)
        return out.view(1, -1)  # Reshape to [1, vocab_size]

    def get_word_emdedding(self, word):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word)

def load_best_model(model_path, vocab_size, embedding_dim):
    """Load the best saved model from checkpoint"""
    checkpoint = torch.load(model_path)
    model = CBOW(vocab_size, embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Best loss: {checkpoint['loss']:.4f}")
    print(f"Best perplexity: {checkpoint['perplexity']:.4f}")
    return model, checkpoint

# Initialize wandb
wandb.init(
    project="mlx-week1",
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "CBOW",
        "dataset": "text8",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "context_size": CONTEXT_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "vocab_size": vocab_size,
        "optimizer": "SGD",
        "loss_function": "NLLLoss",
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
)

# Initialize model and training components
model = CBOW(vocab_size, EMBEDDING_DIM)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Best model tracking
best_loss = float('inf')
best_model_path = 'best_cbow_model.pth'
model_info_path = 'model_info.json'

# Log model architecture
wandb.watch(model, log="all")

# TRAINING
for epoch in range(EPOCHS):
    total_loss = 0
    num_samples = len(data)
    
    # Shuffle data at the start of each epoch
    indices = torch.randperm(len(data))
    
    # Process data in batches
    for batch_start in range(0, len(data), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(data))
        batch_indices = indices[batch_start:batch_end]
        
        batch_loss = 0
        optimizer.zero_grad()
        
        # Process each sample in the batch
        for idx in batch_indices:
            context, target = data[idx]
            context_vector = make_context_vector(context, word_to_ix)
            target_tensor = torch.tensor([word_to_ix[target]], dtype=torch.long)  # Add batch dimension and specify dtype

            log_probs = model(context_vector)
            loss = loss_function(log_probs, target_tensor)
            
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            batch_loss += loss.item() * ACCUMULATION_STEPS
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Clear memory
        gc.collect()
        
        total_loss += batch_loss

    # Calculate metrics
    avg_loss = total_loss / num_samples
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "total_loss": total_loss,
        "average_loss": avg_loss,
        "perplexity": perplexity,
        "samples_processed": num_samples
    })
    
    # Check if this is the best model so far
    is_best = avg_loss < best_loss
    if is_best:
        best_loss = avg_loss
        # Save the best model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'perplexity': perplexity
        }, best_model_path)
        
        # Save model info as JSON
        model_info = {
            'epoch': epoch + 1,
            'loss': float(best_loss),
            'perplexity': float(perplexity),
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'total_epochs': EPOCHS,
            'model_file': best_model_path,
            'max_epochs': EPOCHS,
            'context_size': CONTEXT_SIZE,
            'embedding_dim': EMBEDDING_DIM,
            'batch_size': BATCH_SIZE,
            'accumulation_steps': ACCUMULATION_STEPS
        }
        
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Log best model to wandb
        wandb.save(best_model_path)
        wandb.save(model_info_path)
    
    # Print epoch metrics
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"  Total Loss: {total_loss:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Samples Processed: {num_samples}")
    if is_best:
        print(f"  *** NEW BEST MODEL SAVED! ***")
    print("-" * 40)

# Training completed - print best model summary
print("="*50)
print("TRAINING COMPLETED")
print("="*50)
if os.path.exists(best_model_path):
    print(f"Best model saved to: {best_model_path}")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Model info saved to: {model_info_path}")
    
    # Load the best model for testing
    print("\nLoading best model for testing...")
    model, checkpoint = load_best_model(best_model_path, vocab_size, EMBEDDING_DIM)
else:
    print("No model was saved (this shouldn't happen)")

print("="*50)

# TESTING
context = ['people','create','to', 'direct']
context_vector = make_context_vector(context, word_to_ix)
a = model(context_vector)

# Print result
print(f'Raw text: {" ".join(raw_text)}\n')
print(f'Context: {context}\n')
print(f'Prediction: {ix_to_word[torch.argmax(a[0]).item()]}')

# Log test results to wandb
wandb.log({
    "test_context": context,
    "test_prediction": ix_to_word[torch.argmax(a[0]).item()]
})

# Finish wandb run
wandb.finish()

# print("Shape of the full embedding matrix:", model.embeddings.weight.shape)
# print("First 5 rows of the full embedding matrix (representing the first 5 words in your vocab):")
# print(model.embeddings.weight[:5].detach().numpy())

print("\n" + "="*30 + "\n") # Separator for clarity

print(data)