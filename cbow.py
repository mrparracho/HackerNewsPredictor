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

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)  # Should show: tensor([1.], device='mps:0')
else:
    print("MPS device not found.")

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

CONTEXT_SIZE = 4  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 128
EPOCHS = 2

raw_text = """We are about to study the idea of a computational process
Computational processes are abstract beings that inhabit computers
As they evolve processes manipulate other abstract things called data
The evolution of a process is directed by a pattern of rules
called a program People create programs to direct processes In effect
we conjure the spirits of the computer with our spells""".lower().split()


# --- Configuration for file loading ---
# TEXT_FILE_PATH = os.path.join('data/', 'text8') # Path to your text file
# print(TEXT_FILE_PATH)
# -----------------------------------



# --- Read raw text from file ---
# raw_text = []
# try:
#     with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
#         # Read the entire file content
#         file_content = f.read()
#         # Split into words, similar to how you did with hardcoded text
#         # You might want more sophisticated tokenization here for real applications
#         raw_text = file_content.lower().split() # Convert to lowercase for consistency
# except FileNotFoundError:
#     print(f"Error: The file '{TEXT_FILE_PATH}' was not found.")
#     print("Please make sure you have created the 'data' folder and 'my_corpus.txt' inside it.")
#     exit() # Exit the script if the file isn't found
# -----------------------------


# By deriving a set from `raw_text`, we deduplicate the array
# vocab = set(raw_text)
# vocab_size = len(vocab)

# word_to_ix = {word:ix for ix, word in enumerate(vocab)}
# ix_to_word = {ix:word for ix, word in enumerate(vocab)}

# load vocab
with open('word_to_lemma_index.json', 'r') as f:
    word_to_ix = json.load(f)
ix_to_word = {ix:word for word, ix in word_to_ix.items()}

vocab = set(word_to_ix.keys()) # all unique words in json
vocab_size = len(set(word_to_ix.values()))


data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
    # print(data)

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        hidden_dim = 128
        #out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation_function1 = nn.ReLU()

        #out: 1 x hidden_dim
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim = -1)
        

    def forward(self, inputs):
        embeds = torch.mean(self.embeddings(inputs), dim=0).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word).view(1,-1)

def load_best_model(model_path, vocab_size, embedding_dim):
    """
    Load the best saved model from checkpoint
    """
    checkpoint = torch.load(model_path)
    
    # Create model with same architecture
    model = CBOW(vocab_size, embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Best loss: {checkpoint['loss']:.4f}")
    print(f"Best perplexity: {checkpoint['perplexity']:.4f}")
    
    return model, checkpoint

# Check for MPS availability
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")

# Move model to device
model = CBOW(vocab_size, EMBEDDING_DIM)
# model.to(device) # uncomment this to use MPS
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Best model tracking
best_loss = float('inf')
best_model_path = 'best_cbow_model.pth'
model_info_path = 'model_info.json'

#TRAINING
for epoch in range(EPOCHS):
    total_loss = 0
    num_samples = len(data)
    
    for context, target in data:
        context_vector = make_context_vector(context, word_to_ix)  

        log_probs = model(context_vector)

        total_loss += loss_function(log_probs, torch.tensor([word_to_ix[target]]))

    # Calculate metrics
    avg_loss = total_loss / num_samples
    perplexity = torch.exp(avg_loss)
    
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
            'embedding_dim': EMBEDDING_DIM
        }
        
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    # Print epoch metrics
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"  Total Loss: {total_loss:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Samples Processed: {num_samples}")
    if is_best:
        print(f"  *** NEW BEST MODEL SAVED! ***")
    print("-" * 40)

    #optimize at the end of each epoch
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

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

#TESTING
context = ['people','create','to', 'direct']
context_vector = make_context_vector(context, word_to_ix)
a = model(context_vector)

#Print result
print(f'Raw text: {" ".join(raw_text)}\n')
print(f'Context: {context}\n')
print(f'Prediction: {ix_to_word[torch.argmax(a[0]).item()]}')

# print("Shape of the full embedding matrix:", model.embeddings.weight.shape)
# print("First 5 rows of the full embedding matrix (representing the first 5 words in your vocab):")
# print(model.embeddings.weight[:5].detach().numpy())

print("\n" + "="*30 + "\n") # Separator for clarity

print(data)