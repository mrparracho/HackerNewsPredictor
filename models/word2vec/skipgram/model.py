import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class SkipGramNegativeSamplingDataset(Dataset):
    def __init__(self, data, window_size, num_negatives):
        self.samples = []
        self.vocab_size = len(set(data))
        self.num_negatives = num_negatives
        
        for i in range(window_size, len(data) - window_size):
            center_word = data[i]
            context_words = data[i - window_size:i] + data[i + 1:i + window_size + 1]
            for context_word in context_words:
                self.samples.append((center_word, context_word))

    def __getitem__(self, idx):
        center_word, context_word = self.samples[idx]
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = random.randint(0, self.vocab_size - 1)
            if neg != context_word:
                negatives.append(neg)
        return torch.tensor(center_word), torch.tensor(context_word), torch.tensor(negatives)

    def __len__(self):
        return len(self.samples)

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, center_idxs, context_idxs, negative_idxs):
        v_c = self.in_embed(center_idxs)              # (B, E)
        u_o = self.out_embed(context_idxs)            # (B, E)
        u_k = self.out_embed(negative_idxs)           # (B, N, E)

        pos = torch.sum(v_c * u_o, dim=1)
        pos_loss = -F.logsigmoid(pos)

        neg = torch.bmm(u_k, v_c.unsqueeze(2)).squeeze()
        neg_loss = -torch.sum(F.logsigmoid(-neg), dim=1)

        return (pos_loss + neg_loss).mean() 