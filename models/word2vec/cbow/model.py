import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

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

class CBOWNS(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

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