import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, original_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(original_dim, hidden_dim)
        self.W_k = nn.Linear(original_dim, hidden_dim)
        self.W_v = nn.Linear(original_dim, hidden_dim)

    def mask(self, x):
        pass
    
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # softmax dim?
        score = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5), dim=-1)
        score = self.mask(score)
        return score @ v



class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_dim):
        assert hidden_dim % n_heads == 0
        super().__init__()
        self.layers = nn.ModuleList([
            Attention(original_dim=hidden_dim, hidden_dim=hidden_dim // n_heads)
            for _ in range(n_heads)
        ])
    
    def forward(self, x):
        return torch.cat([layer(x) for layer in self.layers], dim=-1)
        

class Transformer(nn.Module):
    def __init__(self, n_heads, n_layers, hidden_dim, vocab_size) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            MultiHeadAttention(n_heads, hidden_dim)
            for _ in range(n_layers)
        ])

        self.w = nn.Linear(hidden_dim, vocab_size)

    def positional_encoding(self, x):
        pos = torch.arange(0, x.shape[1]) / 10000
        power = 2 * torch.arange(0, self.hidden_dim) / self.hidden_dim
        pos = pos[:, None] ** power[None, :]
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        return x + pos[None, :, :]

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            y = layer(x)
            x = nn.LayerNorm(x + y)

            y = nn.FFN(x)  # TODO
            x = nn.LayerNorm(x + y)
        x = self.w(x)  # TODO: apply to the last token of the squence
        x = F.softmax(x, dim=-1)
        return x 
        
import json
### Load vocab
with open("vocab.json", "r") as f:
    vocab = json.load(f)
with open("data.json", "r") as f:
    data = json.load(f)

max_seq_len = 5
batch_size = 2
vocab_size= 256

x = []
for i in range(batch_size):
    x.append(data[i * max_seq_len: (i + 1) * max_seq_len])
x = torch.tensor(x)

model = Transformer(n_heads=4, n_layers=4, hidden_dim=16, vocab_size=vocab_size)
model(x)



