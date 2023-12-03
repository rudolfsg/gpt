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

    def mask(self, score):
        mask = torch.tril(torch.ones(score.shape)).type(torch.uint8)
        score = score.masked_fill(mask == 0, -1e9)
        return score 
    
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        score = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        score = self.mask(score)
        score = F.softmax(score, dim=-1)
        return torch.bmm(score, v)

class FeedForward(nn.Module):

    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.w_1 = nn.Linear(hidden_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x):
        return self.w_2(self.w_1(x).relu())

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_dim, ffn_dim):
        assert hidden_dim % n_heads == 0
        super().__init__()
        self.layers = nn.ModuleList([
            Attention(original_dim=hidden_dim, hidden_dim=hidden_dim // n_heads)
            for _ in range(n_heads)
        ])
        self.ffn = FeedForward(hidden_dim, ffn_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    
    def forward(self, x):
        y = torch.cat([layer(x) for layer in self.layers], dim=-1)
        x = self.ln1(x + y)
        y = self.ffn(x)
        x = self.ln2(x + y)
        return x
        

class Transformer(nn.Module):
    def __init__(self, n_heads, n_layers, hidden_dim, vocab_size, ffn_dim):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            MultiHeadAttention(n_heads, hidden_dim, ffn_dim=ffn_dim)
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
        x = self.embedding(x) * (self.hidden_dim ** 0.5)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)
 
        x = self.w(x) 
        x = F.softmax(x, dim=-1)
        return x 
    



