import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, original_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(original_dim, hidden_dim)
        self.W_k = nn.Linear(original_dim, hidden_dim)
        self.W_v = nn.Linear(original_dim, hidden_dim)
        self.dropout = dropout

    def mask(self, score):
        mask = torch.tril(torch.ones(score.shape)).type(torch.uint8)
        score = score.masked_fill(mask == 0, -1 * torch.inf)
        return score 
    
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        score = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        score = self.mask(score)
        score = F.softmax(score, dim=-1)
        score = F.dropout(score, p=self.dropout)
        return torch.bmm(score, v)

class FeedForward(nn.Module):

    def __init__(self, hidden_dim, ff_dim, dropout):
        super().__init__()
        self.w_1 = nn.Linear(hidden_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = dropout

    def forward(self, x):
        x = self.w_1(x)
        x = self.gelu(x)
        x = self.w_2(x) 
        x = F.dropout(x, p=self.dropout)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_dim, ffn_dim, dropout):
        assert hidden_dim % n_heads == 0
        super().__init__()
        self.layers = nn.ModuleList([
            Attention(original_dim=hidden_dim, hidden_dim=hidden_dim // n_heads, dropout=dropout)
            for _ in range(n_heads)
        ])
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def attn(self, x):
        x = torch.cat([layer(x) for layer in self.layers], dim=-1)
        x = F.dropout(x, p=self.dropout)
        return x

    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
        

class Transformer(nn.Module):
    def __init__(self, n_heads, n_layers, hidden_dim, vocab_size, ffn_dim, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            MultiHeadAttention(n_heads, hidden_dim, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(hidden_dim)
        self.w = nn.Linear(hidden_dim, vocab_size)
        self.dropout = dropout

        # Weight tying
        self.embedding.weight = self.w.weight

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
        x = F.dropout(x, p=self.dropout)

        for layer in self.layers:
            x = layer(x)

        x = self.ln(x)
        x = self.w(x) 
        return x 
    



