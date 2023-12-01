import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, original_dim, hidden_dim):
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
        # batch dimensions??
        score = F.softmax(q @ k.T / (self.hidden_dim ** 0.5), dim=-1)
        score = self.mask(score)
        return score @ v



class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_dim):
        assert hidden_dim % n_heads == 0
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
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            MultiHeadAttention(n_heads, hidden_dim)
            for _ in range(n_layers)
        ])

        self.w = nn.Linear(hidden_dim, vocab_size)

    def positional_encoding(self, x):
        pass

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
        
