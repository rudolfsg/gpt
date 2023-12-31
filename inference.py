import torch 
from dataset import TextDataset

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", device)
torch.set_default_device(device)

# load model from pt
model = torch.load("big_model.pt")

### Compiled model state load didn't work for some reason
# from model import Transformer
# max_seq_len = 256
# batch_size = 64
# vocab_size = 256

# n_heads = 8
# n_layers = 8
# hidden_dim = 256
# ffn_dim = 4*hidden_dim

# epochs = 50
# learning_rate = 1e-3
# dropout = 0.2
# max_iters = 5000
# eval_iters = 200
# grad_clip = 1.0
# model = Transformer(n_heads=n_heads, n_layers=n_layers, hidden_dim=hidden_dim, vocab_size=vocab_size, ffn_dim=ffn_dim, dropout=dropout)
# model.load_state_dict(torch.load("model_state.pt"), strict=False)


model.eval()
data = TextDataset(max_seq_len=32)

text = "I love"
ids = data.text_to_ids(text)
ids = torch.tensor(ids).reshape(1, -1)
print(ids)
# Simple greedy sampling 
with torch.no_grad():
    for _ in range(200):
        logits = model(ids)
        ids = torch.cat([ids, logits[-1, -1].argmax().reshape(1, 1)], dim=-1)

print(data.ids_to_text(ids.view(-1)))
