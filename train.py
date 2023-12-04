import torch
import torch.nn as nn
from model import Transformer
from dataset import TextDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import time 

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# TODO: amp autocast context

max_seq_len = 256
batch_size = 64
vocab_size = 256

n_heads = 8
n_layers = 8
hidden_dim = 256
ffn_dim = 4*hidden_dim

# max_seq_len = 32
# batch_size = 64
# vocab_size = 256

# n_heads = 4
# n_layers = 4
# hidden_dim = 64
# ffn_dim = 4*hidden_dim

epochs = 50
eval_iters = 100
grad_clip = 1.0

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", device)
torch.set_default_device(device)

data = TextDataset(max_seq_len=max_seq_len)
test_size = 0.1
train, test = torch.utils.data.random_split(data, [1 - test_size, test_size], generator=torch.Generator(device=device))

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))


model = Transformer(n_heads=n_heads, n_layers=n_layers, hidden_dim=hidden_dim, vocab_size=vocab_size, ffn_dim=ffn_dim)
model = torch.compile(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99)
)

# TODO: nanogpt only scales float16, not bfloat?
scaler = torch.cuda.amp.GradScaler(enabled=True)

def calc_loss(logits, x, y):
        target = nn.functional.one_hot(torch.cat([x[:, 1:], y[:, None]], dim=-1), num_classes=vocab_size).float()
        loss = F.cross_entropy(logits.view(batch_size * max_seq_len, -1), target.view(batch_size * max_seq_len, -1))
        return loss

i = 0
t0 = time.time()
for epoch in range(epochs):
    itr = tqdm(train_loader)
    # rolling_loss = torch.zeros(len(train_loader) // batch_size + 1)
    for (x, y) in itr:

        optimizer.zero_grad()

        logits = model(x)
        loss = calc_loss(logits, x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        itr.set_description(f"Loss={loss.item():.4f} Epoch={epoch}")

        if i % eval_iters == 0:
            model.eval()
            test_loss = torch.zeros(len(test_loader))
            with torch.no_grad():
                for k, (x_test, y_test) in enumerate(test_loader):
                    logits = model(x_test)
                    test_loss[k] = calc_loss(logits, x_test, y_test).item()
            print(f"Test loss={test_loss.mean().item():.4f}")

        i += 1
print(f"Finished in {(time.time() - t0)/ 60 : .1f} min")
# torch.save(model.state_dict(), "model.pt")
torch.save(model, "model.pt")

