import torch
import torch.nn as nn
from model import Transformer
from dataset import TextDataset
from torch.utils.data import DataLoader

max_seq_len = 256
batch_size = 64
vocab_size = 256
epochs = 10

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", device)
torch.set_default_device(device)

data = TextDataset(max_seq_len=max_seq_len)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

model = Transformer(n_heads=4, n_layers=4, hidden_dim=384, vocab_size=vocab_size, ffn_dim=512)

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-3, betas=(0.9, 0.95), eps=1e-9
)
loss = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for (x, y) in dataloader:

        optimizer.zero_grad()

        pred = model(x)
        target = nn.functional.one_hot(torch.cat([x[:, 1:], y[:, None]], dim=-1), num_classes=vocab_size).float()
        output = loss(pred, target)
        # TODO: loss changes depending on seq len, should be normalized
        output.backward()
        optimizer.step()

        print(output.item())
