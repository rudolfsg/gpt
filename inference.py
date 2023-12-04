import torch 
from dataset import TextDataset

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", device)
torch.set_default_device(device)

# load model from pt
model = torch.load("model.pt")
model.eval()

data = TextDataset(max_seq_len=32)

text = "My name is "
ids = data.text_to_ids(text)
ids = torch.tensor(ids).reshape(1, -1)

with torch.no_grad():
    for _ in range(100):
        logits = model(ids)
        ids = torch.cat([ids, logits[-1, -1].argmax().reshape(1, 1)], dim=-1)

print(data.ids_to_text(ids.view(-1)))
