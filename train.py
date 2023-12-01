import json
import torch 


### Load vocab
with open("vocab.json", "r") as f:
    vocab = json.load(f)
with open("data.json", "r") as f:
    data = json.load(f)

print(
    "".join(
        [vocab[str(i)] for i in data[:500]]
    )
)