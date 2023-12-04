import torch 
from torch.utils.data import Dataset, DataLoader
import json

class TextDataset(Dataset):
    def __init__(self, max_seq_len):
        with open("vocab.json", "r") as f:
            self.vocab = json.load(f)
        with open("data.json", "r") as f:
            self.data = json.load(f)
        self.data = torch.tensor(self.data)
        self.max_seq_len = max_seq_len
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def text_to_ids(self, text, max_tok_size=10):
        result = []
        i = 0
        while i < len(text):
            for j in range(max_tok_size, 0, -1):
                if text[i:i+j] in self.inverse_vocab:
                    result.append(int(self.inverse_vocab[text[i:i+j]]))
                    i += j
                    break
            else:
                print("Unknown token", text[i])
                i += 1
        return torch.tensor(result)
    
    def ids_to_text(self, ids):
        return "".join([self.vocab[str(i.item())] for i in ids])

    def __len__(self):
        return (len(self.data) // self.max_seq_len) - 1

    def __getitem__(self, i):
        return self.data[i * self.max_seq_len: (i + 1) * self.max_seq_len], self.data[(i + 1) * self.max_seq_len]