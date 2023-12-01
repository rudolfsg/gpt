from collections import Counter
import json 

def tokenize_dataset(data, vocab_size):
    vocab = {
        i: chr(i) for i in set(data)
    }
    # For now assume the text is continuous, i.e. not gathered from multiple contexts
    # Hence don't use endoftext or other special tokens
    special_tokens = {
        0: "<|endoftext|>",
    }
    for key, value in special_tokens.items():
        assert key not in vocab
        vocab[key] = value

    current_vocab_size = len(vocab)
    new_token = max(data) + 1

    while current_vocab_size < vocab_size:

        pairs = [
            tuple(data[i:i+2]) for i in range(len(data) - 1)
        ]
        new_pair = Counter(pairs).most_common(1)[0][0]
        vocab[new_token] = vocab[new_pair[0]] + vocab[new_pair[1]]
        print("added", vocab[new_token], f" -> {new_pair}")
        
        new_data = []
        i=0
        while i < len(data):
            if data[i] == new_pair[0] and i != len(data) -1 and data[i+1] == new_pair[1]:
                new_data.append(new_token)
                i += 2
            else:
                new_data.append(data[i])
                i += 1

        data = new_data
        current_vocab_size = len(set(data)) + len(special_tokens)
        new_token += 1

    # Remove items which got merged and removed from the dataset
    vocab = {k: v for k, v in vocab.items() if k in set(data) or k in special_tokens}
    assert len(vocab) == vocab_size
    # Make token values sequential
    vocab = {old_id: (new_id, v) for new_id, (old_id, v) in enumerate(vocab.items())}
    data = [vocab[t][0] for t in data]
    vocab = {new_id: v for _, (new_id, v) in vocab.items()}
    return data, vocab 

### Generate vocab
# with open("input.txt", "r") as f:
#     data = f.read()

# data = data.encode(encoding="ascii")
# data, vocab = tokenize_dataset(data, vocab_size=256)

# with open("vocab.json", "w") as f:
#     json.dump(vocab, f)
# with open("data.json", "w") as f:
#     json.dump(data, f)




