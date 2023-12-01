from collections import Counter
# EOS tokens?
# Block size?
vocab_size = 100

with open("input.txt", "r") as f:
    data = f.read()

data = data.encode(encoding="ascii")

vocab = {
    i: chr(i) for i in set(data)
}
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
    current_vocab_size = len(set(data))
    new_token += 1

# Remove items which got merged and removed from the dataset
vocab = {k: v for k, v in vocab.items() if k in set(data)}
assert len(vocab) == vocab_size



