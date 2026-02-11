import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split

# ----------------------------
# 1️⃣ Load data
# ----------------------------
print("Loading GDELT JSON data...")
with open("pos_data.json", "r") as f:
    data = json.load(f)

print("Loaded", len(data), "records")

# Extract tokens, tags, and context
samples = []
for row in data:
    token = row.get("token", "").lower()
    pos = row.get("posTag", "")
    context = (row.get("example_snippet", {}) or {}).get("context", "")
    # Clean context to keep only words
    context_tokens = re.findall(r"[A-Za-z']+", context.lower())
    if len(context_tokens) < 3:
        continue
    samples.append((context_tokens, token, pos))

print(f"{len(samples)} usable samples")

# ----------------------------
# 2️⃣ Build vocabularies
# ----------------------------
word_counter = Counter(w for ctx, _, _ in samples for w in ctx)
pos_tags = sorted(set(tag for _, _, tag in samples))

PAD, UNK = "<PAD>", "<UNK>"
word_vocab = {PAD: 0, UNK: 1}
for w, _ in word_counter.most_common(20000):
    word_vocab[w] = len(word_vocab)

tag_vocab = {t: i for i, t in enumerate(pos_tags)}

print(f"Vocab size: {len(word_vocab)}, POS tags: {len(tag_vocab)}")

# ----------------------------
# 3️⃣ Dataset
# ----------------------------
class POSContextDataset(Dataset):
    def __init__(self, samples, word_vocab, tag_vocab, max_len=10):
        self.samples = samples
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def encode_words(self, tokens):
        ids = [self.word_vocab.get(w, self.word_vocab[UNK]) for w in tokens]
        # Pad/truncate
        if len(ids) < self.max_len:
            ids += [self.word_vocab[PAD]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids)

    def __getitem__(self, idx):
        ctx, token, tag = self.samples[idx]
        x = self.encode_words(ctx)
        y = torch.tensor(self.tag_vocab[tag])
        return x, y

train_data, val_data = train_test_split(samples, test_size=0.1, random_state=42)
train_ds = POSContextDataset(train_data, word_vocab, tag_vocab)
val_ds = POSContextDataset(val_data, word_vocab, tag_vocab)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64)

# ----------------------------
# 4️⃣ Model
# ----------------------------
class POSContextModel(nn.Module):
    def __init__(self, vocab_size, tag_size, emb_dim=64, hid_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, tag_size)

    def forward(self, x):
        e = self.emb(x)
        o, _ = self.lstm(e)
        # Use the last hidden state (aggregated context)
        pooled = o.mean(dim=1)
        return self.fc(pooled)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = POSContextModel(len(word_vocab), len(tag_vocab)).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# 5️⃣ Training
# ----------------------------
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}")

# ----------------------------
# 6️⃣ Export to ONNX
# ----------------------------
example_input = torch.randint(0, len(word_vocab), (1, 10)).to(device)
torch.onnx.export(
    model,
    example_input,
    "pos_context_tagger.onnx",
    input_names=["tokens"],
    output_names=["logits"],
    dynamic_axes={"tokens": {1: "seq_len"}, "logits": {1: "seq_len"}},
    opset_version=17,
)
print("Exported pos_context_tagger.onnx")

# ----------------------------
# 7️⃣ Save metadata
# ----------------------------
import json
meta = {"word_vocab": word_vocab, "tag_vocab": tag_vocab}
with open("pos_context_meta.json", "w") as f:
    json.dump(meta, f)
print("Saved pos_context_meta.json")