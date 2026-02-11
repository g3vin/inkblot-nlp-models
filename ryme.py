import io
import random
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ---------------------
# Config / paths
# ---------------------
RAW_CSV_URL = "https://raw.githubusercontent.com/scrosseye/PeRDict-database/main/full_data_elp_perfect_rhymes.csv"
CSV_PATH = None  # Optional local path if download fails

EMBED_DIM = 128
BATCH_SIZE = 1024
LR = 1e-3
EPOCHS = 8
NEGATIVE_RATIO = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Load CSV
# ---------------------
def load_csv():
    if CSV_PATH:
        df = pd.read_csv(CSV_PATH)
    else:
        import requests
        r = requests.get(RAW_CSV_URL)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    return df

# ---------------------
# Build rhyme pairs
# ---------------------
def build_pairs_from_df(df: pd.DataFrame) -> Tuple[List[Tuple[str, str]], List[str]]:
    cols = [c.lower() for c in df.columns]
    pairs = []

    if "word" in cols and ("perfect_rhyme" in cols or "rhyme" in cols or "rhyme_word" in cols):
        w_col = [c for c in df.columns if c.lower() == "word"][0]
        rhyme_col = [c for c in df.columns if c.lower() in ("perfect_rhyme", "rhyme", "rhyme_word")][0]
        for _, r in df[[w_col, rhyme_col]].dropna().iterrows():
            w1 = str(r[w_col]).strip().lower()
            w2 = str(r[rhyme_col]).strip().lower()
            if w1 and w2:
                pairs.append((w1, w2))
    elif df.shape[1] >= 2:
        col1, col2 = df.columns[:2]
        for _, r in df[[col1, col2]].dropna().iterrows():
            w = str(r[col1]).strip().lower()
            rhymes = [x.strip() for x in str(r[col2]).replace(";", ",").split(",") if x.strip()]
            for rr in rhymes:
                pairs.append((w, rr))
    else:
        raise ValueError("CSV format not recognized — check the dataset.")

    vocab = sorted({w for p in pairs for w in p})
    return pairs, vocab

# ---------------------
# Dataset
# ---------------------
class PairDataset(Dataset):
    def __init__(self, pos_pairs: List[Tuple[int, int]], vocab_size: int, neg_ratio: int = 1):
        self.pos = pos_pairs
        self.vocab_size = vocab_size
        self.neg_ratio = neg_ratio

    def __len__(self):
        return len(self.pos) * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.neg_ratio)
        mode = idx % (1 + self.neg_ratio)

        if mode == 0:
            a, b = self.pos[pos_idx]
            label = 1.0
        else:
            a, _ = self.pos[pos_idx]
            b = random.randrange(self.vocab_size)
            label = 0.0
        return (
            torch.tensor(a, dtype=torch.long),
            torch.tensor(b, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )

# ---------------------
# Model
# ---------------------
class RhymeModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, idx_a, idx_b):
        ea, eb = self.emb(idx_a), self.emb(idx_b)
        return self.mlp(torch.cat([ea, eb], dim=-1)).squeeze(-1)

# ---------------------
# Train
# ---------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for a, b, y in train_loader:
            a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(a, b), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * a.size(0)
        total_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for a, b, y in val_loader:
                a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
                loss = criterion(model(a, b), y)
                val_loss += loss.item() * a.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs} | Train: {total_loss:.4f} | Val: {val_loss:.4f}")
    return model

# ---------------------
# Main
# ---------------------
def main():
    print("📥 Loading dataset...")
    df = load_csv()
    pairs, vocab = build_pairs_from_df(df)
    print(f"✅ Loaded {len(pairs)} rhyme pairs | Vocab size: {len(vocab)}")

    word2idx = {w: i for i, w in enumerate(vocab)}
    pos_pairs_idx = list({(word2idx[a], word2idx[b]) for a, b in pairs if a in word2idx and b in word2idx})

    train_pairs, val_pairs = train_test_split(pos_pairs_idx, test_size=0.1, random_state=42)
    train_loader = DataLoader(PairDataset(train_pairs, len(vocab), NEGATIVE_RATIO), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PairDataset(val_pairs, len(vocab), NEGATIVE_RATIO), BATCH_SIZE, shuffle=False)

    model = RhymeModel(vocab_size=len(vocab), embed_dim=EMBED_DIM)
    print("🚀 Training model...")
    model = train_model(model, train_loader, val_loader)

    # ---------------------
    # Save model as ONNX + vocab
    # ---------------------
    print("💾 Exporting to ONNX...")
    dummy_a = torch.randint(0, len(vocab), (1,), dtype=torch.long)
    dummy_b = torch.randint(0, len(vocab), (1,), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_a, dummy_b),
        "rhyme_model.onnx",
        input_names=["word_a", "word_b"],
        output_names=["rhyme_score"],
        dynamic_axes={"word_a": {0: "batch"}, "word_b": {0: "batch"}},
        opset_version=17,
    )
    print("✅ Saved model to rhyme_model.onnx")

    with open("vocab.json", "w") as f:
        json.dump(vocab, f)
    print("✅ Saved vocab.json")

if __name__ == "__main__":
    main()