"""
bert_kaggle_syn_ant.py

End-to-end: Kaggle synonyms/antonyms -> contrastive fine-tune BERT -> ONNX export
"""

import os
import json
import random
import math
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, logging as hf_logging

hf_logging.set_verbosity_error()  # quieter transformers output

# ---------------------------
# CONFIGURABLE PARAMETERS
# ---------------------------
DATA_FILE = "kaggle_syn_ant.json"   # <-- set this to your Kaggle file path
MODEL_NAME = "bert-base-uncased"
ONNX_OUT = "bert_kaggle_syn_ant_context.onnx"

# thresholds (tune these)
SYN_SCORE_THRESHOLD = 0.5   # score >= this => treat as synonym
ANT_SCORE_THRESHOLD = -0.5  # score <= this => treat as antonym

# training params
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
MARGIN = 0.4

# runtime device (MPS / CUDA / CPU)
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[INFO] Using device: {DEVICE}")

# ---------------------------
# UTIL: load Kaggle dataset format (support JSON or JSONL)
# ---------------------------
def load_kaggle_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the Kaggle dataset JSON here or change DATA_FILE.")
    # Try to determine file format: JSON (one dict) or JSONL (one JSON per line)
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        data = json.loads(text)
        # Expecting either dict: { "word": { related:score, ... }, ... }
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Otherwise try JSONL / line-delimited
    data = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # if obj is dict with single top-level word
                # merge into data
                for k, v in obj.items():
                    if isinstance(v, dict):
                        data[k] = v
                    else:
                        data[k] = v
            except Exception:
                # fallback: treat line as "word <tab> json"
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    w, j = parts
                    try:
                        data[w] = json.loads(j)
                    except Exception:
                        pass
    return data

# ---------------------------
# Build synonym & antonym pairs from the Kaggle mapping
# Format expected (example): { "gotten": { "net": 0.843886, "dress": 0.5, "gratify": -0.625, ...}, ... }
# ---------------------------
def build_pairs_from_kaggle(mapping, syn_thr=SYN_SCORE_THRESHOLD, ant_thr=ANT_SCORE_THRESHOLD, max_pairs=None):
    synonym_pairs = []
    antonym_pairs = []
    vocab_set = set()
    for head, related in mapping.items():
        head = head.strip()
        vocab_set.add(head)
        if not isinstance(related, dict):
            continue
        for rword, score in related.items():
            r = rword.strip()
            vocab_set.add(r)
            # convert score to float (robust)
            try:
                s = float(score)
            except Exception:
                continue
            if s >= syn_thr:
                synonym_pairs.append((head, r))
            elif s <= ant_thr:
                antonym_pairs.append((head, r))
            # else ignore borderline
    # Optionally cap pairs for performance
    if max_pairs:
        if len(synonym_pairs) > max_pairs:
            synonym_pairs = random.sample(synonym_pairs, max_pairs)
        if len(antonym_pairs) > max_pairs:
            antonym_pairs = random.sample(antonym_pairs, max_pairs)
    return synonym_pairs, antonym_pairs, sorted(vocab_set)

# ---------------------------
# Dataset for pairs (we provide each side as a short "context" sentence)
# We'll create small contexts like "the word: <word>" so BERT has tokens and some context.
# ---------------------------
class WordPairDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer, template="the word: {}"):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.template = template
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        la = self.labels[idx]
        ta = self.template.format(a)
        tb = self.template.format(b)
        return ta, tb, la
    @staticmethod
    def collate(batch, tokenizer):
        a_texts, b_texts, labels = zip(*batch)
        enc_a = tokenizer(list(a_texts), padding=True, truncation=True, return_tensors="pt")
        enc_b = tokenizer(list(b_texts), padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(labels, dtype=torch.float)
        return enc_a, enc_b, labels

# ---------------------------
# Training function (contrastive with MarginRankingLoss)
# ---------------------------
def train_contrastive(bert, tokenizer, synonym_pairs, antonym_pairs,
                      device=DEVICE, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, margin=MARGIN):
    # Combine and label: synonyms -> +1, antonyms -> -1
    pairs = synonym_pairs + antonym_pairs
    labels = [1]*len(synonym_pairs) + [-1]*len(antonym_pairs)
    # shuffle
    idxs = list(range(len(pairs)))
    random.shuffle(idxs)
    pairs = [pairs[i] for i in idxs]
    labels = [labels[i] for i in idxs]
    ds = WordPairDataset(pairs, labels, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: WordPairDataset.collate(b, tokenizer))
    opt = torch.optim.AdamW(bert.parameters(), lr=lr)
    criterion = nn.MarginRankingLoss(margin=margin)
    bert = bert.to(device)
    bert.train()
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for enc_a, enc_b, labs in tqdm(loader, desc=f"Train epoch {epoch+1}/{epochs}"):
            # move to device
            enc_a = {k:v.to(device) for k,v in enc_a.items()}
            enc_b = {k:v.to(device) for k,v in enc_b.items()}
            labs = labs.to(device)
            opt.zero_grad()
            out_a = bert(**enc_a).last_hidden_state[:,0]  # [B, D]
            out_b = bert(**enc_b).last_hidden_state[:,0]  # [B, D]
            sim = F.cosine_similarity(out_a, out_b)       # [B]
            # We want sim(a,b) > sim_threshold for synonyms and sim(a,b) < -something for antonyms.
            # MarginRankingLoss expects inputs x1,x2 and target y in {-1,1} and computes:
            # loss = max(0, -y * (x1 - x2) + margin)
            # We'll set x1 = sim, x2 = zeros, so for y=1 (synonym) we want sim > 0 by margin
            zeros = torch.zeros_like(sim).to(device)
            loss = criterion(sim, zeros, labs)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        avg = running_loss / max(1, n_batches)
        print(f"Epoch {epoch+1} avg loss: {avg:.6f}")
    return bert

# ---------------------------
# Build vocab embeddings (encode each vocab word via BERT)
# ---------------------------
def build_vocab_embeddings(bert, tokenizer, vocab, device=DEVICE, batch=64):
    bert.eval()
    embs = []
    for i in tqdm(range(0, len(vocab), batch), desc="Encoding vocab"):
        batch_words = vocab[i:i+batch]
        # create short context to help BERT (the same template used during training)
        texts = [f"the word: {w}" for w in batch_words]
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = bert(**enc).last_hidden_state[:,0]  # [B, D]
        embs.append(out.cpu())
    embs = torch.cat(embs, dim=0)  # [V, D]
    return embs

# ---------------------------
# ONNX wrapper: takes (input_ids, attention_mask) and outputs [B, V] cosine similarities
# The embedding matrix is frozen (from fine-tuned BERT) for fast similarity compute
# ---------------------------
class BertContextCosine(nn.Module):
    def __init__(self, base_bert, fixed_embeddings: torch.Tensor):
        super().__init__()
        self.bert = base_bert
        # fixed_embeddings: [V, D] (torch.FloatTensor)
        self.emb = nn.Embedding.from_pretrained(fixed_embeddings, freeze=True)
    def forward(self, input_ids, attention_mask):
        # input_ids, attention_mask are tensors (B, seq_len)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        vec = out.last_hidden_state[:,0]            # [B, D]
        vec_n = F.normalize(vec, p=2, dim=1)       # [B, D]
        emb_n = F.normalize(self.emb.weight, p=2, dim=1)  # [V, D]
        sims = torch.matmul(vec_n, emb_n.T)        # [B, V]
        return sims

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    # load Kaggle data
    mapping = load_kaggle_file(DATA_FILE)
    print(f"[INFO] Loaded {len(mapping)} top-level entries from Kaggle file")

    # build pairs
    synonyms, antonyms, vocab = build_pairs_from_kaggle(mapping,
                                                       syn_thr=SYN_SCORE_THRESHOLD,
                                                       ant_thr=ANT_SCORE_THRESHOLD,
                                                       max_pairs=None)
    print(f"[INFO] Built {len(synonyms)} synonym pairs and {len(antonyms)} antonym pairs")
    print(f"[INFO] Vocab size from dataset: {len(vocab)}")

    # trim or sample pairs if dataset huge (optional)
    MAX_PAIRS_PER_TYPE = 50000
    if len(synonyms) > MAX_PAIRS_PER_TYPE:
        synonyms = random.sample(synonyms, MAX_PAIRS_PER_TYPE)
    if len(antonyms) > MAX_PAIRS_PER_TYPE:
        antonyms = random.sample(antonyms, MAX_PAIRS_PER_TYPE)

    # load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME)
    bert.to(DEVICE)

    # fine-tune contrastively
    fine_tuned = train_contrastive(bert, tokenizer, synonyms, antonyms,
                                   device=DEVICE, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, margin=MARGIN)

    # build vocab embeddings (from fine-tuned BERT)
    embeddings = build_vocab_embeddings(fine_tuned, tokenizer, vocab, device=DEVICE, batch=64)
    print(f"[INFO] Built embeddings shape: {embeddings.shape}")

    # export ONNX
    wrapper = BertContextCosine(fine_tuned, embeddings)
    wrapper.eval()

    # example input to trace
    example_text = "I felt very happy when the news came"
    enc = tokenizer(example_text, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Move example inputs to CPU for export (ONNX export uses CPU tensors)
    # (transformers models are Torch modules; if bert is on GPU, we can temporarily move it to cpu for export)
    wrapper_cpu = wrapper.to("cpu")
    input_ids_cpu = input_ids
    attention_mask_cpu = attention_mask

    print("[INFO] Exporting ONNX model (this may take a minute)...")
    torch.onnx.export(
        wrapper_cpu,
        (input_ids_cpu, attention_mask_cpu),
        ONNX_OUT,
        input_names=["input_ids", "attention_mask"],
        output_names=["cosine_similarity"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "cosine_similarity": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True
    )
    print(f"[OK] ONNX exported to {ONNX_OUT}")

    # Save vocabulary mapping to disk for inference usage
    import json
    with open("vocab_kaggle.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("[OK] Saved vocab_kaggle.json (index -> word)")

    # Simple local helper: top-k from embeddings (numpy)
    weights = embeddings.cpu().numpy()
    word2idx = {w: i for i, w in enumerate(vocab)}

    def get_similar_from_weights(text, topn=10):
        enc = tokenizer(text, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = fine_tuned(**enc).last_hidden_state[:,0]
            vec = out.cpu().numpy()[0]
        sims = weights @ vec
        norms = np.linalg.norm(weights, axis=1) * np.linalg.norm(vec) + 1e-12
        sims = sims / norms
        idxs = np.argsort(sims)[::-1][:topn]
        return [(vocab[i], float(sims[i])) for i in idxs]

    # demo
    demo_ctx = "I felt so happy after getting good news"
    print("\nTop related words (demo):")
    for w, s in get_similar_from_weights(demo_ctx, topn=10):
        print(f"  {w:20s} {s:.3f}")

if __name__ == "__main__":
    main()