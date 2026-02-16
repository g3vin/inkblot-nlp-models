"""
Fast synonym/antonym fine-tuning (optimized for local/MPS)
----------------------------------------------------------
- Uses MiniLM sentence-transformer (small & fast)
- Short context (max_length=32)
- Sharded training for large datasets
- Freezes lower encoder layers
- MPS / CPU friendly (no AMP)
"""

import json, random, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ---------------------------
# CONFIG
# ---------------------------
DATA_FILE = "kaggle_syn_ant.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_OUT = "synonym.onnx"
BATCH_SIZE = 64
EPOCHS = 1
LR = 3e-5
MARGIN = 0.3
SYN_THR, ANT_THR = 0.5, -0.5
SHARD_SIZE = 10_000
MAX_LEN = 32

# ---------------------------
# DEVICE
# ---------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[INFO] Using device: {DEVICE}")

torch.set_num_threads(4)

# ---------------------------
# Load Kaggle dataset
# ---------------------------
def load_kaggle_json(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    data = json.loads(text)
    return data

def build_pairs(mapping):
    syn, ant, vocab = [], [], set()
    for w, rels in mapping.items():
        if not isinstance(rels, dict):
            continue
        for rw, s in rels.items():
            try:
                sc = float(s)
            except:
                continue
            vocab.update([w.strip(), rw.strip()])
            if sc >= SYN_THR:
                syn.append((w, rw))
            elif sc <= ANT_THR:
                ant.append((w, rw))
    random.shuffle(syn); random.shuffle(ant)
    return syn, ant, sorted(vocab)

# ---------------------------
# Dataset + collate
# ---------------------------
class WordPairDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer):
        self.pairs, self.labels, self.tok = pairs, labels, tokenizer
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        label = self.labels[i]
        ctx_a = f"Context: ... {a} ..."
        ctx_b = f"Context: ... {b} ..."
        return ctx_a, a, ctx_b, b, label

def wordpair_collate(batch, tok):
    ctx_a, targ_a, ctx_b, targ_b, labels = zip(*batch)

    enc_a = tok(list(ctx_a), return_tensors="pt", padding="max_length",
                truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    enc_b = tok(list(ctx_b), return_tensors="pt", padding="max_length",
                truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)

    def find_positions(enc, ctxs, targets):
        offmaps = enc.pop("offset_mapping")
        pos_list = []
        for i, (ctx, tgt) in enumerate(zip(ctxs, targets)):
            start_char = ctx.find(tgt)
            pos = 0
            if start_char != -1:
                for tok_idx, (s, e) in enumerate(offmaps[i].tolist()):
                    if s <= start_char < e:
                        pos = tok_idx
                        break
            pos_list.append(pos)
        return torch.tensor(pos_list, dtype=torch.long)

    pos_a = find_positions(enc_a, ctx_a, targ_a)
    pos_b = find_positions(enc_b, ctx_b, targ_b)
    return enc_a, pos_a, enc_b, pos_b, torch.tensor(labels, dtype=torch.float)

def shard_pairs(pairs, labels, shard_size=SHARD_SIZE):
    for i in range(0, len(pairs), shard_size):
        yield pairs[i:i+shard_size], labels[i:i+shard_size]

# ---------------------------
# Training
# ---------------------------
def train_fast(model, tokenizer, syn, ant):
    pairs = syn + ant
    labels = [1]*len(syn) + [-1]*len(ant)
    mix = list(zip(pairs, labels))
    random.shuffle(mix)
    pairs, labels = zip(*mix)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = nn.MarginRankingLoss(margin=MARGIN)
    model.train()

    for epoch in range(EPOCHS):
        shard_losses = []
        for shard_idx, (p_shard, l_shard) in enumerate(shard_pairs(pairs, labels)):
            ds = WordPairDataset(p_shard, l_shard, tokenizer)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=lambda b: wordpair_collate(b, tokenizer),
                            num_workers=0)

            tot = 0.0
            for enc_a, pos_a, enc_b, pos_b, labs in tqdm(
                dl, desc=f"Epoch {epoch+1}/{EPOCHS} – shard {shard_idx+1}"
            ):
                enc_a = {k: v.to(DEVICE) for k, v in enc_a.items()}
                enc_b = {k: v.to(DEVICE) for k, v in enc_b.items()}
                pos_a, pos_b, labs = pos_a.to(DEVICE), pos_b.to(DEVICE), labs.to(DEVICE)

                opt.zero_grad()
                # merge both
                input_ids = torch.cat([enc_a["input_ids"], enc_b["input_ids"]], dim=0)
                attention_mask = torch.cat(
                    [enc_a["attention_mask"], enc_b["attention_mask"]], dim=0
                )

                out = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state
                B = enc_a["input_ids"].size(0)
                hidden_a, hidden_b = hidden[:B], hidden[B:]
                idx = torch.arange(B, device=hidden.device)
                vec_a = hidden_a[idx, pos_a]
                vec_b = hidden_b[idx, pos_b]
                sim = F.cosine_similarity(vec_a, vec_b)
                loss = loss_fn(sim, torch.zeros_like(sim), labs)

                loss.backward()
                opt.step()
                tot += loss.item()

            avg_loss = tot / len(dl)
            shard_losses.append(avg_loss)
            print(f"[INFO] Finished shard {shard_idx+1}, avg loss {avg_loss:.4f}")
            torch.save(model.state_dict(), f"checkpoint_shard{shard_idx+1}.pt")

        print(f"Epoch {epoch+1} mean loss: {np.mean(shard_losses):.4f}")
    return model

# ---------------------------
# Build vocab embeddings
# ---------------------------
def build_vocab_emb(model, tokenizer, vocab):
    model.eval()
    allv = []
    for i in tqdm(range(0, len(vocab), 128), desc="Encoding vocab"):
        words = vocab[i:i+128]
        ctxs = [f"Context: ... {w} ..." for w in words]
        enc = tokenizer(ctxs, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=MAX_LEN,
                        return_offsets_mapping=True).to(DEVICE)
        offmaps = enc.pop("offset_mapping")
        pos_list = []
        for j, (ctx, w) in enumerate(zip(ctxs, words)):
            start = ctx.find(w)
            pos = 0
            if start != -1:
                for tok_idx, (s, e) in enumerate(offmaps[j].tolist()):
                    if s <= start < e:
                        pos = tok_idx
                        break
            pos_list.append(pos)
        pos = torch.tensor(pos_list, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model(**enc)
            hidden = out.last_hidden_state
            idx = torch.arange(hidden.size(0), device=hidden.device)
            vecs = hidden[idx, pos]
        allv.append(vecs.cpu())
    return torch.cat(allv, dim=0)

# ---------------------------
# Target-aware ONNX wrapper
# ---------------------------
class BertTargetCosine(nn.Module):
    def __init__(self, bert, emb, tokenizer):
        super().__init__()
        self.bert = bert
        self.emb = nn.Embedding.from_pretrained(emb, freeze=True)
        self.tokenizer = tokenizer
    def forward(self, input_ids, attention_mask, target_pos):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        idx = torch.arange(hidden.size(0), device=hidden.device)
        vec = hidden[idx, target_pos]
        vec_n = F.normalize(vec, p=2, dim=1)
        emb_n = F.normalize(self.emb.weight, p=2, dim=1)
        sims = torch.matmul(vec_n, emb_n.T)
        return sims

# ---------------------------
# MAIN
# ---------------------------
def main():
    data = load_kaggle_json(DATA_FILE)
    syn, ant, vocab = build_pairs(data)
    print(f"[INFO] {len(syn)} syn, {len(ant)} ant, vocab={len(vocab)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME)

    # Freeze all but top 2 encoder layers
    for name, param in bert.named_parameters():
        if not any(x in name for x in ["layer.5", "layer.4", "pooler", "classifier"]):
            param.requires_grad = False

    bert.to(DEVICE)
    ckpt_path = "checkpoint_shard424.pt"
    if Path(ckpt_path).exists():
        print(f"[INFO] Loading checkpoint {ckpt_path}")
        bert.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        fine = bert
    else:
        fine = train_fast(bert, tok, syn, ant)

    emb = build_vocab_emb(fine, tok, vocab)
    print(f"[INFO] Embeddings: {emb.shape}")

    wrapper = BertTargetCosine(fine, emb, tok)
    wrapper.eval()

    text = "She felt happy after good news"
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    target_pos = torch.tensor([4], dtype=torch.long)
    wrapper.to("cpu")
    print("[INFO] Exporting ONNX...")
    torch.onnx.export(wrapper, (enc["input_ids"], enc["attention_mask"], target_pos),
                      ONNX_OUT,
                      input_names=["input_ids","attention_mask","target_pos"],
                      output_names=["cosine_similarity"],
                      dynamic_axes={"input_ids":{0:"batch",1:"seq"},
                                    "attention_mask":{0:"batch",1:"seq"},
                                    "cosine_similarity":{0:"batch"}},
                      opset_version=18)
    print(f"[OK] Exported {ONNX_OUT}")

    with open("vocab_kaggle.json","w") as f:
        json.dump(vocab,f,indent=2)
    print("[OK] Saved vocab_kaggle.json")

if __name__ == "__main__":
    main()