import json
import re
import numpy as np
import onnxruntime as ort
import spacy
from lemminflect import getInflection
from transformers import AutoTokenizer

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "bert_kaggle_target_fast.onnx"
VOCAB_PATH = "vocab_kaggle.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 32

NUM_SYNONYMS = 10
NUM_ANTONYMS = 10


# -----------------------
# Load NLP tools
# -----------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------
# Load vocab
# -----------------------
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

print(f"[INFO] Loaded vocab size: {len(vocab)}")

# -----------------------
# Precompute morphology cache (fast filtering)
# -----------------------
print("[INFO] Building vocab morphology cache...")

vocab_cache = {}

print("[INFO] Morphology cache built.")

# -----------------------
# Load tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -----------------------
# Load ONNX model
# -----------------------
session = ort.InferenceSession(MODEL_PATH)
print("[INFO] ONNX model loaded.")

# -----------------------
# Utility functions
# -----------------------
def find_target_position(text, target_word):
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LEN
    )

    offsets = enc["offset_mapping"]
    start_char = text.find(target_word)

    if start_char == -1:
        return 0

    for idx, (s, e) in enumerate(offsets):
        if s <= start_char < e:
            return idx

    return 0


def valid_surface_form(target, candidate):
    target = target.lower()
    candidate = candidate.lower()

    # letters only
    if not re.fullmatch(r"[a-zA-Z]+", candidate):
        return False

    # no phrases
    if " " in candidate:
        return False

    # exact match
    if candidate == target:
        return False

    # block derivations like happiness
    if target in candidate or candidate in target:
        return False

    return True


def match_morphology(target_word, candidate):
    doc_target = nlp(target_word)
    if len(doc_target) != 1:
        return False

    t = doc_target[0]
    target_pos = t.pos_
    target_tag = t.tag_
    target_lemma = t.lemma_

    doc_cand = nlp(candidate)
    if len(doc_cand) != 1:
        return False

    c = doc_cand[0]
    cand_pos = c.pos_
    cand_tag = c.tag_
    cand_lemma = c.lemma_

    # POS must match
    if cand_pos != target_pos:
        return False

    # 🚫 block same lemma (this fixes your bug)
    if cand_lemma == target_lemma:
        return False

    # Match morphology
    if target_pos in {"VERB", "NOUN", "ADJ"}:
        if cand_tag != target_tag:
            infl = getInflection(cand_lemma, tag=target_tag)
            if infl:
                return infl[0]
            return False

    return candidate



def run_example(text, target_word):
    enc = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )

    target_pos_arr = np.array(
        [find_target_position(text, target_word)],
        dtype=np.int64
    )

    outputs = session.run(
        None,
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target_pos": target_pos_arr,
        },
    )

    sims = outputs[0][0]

    # Sort once
    sorted_high = np.argsort(-sims)   # synonyms
    sorted_low = np.argsort(sims)     # antonyms

    print("\n" + "=" * 60)
    print(f"Text: {text}")
    print(f"Target word: {target_word}")
    print("=" * 60)

    # -------------------
    # SYNONYMS
    # -------------------
    print("\n🔵 Synonyms")
    print("-" * 40)

    count = 0
    seen = set()

    for idx in sorted_high:
        word = vocab[idx]
        score = sims[idx]

        if not valid_surface_form(target_word, word):
            continue

        morph = match_morphology(target_word, word)
        if not morph:
            continue

        if morph in seen:
            continue

        seen.add(morph)
        print(f"{morph:15s}  score={score:.4f}")
        count += 1

        if count >= NUM_SYNONYMS:
            break

    # -------------------
    # ANTONYMS
    # -------------------
    print("\n🔴 Antonyms")
    print("-" * 40)

    count = 0
    seen = set()

    for idx in sorted_low:
        word = vocab[idx]
        score = sims[idx]

        if not valid_surface_form(target_word, word):
            continue

        morph = match_morphology(target_word, word)
        if not morph:
            continue

        if morph in seen:
            continue

        seen.add(morph)
        print(f"{morph:15s}  score={score:.4f}")
        count += 1

        if count >= NUM_ANTONYMS:
            break



# -----------------------
# Interactive mode
# -----------------------
if __name__ == "__main__":

    run_example("I am very happy today, the sun is shining brightly.", "happy")