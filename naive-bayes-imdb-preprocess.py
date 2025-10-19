from __future__ import annotations
import os
import re
import json
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional
import numpy as np
"""""
- **os**: Used for operating-system-level file and directory operations such as joining paths (os.path.join) and checking directory existence (os.path.isdir). This ensures the code works on Windows, macOS, and Linux without hard-coded separators.
- **re**: The built-in regular expression module for tokenizing text. It identifies word-like patterns and splits text accordingly (e.g., re.findall(TOKEN_PATTERN, text) finds all alphanumeric word tokens).
- **json**: Used to serialize (json.dump) and deserialize (json.load) the vocabulary dictionary to and from disk in a human-readable format.
- **collections.Counter**: A dictionary subclass used for counting word frequencies efficiently when building the vocabulary.
- **dataclasses.dataclass**: Provides a lightweight way to define data structures (e.g., EncodedDataset) with automatic initialization and representation methods.
"""
DATA_DIR: str = os.environ.get("IMDB_DATA_DIR", "./data")
ACL_DIR: str = os.path.join(DATA_DIR, "aclImdb")
# Laplace smoothing parameter
ALPHA = 1.0

# Vocabulary + encoding
MIN_FREQ: int = 2 # tokens with freq < MIN_FREQ become <UNK>
MAX_DOC_LEN: Optional[int] = None # set None to disable pad/truncate


a_POS, a_NEG = 1, 0 # labels: 1 = positive, 0 = negative


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# Regex for very simple tokenization: split on non-alphanumeric (keeps numbers, strips punctuation)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
@dataclass
class EncodedDataset:
    X: np.ndarray # shape: (N, L) if padded; or dtype=object array of lists if variable length
    y: np.ndarray # shape: (N,)
    vocab: Dict[str, int] # token -> id
    id2tok: List[str] # reverse mapping




# ---------------------
# Utility: Tokenization
# ---------------------
"""Tokenize with a light regex, lowercase, and basic punctuation stripping.


- Lowercasing merges variants like "Good" and "good".
  """
def simple_tokenize(text: str) -> List[str]:

    text = text.lower()
    return TOKEN_PATTERN.findall(text)

def load_split(split_dir: str) -> List[Tuple[str, int]]:
    """Load raw texts and labels from a split directory.

    Expected structure under `split_dir`:
      split_dir/pos/*.txt, split_dir/neg/*.txt

    Returns a list of (text, label) where label ∈ {a_POS, a_NEG}.
    Also prints per-class counts for debugging purposes
    """
    import glob

    examples: List[Tuple[str, int]] = []
    for label_name, label_id in [("pos", a_POS), ("neg", a_NEG)]:
        d = os.path.join(split_dir, label_name)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected directory not found: {d}")
        # Used glob to ensure only .txt files are read and avoid a previous issue with reading the whole training/test set.
        files = sorted(glob.glob(os.path.join(d, "*.txt")))
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                txt = f.read()
            examples.append((txt, label_id))
        print(f"  Loaded {len(files):5d} files from {label_name} in {split_dir}")
    print(f"  -> Total loaded from {split_dir}: {len(examples)}")
    return examples

def build_vocab(tokenized_docs: Iterable[List[str]], min_freq: int = MIN_FREQ) -> Dict[str, int]:
    """Create a token->id vocabulary from tokenized training docs only.

    - Tokens with frequency < min_freq are mapped to <UNK> during encoding.
    - Reserve 0 for <PAD>, 1 for <UNK> (useful if padding is enabled).
    """
    counter = Counter()
    for toks in tokenized_docs:
        counter.update(toks)

    # Start with specials
    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, freq in counter.most_common():
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab
def encode(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """Map tokens -> ids using the vocabulary; unknowns -> <UNK>.
    This is the Multinomial NB bag-of-words baseline, but I kept sequence form for flexibility in case of low accuracy requiring additional changes.
    """
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(t, unk_id) for t in tokens]
def pad_truncate(ids: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    """Pad (right) or truncate a list of ids to `max_len`.
    Useful for batching; Naive Bayes itself does not require fixed length.
    """
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids)) #this may need debugging but on pen and paper looks fine
def as_numpy(seqs: List[List[int]], max_len: Optional[int]) -> np.ndarray:
    """Convert a list of variable-length sequences to a numpy array.
    - If max_len is None -> returns an object array of Python lists (still fine for NB counts).
    - Else -> returns shape (N, max_len) int64 array.
    """
    if max_len is None:
        arr = np.empty(len(seqs), dtype=object)
        for i, s in enumerate(seqs):
            arr[i] = s
        return arr
    else:
        mat = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, s in enumerate(seqs):
            padded = np.array(pad_truncate(s, max_len), dtype=np.int64)  # length == max_len
            mat[i, :max_len] = padded  # assign full padded row
        return mat


def preprocess(split: str, vocab: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], List[str]]:
    """Preprocess a dataset split ("train" or "test").

    Steps (baseline):
    1) Load raw texts
    2) Tokenize + lowercase
    3) Build vocab (train only) with MIN_FREQ and specials
    4) Encode tokens -> ids with <UNK>
    5) Optional pad/truncate to MAX_DOC_LEN

    Returns: (X, y, vocab, id2tok)
      - X: numpy array (N, L) if padded; or object array of lists if variable lengths
      - y: numpy array (N,)
      - vocab: token->id
      - id2tok: reverse mapping list
    """
    split_dir = os.path.join(ACL_DIR, split)
    raw = load_split(split_dir)  # List[(text, label)]

    texts, labels = zip(*raw)  # type: ignore
    tokenized = [simple_tokenize(t) for t in texts]

    # Build vocab only on training data
    if split == "train":
        vocab = build_vocab(tokenized, min_freq=MIN_FREQ)
    elif vocab is None:
        raise ValueError("For non-train splits, provide a `vocab` built on training data.")

    # Encode
    X_ids = [encode(toks, vocab) for toks in tokenized]
    y = np.array(labels, dtype=np.int64)

    # Optional padding
    X = as_numpy(X_ids, MAX_DOC_LEN)

    # Reverse map for convenience/debugging
    id2tok = [None] * len(vocab)
    for tok, idx in vocab.items():
        id2tok[idx] = tok

    return X, y, vocab, id2tok  # type: ignore

# -----------------------------
# Orchestration + simple export
# -----------------------------
def save_vocab(vocab: Dict[str, int], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def save_numpy(arr: np.ndarray, out_path: str) -> None:
    # .npy is fast and portable; can be loaded with np.load(mmap_mode='r')
    np.save(out_path, arr)

def main() -> None:
    # Sanity checks for directory layout
    if not os.path.isdir(ACL_DIR):
        raise FileNotFoundError(
            f"IMDb dataset not found under {ACL_DIR}. Set DATA_DIR or place this file inside the dataset folder."
        )

    print("\n[1/3] Preprocessing TRAIN…")
    X_train, y_train, vocab, id2tok = preprocess("train")
    print(f"Train: X shape={X_train.shape}, y shape={y_train.shape}, vocab size={len(vocab)}")

    print("[2/3] Preprocessing TEST…")
    X_test, y_test, _, _ = preprocess("test", vocab=vocab)
    print(f"Test:  X shape={X_test.shape}, y shape={y_test.shape}")

    # Export (optional)
    out_dir = os.path.join(ACL_DIR, "preprocessed_baseline")
    os.makedirs(out_dir, exist_ok=True)

    save_vocab(vocab, os.path.join(out_dir, "vocab.json"))
    save_numpy(X_train, os.path.join(out_dir, "X_train.npy"))
    save_numpy(y_train, os.path.join(out_dir, "y_train.npy"))
    save_numpy(X_test, os.path.join(out_dir, "X_test.npy"))
    save_numpy(y_test, os.path.join(out_dir, "y_test.npy"))

    print("[3/3] Saved preprocessed arrays and vocab to:", out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
