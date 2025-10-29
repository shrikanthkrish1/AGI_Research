# arc_tokenization.py
"""
Structured tokenization utilities for ARC grid tasks.

Key functions:
 - grid_to_structured_text(input_grid, output_grid=None, include_delta=False, normalize_colors=True)
 - add_arc_special_tokens(tokenizer, model=None, max_color_tokens=50)
 - encode_arc_pair_batch(tokenizer, pairs, max_length=512, return_tensors='pt')
 - structured_text_to_grid(text)  # best-effort parse for debugging
 - grid_hash(grid)

Usage:
  from arc_tokenization import add_arc_special_tokens, grid_to_structured_text, encode_arc_pair_batch
"""
import hashlib
import numpy as np
from typing import Sequence, Optional, Tuple, List, Dict, Any
import re

# --------------------------
# Utilities for canonical color mapping and hashing
# --------------------------
def canonical_color_map(grid: Sequence[Sequence[int]]) -> Dict[int, str]:
    """
    Map numeric colors to canonical tokens COLOR_0, COLOR_1... based on frequency.
    Deterministic ordering by frequency then by numeric value.
    """
    flat = [int(c) for row in grid for c in row]
    if len(flat) == 0:
        return {}
    unique, counts = np.unique(flat, return_counts=True)
    # sort by (-count, value)
    order = sorted(zip(-counts, unique), key=lambda x: (x[0], x[1]))
    ordered_vals = [v for _, v in order]
    cmap = {int(v): f"COLOR_{i}" for i, v in enumerate(ordered_vals)}
    return cmap

def grid_hash(grid: Sequence[Sequence[int]]) -> str:
    """SHA256 hash for a numeric grid (row-major)."""
    arr = np.array(grid, dtype=np.int32)
    return hashlib.sha256(arr.tobytes()).hexdigest()

# --------------------------
# Structured text format
# --------------------------
def grid_to_structured_text(input_grid: Sequence[Sequence[int]],
                            output_grid: Optional[Sequence[Sequence[int]]] = None,
                            include_delta: bool = False,
                            normalize_colors: bool = True,
                            include_size: bool = True) -> str:
    """
    Convert a numeric grid into structured token text.
    - input_grid: H x W numeric grid (list of lists)
    - output_grid: optional H x W grid (target) — if provided, appended in [OUTPUT] block
    - include_delta: append per-cell DELTA tokens if output_grid provided
    - normalize_colors: map raw integer colors to canonical COLOR_x tokens
    Returns a single string that is stable and suitable for HF tokenizers.
    """
    H = len(input_grid)
    W = len(input_grid[0]) if H > 0 else 0
    cmap = canonical_color_map(input_grid) if normalize_colors else None

    def tok_for(r: int, c: int, val: int) -> str:
        if cmap is not None:
            color_tok = cmap.get(int(val), f"COLOR_{val}")
        else:
            color_tok = f"COLOR_{val}"
        return f"R{r}C{c}_{color_tok}"

    parts = []
    parts.append("[START_GRID]")
    if include_size:
        parts.append(f"[SIZE]_{H}x{W}")

    # row-major tokens: use explicit row separators
    for r in range(H):
        row_tokens = [tok_for(r, c, input_grid[r][c]) for c in range(W)]
        parts.append("[ROW]" + " " + " ".join(row_tokens))

    parts.append("[END_GRID]")

    if output_grid is not None:
        parts.append("[SEP]")
        parts.append("[OUTPUT]")
        for r in range(H):
            row_tokens = [tok_for(r, c, output_grid[r][c]) for c in range(W)]
            parts.append("[ROW]" + " " + " ".join(row_tokens))
        parts.append("[END_OUTPUT]")

    if include_delta and output_grid is not None:
        parts.append("[DELTA]")
        for r in range(H):
            for c in range(W):
                inv = int(input_grid[r][c])
                outv = int(output_grid[r][c])
                if inv == outv:
                    parts.append(f"NOP_R{r}C{c}")
                else:
                    parts.append(f"DELTA_R{r}C{c}_FROM_{inv}_TO_{outv}")
        parts.append("[END_DELTA]")

    return " ".join(parts)

# --------------------------
# Special tokens handling: add to tokenizer (Hugging Face)
# --------------------------
def build_arc_special_tokens(max_h=30, max_w=30, max_color_tokens=50, include_delta_tokens=True):
    """
    Create a list of special tokens we recommend adding to the tokenizer.
    - max_h, max_w: expected maximum grid sizes (for row/col tokens)
    - max_color_tokens: how many canonical color tokens to add (COLOR_0..COLOR_N)
    """
    toks = set()
    # structural tokens
    toks.update(["[START_GRID]", "[END_GRID]", "[ROW]", "[SIZE]", "[SEP]", "[OUTPUT]", "[END_OUTPUT]", "[DELTA]", "[END_DELTA]"])
    # row/col tokens
    for r in range(max_h):
        toks.add(f"[R{r}]")  # optional; we also use inline R{r}C{c}_ tokens
    for c in range(max_w):
        toks.add(f"[C{c}]")
    # color tokens (basic)
    for i in range(max_color_tokens):
        toks.add(f"COLOR_{i}")
    # explicit cell tokens are many (R{r}C{c}_COLOR_x); we don't pre-add all but the base pieces help
    if include_delta_tokens:
        toks.add("NOP")
        toks.add("DELTA")
    return sorted(list(toks))

def add_arc_special_tokens(tokenizer, model=None, max_h=30, max_w=30, max_color_tokens=20, include_delta_tokens=True):
    """
    Add structured ARC tokens to the HF tokenizer and resize model embeddings if provided.
    Returns: dict with { 'num_added': N }
    CAUTION: adding many tokens increases model.embedding size — do this once before heavy fine-tuning.
    """
    new_tokens = build_arc_special_tokens(max_h=max_h, max_w=max_w, max_color_tokens=max_color_tokens, include_delta_tokens=include_delta_tokens)
    # Only add tokens that are not present
    to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if len(to_add) == 0:
        return {'num_added':0}
    tokenizer.add_tokens(to_add)
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
    return {'num_added': len(to_add), 'tokens_added': to_add}

# --------------------------
# Encoding helpers - batch-friendly
# --------------------------
def encode_arc_pair_batch(tokenizer, pairs: List[Tuple[str, Optional[str]]], max_length: int = 512, return_tensors: str = 'pt', padding: str='longest'):
    """
    Encode a batch of (structured_input_text, structured_output_text or None) pairs.
    pairs: list of (input_text, output_text_or_None)
    Returns: dict with input_ids, attention_mask, optionally labels (for supervised)
    If output_text provided, labels = token ids for output placed after SEP (useful for teacher forcing / seq2seq)
    Note: this function treats the combined string as a single sequence; for encoder-decoder models adapt accordingly.
    """
    texts = []
    labels = []
    for inp, out in pairs:
        if out is None:
            texts.append(inp)
            labels.append(None)
        else:
            # create combined sequence input with output appended (model conditioned on whole sequence)
            combined = inp + " [SEP] " + out
            texts.append(combined)
            # label mask: we will create labels that are -100 for input part and token ids for output part
            labels.append(out)

    enc = tokenizer(texts, return_tensors=return_tensors, truncation=True, max_length=max_length, padding=padding)
    result = dict(enc)

    # If labels were requested, create labels aligned to tokens (masking input tokens)
    if any(l is not None for l in labels):
        # Tokenize inputs and outputs separately to find boundary indices
        label_ids = []
        for i, (inp, out) in enumerate(pairs):
            if out is None:
                label_ids.append([-100] * enc['input_ids'].shape[1])
                continue
            # token lengths of inp + sep
            inp_enc = tokenizer(inp + " [SEP]", return_tensors='pt', truncation=True, max_length=max_length)
            inp_len = inp_enc['input_ids'].shape[1]
            out_enc = tokenizer(out, return_tensors='pt', truncation=True, max_length=max_length)
            out_ids = out_enc['input_ids'][0].tolist()
            # create -100 pad for input prefix
            lbl = [-100] * inp_len
            lbl.extend(out_ids)
            # pad / truncate to enc length
            if len(lbl) < enc['input_ids'].shape[1]:
                lbl = lbl + [-100] * (enc['input_ids'].shape[1] - len(lbl))
            else:
                lbl = lbl[:enc['input_ids'].shape[1]]
            label_ids.append(lbl)
        import torch
        result['labels'] = torch.tensor(label_ids, dtype=torch.long)
    return result

# --------------------------
# Best-effort parsing to grid (debugging)
# --------------------------
def structured_text_to_grid(text: str) -> List[List[int]]:
    """
    Best-effort attempt to parse structured text back into a numeric grid.
    This function expects tokens like R{r}C{c}_COLOR_x and extracts numeric color x if possible.
    Returns a list of rows or raises ValueError if parsing fails.
    """
    # find [SIZE]_HxW first if present
    size_match = re.search(r"\[SIZE\]_(\d+)x(\d+)", text)
    if size_match:
        H = int(size_match.group(1)); W = int(size_match.group(2))
    else:
        # fallback: find the first [ROW] occurrences and infer width by tokens
        rows = re.split(r"\[ROW\]", text)
        rows = [r.strip() for r in rows if r.strip()]
        if not rows:
            raise ValueError("Can't parse grid: no [ROW] markers")
        grid = []
        for r in rows:
            toks = r.split()
            row_vals = []
            for tok in toks:
                m = re.search(r"COLOR_(\d+)", tok)
                if m:
                    row_vals.append(int(m.group(1)))
                else:
                    # unknown token; put -1
                    row_vals.append(-1)
            grid.append(row_vals)
        return grid

    # find the block between [START_GRID] and [END_GRID]
    m = re.search(r"\[START_GRID\](.*?)\[END_GRID\]", text, flags=re.S)
    if not m:
        raise ValueError("No grid block found")
    block = m.group(1)
    rows = [r.strip() for r in block.split("[ROW]") if r.strip()]
    grid = []
    for r in rows:
        toks = r.split()
        row_vals = []
        for tok in toks:
            m = re.search(r"COLOR_(\d+)", tok)
            if m:
                row_vals.append(int(m.group(1)))
            else:
                row_vals.append(-1)
        grid.append(row_vals)
    # enforce shape HxW by trimming/padding
    out = []
    for r in grid[:H]:
        row = r[:W] + [-1] * max(0, W - len(r))
        out.append(row)
    while len(out) < H:
        out.append([-1] * W)
    return out
