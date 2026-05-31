"""
src/probing/extract_hidden_states.py

Extracts per-layer hidden states from transformer models for each
expression in the processed dataset. For each expression we produce
two representations:
  - figurative sentence hidden states
  - literal control sentence hidden states (if available)

States are saved as numpy .npz files to outputs/hidden_states/.

Usage:
    python src/probing/extract_hidden_states.py --model bert
    python src/probing/extract_hidden_states.py --model all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    MODELS, MODEL_LAYERS, PROCESSED_DIR, HIDDEN_STATES_DIR,
    EXTRACTION_BATCH_SIZE, POOLING_STRATEGY, SEED,
)

console = Console()
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model_and_tokenizer(model_key: str):
    """
    Load a HuggingFace model + tokenizer by short key (e.g. 'bert').
    Sets output_hidden_states=True automatically.
    Returns (model, tokenizer, n_layers).
    """
    model_id = MODELS[model_key]
    console.print(f"[cyan]Loading {model_id}...[/cyan]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
    model.eval()

    # Some tokenizers (GPT-2) don't have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = MODEL_LAYERS[model_key]
    console.print(f"  ✓ {model_id} loaded ({n_layers} layers, CPU mode)")
    return model, tokenizer, n_layers


# ─────────────────────────────────────────────
# Span token identification
# ─────────────────────────────────────────────

def get_span_token_indices(
    tokenizer,
    sentence: str,
    span_start: int,
    span_end: int,
) -> list[int]:
    """
    Return the token indices (within the tokenized sentence) that correspond
    to the character span [span_start, span_end].

    For sarcasm (span_start=0, span_end=len(sentence)), returns all tokens.
    Falls back to all tokens if span is not found.
    """
    encoding = tokenizer(sentence, return_offsets_mapping=True, truncation=True, max_length=128)
    offsets  = encoding["offset_mapping"]  # list of (char_start, char_end)

    span_indices = []
    for i, (cs, ce) in enumerate(offsets):
        if cs == 0 and ce == 0:
            continue  # special tokens ([CLS], [SEP], <|endoftext|>)
        # Token overlaps with the target span
        if cs < span_end and ce > span_start:
            span_indices.append(i)

    return span_indices if span_indices else list(range(1, len(offsets) - 1))


# ─────────────────────────────────────────────
# Hidden state pooling
# ─────────────────────────────────────────────

def pool_span(
    hidden_state: np.ndarray,   # shape: (seq_len, hidden_dim)
    span_indices: list[int],
    strategy: str = POOLING_STRATEGY,
) -> np.ndarray:
    """
    Pool the hidden state over span token indices.

    Strategies:
        mean  — average all span token vectors (default)
        first — first span token
        last  — last span token
        max   — element-wise max over span tokens
    """
    if not span_indices:
        span_indices = [0]

    span_vecs = hidden_state[span_indices]  # (n_span_tokens, hidden_dim)

    if strategy == "mean":
        return span_vecs.mean(axis=0)
    elif strategy == "first":
        return span_vecs[0]
    elif strategy == "last":
        return span_vecs[-1]
    elif strategy == "max":
        return span_vecs.max(axis=0)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


# ─────────────────────────────────────────────
# Single-sentence extraction
# ─────────────────────────────────────────────

def extract_for_sentence(
    model,
    tokenizer,
    sentence: str,
    span_start: int,
    span_end: int,
    n_layers: int,
) -> np.ndarray:
    """
    Forward pass through the model and extract per-layer span representations.

    Returns array of shape (n_layers + 1, hidden_dim):
        - index 0: embedding layer
        - index k: output of transformer layer k
    """
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False,
    )

    span_indices = get_span_token_indices(tokenizer, sentence, span_start, span_end)

    with torch.no_grad():
        outputs = model(**inputs)

    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states  # always available (output_hidden_states=True)

    layer_reps = []
    for hs in hidden_states:
        hs_np  = hs.squeeze(0).numpy()   # (seq_len, hidden_dim)
        pooled = pool_span(hs_np, span_indices)
        layer_reps.append(pooled)

    return np.stack(layer_reps)   # (n_layers+1, hidden_dim)


# ─────────────────────────────────────────────
# Full dataset extraction
# ─────────────────────────────────────────────

def extract_for_model(model_key: str, records: list[dict]) -> Path:
    """
    Extract hidden states for all records using the given model.
    Saves results to outputs/hidden_states/{model_key}.npz

    The .npz file has the following arrays:
        ids          — (N,) string array of record IDs
        labels       — (N,) int array (1=figurative, 0=literal control)
        types        — (N,) string array of figurative types
        hidden_states — (N, n_layers+1, hidden_dim) float32 array

    We extract both the figurative sentence AND the literal partner (if present)
    so the probe can be trained with contrastive pairs.
    """
    model, tokenizer, n_layers = load_model_and_tokenizer(model_key)

    out_path = HIDDEN_STATES_DIR / f"{model_key}.npz"
    if out_path.exists():
        console.print(f"[green]✓[/green] Hidden states already extracted → {out_path}")
        return out_path

    all_ids:    list[str]       = []
    all_labels: list[int]       = []
    all_types:  list[str]       = []
    all_hs:     list[np.ndarray] = []

    for record in track(records, description=f"Extracting [{model_key}]"):
        # ── Figurative sentence ────────────────────────────────────
        try:
            hs_fig = extract_for_sentence(
                model, tokenizer,
                record["sentence"],
                record.get("span_start", 0),
                record.get("span_end",   len(record["sentence"])),
                n_layers,
            )
            all_ids.append(record["id"])
            all_labels.append(1)
            all_types.append(record["figurative_type"])
            all_hs.append(hs_fig)
        except Exception as e:
            console.print(f"[yellow]⚠ Skip {record['id']} (fig): {e}[/yellow]")

        # ── Literal partner ────────────────────────────────────────
        lit_sentence = record.get("literal_partner")
        if lit_sentence:
            expr = record.get("expression", "")
            lit_start = lit_sentence.lower().find(expr.lower()) if expr else 0
            lit_end   = lit_start + len(expr) if lit_start >= 0 else len(lit_sentence)

            try:
                hs_lit = extract_for_sentence(
                    model, tokenizer,
                    lit_sentence,
                    max(0, lit_start),
                    max(1, lit_end),
                    n_layers,
                )
                all_ids.append(f"{record['id']}_literal")
                all_labels.append(0)
                all_types.append(record["figurative_type"])
                all_hs.append(hs_lit)
            except Exception as e:
                console.print(f"[yellow]⚠ Skip {record['id']} (lit): {e}[/yellow]")

    if not all_hs:
        raise RuntimeError(f"No hidden states extracted for model: {model_key}")

    hs_array = np.stack(all_hs).astype(np.float32)   # (N, n_layers+1, hidden_dim)

    np.savez(
        out_path,
        ids=np.array(all_ids),
        labels=np.array(all_labels, dtype=np.int32),
        types=np.array(all_types),
        hidden_states=hs_array,
    )

    console.print(
        f"[green]✓[/green] Saved {len(all_ids)} representations "
        f"(shape {hs_array.shape}) → {out_path}"
    )
    return out_path


def load_hidden_states(model_key: str) -> dict:
    """
    Load saved hidden states for a model.
    Returns dict with keys: ids, labels, types, hidden_states.
    """
    path = HIDDEN_STATES_DIR / f"{model_key}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Hidden states not found for {model_key}: {path}\n"
            f"Run: python src/probing/extract_hidden_states.py --model {model_key}"
        )
    data = np.load(path, allow_pickle=True)
    return {
        "ids":           data["ids"],
        "labels":        data["labels"],
        "types":         data["types"],
        "hidden_states": data["hidden_states"],
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract transformer hidden states")
    parser.add_argument(
        "--model", default="bert",
        choices=list(MODELS.keys()) + ["all"],
        help="Model to use (or 'all' for all models)",
    )
    args = parser.parse_args()

    # Load processed dataset
    processed_path = PROCESSED_DIR / "expressions.jsonl"
    if not processed_path.exists():
        console.print("[red]✗ Processed dataset not found.[/red]")
        console.print("Run: python src/dataset/preprocess.py")
        sys.exit(1)

    records = []
    with open(processed_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    console.print(f"Loaded {len(records)} expressions.")

    # Extract
    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    for key in model_keys:
        console.rule(f"[bold]Model: {key}[/bold]")
        extract_for_model(key, records)

    console.rule()
    console.print("[bold green]Hidden state extraction complete.[/bold green]")


if __name__ == "__main__":
    main()
