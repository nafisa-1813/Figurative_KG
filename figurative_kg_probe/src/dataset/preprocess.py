"""
src/dataset/preprocess.py

Loads raw JSONL files for all three figurative types, applies
cleaning, deduplication, balanced sampling, and saves a unified
processed dataset to data/processed/expressions.jsonl.

Each output record has the schema:
    {
      "id":              str,          # unique identifier
      "expression":      str,          # the figurative expression / span
      "sentence":        str,          # full sentence context
      "label":           str,          # "figurative" | "literal"
      "figurative_type": str,          # "idiom" | "metaphor" | "sarcasm"
      "span_start":      int,          # char offset of expression in sentence
      "span_end":        int,          # char offset end
      "source":          str,          # dataset provenance
      "literal_partner": str | null,   # matched literal sentence (same expression)
    }

Run directly:
    python src/dataset/preprocess.py
"""

import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RAW_DIR, PROCESSED_DIR, TARGET_PER_TYPE, SEED

console = Console()
random.seed(SEED)


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_raw(figurative_type: str) -> list[dict]:
    """Load raw records for a figurative type. Returns list of dicts."""
    filename_map = {
        "idiom":    "magpie.jsonl",
        "metaphor": "vua_metaphors.jsonl",
        "sarcasm":  "semeval_sarcasm.jsonl",
    }
    path = RAW_DIR / filename_map[figurative_type]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {path}\n"
            f"Run: python src/dataset/download.py"
        )
    return list(_iter_jsonl(path))


# ─────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Normalise whitespace, remove control characters, strip leading/trailing
    whitespace. Does NOT lowercase — we preserve case for the model tokeniser.
    """
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_valid(record: dict) -> bool:
    """Return True if the record passes basic quality filters."""
    sentence = record.get("sentence", "")
    if not sentence or len(sentence.split()) < 4:
        return False
    if len(sentence) > 512:
        return False
    label = record.get("label", "")
    if label not in ("figurative", "literal"):
        return False
    return True


def clean_records(records: list[dict]) -> list[dict]:
    """Apply cleaning to a list of raw records."""
    cleaned = []
    for r in records:
        r["sentence"]   = _clean_text(r.get("sentence", ""))
        r["expression"] = _clean_text(r.get("expression", ""))
        if _is_valid(r):
            cleaned.append(r)
    return cleaned


# ─────────────────────────────────────────────
# Span detection
# ─────────────────────────────────────────────

def _find_span(sentence: str, expression: str) -> tuple[int, int]:
    """
    Find the character span of `expression` inside `sentence`.
    Case-insensitive search. Returns (start, end) or (-1, -1) if not found.
    """
    if not expression:
        return (0, len(sentence))  # sarcasm: whole sentence is the span

    pattern = re.compile(re.escape(expression), re.IGNORECASE)
    match = pattern.search(sentence)
    if match:
        return match.start(), match.end()
    return -1, -1


def add_spans(records: list[dict]) -> list[dict]:
    """Annotate each record with span_start / span_end of the expression."""
    result = []
    for r in records:
        start, end = _find_span(r["sentence"], r["expression"])
        r["span_start"] = start
        r["span_end"]   = end
        result.append(r)
    return result


# ─────────────────────────────────────────────
# Pairing figurative ↔ literal
# ─────────────────────────────────────────────

def pair_figurative_literal(records: list[dict]) -> list[dict]:
    """
    For each figurative sentence, find its literal partner (same expression).
    Adds a `literal_partner` field to figurative records.
    Literal records without a figurative partner are dropped.
    """
    by_expression: dict[str, dict] = defaultdict(lambda: {"figurative": [], "literal": []})

    for r in records:
        key = r["expression"].lower().strip() if r["expression"] else r["sentence"][:40]
        by_expression[key][r["label"]].append(r["sentence"])

    paired = []
    for r in records:
        key = r["expression"].lower().strip() if r["expression"] else r["sentence"][:40]
        partners = by_expression[key]

        if r["label"] == "figurative":
            lit = partners["literal"]
            r["literal_partner"] = lit[0] if lit else None
            paired.append(r)
        # literal records are included only as partners, not standalone

    return paired


# ─────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────

def _record_hash(r: dict) -> str:
    key = f"{r['expression'].lower()}|{r['sentence'].lower()}"
    return hashlib.md5(key.encode()).hexdigest()


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove exact-duplicate (expression, sentence) pairs."""
    seen: set[str] = set()
    unique = []
    for r in records:
        h = _record_hash(r)
        if h not in seen:
            seen.add(h)
            unique.append(r)
    return unique


# ─────────────────────────────────────────────
# Balanced sampling
# ─────────────────────────────────────────────

def sample(records: list[dict], n: int, figurative_type: str) -> list[dict]:
    """
    Randomly sample up to `n` records. If fewer than `n` are available,
    return all (with a warning). Prefers records with a literal_partner.
    """
    # Prefer paired records
    paired   = [r for r in records if r.get("literal_partner")]
    unpaired = [r for r in records if not r.get("literal_partner")]

    pool = paired + unpaired
    if len(pool) < n:
        console.print(
            f"[yellow]⚠ {figurative_type}: only {len(pool)} records available "
            f"(target {n}). Using all.[/yellow]"
        )
        return pool

    return random.sample(pool, n)


# ─────────────────────────────────────────────
# ID assignment
# ─────────────────────────────────────────────

def assign_ids(records: list[dict]) -> list[dict]:
    """Assign a stable unique ID to each record."""
    for i, r in enumerate(records):
        r["id"] = f"{r['figurative_type']}_{i:04d}"
    return records


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def preprocess_all() -> Path:
    """
    Full preprocessing pipeline for all three figurative types.
    Saves unified dataset to data/processed/expressions.jsonl.
    Returns the output path.
    """
    console.rule("[bold]Dataset Preprocessing[/bold]")
    all_records = []
    stats: dict[str, dict] = {}

    for ftype in ["idiom", "metaphor", "sarcasm"]:
        console.print(f"\n[bold cyan]Processing: {ftype}[/bold cyan]")

        raw      = load_raw(ftype)
        cleaned  = clean_records(raw)
        spanned  = add_spans(cleaned)
        paired   = pair_figurative_literal(spanned)
        unique   = deduplicate(paired)
        sampled  = sample(unique, TARGET_PER_TYPE[ftype], ftype)

        stats[ftype] = {
            "raw":      len(raw),
            "cleaned":  len(cleaned),
            "paired":   len(paired),
            "unique":   len(unique),
            "sampled":  len(sampled),
        }
        all_records.extend(sampled)

    # Assign IDs across the unified pool
    all_records = assign_ids(all_records)

    # Save
    out_path = PROCESSED_DIR / "expressions.jsonl"
    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # Report
    table = Table(title="Preprocessing Summary", show_lines=True)
    table.add_column("Type",    style="bold")
    table.add_column("Raw",     justify="right")
    table.add_column("Cleaned", justify="right")
    table.add_column("Paired",  justify="right")
    table.add_column("Unique",  justify="right")
    table.add_column("Final",   justify="right", style="green")

    for ftype, s in stats.items():
        table.add_row(
            ftype,
            str(s["raw"]),
            str(s["cleaned"]),
            str(s["paired"]),
            str(s["unique"]),
            str(s["sampled"]),
        )

    console.print(table)
    console.print(f"\n[bold green]Total expressions: {len(all_records)}[/bold green]")
    console.print(f"Saved → {out_path}")
    return out_path


def load_processed() -> list[dict]:
    """Load the processed expressions dataset. Run preprocess_all() first."""
    path = PROCESSED_DIR / "expressions.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\n"
            f"Run: python src/dataset/preprocess.py"
        )
    return list(_iter_jsonl(path))


if __name__ == "__main__":
    preprocess_all()
