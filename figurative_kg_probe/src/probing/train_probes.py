"""
src/probing/train_probes.py

Trains a linear probe (logistic regression) at each transformer layer
to classify representations as figurative vs. literal.

For each model × layer combination, we:
  1. Extract the (N, hidden_dim) matrix at that layer
  2. Train a 5-fold cross-validated logistic regression
  3. Record mean accuracy (= probe accuracy at this layer)

The result is a "figurativeness curve" — probe accuracy as a function
of layer depth — for each model and figurative type.

Outputs (outputs/curves/):
    {model_key}_curves.json — full curve data

Usage:
    python src/probing/train_probes.py --model bert
    python src/probing/train_probes.py --model all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import track
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    MODELS, MODEL_LAYERS, CURVES_DIR, RESULTS_DIR,
    PROBE_TYPE, PROBE_MAX_ITER, PROBE_C, PROBE_SOLVER,
    PROBE_CV_FOLDS, TRANSITION_THRESHOLD, SEED,
    FIGURATIVE_TYPES,
)
from src.probing.extract_hidden_states import load_hidden_states

console = Console()


# ─────────────────────────────────────────────
# Probe
# ─────────────────────────────────────────────

def make_probe() -> LogisticRegression:
    """Construct the linear probe (logistic regression classifier)."""
    return LogisticRegression(
        C=PROBE_C,
        max_iter=PROBE_MAX_ITER,
        solver=PROBE_SOLVER,
        random_state=SEED,
        class_weight="balanced",   # handle class imbalance
    )


def probe_at_layer(
    X: np.ndarray,   # (N, hidden_dim)
    y: np.ndarray,   # (N,) binary labels
    n_folds: int = PROBE_CV_FOLDS,
) -> dict:
    """
    Train a linear probe on (X, y) using stratified k-fold CV.

    Returns:
        {
            "mean_accuracy": float,
            "std_accuracy":  float,
            "per_fold":      list[float],
            "n_samples":     int,
            "n_positive":    int,
        }
    """
    if len(np.unique(y)) < 2:
        return {
            "mean_accuracy": float(np.mean(y == y[0])),
            "std_accuracy":  0.0,
            "per_fold":      [],
            "n_samples":     len(y),
            "n_positive":    int(y.sum()),
        }

    # Standardize features per-layer (important for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=min(n_folds, len(y) // 2), shuffle=True, random_state=SEED)
    probe = make_probe()

    scores = cross_val_score(probe, X_scaled, y, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy":  float(scores.std()),
        "per_fold":      scores.tolist(),
        "n_samples":     len(y),
        "n_positive":    int(y.sum()),
    }


# ─────────────────────────────────────────────
# Transition layer detection
# ─────────────────────────────────────────────

def find_transition_layer(
    layer_accuracies: list[float],
    threshold: float = TRANSITION_THRESHOLD,
) -> int:
    """
    Find T*: the first layer at which probe accuracy exceeds `threshold`.
    If no layer reaches the threshold, returns the layer with maximum accuracy.
    Falls back to the last layer if accuracies list is empty.
    """
    for i, acc in enumerate(layer_accuracies):
        if acc >= threshold:
            return i

    # Fallback: layer with max accuracy
    if layer_accuracies:
        return int(np.argmax(layer_accuracies))
    return len(layer_accuracies) - 1


def compute_transition_slope(layer_accuracies: list[float]) -> list[float]:
    """
    Compute the slope (finite difference) of the figurativeness curve.
    Returns a list of length (n_layers - 1).
    """
    return [
        layer_accuracies[i + 1] - layer_accuracies[i]
        for i in range(len(layer_accuracies) - 1)
    ]


# ─────────────────────────────────────────────
# Full probe training for one model
# ─────────────────────────────────────────────

def train_probes_for_model(model_key: str) -> dict:
    """
    Train probes at every layer for a given model.

    Returns a nested dict:
    {
        "model":   str,
        "n_layers": int,
        "overall": {
            "layer_accuracies":  list[float],    # mean over all types
            "layer_std":         list[float],
            "transition_layer":  int,
            "transition_slope":  list[float],
        },
        "by_type": {
            "idiom":    { same structure as overall },
            "metaphor": { ... },
            "sarcasm":  { ... },
        },
        "by_layer": [                            # one entry per layer
            {
                "layer": int,
                "overall_accuracy": float,
                "by_type": { "idiom": float, ... }
            },
            ...
        ]
    }
    """
    console.print(f"\n[bold cyan]Training probes: {model_key}[/bold cyan]")

    data = load_hidden_states(model_key)
    ids    = data["ids"]
    labels = data["labels"]
    types  = data["types"]
    hs     = data["hidden_states"]   # (N, n_layers+1, hidden_dim)

    n_total_layers = hs.shape[1]   # includes embedding (layer 0)
    n_layers       = MODEL_LAYERS[model_key]

    results_by_type: dict[str, dict] = {}

    # ── Per-type probing ───────────────────────────────────────────
    for ftype in FIGURATIVE_TYPES:
        # Select indices for this figurative type
        mask = np.array([t == ftype for t in types])
        if mask.sum() < 4:
            console.print(f"  [yellow]⚠ Not enough {ftype} samples. Skipping.[/yellow]")
            continue

        X_type = hs[mask]        # (N_type, n_layers+1, hidden_dim)
        y_type = labels[mask]    # (N_type,)

        layer_accs  = []
        layer_stds  = []

        for layer_idx in range(n_total_layers):
            X_layer = X_type[:, layer_idx, :]   # (N_type, hidden_dim)
            layer_result = probe_at_layer(X_layer, y_type)
            layer_accs.append(layer_result["mean_accuracy"])
            layer_stds.append(layer_result["std_accuracy"])

        t_star = find_transition_layer(layer_accs[1:])  # skip embedding layer (idx 0)
        t_star += 1   # offset back (we skipped idx 0)
        slope  = compute_transition_slope(layer_accs)

        results_by_type[ftype] = {
            "layer_accuracies": layer_accs,
            "layer_std":        layer_stds,
            "transition_layer": t_star,
            "transition_slope": slope,
            "n_samples":        int(mask.sum()),
        }
        console.print(
            f"  {ftype:10s} | T* = layer {t_star:2d} | "
            f"max acc = {max(layer_accs):.3f}"
        )

    # ── Overall (all types combined) ────────────────────────────────
    layer_accs_all  = []
    layer_stds_all  = []

    for layer_idx in track(range(n_total_layers), description="  Overall layers"):
        X_layer = hs[:, layer_idx, :]
        layer_result = probe_at_layer(X_layer, labels)
        layer_accs_all.append(layer_result["mean_accuracy"])
        layer_stds_all.append(layer_result["std_accuracy"])

    t_star_all = find_transition_layer(layer_accs_all[1:]) + 1
    slope_all  = compute_transition_slope(layer_accs_all)

    overall = {
        "layer_accuracies": layer_accs_all,
        "layer_std":        layer_stds_all,
        "transition_layer": t_star_all,
        "transition_slope": slope_all,
        "n_samples":        len(labels),
    }

    console.print(
        f"  {'overall':10s} | T* = layer {t_star_all:2d} | "
        f"max acc = {max(layer_accs_all):.3f}"
    )

    # ── Per-layer summary ────────────────────────────────────────────
    by_layer = []
    for layer_idx in range(n_total_layers):
        entry = {
            "layer":            layer_idx,
            "overall_accuracy": layer_accs_all[layer_idx],
            "by_type":          {
                ftype: results_by_type[ftype]["layer_accuracies"][layer_idx]
                for ftype in results_by_type
            },
        }
        by_layer.append(entry)

    # ── Assemble output ───────────────────────────────────────────────
    output = {
        "model":    model_key,
        "n_layers": n_layers,
        "overall":  overall,
        "by_type":  results_by_type,
        "by_layer": by_layer,
    }

    # Save
    out_path = CURVES_DIR / f"{model_key}_curves.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"[green]✓[/green] Curves saved → {out_path}")

    return output


def load_curves(model_key: str) -> dict:
    """Load saved probe curves for a model."""
    path = CURVES_DIR / f"{model_key}_curves.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Curves not found for {model_key}: {path}\n"
            f"Run: python src/probing/train_probes.py --model {model_key}"
        )
    with open(path) as f:
        return json.load(f)


def load_all_curves() -> dict[str, dict]:
    """Load all available model curves."""
    curves = {}
    for key in MODELS:
        try:
            curves[key] = load_curves(key)
        except FileNotFoundError:
            pass
    return curves


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train linear probes at each transformer layer")
    parser.add_argument(
        "--model", default="bert",
        choices=list(MODELS.keys()) + ["all"],
        help="Model to probe (or 'all' for all models)",
    )
    args = parser.parse_args()

    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]

    all_results = {}
    for key in model_keys:
        console.rule(f"[bold]Model: {key}[/bold]")
        all_results[key] = train_probes_for_model(key)

    # Summary table
    from rich.table import Table
    table = Table(title="Probe Summary — Transition Layers (T*)", show_lines=True)
    table.add_column("Model",    style="bold")
    table.add_column("Overall",  justify="center")
    table.add_column("Idiom",    justify="center")
    table.add_column("Metaphor", justify="center")
    table.add_column("Sarcasm",  justify="center")
    table.add_column("Max Acc",  justify="center", style="green")

    for key, r in all_results.items():
        bt = r.get("by_type", {})
        table.add_row(
            key,
            str(r["overall"]["transition_layer"]),
            str(bt.get("idiom",    {}).get("transition_layer", "—")),
            str(bt.get("metaphor", {}).get("transition_layer", "—")),
            str(bt.get("sarcasm",  {}).get("transition_layer", "—")),
            f"{max(r['overall']['layer_accuracies']):.3f}",
        )

    console.print(table)
    console.rule()
    console.print("[bold green]Probe training complete.[/bold green]")


if __name__ == "__main__":
    main()
