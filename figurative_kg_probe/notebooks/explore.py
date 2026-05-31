"""
notebooks/explore.py

Exploratory analysis notebook (runnable as a plain Python script).
Useful for interactive inspection in IPython / Jupyter by running
individual cells, or as a standalone script for a quick sanity check.

Usage:
    python notebooks/explore.py
    # or in Jupyter: jupytext --to notebook explore.py
"""

# %% [markdown]
# # Figurative Language KG Probe — Exploration Notebook
# %%

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# %%
# ── 1. Inspect processed dataset ──────────────────────────────────────────

from src.dataset.preprocess import load_processed

try:
    records = load_processed()
    print(f"Total records: {len(records)}")

    from collections import Counter
    type_counts = Counter(r["figurative_type"] for r in records)
    print(f"\nBy type: {dict(type_counts)}")

    label_counts = Counter(r["label"] for r in records)
    print(f"By label: {dict(label_counts)}")

    # Show a sample record
    print("\nSample record:")
    print(json.dumps(records[0], indent=2))
except FileNotFoundError as e:
    print(f"⚠ {e}")

# %%
# ── 2. Inspect KG ─────────────────────────────────────────────────────────

from src.kg.build_kg import FigurativeKG

try:
    kg = FigurativeKG.load()
    G  = kg.G

    print(f"\nKG stats:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Node type distribution
    node_types = [d.get("node_type", "unknown") for _, d in G.nodes(data=True)]
    from collections import Counter
    print(f"\n  Node types: {dict(Counter(node_types))}")

    # Show example idiom neighbourhood
    expr_node = "expr:kick the bucket"
    if expr_node in G:
        neighbours = list(G.neighbors(expr_node))
        print(f"\n  Neighbours of '{expr_node}':")
        for n in neighbours[:8]:
            rel = G.edges[expr_node, n].get("relation", "?")
            lbl = G.nodes[n].get("label", n)
            print(f"    --[{rel}]--> {lbl}")
except FileNotFoundError as e:
    print(f"⚠ {e}")

# %%
# ── 3. Inspect FCD scores ──────────────────────────────────────────────────

from src.kg.build_kg import load_fcd_scores

try:
    fcd_scores = load_fcd_scores()
    scores = list(fcd_scores.values())

    print(f"\nFCD score stats:")
    print(f"  Count:  {len(scores)}")
    print(f"  Min:    {min(scores):.2f}")
    print(f"  Max:    {max(scores):.2f}")
    print(f"  Mean:   {np.mean(scores):.2f}")
    print(f"  Median: {np.median(scores):.2f}")
    print(f"  Std:    {np.std(scores):.2f}")

    # Quick histogram
    plt.figure(figsize=(6, 3))
    plt.hist(scores, bins=15, color="#3D405B", edgecolor="white", alpha=0.85)
    plt.xlabel("FCD Score")
    plt.ylabel("Count")
    plt.title("Distribution of Figurative Conceptual Distance (FCD)")
    plt.tight_layout()
    plt.savefig("outputs/plots/explore_fcd_hist.png", dpi=120)
    plt.show()
    print("  Histogram saved.")
except FileNotFoundError as e:
    print(f"⚠ {e}")

# %%
# ── 4. Inspect probe curves ────────────────────────────────────────────────

from src.probing.train_probes import load_all_curves

try:
    curves = load_all_curves()
    print(f"\nAvailable model curves: {list(curves.keys())}")

    for model_key, data in curves.items():
        t_star   = data["overall"]["transition_layer"]
        max_acc  = max(data["overall"]["layer_accuracies"])
        print(f"  {model_key:12s}: T* = {t_star:2d}, max accuracy = {max_acc:.3f}")
except FileNotFoundError as e:
    print(f"⚠ {e}")

# %%
# ── 5. Quick correlation check ─────────────────────────────────────────────

from config import RESULTS_DIR

results_path = RESULTS_DIR / "correlation_results.json"
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    corr = results.get("correlations", {}).get("overall", {})
    print(f"\nOverall FCD ↔ T* Spearman ρ = {corr.get('coefficient', '?')}")
    print(f"p-value = {corr.get('pvalue', '?')}")
    print(f"Significant: {corr.get('significant', '?')}")

    by_type = results.get("correlations", {}).get("by_type", {})
    for ftype, r in by_type.items():
        print(f"  {ftype:10s}: ρ = {r.get('coefficient', '?')}, p = {r.get('pvalue', '?')}")
else:
    print("⚠ Run correlation analysis first (src/analysis/correlate_fcd_transition.py)")

# %% [markdown]
# ---
# End of exploratory notebook.
# Run `python src/visualization/plot_all.py` to generate all paper figures.
