"""
run_pipeline.py

End-to-end runner for the Figurative Language KG Probe project.
Runs all stages in order:

  Stage 1: Download datasets
  Stage 2: Preprocess + curate
  Stage 3: Build KG + compute FCD scores
  Stage 4: Extract hidden states (per model)
  Stage 5: Train linear probes (per model)
  Stage 6: Correlation analysis
  Stage 7: Generate all plots

Usage:
    python run_pipeline.py                          # default: bert only
    python run_pipeline.py --models bert roberta    # specific models
    python run_pipeline.py --models all             # all 4 models
    python run_pipeline.py --skip-download          # if data already downloaded
    python run_pipeline.py --stages 4 5 6 7         # run specific stages only
"""

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# ── Stage runner ───────────────────────────────────────────────────────────

def run_stage(name: str, fn, *args, **kwargs):
    """Run a pipeline stage with timing and error handling."""
    console.print(Panel(f"[bold]{name}[/bold]", expand=False))
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        console.print(f"[green]✓[/green] Done in {elapsed:.1f}s\n")
        return result
    except Exception as e:
        console.print(f"[red]✗ FAILED: {e}[/red]")
        raise


# ── Stages ─────────────────────────────────────────────────────────────────

def stage_download():
    from src.dataset.download import download_all
    return download_all()


def stage_preprocess():
    from src.dataset.preprocess import preprocess_all
    return preprocess_all()


def stage_build_kg():
    import json
    from src.kg.build_kg import FigurativeKG, compute_all_fcd

    records = []
    with open("data/processed/expressions.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    kg = FigurativeKG()
    kg.build_from_dataset(records)
    kg.save()
    compute_all_fcd(records, kg)


def stage_extract(model_keys: list[str]):
    import json
    from src.probing.extract_hidden_states import extract_for_model

    records = []
    with open("data/processed/expressions.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    for key in model_keys:
        console.rule(f"Extracting: {key}")
        extract_for_model(key, records)


def stage_probe(model_keys: list[str]):
    from src.probing.train_probes import train_probes_for_model

    for key in model_keys:
        console.rule(f"Probing: {key}")
        train_probes_for_model(key)


def stage_analyze():
    from src.analysis.correlate_fcd_transition import main as analyze_main
    return analyze_main()


def stage_plot():
    from src.visualization.plot_all import plot_all
    plot_all()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full Figurative Language KG Probe pipeline"
    )
    parser.add_argument(
        "--models", nargs="+", default=["bert"],
        choices=["bert", "roberta", "gpt2", "distilbert", "all"],
        help="Models to run (default: bert)",
    )
    parser.add_argument(
        "--stages", nargs="+", type=int,
        default=list(range(1, 8)),
        help="Stages to run (1–7). Default: all.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Stage 1 (dataset download).",
    )
    args = parser.parse_args()

    # Resolve model keys
    from config import MODELS
    model_keys = list(MODELS.keys()) if "all" in args.models else args.models

    stages = set(args.stages)
    if args.skip_download:
        stages.discard(1)

    console.print(Panel(
        f"[bold cyan]Figurative Language KG Probe — Pipeline[/bold cyan]\n"
        f"Models: {model_keys}\n"
        f"Stages: {sorted(stages)}",
        title="[bold]Starting[/bold]",
    ))

    t_total = time.time()

    if 1 in stages:
        run_stage("Stage 1 — Download Datasets", stage_download)

    if 2 in stages:
        run_stage("Stage 2 — Preprocess & Curate", stage_preprocess)

    if 3 in stages:
        run_stage("Stage 3 — Build KG & Compute FCD", stage_build_kg)

    if 4 in stages:
        run_stage("Stage 4 — Extract Hidden States", stage_extract, model_keys)

    if 5 in stages:
        run_stage("Stage 5 — Train Linear Probes", stage_probe, model_keys)

    if 6 in stages:
        run_stage("Stage 6 — Correlation Analysis", stage_analyze)

    if 7 in stages:
        run_stage("Stage 7 — Generate Plots", stage_plot)

    elapsed = time.time() - t_total
    console.print(Panel(
        f"[bold green]Pipeline complete in {elapsed:.1f}s[/bold green]\n"
        f"Results   → outputs/results/\n"
        f"Plots     → outputs/plots/\n"
        f"Curves    → outputs/curves/",
        title="[bold green]Done[/bold green]",
    ))


if __name__ == "__main__":
    main()
