"""
src/analysis/correlate_fcd_transition.py

Core analysis module: computes the Spearman correlation between
Figurative Conceptual Distance (FCD) and Transition Layer (T*)
across all expressions and models.

Also runs:
  - Per-type breakdown (idiom / metaphor / sarcasm)
  - Cross-model T* comparison
  - Error analysis (expressions that never reach threshold)

Outputs (outputs/results/):
    correlation_results.json   — all correlation statistics
    cross_model_comparison.json — T* by model and type
    error_analysis.json        — failed / extreme expressions

Usage:
    python src/analysis/correlate_fcd_transition.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    MODELS, RESULTS_DIR, FIGURATIVE_TYPES,
    CORRELATION_METHOD, ALPHA,
)
from src.kg.build_kg import load_fcd_scores
from src.probing.train_probes import load_all_curves

console = Console()


# ─────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────

def build_analysis_df(
    fcd_scores: dict[str, float],
    curves: dict[str, dict],
    records: list[dict],
) -> list[dict]:
    """
    Join FCD scores and T* values per expression across all models.

    Returns a list of dicts, one per (expression, model) pair:
    {
        id, expression, figurative_type, fcd,
        model, transition_layer, max_accuracy
    }
    """
    rows = []
    # Map record ID → record metadata
    id_to_record = {r["id"]: r for r in records}

    for model_key, curve_data in curves.items():
        by_type = curve_data.get("by_type", {})

        # We need per-expression T*, not just per-type average.
        # We proxy this using the per-type T* for now (since probing
        # is done at the type level). Future work: per-instance probing.
        for ftype in FIGURATIVE_TYPES:
            if ftype not in by_type:
                continue

            t_star   = by_type[ftype]["transition_layer"]
            max_acc  = max(by_type[ftype]["layer_accuracies"])

            # All expressions of this type get this model's T* for the type
            for rec_id, fcd in fcd_scores.items():
                record = id_to_record.get(rec_id)
                if record is None or record["figurative_type"] != ftype:
                    continue

                rows.append({
                    "id":               rec_id,
                    "expression":       record["expression"],
                    "figurative_type":  ftype,
                    "fcd":              fcd,
                    "model":            model_key,
                    "transition_layer": t_star,
                    "max_accuracy":     max_acc,
                })

    return rows


# ─────────────────────────────────────────────
# Correlation
# ─────────────────────────────────────────────

def correlate(x: list[float], y: list[float], method: str = CORRELATION_METHOD) -> dict:
    """
    Compute correlation between x and y.
    Returns {coefficient, pvalue, method, n, significant}.
    """
    if len(x) < 3:
        return {"coefficient": None, "pvalue": None, "method": method,
                "n": len(x), "significant": False}

    if method == "spearman":
        coef, pval = stats.spearmanr(x, y)
    elif method == "pearson":
        coef, pval = stats.pearsonr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "coefficient": round(float(coef), 4),
        "pvalue":      round(float(pval), 6),
        "method":      method,
        "n":           len(x),
        "significant": bool(pval < ALPHA),
    }


def run_correlations(rows: list[dict]) -> dict:
    """
    Run FCD ↔ T* correlations:
      - Overall (all types, all models)
      - Per-type
      - Per-model
      - Per-type × per-model

    Returns a nested dict of correlation results.
    """
    results = {}

    def get_fcd_tstar(subset):
        fcd    = [r["fcd"]              for r in subset]
        t_star = [r["transition_layer"] for r in subset]
        return fcd, t_star

    # Overall
    fcd, t_star = get_fcd_tstar(rows)
    results["overall"] = correlate(fcd, t_star)

    # Per type
    results["by_type"] = {}
    for ftype in FIGURATIVE_TYPES:
        subset = [r for r in rows if r["figurative_type"] == ftype]
        fcd, t_star = get_fcd_tstar(subset)
        results["by_type"][ftype] = correlate(fcd, t_star)

    # Per model
    results["by_model"] = {}
    for model_key in MODELS:
        subset = [r for r in rows if r["model"] == model_key]
        fcd, t_star = get_fcd_tstar(subset)
        results["by_model"][model_key] = correlate(fcd, t_star)

    # Per type × per model
    results["by_type_and_model"] = {}
    for ftype in FIGURATIVE_TYPES:
        results["by_type_and_model"][ftype] = {}
        for model_key in MODELS:
            subset = [r for r in rows
                      if r["figurative_type"] == ftype and r["model"] == model_key]
            fcd, t_star = get_fcd_tstar(subset)
            results["by_type_and_model"][ftype][model_key] = correlate(fcd, t_star)

    return results


# ─────────────────────────────────────────────
# Cross-model T* comparison
# ─────────────────────────────────────────────

def cross_model_comparison(curves: dict[str, dict]) -> dict:
    """
    Build a cross-model comparison of T* values by figurative type.
    Also checks: do encoder models (BERT, RoBERTa) transition earlier than GPT-2?
    """
    comparison = {}
    for model_key, curve_data in curves.items():
        comparison[model_key] = {
            "overall_t_star": curve_data["overall"]["transition_layer"],
            "overall_max_acc": max(curve_data["overall"]["layer_accuracies"]),
            "by_type": {},
        }
        for ftype in FIGURATIVE_TYPES:
            if ftype in curve_data.get("by_type", {}):
                comparison[model_key]["by_type"][ftype] = {
                    "t_star":   curve_data["by_type"][ftype]["transition_layer"],
                    "max_acc":  max(curve_data["by_type"][ftype]["layer_accuracies"]),
                }

    # H5: GPT-2 transitions later than BERT?
    if "bert" in comparison and "gpt2" in comparison:
        t_bert = comparison["bert"]["overall_t_star"]
        t_gpt2 = comparison["gpt2"]["overall_t_star"]
        comparison["_hypothesis_H5"] = {
            "bert_t_star":  t_bert,
            "gpt2_t_star":  t_gpt2,
            "gpt2_later":   t_gpt2 > t_bert,
            "difference":   t_gpt2 - t_bert,
        }

    return comparison


# ─────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────

def error_analysis(fcd_scores: dict[str, float], curves: dict[str, dict],
                   records: list[dict]) -> dict:
    """
    Identify problematic expressions:
      - Never reach threshold (max accuracy < 0.65 across all models)
      - Extreme FCD (top/bottom 10%)
      - Sarcasm plateau analysis
    """
    threshold = 0.65
    id_to_record = {r["id"]: r for r in records}

    # Find IDs where no model ever exceeds threshold
    low_accuracy_ids = set(fcd_scores.keys())
    for model_key, curve_data in curves.items():
        for ftype, type_data in curve_data.get("by_type", {}).items():
            max_acc = max(type_data["layer_accuracies"])
            if max_acc >= threshold:
                # This model+type pair is OK — remove matching records from concern set
                for rec_id in list(low_accuracy_ids):
                    rec = id_to_record.get(rec_id)
                    if rec and rec["figurative_type"] == ftype:
                        low_accuracy_ids.discard(rec_id)

    # FCD extremes
    all_fcd = list(fcd_scores.values())
    p10 = np.percentile(all_fcd, 10)
    p90 = np.percentile(all_fcd, 90)

    extreme_low  = {k: v for k, v in fcd_scores.items() if v <= p10}
    extreme_high = {k: v for k, v in fcd_scores.items() if v >= p90}

    # Sarcasm plateau: do sarcasm probes plateau earlier?
    sarcasm_analysis = {}
    for model_key, curve_data in curves.items():
        if "sarcasm" in curve_data.get("by_type", {}):
            sarc = curve_data["by_type"]["sarcasm"]
            accs = sarc["layer_accuracies"]
            # Plateau: max minus final accuracy
            plateau_delta = max(accs) - accs[-1]
            sarcasm_analysis[model_key] = {
                "t_star":       sarc["transition_layer"],
                "max_accuracy": max(accs),
                "final_accuracy": accs[-1],
                "plateau_delta": round(plateau_delta, 4),
                "plateaued": plateau_delta > 0.05,
            }

    return {
        "low_accuracy_expressions": list(low_accuracy_ids)[:20],  # top 20
        "extreme_fcd_low":  dict(list(extreme_low.items())[:10]),
        "extreme_fcd_high": dict(list(extreme_high.items())[:10]),
        "sarcasm_plateau":  sarcasm_analysis,
    }


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def print_correlation_table(correlation_results: dict):
    """Pretty-print the correlation results as a Rich table."""
    table = Table(title="FCD ↔ T* Spearman Correlation", show_lines=True)
    table.add_column("Scope",        style="bold")
    table.add_column("ρ",            justify="right")
    table.add_column("p-value",      justify="right")
    table.add_column("n",            justify="right")
    table.add_column("Significant?", justify="center")

    def add_row(label, corr_dict):
        coef = corr_dict.get("coefficient")
        pval = corr_dict.get("pvalue")
        n    = corr_dict.get("n", "—")
        sig  = "✓" if corr_dict.get("significant") else "✗"
        table.add_row(
            label,
            f"{coef:.4f}" if coef is not None else "—",
            f"{pval:.4f}" if pval is not None else "—",
            str(n),
            sig,
        )

    add_row("Overall", correlation_results["overall"])
    for ftype in FIGURATIVE_TYPES:
        if ftype in correlation_results["by_type"]:
            add_row(f"  {ftype}", correlation_results["by_type"][ftype])
    for model_key in MODELS:
        if model_key in correlation_results["by_model"]:
            add_row(f"  {model_key}", correlation_results["by_model"][model_key])

    console.print(table)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    console.rule("[bold]FCD ↔ Transition Layer Correlation Analysis[/bold]")

    # Load data
    fcd_scores = load_fcd_scores()
    curves     = load_all_curves()

    if not curves:
        console.print("[red]✗ No probe curves found.[/red]")
        console.print("Run: python src/probing/train_probes.py --model all")
        sys.exit(1)

    records_path = Path("data/processed/expressions.jsonl")
    if not records_path.exists():
        console.print("[red]✗ Processed dataset not found.[/red]")
        sys.exit(1)

    records = []
    with open(records_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    console.print(f"Loaded {len(fcd_scores)} FCD scores, {len(curves)} model curves, {len(records)} records.")

    # Build analysis dataframe
    rows = build_analysis_df(fcd_scores, curves, records)
    console.print(f"Analysis rows: {len(rows)}")

    # Correlations
    console.print("\n[bold cyan]Running correlations...[/bold cyan]")
    correlation_results = run_correlations(rows)
    print_correlation_table(correlation_results)

    # Cross-model comparison
    comparison = cross_model_comparison(curves)

    # Error analysis
    errors = error_analysis(fcd_scores, curves, records)

    # Save all results
    all_results = {
        "correlations":       correlation_results,
        "cross_model":        comparison,
        "error_analysis":     errors,
        "n_expressions":      len(fcd_scores),
        "n_models":           len(curves),
        "n_analysis_rows":    len(rows),
    }

    out_path = RESULTS_DIR / "correlation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[green]✓[/green] Results saved → {out_path}")

    # H5 report
    if "_hypothesis_H5" in comparison:
        h5 = comparison["_hypothesis_H5"]
        console.print(
            f"\n[bold]H5 (GPT-2 transitions later than BERT):[/bold] "
            f"BERT T* = {h5['bert_t_star']}, GPT-2 T* = {h5['gpt2_t_star']} → "
            f"{'[green]SUPPORTED[/green]' if h5['gpt2_later'] else '[red]NOT SUPPORTED[/red]'}"
        )

    console.rule()
    console.print("[bold green]Analysis complete.[/bold green]")

    return all_results


if __name__ == "__main__":
    main()
