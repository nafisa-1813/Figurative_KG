# Figurative Language KG Probe

> *How Deep is Figurative? Knowledge Graph Distance as a Probe for Non-Literal Understanding in Transformers*

A research codebase that investigates **at which transformer layer** a model transitions from literal to figurative interpretation — using **Knowledge Graph conceptual distance (FCD)** as a semantic ruler.

---

## Research Question

> Does the structural distance between literal and figurative meanings in a knowledge graph predict how deep a transformer must process to "understand" the figurative expression?

---

## Pipeline Overview

```
Raw Datasets          KG Construction         Probing              Analysis
─────────────         ───────────────         ───────              ────────
MAGPIE (idioms)  ──->  ConceptNet edges   ──->  BERT hidden   ──->   FCD vs T*
VUA (metaphors)  ──->  WordNet synsets    ──->  states per    ──->   correlation
SemEval (sarc.)  ─-> FCD computation   ->  layer probe   ──-->   curves +
                       KG serialization       accuracy            plots
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python src/dataset/download.py

# 3. Build the knowledge graph
python src/kg/build_kg.py

# 4. Extract hidden states
python src/probing/extract_hidden_states.py --model bert-base-uncased

# 5. Train probes
python src/probing/train_probes.py --model bert-base-uncased

# 6. Run full analysis
python src/analysis/correlate_fcd_transition.py

# 7. Generate all plots
python src/visualization/plot_all.py
```

Or run the full pipeline end-to-end:

```bash
python run_pipeline.py --models bert roberta gpt2 distilbert
```

---

## Project Structure

```
figurative_kg_probe/
├── data/
│   ├── raw/              # Downloaded datasets
│   ├── processed/        # Cleaned, unified expression dataset
│   └── kg/               # KG nodes, edges, FCD scores
├── src/
│   ├── dataset/          # Dataset loading, curation, preprocessing
│   ├── kg/               # KG construction, FCD computation
│   ├── probing/          # Hidden state extraction, linear probes
│   ├── analysis/         # FCD↔T* correlation, cross-model stats
│   └── visualization/    # All plotting utilities
├── outputs/
│   ├── curves/           # Per-model figurativeness curves (JSON)
│   ├── plots/            # All generated figures (PNG/PDF)
│   └── results/          # Correlation tables, summary stats
├── notebooks/            # Exploratory analysis notebooks
├── tests/                # Unit tests
├── run_pipeline.py       # End-to-end runner
├── config.py             # All hyperparameters and paths
└── requirements.txt
```

---

## Models Supported (CPU-Friendly)

| Model | HuggingFace ID | Layers |
|---|---|---|
| BERT | `bert-base-uncased` | 12 |
| RoBERTa | `roberta-base` | 12 |
| GPT-2 | `gpt2` | 12 |
| DistilBERT | `distilbert-base-uncased` | 6 |

---

## Key Concepts

**FCD (Figurative Conceptual Distance)**: Shortest path in the KG between the set of literal constituent concepts and the figurative meaning node. Higher = more semantically distant = harder to interpret.

**T\* (Transition Layer)**: The transformer layer at which a linear probe's accuracy on figurative vs. literal classification first exceeds a threshold (default 75%) or shows maximum slope.

**Core Hypothesis**: `FCD ↔ T*` — expressions with greater KG conceptual distance require deeper layers for figurative interpretation.

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{figurative_kg_probe_2026,
  title   = {How Deep is Figurative? KG Distance as a Probe for Non-Literal Understanding},
  year    = {2026},
  note    = {Independent Research}
}
```
