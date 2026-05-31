"""
config.py — Central configuration for Figurative Language KG Probe.

All paths, model IDs, hyperparameters, and dataset settings live here.
Import this module anywhere in the project instead of hardcoding values.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()

DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
KG_DIR          = DATA_DIR / "kg"

OUTPUTS_DIR     = ROOT / "outputs"
CURVES_DIR      = OUTPUTS_DIR / "curves"
PLOTS_DIR       = OUTPUTS_DIR / "plots"
RESULTS_DIR     = OUTPUTS_DIR / "results"

# Ensure all directories exist
for _dir in [RAW_DIR, PROCESSED_DIR, KG_DIR, CURVES_DIR, PLOTS_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

MODELS = {
    "bert":       "bert-base-uncased",
    "roberta":    "roberta-base",
    "gpt2":       "gpt2",
    "distilbert": "distilbert-base-uncased",
}

# Number of transformer layers per model (excludes embedding layer)
MODEL_LAYERS = {
    "bert":       12,
    "roberta":    12,
    "gpt2":       12,
    "distilbert": 6,
}

# Hidden state dimension
MODEL_HIDDEN_DIM = {
    "bert":       768,
    "roberta":    768,
    "gpt2":       768,
    "distilbert": 768,
}


# ─────────────────────────────────────────────
# Dataset settings
# ─────────────────────────────────────────────

FIGURATIVE_TYPES = ["idiom", "metaphor", "sarcasm"]

# Target number of expressions per type
TARGET_PER_TYPE = {
    "idiom":    120,
    "metaphor": 120,
    "sarcasm":  60,
}

TOTAL_EXPRESSIONS = sum(TARGET_PER_TYPE.values())  # 300

# Random seed for reproducibility
SEED = 42


# ─────────────────────────────────────────────
# KG settings
# ─────────────────────────────────────────────

# ConceptNet API (no key needed for lookup endpoint)
CONCEPTNET_API = "http://api.conceptnet.io/query"

# Relations to traverse when computing shortest paths
KG_RELATIONS = [
    "RelatedTo", "IsA", "PartOf", "HasA", "UsedFor",
    "CapableOf", "Causes", "HasProperty", "SimilarTo",
    "Antonym", "DerivedFrom", "Synonym",
]

# Maximum path length when computing FCD
# (expressions beyond this are marked as distance = MAX_PATH + 1)
MAX_KG_PATH_LENGTH = 8

# ConceptNet local cache file (to avoid re-fetching)
CONCEPTNET_CACHE = KG_DIR / "conceptnet_cache.json"

# Serialized KG files
KG_NODES_FILE  = KG_DIR / "kg_nodes.json"
KG_EDGES_FILE  = KG_DIR / "kg_edges.json"
KG_GRAPH_FILE  = KG_DIR / "kg_graph.graphml"
FCD_SCORES_FILE = KG_DIR / "fcd_scores.json"


# ─────────────────────────────────────────────
# Probing settings
# ─────────────────────────────────────────────

# Probe type: 'logistic' (linear probe) or 'mlp' (non-linear upper bound)
PROBE_TYPE = "logistic"

# Logistic regression settings
PROBE_MAX_ITER   = 1000
PROBE_C          = 1.0      # Regularization (inverse). 1.0 = moderate.
PROBE_SOLVER     = "lbfgs"
PROBE_CV_FOLDS   = 5        # Cross-validation folds

# Hidden state pooling strategy for span tokens
# 'mean': average all span token vectors
# 'first': use first token (good for BERT [CLS]-like behavior)
# 'last': use last token (better for GPT-2 causal direction)
POOLING_STRATEGY = "mean"

# Transition layer threshold: probe accuracy must exceed this to count as T*
TRANSITION_THRESHOLD = 0.75

# Batch size for hidden state extraction (keep small for CPU)
EXTRACTION_BATCH_SIZE = 8

# Cached hidden states directory (large files)
HIDDEN_STATES_DIR = OUTPUTS_DIR / "hidden_states"
HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Analysis settings
# ─────────────────────────────────────────────

# Minimum number of expressions required per figurative type for analysis
MIN_EXPRESSIONS_PER_TYPE = 20

# Correlation method: 'spearman' or 'pearson'
CORRELATION_METHOD = "spearman"

# Significance threshold for p-values
ALPHA = 0.05


# ─────────────────────────────────────────────
# Visualization settings
# ─────────────────────────────────────────────

PLOT_DPI    = 150
PLOT_FORMAT = "png"  # 'png' or 'pdf'

# Color palette per figurative type
TYPE_COLORS = {
    "idiom":    "#E07A5F",   # terracotta
    "metaphor": "#3D405B",   # slate blue
    "sarcasm":  "#81B29A",   # sage green
    "all":      "#F2CC8F",   # warm yellow
}

# Color palette per model
MODEL_COLORS = {
    "bert":       "#264653",
    "roberta":    "#2A9D8F",
    "gpt2":       "#E9C46A",
    "distilbert": "#E76F51",
}
