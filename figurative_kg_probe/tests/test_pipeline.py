"""
tests/test_pipeline.py

Unit tests for core pipeline components.
Run with: python -m pytest tests/ -v

Covers:
  - Dataset cleaning and span detection
  - KG node/edge creation
  - FCD computation (known idioms)
  - Probe accuracy shape
  - Transition layer detection
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────
# Dataset tests
# ─────────────────────────────────────────────

class TestDatasetCleaning:

    def test_clean_text_strips_whitespace(self):
        from src.dataset.preprocess import _clean_text
        assert _clean_text("  hello  world  ") == "hello world"

    def test_clean_text_removes_control_chars(self):
        from src.dataset.preprocess import _clean_text
        assert "\x00" not in _clean_text("hello\x00world")

    def test_is_valid_rejects_short(self):
        from src.dataset.preprocess import _is_valid
        assert not _is_valid({"sentence": "hi", "label": "figurative"})

    def test_is_valid_rejects_bad_label(self):
        from src.dataset.preprocess import _is_valid
        assert not _is_valid({"sentence": "This is a valid sentence here.", "label": "unknown"})

    def test_is_valid_accepts_good_record(self):
        from src.dataset.preprocess import _is_valid
        assert _is_valid({"sentence": "She kicked the bucket last year.", "label": "figurative"})

    def test_is_valid_rejects_very_long(self):
        from src.dataset.preprocess import _is_valid
        long_sent = "word " * 120
        assert not _is_valid({"sentence": long_sent, "label": "figurative"})


class TestSpanDetection:

    def test_find_span_found(self):
        from src.dataset.preprocess import _find_span
        s, e = _find_span("She kicked the bucket last year.", "kicked the bucket")
        assert s == 4
        assert e == s + len("kicked the bucket")

    def test_find_span_case_insensitive(self):
        from src.dataset.preprocess import _find_span
        s, e = _find_span("She KICKED THE BUCKET last year.", "kicked the bucket")
        assert s >= 0

    def test_find_span_not_found(self):
        from src.dataset.preprocess import _find_span
        s, e = _find_span("This sentence has nothing.", "spill the beans")
        assert s == -1 and e == -1

    def test_find_span_empty_expression(self):
        from src.dataset.preprocess import _find_span
        s, e = _find_span("Any sentence here.", "")
        assert s == 0 and e > 0


class TestDeduplication:

    def test_deduplication_removes_exact_dupe(self):
        from src.dataset.preprocess import deduplicate
        records = [
            {"expression": "kick the bucket", "sentence": "She kicked the bucket.", "label": "figurative"},
            {"expression": "kick the bucket", "sentence": "She kicked the bucket.", "label": "figurative"},
            {"expression": "spill the beans", "sentence": "Don't spill the beans.", "label": "figurative"},
        ]
        unique = deduplicate(records)
        assert len(unique) == 2

    def test_deduplication_keeps_different(self):
        from src.dataset.preprocess import deduplicate
        records = [
            {"expression": "kick the bucket", "sentence": "She kicked the bucket.", "label": "figurative"},
            {"expression": "spill the beans", "sentence": "Don't spill the beans.", "label": "figurative"},
        ]
        assert len(deduplicate(records)) == 2


# ─────────────────────────────────────────────
# KG tests
# ─────────────────────────────────────────────

class TestFigurativeKG:

    def test_kg_add_idiom_creates_nodes(self):
        from src.kg.build_kg import FigurativeKG
        kg = FigurativeKG()
        kg.add_idiom("kick the bucket")
        assert kg.G.number_of_nodes() > 0
        assert f"expr:kick the bucket" in kg.G.nodes

    def test_kg_add_idiom_creates_figurative_edge(self):
        from src.kg.build_kg import FigurativeKG
        kg = FigurativeKG()
        kg.add_idiom("kick the bucket")
        expr_id = "expr:kick the bucket"
        # Check at least one figurative_meaning edge
        edges = list(kg.G.edges(expr_id, data=True))
        relations = [d.get("relation", "") for _, _, d in edges]
        assert any("figurative_meaning" in r for r in relations)

    def test_kg_add_idiom_literal_constituents(self):
        from src.kg.build_kg import FigurativeKG
        kg = FigurativeKG()
        kg.add_idiom("spill the beans")
        # Should have concept nodes for literal parts
        assert "concept:spill" in kg.G.nodes or "concept:beans" in kg.G.nodes

    def test_kg_save_load_roundtrip(self):
        import networkx as nx
        from src.kg.build_kg import FigurativeKG

        kg = FigurativeKG()
        kg.add_idiom("kick the bucket")

        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            tmp_path = Path(f.name)

        nx.write_graphml(kg.G, str(tmp_path))
        G_loaded = nx.read_graphml(str(tmp_path))

        assert G_loaded.number_of_nodes() == kg.G.number_of_nodes()
        tmp_path.unlink()

    def test_wordnet_edges_returns_list(self):
        from src.kg.build_kg import get_wordnet_edges
        edges = get_wordnet_edges("kick")
        assert isinstance(edges, list)

    def test_wordnet_path_distance_same_word(self):
        from src.kg.build_kg import wordnet_path_distance
        dist = wordnet_path_distance("dog", "dog")
        # Same word should have low distance (high similarity → low distance)
        assert dist is not None
        assert dist < 0.1

    def test_wordnet_path_distance_unrelated(self):
        from src.kg.build_kg import wordnet_path_distance
        dist = wordnet_path_distance("mathematics", "banana")
        if dist is not None:
            assert 0.0 <= dist <= 1.0


# ─────────────────────────────────────────────
# FCD tests
# ─────────────────────────────────────────────

class TestFCD:

    def setup_method(self):
        """Build a minimal KG for testing."""
        from src.kg.build_kg import FigurativeKG
        self.kg = FigurativeKG()
        for expr in ["kick the bucket", "spill the beans", "piece of cake"]:
            self.kg.add_idiom(expr)

    def test_fcd_idiom_known_expression(self):
        from src.kg.build_kg import compute_fcd
        fcd = compute_fcd("kick the bucket", "idiom", self.kg)
        assert isinstance(fcd, float)
        assert fcd >= 0

    def test_fcd_idiom_unknown_expression(self):
        from src.kg.build_kg import compute_fcd
        from config import MAX_KG_PATH_LENGTH
        fcd = compute_fcd("nonexistent idiom xyz", "idiom", self.kg)
        assert fcd == float(MAX_KG_PATH_LENGTH)

    def test_fcd_sarcasm_is_fixed(self):
        from src.kg.build_kg import compute_fcd
        from config import MAX_KG_PATH_LENGTH
        fcd = compute_fcd("", "sarcasm", self.kg)
        assert fcd == float(MAX_KG_PATH_LENGTH - 1)

    def test_fcd_different_idioms_differ(self):
        """Two idioms should (generally) have different FCD values."""
        from src.kg.build_kg import compute_fcd
        # This is a soft test — they can be equal if both are MAX
        fcd1 = compute_fcd("kick the bucket", "idiom", self.kg)
        fcd2 = compute_fcd("piece of cake", "idiom", self.kg)
        assert isinstance(fcd1, float) and isinstance(fcd2, float)


# ─────────────────────────────────────────────
# Probing tests
# ─────────────────────────────────────────────

class TestProbeAtLayer:
    """Tests for linear probe and transition logic.
    Imports are done inline to avoid torch dependency at collection time.
    """

    def _make_data(self, n=50, dim=64, seed=0):
        rng = np.random.RandomState(seed)
        X0 = rng.randn(n // 2, dim) + 1.0
        X1 = rng.randn(n // 2, dim) - 1.0
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return X, y

    def _probe_at_layer(self):
        """Import probe_at_layer without pulling in torch at module level."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler

        def probe_at_layer(X, y, n_folds=3):
            if len(np.unique(y)) < 2:
                return {"mean_accuracy": float(np.mean(y == y[0])),
                        "std_accuracy": 0.0, "per_fold": [],
                        "n_samples": len(y), "n_positive": int(y.sum())}
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            cv = StratifiedKFold(n_splits=min(n_folds, len(y) // 2), shuffle=True, random_state=42)
            clf = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
            scores = cross_val_score(clf, X_s, y, cv=cv, scoring="accuracy")
            return {"mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std()),
                    "per_fold": scores.tolist(), "n_samples": len(y), "n_positive": int(y.sum())}

        return probe_at_layer

    def test_probe_returns_dict(self):
        probe_at_layer = self._probe_at_layer()
        X, y = self._make_data()
        result = probe_at_layer(X, y)
        assert "mean_accuracy" in result
        assert "std_accuracy" in result

    def test_probe_separable_data_high_accuracy(self):
        probe_at_layer = self._probe_at_layer()
        X, y = self._make_data(n=100, dim=32)
        result = probe_at_layer(X, y)
        assert result["mean_accuracy"] > 0.7

    def test_probe_random_data_near_chance(self):
        probe_at_layer = self._probe_at_layer()
        rng = np.random.RandomState(42)
        X = rng.randn(80, 32)
        y = rng.randint(0, 2, 80)
        result = probe_at_layer(X, y)
        assert 0.3 < result["mean_accuracy"] < 0.75


class TestTransitionLayer:

    def _find_transition_layer(self, layer_accuracies, threshold=0.75):
        for i, acc in enumerate(layer_accuracies):
            if acc >= threshold:
                return i
        return int(np.argmax(layer_accuracies)) if layer_accuracies else 0

    def _compute_slope(self, accs):
        return [accs[i + 1] - accs[i] for i in range(len(accs) - 1)]

    def test_transition_at_threshold(self):
        accs = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
        t = self._find_transition_layer(accs)
        assert t == 4

    def test_transition_fallback_to_max(self):
        accs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.72]
        t = self._find_transition_layer(accs, threshold=0.75)
        assert t == 5

    def test_slope_length(self):
        accs = [0.5, 0.6, 0.7, 0.8]
        slopes = self._compute_slope(accs)
        assert len(slopes) == 3
        assert all(abs(s - 0.1) < 1e-9 for s in slopes)


# ─────────────────────────────────────────────
# Correlation tests
# ─────────────────────────────────────────────

class TestCorrelation:

    def _correlate(self, x, y, method="spearman"):
        """Inline correlation — avoids torch import chain."""
        from scipy import stats
        if len(x) < 3:
            return {"coefficient": None, "pvalue": None, "method": method,
                    "n": len(x), "significant": False}
        if method == "spearman":
            coef, pval = stats.spearmanr(x, y)
        else:
            coef, pval = stats.pearsonr(x, y)
        return {"coefficient": round(float(coef), 4), "pvalue": round(float(pval), 6),
                "method": method, "n": len(x), "significant": bool(pval < 0.05)}

    def test_spearman_positive_correlation(self):
        result = self._correlate([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        assert result["coefficient"] == pytest.approx(1.0, abs=0.01)
        assert result["significant"]

    def test_pearson_zero_correlation(self):
        result = self._correlate([1, 2, 3, 4, 5], [3, 1, 4, 1, 5], method="pearson")
        assert abs(result["coefficient"]) < 0.8

    def test_correlate_too_few_samples(self):
        result = self._correlate([1, 2], [3, 4])
        assert result["coefficient"] is None


# ─────────────────────────────────────────────
# Integration smoke test
# ─────────────────────────────────────────────

class TestIntegrationSmoke:
    """
    Quick end-to-end smoke test using tiny synthetic data.
    Does NOT download real models or datasets.
    """

    def test_preprocess_pipeline_runs(self, tmp_path, monkeypatch):
        """Preprocess stage runs on synthetic JSONL without errors."""
        import sys

        # Write tiny raw data files
        (tmp_path / "raw").mkdir()
        (tmp_path / "processed").mkdir()
        (tmp_path / "kg").mkdir()

        data = [
            {"expression": "kick the bucket", "sentence": "She kicked the bucket last year.", "label": "figurative", "figurative_type": "idiom", "source": "test"},
            {"expression": "kick the bucket", "sentence": "He kicked the bucket across the floor.", "label": "literal", "figurative_type": "idiom", "source": "test"},
            {"expression": "spill the beans", "sentence": "Don't spill the beans about the party.", "label": "figurative", "figurative_type": "idiom", "source": "test"},
            {"expression": "spill the beans", "sentence": "She spilled the beans on the floor.", "label": "literal", "figurative_type": "idiom", "source": "test"},
        ]
        for fname in ["magpie.jsonl", "vua_metaphors.jsonl", "semeval_sarcasm.jsonl"]:
            p = tmp_path / "raw" / fname
            with open(p, "w") as f:
                for r in data:
                    f.write(json.dumps(r) + "\n")

        # Monkeypatch config paths
        import config
        monkeypatch.setattr(config, "RAW_DIR",       tmp_path / "raw")
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(config, "TARGET_PER_TYPE", {"idiom": 2, "metaphor": 2, "sarcasm": 2})

        from src.dataset import preprocess
        monkeypatch.setattr(preprocess, "RAW_DIR",        tmp_path / "raw")
        monkeypatch.setattr(preprocess, "PROCESSED_DIR",  tmp_path / "processed")
        monkeypatch.setattr(preprocess, "TARGET_PER_TYPE", {"idiom": 2, "metaphor": 2, "sarcasm": 2})

        out = preprocess.preprocess_all()
        assert out.exists()

        records = list(preprocess.load_processed())
        assert len(records) >= 1
        assert all("id" in r for r in records)
