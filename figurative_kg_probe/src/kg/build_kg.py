"""
src/kg/build_kg.py

Constructs the Figurative Meaning Knowledge Graph (FMKG) from three sources:
  1. ConceptNet API   — common-sense relations between concepts
  2. WordNet (NLTK)   — synsets, hypernyms, hyponyms (lexical hierarchy)
  3. Curated edges    — manual figurative→meaning mappings for idioms

Then computes the Figurative Conceptual Distance (FCD) for every
expression in the processed dataset.

Output files (data/kg/):
    kg_nodes.json     — list of node dicts {id, label, type}
    kg_edges.json     — list of edge dicts {source, target, relation, weight}
    kg_graph.graphml  — full graph (NetworkX compatible)
    fcd_scores.json   — {expression_id: fcd_score} mapping

Run directly:
    python src/kg/build_kg.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import requests
from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    KG_DIR, KG_NODES_FILE, KG_EDGES_FILE, KG_GRAPH_FILE, FCD_SCORES_FILE,
    CONCEPTNET_CACHE, CONCEPTNET_API, KG_RELATIONS, MAX_KG_PATH_LENGTH,
    PROCESSED_DIR,
)

console = Console()


# ─────────────────────────────────────────────
# ConceptNet interface
# ─────────────────────────────────────────────

class ConceptNetClient:
    """
    Thin wrapper around the ConceptNet 5 REST API.
    Caches all responses locally to avoid re-fetching.
    """

    BASE = "http://api.conceptnet.io"

    def __init__(self, cache_path: Path = CONCEPTNET_CACHE):
        self.cache_path = cache_path
        self._cache: dict = {}
        if cache_path.exists():
            with open(cache_path) as f:
                self._cache = json.load(f)

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f)

    def get_edges(self, concept: str, limit: int = 20) -> list[dict]:
        """
        Return ConceptNet edges for a given concept (English, lowercased).
        Returns list of {relation, start, end, weight}.
        Falls back to empty list on network error.
        """
        key = f"edges:{concept}"
        if key in self._cache:
            return self._cache[key]

        url = f"{self.BASE}/c/en/{concept.lower().replace(' ', '_')}"
        try:
            resp = requests.get(url, params={"limit": limit}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            edges = []
            for edge in data.get("edges", []):
                rel   = edge.get("rel", {}).get("label", "")
                start = edge.get("start", {}).get("label", "")
                end   = edge.get("end",   {}).get("label", "")
                w     = edge.get("weight", 1.0)
                if rel and start and end:
                    edges.append({"relation": rel, "start": start, "end": end, "weight": w})
            self._cache[key] = edges
            self._save_cache()
            time.sleep(0.3)   # polite rate limit
            return edges
        except Exception:
            self._cache[key] = []
            return []

    def get_related(self, concept: str, limit: int = 10) -> list[str]:
        """Return a flat list of concepts related to `concept`."""
        edges = self.get_edges(concept, limit=limit)
        related = set()
        for e in edges:
            related.add(e["start"].lower())
            related.add(e["end"].lower())
        related.discard(concept.lower())
        return list(related)


# ─────────────────────────────────────────────
# WordNet interface
# ─────────────────────────────────────────────

def _ensure_wordnet():
    """Download WordNet if not already present."""
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def get_wordnet_edges(word: str) -> list[tuple[str, str, str]]:
    """
    Return (source, target, relation) triples from WordNet for `word`.
    Relations: hypernym, hyponym, holonym, meronym, antonym, synonym.
    """
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    triples = []
    for syn in wn.synsets(word):
        name = syn.name().split(".")[0].replace("_", " ")

        for hyper in syn.hypernyms():
            t = hyper.name().split(".")[0].replace("_", " ")
            triples.append((name, t, "hypernym"))

        for hypo in syn.hyponyms():
            t = hypo.name().split(".")[0].replace("_", " ")
            triples.append((name, t, "hyponym"))

        for holo in syn.member_holonyms() + syn.substance_holonyms():
            t = holo.name().split(".")[0].replace("_", " ")
            triples.append((name, t, "holonym"))

        for mero in syn.part_meronyms():
            t = mero.name().split(".")[0].replace("_", " ")
            triples.append((name, t, "meronym"))

        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                t = ant.name().replace("_", " ")
                triples.append((name, t, "antonym"))

    return triples


def wordnet_path_distance(word_a: str, word_b: str) -> Optional[float]:
    """
    Return the Wu-Palmer similarity distance between two words using WordNet.
    Distance = 1 - similarity (so 0 = identical, 1 = completely unrelated).
    Returns None if no synset exists for either word.
    """
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    syns_a = wn.synsets(word_a)
    syns_b = wn.synsets(word_b)
    if not syns_a or not syns_b:
        return None

    best_sim = max(
        (s_a.wup_similarity(s_b) or 0.0)
        for s_a in syns_a[:3]
        for s_b in syns_b[:3]
    )
    return round(1.0 - best_sim, 4)


# ─────────────────────────────────────────────
# Curated figurative meaning mappings
# ─────────────────────────────────────────────

# Hand-curated: idiom_expression → {literal_concepts, figurative_meaning, domain}
IDIOM_MEANINGS: dict[str, dict] = {
    "kick the bucket":           {"literal": ["kick", "bucket"],       "figurative": ["die", "death"],           "domain": "mortality"},
    "spill the beans":           {"literal": ["spill", "beans"],       "figurative": ["reveal", "secret"],       "domain": "disclosure"},
    "bite the bullet":           {"literal": ["bite", "bullet"],       "figurative": ["endure", "hardship"],     "domain": "resilience"},
    "break a leg":               {"literal": ["break", "leg"],         "figurative": ["good luck", "success"],   "domain": "performance"},
    "burn bridges":              {"literal": ["burn", "bridge"],       "figurative": ["ruin relationship"],      "domain": "social"},
    "cost an arm and a leg":     {"literal": ["cost", "arm", "leg"],   "figurative": ["expensive"],              "domain": "economics"},
    "hit the nail on the head":  {"literal": ["hit", "nail", "head"],  "figurative": ["correct", "accurate"],    "domain": "evaluation"},
    "let the cat out of the bag":{"literal": ["cat", "bag"],           "figurative": ["reveal", "secret"],       "domain": "disclosure"},
    "under the weather":         {"literal": ["weather"],              "figurative": ["sick", "unwell"],         "domain": "health"},
    "bite off more than you can chew": {"literal": ["bite", "chew"],   "figurative": ["overcommit"],             "domain": "capacity"},
    "add fuel to the fire":      {"literal": ["fuel", "fire"],         "figurative": ["worsen", "conflict"],     "domain": "conflict"},
    "hit the sack":              {"literal": ["hit", "sack"],          "figurative": ["sleep", "rest"],          "domain": "rest"},
    "once in a blue moon":       {"literal": ["moon"],                 "figurative": ["rarely", "infrequent"],   "domain": "frequency"},
    "pull someone's leg":        {"literal": ["pull", "leg"],          "figurative": ["joke", "tease"],          "domain": "humour"},
    "the tip of the iceberg":    {"literal": ["iceberg", "tip"],       "figurative": ["small part", "hidden"],   "domain": "visibility"},
    "jump on the bandwagon":     {"literal": ["bandwagon", "jump"],    "figurative": ["follow trend"],           "domain": "conformity"},
    "read between the lines":    {"literal": ["read", "lines"],        "figurative": ["infer", "implicit"],      "domain": "inference"},
    "break the ice":             {"literal": ["break", "ice"],         "figurative": ["start conversation"],     "domain": "social"},
    "go back to the drawing board": {"literal": ["drawing board"],     "figurative": ["restart", "redesign"],    "domain": "planning"},
    "hit the ground running":    {"literal": ["ground", "run"],        "figurative": ["start quickly"],          "domain": "initiation"},
    "learn the ropes":           {"literal": ["ropes"],                "figurative": ["learn", "procedure"],     "domain": "learning"},
    "on thin ice":               {"literal": ["ice", "thin"],          "figurative": ["risky", "precarious"],    "domain": "risk"},
    "see eye to eye":            {"literal": ["eye"],                  "figurative": ["agree", "consensus"],     "domain": "agreement"},
    "sit on the fence":          {"literal": ["fence", "sit"],         "figurative": ["undecided", "neutral"],   "domain": "decision"},
    "steal someone's thunder":   {"literal": ["thunder"],              "figurative": ["upstage", "attention"],   "domain": "social"},
    "the last straw":            {"literal": ["straw"],                "figurative": ["limit", "breaking point"],"domain": "tolerance"},
    "throw in the towel":        {"literal": ["towel"],                "figurative": ["give up", "surrender"],   "domain": "persistence"},
    "up in the air":             {"literal": ["air"],                  "figurative": ["uncertain", "undecided"], "domain": "certainty"},
    "piece of cake":             {"literal": ["cake"],                 "figurative": ["easy", "simple"],         "domain": "difficulty"},
    "raining cats and dogs":     {"literal": ["rain", "cat", "dog"],   "figurative": ["heavy rain"],             "domain": "weather"},
    "wolf in sheep's clothing":  {"literal": ["wolf", "sheep"],        "figurative": ["deceptive", "disguise"],  "domain": "deception"},
    "burn the midnight oil":     {"literal": ["oil", "midnight"],      "figurative": ["work late", "effort"],    "domain": "effort"},
    "cut corners":               {"literal": ["corner", "cut"],        "figurative": ["shortcut", "reduce quality"],"domain": "quality"},
    "go the extra mile":         {"literal": ["mile"],                 "figurative": ["extra effort"],           "domain": "effort"},
    "kill two birds with one stone":{"literal": ["bird", "stone"],     "figurative": ["efficient", "two goals"], "domain": "efficiency"},
    "miss the boat":             {"literal": ["boat"],                 "figurative": ["miss opportunity"],       "domain": "opportunity"},
    "not my cup of tea":         {"literal": ["cup", "tea"],           "figurative": ["dislike", "preference"],  "domain": "preference"},
}

# Conceptual metaphor source→target domain mappings
METAPHOR_DOMAINS: dict[str, dict] = {
    "time is money":        {"source": "economics",   "target": "time",        "bridge_concepts": ["value", "resource"]},
    "argument is war":      {"source": "combat",      "target": "discourse",   "bridge_concepts": ["strategy", "attack"]},
    "ideas are food":       {"source": "consumption", "target": "cognition",   "bridge_concepts": ["nourishment", "digest"]},
    "life is a journey":    {"source": "travel",      "target": "existence",   "bridge_concepts": ["path", "destination"]},
    "mind is a machine":    {"source": "mechanics",   "target": "cognition",   "bridge_concepts": ["process", "function"]},
    "knowledge is light":   {"source": "optics",      "target": "cognition",   "bridge_concepts": ["illumination", "clarity"]},
    "emotions as temperature":{"source": "thermal",   "target": "affect",      "bridge_concepts": ["warmth", "intensity"]},
    "happiness is up":      {"source": "spatial",     "target": "affect",      "bridge_concepts": ["elevation", "positive"]},
    "theories are buildings":{"source": "architecture","target": "epistemology","bridge_concepts": ["structure", "foundation"]},
}


# ─────────────────────────────────────────────
# KG construction
# ─────────────────────────────────────────────

class FigurativeKG:
    """
    The Figurative Meaning Knowledge Graph (FMKG).

    Node types:
        concept          — a general English concept word
        figurative_expr  — a full idiom / metaphorical expression
        figurative_meaning — the non-literal meaning (e.g. "die", "reveal")
        domain           — abstract semantic domain (e.g. "mortality")

    Edge types (relations):
        literal_constituent    — expr→concept (literal parts of the expression)
        figurative_meaning     — expr→meaning (non-literal interpretation)
        domain_of              — expr/meaning→domain
        conceptnet_*           — edges imported from ConceptNet
        wordnet_hypernym, etc. — edges from WordNet
        cross_domain_map       — metaphor source domain → target domain
    """

    def __init__(self):
        self.G = nx.Graph()
        self._cn = ConceptNetClient()

    # ── Node helpers ──────────────────────────────────────────────

    def add_node(self, node_id: str, label: str, node_type: str, **attrs):
        self.G.add_node(node_id, label=label, node_type=node_type, **attrs)

    def add_edge(self, src: str, dst: str, relation: str, weight: float = 1.0):
        # Ensure both nodes exist (anonymous concept node if absent)
        for n in (src, dst):
            if n not in self.G:
                self.G.add_node(n, label=n, node_type="concept")
        self.G.add_edge(src, dst, relation=relation, weight=weight)

    # ── Idiom subgraphs ──────────────────────────────────────────

    def add_idiom(self, expression: str):
        """
        Add a complete idiom subgraph:
        expression → literal constituents
        expression → figurative meaning node(s)
        expression → domain node
        ConceptNet edges for each constituent
        WordNet edges for each constituent
        """
        if expression not in IDIOM_MEANINGS:
            return

        meta = IDIOM_MEANINGS[expression]
        expr_id = f"expr:{expression}"
        self.add_node(expr_id, label=expression, node_type="figurative_expr",
                      figurative_type="idiom")

        # Literal constituent edges
        for concept in meta["literal"]:
            cid = f"concept:{concept}"
            self.add_node(cid, label=concept, node_type="concept")
            self.add_edge(expr_id, cid, "literal_constituent", weight=1.0)

            # Enrich from ConceptNet
            for cn_edge in self._cn.get_edges(concept, limit=15):
                src_label = cn_edge["start"].lower()
                dst_label = cn_edge["end"].lower()
                src_id = f"concept:{src_label}"
                dst_id = f"concept:{dst_label}"
                self.add_edge(src_id, dst_id,
                              f"conceptnet_{cn_edge['relation']}",
                              weight=cn_edge["weight"])

            # Enrich from WordNet
            for (w_src, w_dst, w_rel) in get_wordnet_edges(concept):
                self.add_edge(f"concept:{w_src}", f"concept:{w_dst}",
                              f"wordnet_{w_rel}", weight=0.8)

        # Figurative meaning edges
        for meaning in meta["figurative"]:
            mid = f"meaning:{meaning}"
            self.add_node(mid, label=meaning, node_type="figurative_meaning")
            self.add_edge(expr_id, mid, "figurative_meaning", weight=2.0)

        # Domain node
        domain_id = f"domain:{meta['domain']}"
        self.add_node(domain_id, label=meta["domain"], node_type="domain")
        self.add_edge(expr_id, domain_id, "domain_of", weight=1.5)

    # ── Metaphor subgraphs ────────────────────────────────────────

    def add_metaphor_domain(self, name: str):
        """
        Add a conceptual metaphor cross-domain mapping subgraph.
        """
        if name not in METAPHOR_DOMAINS:
            return

        meta = METAPHOR_DOMAINS[name]
        src_id    = f"domain:{meta['source']}"
        dst_id    = f"domain:{meta['target']}"
        meta_id   = f"metaphor:{name}"

        self.add_node(src_id,  label=meta["source"], node_type="domain")
        self.add_node(dst_id,  label=meta["target"], node_type="domain")
        self.add_node(meta_id, label=name, node_type="figurative_expr",
                      figurative_type="metaphor")

        self.add_edge(meta_id, src_id, "source_domain", weight=1.0)
        self.add_edge(meta_id, dst_id, "target_domain", weight=1.0)
        self.add_edge(src_id, dst_id, "cross_domain_map", weight=3.0)

        for bridge in meta["bridge_concepts"]:
            bid = f"concept:{bridge}"
            self.add_node(bid, label=bridge, node_type="concept")
            self.add_edge(meta_id, bid, "bridge_concept", weight=1.5)
            for cn_edge in self._cn.get_edges(bridge, limit=10):
                self.add_edge(
                    f"concept:{cn_edge['start'].lower()}",
                    f"concept:{cn_edge['end'].lower()}",
                    f"conceptnet_{cn_edge['relation']}",
                    weight=cn_edge["weight"],
                )

    # ── Sarcasm nodes ─────────────────────────────────────────────

    def add_sarcasm_nodes(self):
        """
        Add abstract sarcasm-related nodes.
        Sarcasm FCD is computed differently (pragmatic distance),
        so we add marker nodes for it.
        """
        for concept in ["irony", "contradiction", "expectation", "violation",
                        "literal meaning", "intended meaning", "pragmatics"]:
            cid = f"concept:{concept}"
            self.add_node(cid, label=concept, node_type="concept")
            for cn_edge in self._cn.get_edges(concept, limit=8):
                self.add_edge(
                    f"concept:{cn_edge['start'].lower()}",
                    f"concept:{cn_edge['end'].lower()}",
                    f"conceptnet_{cn_edge['relation']}",
                    weight=cn_edge["weight"],
                )

    # ── Build from dataset ────────────────────────────────────────

    def build_from_dataset(self, records: list[dict]):
        """Build the full KG from the processed dataset."""
        console.print("[bold cyan]Building KG from dataset...[/bold cyan]")

        # Collect unique expressions per type
        by_type: dict[str, set] = {"idiom": set(), "metaphor": set(), "sarcasm": set()}
        for r in records:
            expr = r["expression"].lower().strip()
            by_type[r["figurative_type"]].add(expr)

        console.print(f"  Idioms found:    {len(by_type['idiom'])}")
        console.print(f"  Metaphors found: {len(by_type['metaphor'])}")
        console.print(f"  Sarcasm items:   {len(by_type['sarcasm'])}")

        # Idioms
        console.print("\n[cyan]Adding idiom subgraphs...[/cyan]")
        for expr in track(IDIOM_MEANINGS.keys(), description="Idiom subgraphs"):
            self.add_idiom(expr)

        # Metaphors
        console.print("\n[cyan]Adding metaphor domain subgraphs...[/cyan]")
        for name in track(METAPHOR_DOMAINS.keys(), description="Metaphor subgraphs"):
            self.add_metaphor_domain(name)

        # Sarcasm
        console.print("\n[cyan]Adding sarcasm nodes...[/cyan]")
        self.add_sarcasm_nodes()

        console.print(
            f"\n[green]KG built:[/green] "
            f"{self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        )

    # ── Serialization ─────────────────────────────────────────────

    def save(self):
        """Save nodes, edges, and GraphML to data/kg/."""
        # Nodes
        nodes = [
            {"id": n, **self.G.nodes[n]}
            for n in self.G.nodes
        ]
        with open(KG_NODES_FILE, "w") as f:
            json.dump(nodes, f, indent=2)

        # Edges
        edges = [
            {"source": u, "target": v, **self.G.edges[u, v]}
            for u, v in self.G.edges
        ]
        with open(KG_EDGES_FILE, "w") as f:
            json.dump(edges, f, indent=2)

        # GraphML
        nx.write_graphml(self.G, str(KG_GRAPH_FILE))

        console.print(f"[green]✓[/green] KG saved → {KG_DIR}")

    @classmethod
    def load(cls) -> "FigurativeKG":
        """Load a previously saved KG from GraphML."""
        kg = cls()
        if not KG_GRAPH_FILE.exists():
            raise FileNotFoundError(
                f"KG not found: {KG_GRAPH_FILE}\n"
                f"Run: python src/kg/build_kg.py"
            )
        kg.G = nx.read_graphml(str(KG_GRAPH_FILE))
        console.print(
            f"[green]✓[/green] KG loaded: "
            f"{kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges"
        )
        return kg


# ─────────────────────────────────────────────
# FCD computation
# ─────────────────────────────────────────────

def compute_fcd(expression: str, figurative_type: str, kg: FigurativeKG) -> float:
    """
    Compute the Figurative Conceptual Distance (FCD) for a single expression.

    For idioms:
        FCD = mean shortest path from each literal constituent node
              to the figurative meaning node(s).

    For metaphors:
        FCD = shortest path between source domain and target domain nodes
              + 1 (for the cross-domain leap).

    For sarcasm:
        FCD is computed as a fixed heuristic (pragmatic distance):
        = MAX_KG_PATH_LENGTH - 1  (sarcasm requires discourse-level pragmatics,
          which is beyond semantic graph distance; we mark it as "far").

    Returns a float. Higher = more conceptually distant from literal meaning.
    """
    G = kg.G

    if figurative_type == "sarcasm":
        return float(MAX_KG_PATH_LENGTH - 1)

    expr_lower = expression.lower().strip()

    if figurative_type == "idiom":
        if expr_lower not in IDIOM_MEANINGS:
            return float(MAX_KG_PATH_LENGTH)  # unknown idiom → max distance

        meta = IDIOM_MEANINGS[expr_lower]
        literal_nodes   = [f"concept:{c}" for c in meta["literal"]]
        figurative_nodes = [f"meaning:{m}" for m in meta["figurative"]]

        distances = []
        for lit_node in literal_nodes:
            for fig_node in figurative_nodes:
                if lit_node not in G or fig_node not in G:
                    distances.append(MAX_KG_PATH_LENGTH)
                    continue
                try:
                    d = nx.shortest_path_length(G, lit_node, fig_node)
                    distances.append(d)
                except nx.NetworkXNoPath:
                    distances.append(MAX_KG_PATH_LENGTH)

        return round(sum(distances) / max(len(distances), 1), 2) if distances else float(MAX_KG_PATH_LENGTH)

    if figurative_type == "metaphor":
        # Find which metaphor domain this expression belongs to
        best_dist = float(MAX_KG_PATH_LENGTH)

        for meta_name, meta in METAPHOR_DOMAINS.items():
            src_id = f"domain:{meta['source']}"
            dst_id = f"domain:{meta['target']}"

            if src_id not in G or dst_id not in G:
                continue
            try:
                d = nx.shortest_path_length(G, src_id, dst_id)
                best_dist = min(best_dist, d + 1)  # +1 for cross-domain leap
            except nx.NetworkXNoPath:
                pass

            # Also check if the expression word itself is in the graph
            expr_node = f"concept:{expr_lower}"
            if expr_node in G and src_id in G:
                try:
                    d2 = nx.shortest_path_length(G, expr_node, dst_id)
                    best_dist = min(best_dist, d2)
                except nx.NetworkXNoPath:
                    pass

        return round(best_dist, 2)

    return float(MAX_KG_PATH_LENGTH)


def compute_all_fcd(records: list[dict], kg: FigurativeKG) -> dict[str, float]:
    """
    Compute FCD for every record in the processed dataset.
    Returns {record_id: fcd_score}.
    """
    console.print("\n[bold cyan]Computing FCD scores...[/bold cyan]")
    fcd_map: dict[str, float] = {}

    for r in track(records, description="Computing FCD"):
        fcd = compute_fcd(r["expression"], r["figurative_type"], kg)
        fcd_map[r["id"]] = fcd

    # Save
    with open(FCD_SCORES_FILE, "w") as f:
        json.dump(fcd_map, f, indent=2)

    # Summary stats
    import statistics
    scores = list(fcd_map.values())
    console.print(f"  Min FCD:  {min(scores):.2f}")
    console.print(f"  Max FCD:  {max(scores):.2f}")
    console.print(f"  Mean FCD: {statistics.mean(scores):.2f}")
    console.print(f"  Median:   {statistics.median(scores):.2f}")
    console.print(f"[green]✓[/green] FCD scores saved → {FCD_SCORES_FILE}")

    return fcd_map


def load_fcd_scores() -> dict[str, float]:
    """Load saved FCD scores. Run build_kg.py first."""
    if not FCD_SCORES_FILE.exists():
        raise FileNotFoundError(
            f"FCD scores not found: {FCD_SCORES_FILE}\n"
            f"Run: python src/kg/build_kg.py"
        )
    with open(FCD_SCORES_FILE) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    console.rule("[bold]KG Construction & FCD Computation[/bold]")

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

    # Build KG
    kg = FigurativeKG()
    kg.build_from_dataset(records)
    kg.save()

    # Compute FCD
    fcd_map = compute_all_fcd(records, kg)

    console.rule()
    console.print(f"[bold green]Done.[/bold green] KG and FCD scores ready in {KG_DIR}")


if __name__ == "__main__":
    main()
