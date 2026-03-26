"""
attention_edges.py — Attention-Derived Edge Extraction

Extracts candidate graph edges from GPT-2 attention weights during a
forward pass. Every attention head already computes a weighted graph over
token positions — this module makes that graph explicit and maps it back
onto RGM concept nodes.

Two independent edge sources in RGM:
  1. Conversation structure  — what concepts co-occur and how (existing)
  2. Model internals         — what the model attends to (this file)

Where both sources agree, edges get high confidence.
Where they disagree, the tension is a signal worth recording.

This module is intentionally decoupled from core RGM. Drop it in or
leave it out — the rest of the architecture is unchanged.

Usage:
    extractor = AttentionEdgeExtractor(model, tokenizer, graph)

    with extractor.hooked():
        _ = model(input_ids, output_attentions=True)

    token_node_map = extractor.build_token_node_map(input_ids, graph)
    candidates     = extractor.get_edges(token_node_map)
    n_injected     = extractor.inject(candidates, graph, conv_pos)

Authors: Stig Norland, Claude (Anthropic)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from library.graph import ConceptGraph, EdgeType


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Attention weight must exceed this to produce a candidate edge.
# Lower = more edges, more noise. Higher = fewer edges, higher precision.
ATTENTION_THRESHOLD = 0.10

# Only heads whose specialisation type is in this set are used.
# Set to None to use all heads (noisier but more complete).
ENABLED_HEAD_TYPES = {"semantic", "induction"}

# Maximum candidate edges to return per forward pass.
MAX_EDGES_PER_PASS = 64

# Weight assigned to a fresh attention-derived edge.
# Kept below 1.0 so it doesn't dominate Hebbian-reinforced edges.
ATTENTION_EDGE_WEIGHT = 0.4

# Minimum label similarity to accept a token→node mapping.
TOKEN_NODE_MIN_SIM = 0.30


# ---------------------------------------------------------------------------
# Head type registry
#
# GPT-2 heads have known functional roles identified by mechanistic
# interpretability research (Olsson et al. 2022, Elhage et al. 2021).
# We label each (layer, head) pair so we can filter by semantic relevance.
#
# Partial map for GPT-2 small (12 layers × 12 heads).
# Unlabelled heads default to "unknown".
# ---------------------------------------------------------------------------

# fmt: off
GPT2_HEAD_TYPES: dict[tuple[int, int], str] = {
    # Induction heads — copy previous context patterns
    (1,  4): "induction",  (1, 10): "induction",
    (2,  2): "induction",  (2,  7): "induction",
    # Positional heads — track token distance
    (0,  1): "positional", (0,  5): "positional",
    (1,  2): "positional",
    # Semantic heads — track meaning relationships
    (3,  0): "semantic",   (3,  2): "semantic",
    (4,  4): "semantic",   (4,  9): "semantic",
    (5,  1): "semantic",   (5,  5): "semantic",
    (6,  0): "semantic",   (6,  8): "semantic",
    (7,  2): "semantic",   (7,  7): "semantic",
    (8,  6): "semantic",   (8, 10): "semantic",
    (9,  3): "semantic",   (9,  9): "semantic",
    (10, 0): "semantic",   (10, 5): "semantic",
    (11, 1): "semantic",   (11, 9): "semantic",
}
# fmt: on


def head_type(layer: int, head: int) -> str:
    return GPT2_HEAD_TYPES.get((layer, head), "unknown")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttentionEdgeCandidate:
    """
    A single candidate edge derived from attention weights.

    source_token / target_token: positions in the input sequence.
    source_node / target_node:   concept node ids (None until mapped).
    weight:    attention weight (0–1).
    layer, head: which head produced this candidate.
    head_type: functional role of the head.
    """
    source_token: int
    target_token: int
    weight:       float
    layer:        int
    head:         int
    head_type:    str
    source_node:  Optional[str] = None
    target_node:  Optional[str] = None


@dataclass
class AttentionPass:
    """Accumulated attention weights from one forward pass."""
    # attention_weights[layer][head] = [(src_tok, tgt_tok, weight), ...]
    attention_weights: list[list[list[tuple[int, int, float]]]] = field(
        default_factory=list
    )
    n_layers: int = 0
    n_heads:  int = 0
    seq_len:  int = 0


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

class _AttentionHook:
    """
    PyTorch forward hook that captures attention weights from one layer.

    GPT-2 attention output with output_attentions=True:
        (attn_output, attn_weights, ...)
    attn_weights shape: (batch, n_heads, seq_len, seq_len)
    """

    def __init__(self, layer_idx: int, store: AttentionPass):
        self._layer = layer_idx
        self._store = store

    def __call__(self, module, inputs, output):
        if not isinstance(output, tuple) or len(output) < 2:
            return
        attn_weights = output[1]
        if attn_weights is None:
            return

        w = attn_weights.detach().cpu()
        batch, n_heads, seq_len, _ = w.shape

        # Grow layer list if needed
        while len(self._store.attention_weights) <= self._layer:
            self._store.attention_weights.append([])

        for head in range(n_heads):
            while len(self._store.attention_weights[self._layer]) <= head:
                self._store.attention_weights[self._layer].append([])

            candidates = []
            for src in range(seq_len):
                for tgt in range(seq_len):
                    weight = float(w[0, head, src, tgt])
                    if weight >= ATTENTION_THRESHOLD:
                        candidates.append((src, tgt, weight))

            self._store.attention_weights[self._layer][head] = candidates

        self._store.n_layers = max(self._store.n_layers, self._layer + 1)
        self._store.n_heads  = max(self._store.n_heads, n_heads)
        self._store.seq_len  = seq_len


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class AttentionEdgeExtractor:
    """
    Extracts concept-level edges from GPT-2 attention weights.

    Registers hooks before a forward pass, captures weights, maps token
    positions back to concept nodes in the RGM graph, and injects or
    reinforces edges. Fully decoupled from core RGM — uses only the
    public graph API (_add_edge, _label_similarity, _adjacency).
    """

    def __init__(self, model, tokenizer, graph: "ConceptGraph"):
        self._model     = model
        self._tokenizer = tokenizer
        self._graph     = graph
        self._store     = AttentionPass()
        self._hooks     = []
        self._stats     = {"passes": 0, "candidates": 0, "injected": 0}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    @contextmanager
    def hooked(self):
        """
        Register attention hooks, yield (run your forward pass here),
        then cleanly remove hooks. No permanent model modification.

            with extractor.hooked():
                output = model(input_ids, output_attentions=True)
        """
        self._store = AttentionPass()
        self._register_hooks()
        try:
            yield
        finally:
            self._remove_hooks()
            self._stats["passes"] += 1

    def _register_hooks(self):
        # GPT-2 HuggingFace layout: model.transformer.h[i].attn
        # nanoGPT layout:           model.blocks[i].attn
        try:
            blocks = self._model.transformer.h
        except AttributeError:
            blocks = getattr(self._model, "blocks", [])

        for layer_idx, block in enumerate(blocks):
            attn_module = getattr(block, "attn", None)
            if attn_module is None:
                continue
            handle = attn_module.register_forward_hook(
                _AttentionHook(layer_idx, self._store)
            )
            self._hooks.append(handle)

    def _remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Token → concept node mapping
    # ------------------------------------------------------------------

    def build_token_node_map(self, input_ids,
                             graph: Optional["ConceptGraph"] = None
                             ) -> dict[int, str]:
        """
        Map token positions in input_ids to concept node ids.

        Decodes tokens to words, accumulates sub-word fragments, then
        matches whole words against graph node labels using the graph's
        own _label_similarity metric. Positions that don't match any
        node above TOKEN_NODE_MIN_SIM are omitted — they produce no edges.

        This is the main integration point. Plug in richer matching logic
        here if needed (e.g. embedding similarity instead of label bigrams).
        """
        g = graph or self._graph
        if g is None:
            return {}

        ids = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        tokens = [self._tokenizer.decode([tid]).strip() for tid in ids]

        token_node_map: dict[int, str] = {}
        word_buffer = ""
        word_start  = 0

        for pos, tok in enumerate(tokens):
            # GPT-2 uses Ġ prefix for space-preceded tokens
            is_word_start = tok.startswith("Ġ") or pos == 0
            clean_tok     = tok.lstrip("Ġ").lower()

            if is_word_start and word_buffer:
                node_id = self._match_word(word_buffer, g)
                if node_id:
                    for i in range(word_start, pos):
                        token_node_map[i] = node_id
                word_buffer = clean_tok
                word_start  = pos
            else:
                word_buffer += clean_tok

        # Final word
        if word_buffer:
            node_id = self._match_word(word_buffer, g)
            if node_id:
                for i in range(word_start, len(tokens)):
                    token_node_map[i] = node_id

        return token_node_map

    def _match_word(self, word: str, graph: "ConceptGraph") -> Optional[str]:
        if len(word) < 3:
            return None
        best_id    = None
        best_score = TOKEN_NODE_MIN_SIM
        for node in graph.nodes.values():
            score = graph._label_similarity(word, node.label)
            if score > best_score:
                best_score = score
                best_id    = node.id
        return best_id

    # ------------------------------------------------------------------
    # Edge extraction
    # ------------------------------------------------------------------

    def get_edges(self, token_node_map: dict[int, str]
                  ) -> list[AttentionEdgeCandidate]:
        """
        Convert raw attention weights into AttentionEdgeCandidates,
        filtered by head type and mapped to concept nodes.

        Returns a list sorted by weight descending, capped at MAX_EDGES_PER_PASS.
        Deduplicates to one edge per (source_node, target_node) pair,
        keeping the highest-weight instance.
        """
        candidates: list[AttentionEdgeCandidate] = []

        for layer_idx, layer in enumerate(self._store.attention_weights):
            for head_idx, triples in enumerate(layer):
                htype = head_type(layer_idx, head_idx)
                if ENABLED_HEAD_TYPES and htype not in ENABLED_HEAD_TYPES:
                    continue

                for src_tok, tgt_tok, weight in triples:
                    src_node = token_node_map.get(src_tok)
                    tgt_node = token_node_map.get(tgt_tok)
                    if src_node is None or tgt_node is None:
                        continue
                    if src_node == tgt_node:
                        continue

                    candidates.append(AttentionEdgeCandidate(
                        source_token = src_tok,
                        target_token = tgt_tok,
                        weight       = weight,
                        layer        = layer_idx,
                        head         = head_idx,
                        head_type    = htype,
                        source_node  = src_node,
                        target_node  = tgt_node,
                    ))

        # Deduplicate: keep highest-weight per (src, tgt) pair
        seen: dict[tuple[str, str], AttentionEdgeCandidate] = {}
        for c in candidates:
            key = (c.source_node, c.target_node)
            if key not in seen or c.weight > seen[key].weight:
                seen[key] = c

        unique = sorted(seen.values(), key=lambda c: -c.weight)
        self._stats["candidates"] += len(unique)
        return unique[:MAX_EDGES_PER_PASS]

    # ------------------------------------------------------------------
    # Graph injection
    # ------------------------------------------------------------------

    def inject(self, candidates: list[AttentionEdgeCandidate],
               graph: Optional["ConceptGraph"] = None,
               conv_pos: int = 0) -> int:
        """
        Write attention-derived edges into the concept graph.

        New edges are created with ATTENTION_EDGE_WEIGHT.
        Existing edges (already present from conversation structure)
        are reinforced — the two sources accumulate via Hebbian update.

        Returns the number of edges injected or reinforced.
        """
        from library.graph import EdgeType

        g = graph or self._graph
        if g is None:
            return 0

        edge_type_map = {
            "semantic":   EdgeType.SEMANTIC,
            "induction":  EdgeType.SEMANTIC,
            "positional": EdgeType.TEMPORAL,
            "unknown":    EdgeType.SEMANTIC,
        }

        n_injected = 0
        for c in candidates:
            if c.source_node not in g.nodes or c.target_node not in g.nodes:
                continue
            etype = edge_type_map.get(c.head_type, EdgeType.SEMANTIC)
            g.conv_pos = conv_pos
            created = g._add_edge(
                source    = c.source_node,
                target    = c.target_node,
                edge_type = etype,
                weight    = ATTENTION_EDGE_WEIGHT,
            )
            if created:
                g._write_count += 1
            n_injected += 1

        self._stats["injected"] += n_injected
        return n_injected

    # ------------------------------------------------------------------
    # Tension detection
    # ------------------------------------------------------------------

    def tension_candidates(self,
                           candidates: list[AttentionEdgeCandidate],
                           graph: Optional["ConceptGraph"] = None
                           ) -> list[tuple[str, str, float]]:
        """
        Find pairs where attention weight is high but no graph edge exists.

        These are concepts the model relates internally that the conversation
        structure hasn't captured — potential new edges or unresolved gaps.

        Returns [(source_label, target_label, attention_weight)] descending.
        """
        g = graph or self._graph
        if g is None:
            return []

        missing = []
        for c in candidates:
            if c.source_node is None or c.target_node is None:
                continue
            if c.target_node not in g._adjacency.get(c.source_node, {}):
                src_label = g.nodes[c.source_node].label
                tgt_label = g.nodes[c.target_node].label
                missing.append((src_label, tgt_label, c.weight))

        return sorted(missing, key=lambda x: -x[2])

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def summary(self) -> str:
        s = self._stats
        return (
            f"AttentionEdgeExtractor: "
            f"{s['passes']} passes, "
            f"{s['candidates']} candidates, "
            f"{s['injected']} injected"
        )