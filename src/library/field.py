"""
field.py — Bioelectric Field Layer (Holographic Memory)

Inspired by Michael Levin's work on bioelectric memory in planaria and
xenobots (2024–2026). In Levin's model, memory is stored as a distributed
bioelectric field across tissue — not in any single cell or nucleus. When
a trained planarian is cut, every fragment regenerates a complete worm
retaining the original learned behaviour, because each fragment holds a
compressed projection of the global field pattern (holographic storage).

Key insight: in Levin's framework, the field is the memory. Individual cells
store *local projections* of a low-dimensional global attractor. Any
sufficiently large fragment can reconstruct the global pattern by summing
its projections — exactly the property a hologram has.

Mapping to RGM:

  Bioelectric field (Vmem gradients)  ←→  Global field vector f ∈ ℝ^d
  Individual cell (engram)            ←→  Mature Concept node
  Local projection per cell           ←→  node.meta["field_projection"]
  Gap junction propagation            ←→  Hebbian edge-weighted averaging
  Fragment regeneration               ←→  Graph-only field reconstruction
  Mnemonic improvisation              ←→  Field rewrite on merge/rejuvenation

Implementation:

  BioelectricField maintains a global vector (d=64 by default) built from
  a random-projection of TF-IDF weighted term frequencies across all turns.
  Pure stdlib — no numpy, no external dependencies.

  On every ingest() call the field is updated incrementally.
  On _stabilize() (called from graph.py), the node's local projection is
  written to node.meta["field_projection"].

  At context assembly time (graph-only mode), the assembler reconstructs
  an approximate global field by averaging the visible nodes' projections,
  then generates a "field reconstruction summary" — a ranked list of the
  most field-active terms, providing the LLM with a holographic hint about
  the full conversation even from a small fragment.

Usage:

    from .field import BioelectricField

    field = BioelectricField(dim=64)
    field.ingest_turn(speaker, text, conv_pos)
    # ... after graph._stabilize(node_id):
    field.project_node(node, graph)
    # At assembly time:
    summary = field.reconstruct(visible_node_ids, graph)

Authors: Stig Norland, Claude (Anthropic)

References:
  Levin, M. (2024). Bioelectric networks: the cognitive glue enabling
    evolutionary scaling from physiology to mind. Animal Cognition.
  Berdugo-Vega, G. et al. (2026). Epigenetic rejuvenation of engram cells
    restores memory accessibility in aged mice. Neuron.
  Tarulli, G. et al. (2026). Nuclear epigenetic traces of original memory
    persist in Alzheimer's-model engram cells. J. Neurochem.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ConceptGraph, Node


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_DIM       = 64       # dimensionality of the global field vector
VOCAB_MAX       = 2000     # max vocabulary size (oldest terms drop out)
IDF_SMOOTH      = 1.0      # Laplace smoothing for IDF
PROJ_SEED       = 2718     # fixed seed for reproducible random projection
DECAY_ALPHA     = 0.02     # per-turn field decay (keeps field current)
TOP_TERMS_K     = 12       # terms in reconstruction summary

# Stopwords for TF computation
_STOPS = frozenset({
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","can","to","of","in","for","on","with",
    "at","by","from","up","about","into","i","you","he","she","it",
    "we","they","what","which","who","that","this","these","those",
    "and","but","or","nor","not","so","yet","as","if","then",
    "because","while","although","just","also","very","more",
})


# ---------------------------------------------------------------------------
# Random projection matrix (Achlioptas 2003, pure stdlib)
# ---------------------------------------------------------------------------

def _build_projection_matrix(vocab_size: int, dim: int,
                               seed: int = PROJ_SEED) -> list[list[float]]:
    """
    Build a stable random projection matrix P ∈ ℝ^{dim × vocab_size}.
    Uses Achlioptas sparse {+1, -1} construction — preserves distances
    (Johnson-Lindenstrauss lemma) with minimal memory and computation.
    The matrix is rebuilt lazily when the vocabulary grows.
    """
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(dim)
    return [
        [scale * (1.0 if rng.random() > 0.5 else -1.0)
         for _ in range(vocab_size)]
        for _ in range(dim)
    ]


# ---------------------------------------------------------------------------
# BioelectricField
# ---------------------------------------------------------------------------

class BioelectricField:
    """
    Holographic field memory for RGM.

    The field is a global low-dimensional vector that encodes the
    distributed "memory" of the entire conversation, updated incrementally
    as turns arrive. Each mature node stores a local projection (its
    "piece of the holographic plate"). Any subset of nodes can reconstruct
    an approximation of the global field and recover the dominant themes
    of the conversation — even if the specific backing-store turns are
    not available.

    This implements Levin's key property: any sufficiently large fragment
    carries enough information to regenerate the whole.
    """

    def __init__(self, dim: int = FIELD_DIM):
        self.dim          = dim
        self._field       = [0.0] * dim      # global field vector f ∈ ℝ^d
        self._vocab:      dict[str, int]  = {}   # term -> index
        self._df:         Counter         = Counter()  # document freq
        self._tf_total:   Counter         = Counter()  # corpus term freq
        self._n_docs:     int             = 0          # turns ingested
        self._proj:       list[list[float]] = []       # projection matrix
        self._proj_vocab_size: int        = 0          # size when built
        self._node_projections: dict[str, list[float]] = {}  # node_id -> proj
        self._node_turns: dict[str, list[int]] = defaultdict(list)  # node -> turn positions
        self._turn_terms: list[list[str]] = []  # terms per turn (for IDF)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_turn(self, speaker: str, text: str, conv_pos: int) -> None:
        """
        Update the global field with a new conversation turn.
        Called once per turn during graph ingestion.
        """
        terms = self._tokenise(text)
        if not terms:
            return

        self._n_docs += 1
        self._turn_terms.append(terms)

        # Update vocabulary and document frequencies
        for term in set(terms):
            if term not in self._vocab:
                if len(self._vocab) >= VOCAB_MAX:
                    # Drop the least frequent term to make room
                    least = min(self._vocab, key=lambda t: self._tf_total[t])
                    idx = self._vocab.pop(least)
                    del self._tf_total[least]
                    del self._df[least]
                self._vocab[term] = len(self._vocab)
            self._df[term] += 1
            self._tf_total[term] += terms.count(term)

        # Build/rebuild projection matrix if vocabulary grew
        if len(self._vocab) > self._proj_vocab_size:
            self._rebuild_projection()

        # Compute TF-IDF vector for this turn
        tfidf = self._tfidf_vector(terms)

        # Project to field dimension
        turn_field = self._project(tfidf)

        # Levin-style update: decay existing field, add new turn's contribution
        # This is analogous to gap-junction propagation — the field integrates
        # new signals while maintaining the global attractor
        for i in range(self.dim):
            self._field[i] = (1.0 - DECAY_ALPHA) * self._field[i] + turn_field[i]

        # Normalise to unit sphere (maintains field as an attractor state)
        self._normalise_field()

    def project_node(self, node: "Node", graph: "ConceptGraph") -> None:
        """
        Write a local field projection onto a node when it stabilises.
        This is the "engram allocation" step — the node receives its
        piece of the holographic plate at the moment it matures.

        The projection is the current global field modulated by the node's
        local connectivity (its neighbourhood in the graph).
        """
        if "field_projection" in node.meta:
            return  # idempotent — don't overwrite existing projection

        # Node's local influence: weight by neighbour edge strengths
        # This implements gap-junction coupling — the node's projection
        # is influenced by its neighbours' activations
        local_weight = [1.0] * self.dim   # default: take global field as-is

        neighbours = graph._get_neighbors(node.id)
        if neighbours:
            # Neighbours that also have projections contribute their field
            nb_fields = []
            for nb_id, w, _ in neighbours:
                if nb_id in self._node_projections:
                    nb_field = self._node_projections[nb_id]
                    nb_fields.append((w, nb_field))

            if nb_fields:
                # Weighted average of neighbour projections
                total_w = sum(w for w, _ in nb_fields)
                coupled = [0.0] * self.dim
                for w, nf in nb_fields:
                    for i in range(self.dim):
                        coupled[i] += (w / total_w) * nf[i]
                # Blend global field with local coupling (50/50)
                local_weight = [
                    0.5 * self._field[i] + 0.5 * coupled[i]
                    for i in range(self.dim)
                ]
            else:
                local_weight = list(self._field)
        else:
            local_weight = list(self._field)

        node.meta["field_projection"] = local_weight
        self._node_projections[node.id] = local_weight

    def reconstruct(self, visible_node_ids: list[str],
                    graph: "ConceptGraph",
                    top_k: int = TOP_TERMS_K) -> str:
        """
        Reconstruct the global field from a fragment of visible nodes.
        Returns a natural-language "field reconstruction summary" suitable
        for injection into graph-only context.

        This is the planaria property: even a fragment of nodes can
        reconstruct the dominant themes of the full conversation.
        """
        if not visible_node_ids:
            return ""

        # Collect projections from visible nodes
        projections = []
        maturities  = []
        for node_id in visible_node_ids:
            if node_id not in graph.nodes:
                continue
            node = graph.nodes[node_id]
            proj = node.meta.get("field_projection")
            if proj:
                projections.append(proj)
                maturities.append(node.maturity)

        if not projections:
            return ""

        # Reconstruct global field: maturity-weighted average of local projections
        total_m = sum(maturities) + 1e-9
        recon   = [0.0] * self.dim
        for proj, m in zip(projections, maturities):
            for i in range(self.dim):
                recon[i] += (m / total_m) * proj[i]

        # Measure reconstruction fidelity (cosine with global field)
        fidelity = self._cosine(recon, self._field)

        # Invert projection to find dominant vocabulary terms
        # Each term's "field activation" = dot(recon, projection_of_term)
        term_activations = {}
        for term, idx in self._vocab.items():
            if idx >= self._proj_vocab_size:
                continue
            # Dot product of recon with this term's projection column
            activation = sum(
                recon[d] * self._proj[d][idx]
                for d in range(self.dim)
            )
            # Weight by IDF so common terms don't dominate
            idf = math.log((self._n_docs + IDF_SMOOTH) /
                           (self._df[term] + IDF_SMOOTH))
            term_activations[term] = activation * idf

        # Top-k most field-active terms
        top_terms = sorted(term_activations.items(),
                           key=lambda x: -x[1])[:top_k]
        top_terms = [t for t, _ in top_terms if t not in _STOPS][:top_k]

        if not top_terms:
            return ""

        # Format as a compact field reconstruction summary
        fidelity_pct = int(fidelity * 100)
        summary = (
            f"[Field reconstruction from {len(projections)} nodes, "
            f"fidelity={fidelity_pct}%] "
            f"Dominant themes: {', '.join(top_terms[:8])}"
        )
        return summary

    def rejuvenate(self, node: "Node", graph: "ConceptGraph") -> None:
        """
        Levin's 'mnemonic improvisation': rewrite a node's field projection
        using the current global field state, without changing its relational
        signature.

        Triggered when a node's accessibility has degraded (e.g. its
        backing-store turns are no longer in the active context window).
        The projection is refreshed from the current field — restoring
        access to the global pattern without erasing the node's content.
        """
        # Remove old projection so project_node will rewrite it
        node.meta.pop("field_projection", None)
        self._node_projections.pop(node.id, None)
        self.project_node(node, graph)

    @property
    def field_vector(self) -> list[float]:
        """Read-only access to the current global field state."""
        return list(self._field)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def n_turns(self) -> int:
        return self._n_docs

    def stats(self) -> dict:
        n_projected = sum(
            1 for nid, proj in self._node_projections.items()
            if proj is not None
        )
        field_norm = sum(x*x for x in self._field) ** 0.5
        return {
            "dim":           self.dim,
            "vocab_size":    self.vocab_size,
            "n_turns":       self.n_turns,
            "n_projected":   n_projected,
            "field_norm":    round(field_norm, 4),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> list[str]:
        """Simple tokeniser: lowercase, alphanumeric only, no stopwords."""
        words = re.findall(r"[a-z]{3,}", text.lower())
        return [w for w in words if w not in _STOPS]

    def _tfidf_vector(self, terms: list[str]) -> list[float]:
        """Compute TF-IDF vector for a list of terms over the current vocab."""
        tf = Counter(terms)
        total_tf = sum(tf.values()) + 1e-9
        vec = [0.0] * len(self._vocab)
        for term, count in tf.items():
            if term in self._vocab:
                idx = self._vocab[term]
                tf_val  = count / total_tf
                idf_val = math.log(
                    (self._n_docs + IDF_SMOOTH) /
                    (self._df[term] + IDF_SMOOTH)
                )
                vec[idx] = tf_val * idf_val
        return vec

    def _project(self, tfidf: list[float]) -> list[float]:
        """Project a TF-IDF vector to field dimension via random projection."""
        if not self._proj or len(tfidf) != self._proj_vocab_size:
            return [0.0] * self.dim
        return [
            sum(self._proj[d][i] * tfidf[i]
                for i in range(len(tfidf)))
            for d in range(self.dim)
        ]

    def _rebuild_projection(self) -> None:
        """Rebuild the projection matrix when vocabulary grows."""
        vocab_size = len(self._vocab)
        self._proj = _build_projection_matrix(vocab_size, self.dim, PROJ_SEED)
        self._proj_vocab_size = vocab_size

    def _normalise_field(self) -> None:
        """L2-normalise the global field vector."""
        norm = sum(x * x for x in self._field) ** 0.5
        if norm > 1e-9:
            self._field = [x / norm for x in self._field]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        na  = sum(x * x for x in a) ** 0.5
        nb  = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-9)
