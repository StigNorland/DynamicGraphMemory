"""
context.py — Graph-to-Context Assembly

Translates relational graph state into token sequences
that nanoGPT can consume. This is the file that replaces
the KV cache as the context assembly mechanism.

The key operation: given a query and a token budget,
traverse the relevant subgraph and assemble context
from graph structure + selective backing store dereference.

Authors: Stig Norland, Claude (Anthropic)
"""

import re
from dataclasses import dataclass
from typing import Optional

from src.graph import ConceptGraph, NodeType, EdgeType

# ---------------------------------------------------------------------------
# Typed-triple extractor (replaces stopword filter)
# ---------------------------------------------------------------------------
try:
    from concept_extractor import extract_concepts as _spacy_extract
    _EXTRACTOR_AVAILABLE = True
except ImportError:
    _EXTRACTOR_AVAILABLE = False


# -------------------------------------------------------------------
# Backing store — the lazy-loaded conversation archive
# -------------------------------------------------------------------

@dataclass
class Turn:
    """A single conversation turn with its graph pointers."""
    pos:     int        # position in conversation
    speaker: str        # "user" or "assistant"
    text:    str        # raw text — the full fidelity record
    nodes:   list[str]  # graph node ids created/activated by this turn


class BackingStore:
    """
    The raw conversation, retained in full.
    Only dereferenced when the graph decides full fidelity is needed.

    This is the archive. The graph is the index.
    """

    def __init__(self):
        self.turns: list[Turn] = []

    def append(self, speaker: str, text: str,
               nodes: list[str] = None) -> int:
        pos = len(self.turns)
        self.turns.append(Turn(
            pos     = pos,
            speaker = speaker,
            text    = text,
            nodes   = nodes or [],
        ))
        return pos

    def get(self, pos: int) -> Optional[Turn]:
        if 0 <= pos < len(self.turns):
            return self.turns[pos]
        return None

    def get_text(self, pos: int) -> str:
        turn = self.get(pos)
        return f"{turn.speaker}: {turn.text}" if turn else ""

    def recent(self, n: int) -> list[Turn]:
        return self.turns[-n:] if self.turns else []

    def __len__(self):
        return len(self.turns)


# -------------------------------------------------------------------
# Context assembler
# -------------------------------------------------------------------

class ContextAssembler:
    """
    Assembles token context from graph state for model consumption.

    Two modes, selectable per inference call:
      - GRAPH:    traverse relevant subgraph, dereference selectively
      - BASELINE: raw recent conversation, standard KV-cache style
    """

    def __init__(self, graph: ConceptGraph, store: BackingStore,
                 tokenizer=None):
        self.graph     = graph
        self.store     = store
        self.tokenizer = tokenizer

    def assemble_graph(self, query: str, token_budget: int) -> str:
        query_labels   = self._extract_concepts(query)
        query_node_ids = []
        for label in query_labels:
            node_id, score = self.graph._find_match(label, NodeType.CONCEPT)
            if node_id and score > 0.3:
                query_node_ids.append((node_id, score))

        if not query_node_ids:
            return self._fallback_context(token_budget)

        context_parts     = []
        tokens_used       = 0
        visited           = set()
        visited_positions = set()

        frontier = sorted(query_node_ids, key=lambda x: -x[1])

        while frontier and tokens_used < token_budget:
            node_id, _ = frontier.pop(0)

            if node_id not in self.graph.nodes:
                continue
            if node_id in visited:
                continue
            visited.add(node_id)

            node = self.graph.nodes[node_id]

            if node.conv_pos not in visited_positions:
                visited_positions.add(node.conv_pos)
                turn_text = self.store.get_text(node.conv_pos)
                if turn_text:
                    fragment_tokens = self._count_tokens(turn_text)
                    if tokens_used + fragment_tokens <= token_budget:
                        context_parts.append((node.maturity, turn_text))
                        tokens_used += fragment_tokens

            neighbors = self.graph._get_neighbors(node_id)
            for neighbor_id, edge_weight in neighbors:
                if neighbor_id not in visited and neighbor_id in self.graph.nodes:
                    neighbor_maturity = self.graph.nodes[neighbor_id].maturity
                    priority = edge_weight * (1.0 + neighbor_maturity * 0.1)
                    frontier.append((neighbor_id, priority))

            frontier.sort(key=lambda x: -x[1])

        merge_context = self._merge_context(token_budget - tokens_used)
        if merge_context:
            context_parts.append((999.0, merge_context))

        context_parts.sort(key=lambda x: -x[0])
        context = "\n".join(text for _, text in context_parts)

        return context.strip()

    def assemble_baseline(self, token_budget: int) -> str:
        turns       = []
        tokens_used = 0

        for turn in reversed(self.store.turns):
            text   = f"{turn.speaker}: {turn.text}"
            tokens = self._count_tokens(text)
            if tokens_used + tokens > token_budget:
                break
            turns.append(text)
            tokens_used += tokens

        turns.reverse()
        return "\n".join(turns).strip()

    # -------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------

    def ingest(self, speaker: str, text: str) -> dict:
        pos = self.store.append(speaker, text)
        self.graph.conv_pos = pos

        concepts    = self._extract_concepts(text)
        written_ids = []
        novelties   = []

        self.graph.begin_pass()

        prev_id = None
        for concept in concepts:
            node_type = self._classify_node_type(concept, text)
            related   = [prev_id] if prev_id else []
            edge_t    = EdgeType.TEMPORAL if prev_id else EdgeType.SEMANTIC

            node_id, is_new = self.graph.write(
                label      = concept,
                node_type  = node_type,
                related_to = related,
                edge_type  = edge_t,
            )

            written_ids.append(node_id)
            novelties.append(is_new)
            prev_id = node_id

        self.store.turns[pos].nodes = written_ids
        self.graph.advance()

        status = self.graph.end_pass()
        status["concepts_extracted"] = len(concepts)
        status["nodes_created"]      = sum(novelties)
        status["graph_summary"]      = self.graph.summary()

        return status

    # -------------------------------------------------------------------
    # Concept extraction — spaCy IS-A extractor (with stopword fallback)
    # -------------------------------------------------------------------

    def _extract_concepts(self, text: str) -> list[str]:
        """
        Extract concept labels from text.

        Primary: spaCy-based typed-triple extractor (concept_extractor.py).
          Returns the subject and object labels of high-confidence IS-A
          triples, preserving full noun-phrase form
          (e.g. "dark_matter", "superfluid_vortex").

        Fallback: simple stopword filter (original implementation),
          used when spaCy is not installed.
        """
        if _EXTRACTOR_AVAILABLE:
            return self._extract_via_spacy(text)
        return self._extract_stopword_fallback(text)

    def _extract_via_spacy(self, text: str) -> list[str]:
        """
        Use the typed-triple extractor to pull IS-A concept labels.

        Both subject and object of each triple are added as distinct
        concept nodes — the IS-A relationship itself is encoded as
        a semantic edge between them when both nodes are written to
        the graph in sequence.
        """
        try:
            concept_dicts = _spacy_extract(text)
        except Exception:
            # Degrade gracefully if extraction fails on a specific turn
            return self._extract_stopword_fallback(text)

        labels = []
        seen   = set()

        for concept in concept_dicts:
            # Add subject
            subj = concept["label"]
            if subj and subj not in seen:
                seen.add(subj)
                labels.append(subj)
            # Add object nodes so they enter the graph as real concepts
            for rel in concept.get("relations", []):
                obj = rel.get("target", "")
                if obj and obj not in seen:
                    seen.add(obj)
                    labels.append(obj)

        return labels

    def _extract_stopword_fallback(self, text: str) -> list[str]:
        """Original stopword-filter extractor — used when spaCy unavailable."""
        text    = text.lower()
        text    = re.sub(r'[^\w\s]', ' ', text)
        words   = text.split()

        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
            'by', 'from', 'up', 'about', 'into', 'through', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'that', 'this', 'these', 'those', 'and', 'but',
            'or', 'nor', 'so', 'yet', 'both', 'either', 'not',
            'only', 'own', 'same', 'than', 'too', 'very', 'just',
            'as', 'if', 'then', 'because', 'while', 'although',
        }

        concepts = [w for w in words if w not in stopwords and len(w) > 2]

        seen   = set()
        unique = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    # -------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------

    def _classify_node_type(self, concept: str, context: str) -> NodeType:
        temporal_markers = {
            'before', 'after', 'when', 'while', 'during',
            'then', 'now', 'later', 'earlier', 'first', 'last',
        }
        causal_markers = {
            'because', 'therefore', 'thus', 'hence', 'since',
            'causes', 'leads', 'results', 'produces', 'prevents',
            'enables', 'requires', 'implies',
        }
        if concept in temporal_markers:
            return NodeType.TEMPORAL
        if concept in causal_markers:
            return NodeType.CAUSAL
        return NodeType.CONCEPT

    def _merge_context(self, remaining_budget: int) -> str:
        if not self.graph.merge_events:
            return ""

        significant = sorted(
            self.graph.merge_events,
            key=lambda e: -e.magnitude
        )

        parts  = []
        tokens = 0
        for event in significant:
            node_a = self.graph.nodes.get(event.merged_into)
            if not node_a:
                continue
            text = (f"[insight: '{event.node_a}' and '{event.node_b}' "
                    f"unified into '{node_a.label}' "
                    f"revealing {event.cascade_depth} new connections]")
            t = self._count_tokens(text)
            if tokens + t <= remaining_budget:
                parts.append(text)
                tokens += t

        return "\n".join(parts)

    def _fallback_context(self, token_budget: int) -> str:
        return self.assemble_baseline(token_budget)

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())