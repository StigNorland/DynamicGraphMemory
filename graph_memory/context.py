"""
context.py — Graph-to-Context Assembly

Translates relational graph state into token sequences
that nanoGPT can consume. This is the file that replaces
the KV cache as the context assembly mechanism.

The key operation: given a query and a token budget,
traverse the relevant subgraph and assemble context
from graph structure + selective backing store dereference.

Authors: Stig [last name], Claude (Anthropic)
"""

import re
from dataclasses import dataclass
from typing import Optional

from .graph import ConceptGraph, NodeType, EdgeType


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
        """
        Add a turn to the backing store.
        Returns position (the pointer stored in graph edges/nodes).
        """
        pos = len(self.turns)
        self.turns.append(Turn(
            pos     = pos,
            speaker = speaker,
            text    = text,
            nodes   = nodes or [],
        ))
        return pos

    def get(self, pos: int) -> Optional[Turn]:
        """Dereference a pointer. Returns None if out of range."""
        if 0 <= pos < len(self.turns):
            return self.turns[pos]
        return None

    def get_text(self, pos: int) -> str:
        """Dereference and return just the text."""
        turn = self.get(pos)
        return f"{turn.speaker}: {turn.text}" if turn else ""

    def recent(self, n: int) -> list[Turn]:
        """Return n most recent turns — used as fallback context."""
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

    Having both in one class gives clean experimental comparison —
    same token budget, same backing store, only context assembly differs.
    """

    def __init__(self, graph: ConceptGraph, store: BackingStore,
                 tokenizer=None):
        self.graph     = graph
        self.store     = store
        self.tokenizer = tokenizer  # optional — used for precise budgeting

    def assemble_graph(self, query: str, token_budget: int) -> str:
        """
        Graph-mode context assembly.

        1. Extract concept labels from query
        2. Find matching nodes in graph
        3. Traverse subgraph outward, weighted by maturity + edge weight
        4. Dereference backing store for high-value nodes
        5. Fit within token budget
        """
        # Step 1: extract query concepts
        query_labels  = self._extract_concepts(query)

        # Step 2: find matching graph nodes
        query_node_ids = []
        for label in query_labels:
            node_id, score = self.graph._find_match(label, NodeType.CONCEPT)
            if node_id and score > 0.3:  # loose threshold for query matching
                query_node_ids.append((node_id, score))

        # Fallback: if no graph match, use recent turns
        if not query_node_ids:
            return self._fallback_context(token_budget)

        # Step 3 & 4: traverse and assemble
        context_parts = []
        tokens_used   = 0
        visited       = set()

        # Priority: high maturity nodes connected to query
        frontier = sorted(query_node_ids, key=lambda x: -x[1])

        while frontier and tokens_used < token_budget:
            node_id, _ = frontier.pop(0)

            if node_id not in self.graph.nodes:
                continue
            if node_id in visited:
                continue
            visited.add(node_id)

            node = self.graph.nodes[node_id]

            # Dereference backing store for this node's origin turn
            turn_text = self.store.get_text(node.conv_pos)
            if turn_text:
                fragment_tokens = self._count_tokens(turn_text)
                if tokens_used + fragment_tokens <= token_budget:
                    context_parts.append((node.maturity, turn_text))
                    tokens_used += fragment_tokens

            # Expand neighbors weighted by edge strength * neighbor maturity
            neighbors = self.graph._get_neighbors(node_id)
            for neighbor_id, edge_weight in neighbors:
                if neighbor_id not in visited and neighbor_id in self.graph.nodes:
                    neighbor_maturity = self.graph.nodes[neighbor_id].maturity
                    priority = edge_weight * (1.0 + neighbor_maturity * 0.1)
                    frontier.append((neighbor_id, priority))

            # Keep frontier sorted by priority
            frontier.sort(key=lambda x: -x[1])

        # Also include merge event context — aha moments are high value
        merge_context = self._merge_context(token_budget - tokens_used)
        if merge_context:
            context_parts.append((999.0, merge_context))  # high priority

        # Sort by maturity (most established concepts first)
        # then assemble into string
        context_parts.sort(key=lambda x: -x[0])
        context = "\n".join(text for _, text in context_parts)

        return context.strip()

    def assemble_baseline(self, token_budget: int) -> str:
        """
        Baseline mode: raw recent conversation.
        Standard KV-cache style — most recent turns up to token budget.

        This is the control condition for experiments.
        Same backing store, no graph involvement.
        """
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
    # Ingestion — process new turns into graph
    # -------------------------------------------------------------------

    def ingest(self, speaker: str, text: str) -> dict:
        """
        Process a new conversation turn:
        1. Add to backing store
        2. Extract concepts
        3. Write to graph
        4. Return graph status

        This is the main entry point during conversation.
        """
        # Add to backing store first — get position pointer
        pos = self.store.append(speaker, text)
        self.graph.conv_pos = pos

        # Extract and write concepts
        concepts     = self._extract_concepts(text)
        written_ids  = []
        novelties    = []

        self.graph.begin_pass()

        prev_id = None
        for i, concept in enumerate(concepts):
            # Determine node type heuristically
            # POC will refine this — for now concept is default
            node_type = self._classify_node_type(concept, text)

            # Write with temporal relation to previous concept
            related = [prev_id] if prev_id else []
            edge_t  = EdgeType.TEMPORAL if prev_id else EdgeType.SEMANTIC

            node_id, is_new = self.graph.write(
                label      = concept,
                node_type  = node_type,
                related_to = related,
                edge_type  = edge_t,
            )

            written_ids.append(node_id)
            novelties.append(is_new)
            prev_id = node_id

        # Update backing store turn with node pointers
        self.store.turns[pos].nodes = written_ids
        self.graph.advance()

        status = self.graph.end_pass()
        status["concepts_extracted"] = len(concepts)
        status["nodes_created"]      = sum(novelties)
        status["graph_summary"]      = self.graph.summary()

        return status

    # -------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------

    def _extract_concepts(self, text: str) -> list[str]:
        """
        Extract concept labels from text.

        Current implementation: simple noun/keyword extraction.
        POC placeholder — in production this would use a lightweight
        NLP model or the graph's own pattern recognition.

        Intentionally naive so it doesn't obscure the graph behavior.
        """
        # Lowercase, strip punctuation, split
        text    = text.lower()
        text    = re.sub(r'[^\w\s]', ' ', text)
        words   = text.split()

        # Filter stopwords — very basic list, expand as needed
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

        concepts = [w for w in words
                    if w not in stopwords and len(w) > 2]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique

    def _classify_node_type(self, concept: str, context: str) -> NodeType:
        """
        Heuristic node type classification.
        Temporal markers → TEMPORAL
        Causal markers   → CAUSAL
        Everything else  → CONCEPT

        POC placeholder — graph's own pattern recognition
        will supersede this as it matures.
        """
        temporal_markers = {
            'before', 'after', 'when', 'while', 'during',
            'then', 'now', 'later', 'earlier', 'first', 'last',
            'yesterday', 'tomorrow', 'today', 'always', 'never',
        }
        causal_markers = {
            'because', 'therefore', 'thus', 'hence', 'since',
            'causes', 'leads', 'results', 'produces', 'prevents',
            'enables', 'requires', 'implies', 'means',
        }

        if concept in temporal_markers:
            return NodeType.TEMPORAL
        if concept in causal_markers:
            return NodeType.CAUSAL
        return NodeType.CONCEPT

    def _merge_context(self, remaining_budget: int) -> str:
        """
        Include context about significant merge events.
        Aha moments are high semantic value — worth including
        even at the cost of other context.
        """
        if not self.graph.merge_events:
            return ""

        # Most significant merge first
        significant = sorted(
            self.graph.merge_events,
            key=lambda e: -e.magnitude
        )

        parts = []
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
        """No graph match — fall back to recent turns."""
        return self.assemble_baseline(token_budget)

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        Uses tokenizer if available, otherwise word count as proxy.
        Word count is good enough for budget management in the POC.
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())