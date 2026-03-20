"""
graph.py — Relational Concept Graph

The primary working memory substrate. Concepts are nodes defined
entirely by their relationships. Meaning is relational structure.

Authors: Stig [last name], Claude (Anthropic)
"""

import hashlib
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from concept_extractor import _spacy_extract
concepts = _spacy_extract(turn_text)

# -------------------------------------------------------------------
# Types
# -------------------------------------------------------------------

class NodeType(Enum):
    CONCEPT  = "concept"
    TEMPORAL = "temporal"
    CAUSAL   = "causal"


class EdgeType(Enum):
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL   = "causal"


# -------------------------------------------------------------------
# Core structures
# -------------------------------------------------------------------

@dataclass
class Node:
    id:          str
    node_type:   NodeType
    label:       str                        # human readable, not identity
    provisional: bool        = True         # identity not yet stable
    maturity:    float       = 0.0
    conv_pos:    int         = 0            # pointer to backing store
    meta:        dict        = field(default_factory=dict)


@dataclass
class Edge:
    source:      str                        # node id
    target:      str                        # node id
    edge_type:   EdgeType
    weight:      float       = 1.0
    conv_pos:    int         = 0            # pointer to backing store
    maturity_at_creation: float = 0.0
    meta:        dict        = field(default_factory=dict)


@dataclass
class MergeEvent:
    node_a:        str
    node_b:        str
    merged_into:   str
    conv_pos:      int
    cascade_depth: int
    magnitude:     float
    new_edges:     list      = field(default_factory=list)


# -------------------------------------------------------------------
# Graph
# -------------------------------------------------------------------

class ConceptGraph:
    """
    Relational concept graph as primary working memory.

    Nodes are language-agnostic concepts. Identity is determined
    by relational signature, not surface form. The write operation
    is the novelty signal — if nothing changes, nothing is new.
    """

    # Tunable — POC will determine good values empirically
    MATURITY_THRESHOLD    = 5.0
    LEARNING_RATE         = 0.1
    SIMILARITY_THRESHOLD  = 0.85   # for concept matching (keep strict)
    HASH_WINDOW           = 8       # recent states for loop detection


    @property
    def maturity_threshold(self) -> float:
        """Scale maturity threshold with graph size."""
        base = 5.0
        size_factor = max(1.0, len(self.nodes) / 60)
        return base * (1.0 / size_factor ** 0.3)

    @property
    def merge_threshold(self) -> float:
        """
        Dynamic merge threshold that scales with graph density.

        A fixed threshold measures graph density as much as semantic
        similarity — the same degree of conceptual overlap produces
        different Jaccard scores depending on how many neighbors
        nodes have accumulated.

        Solution: scale threshold inversely with mean node degree.
        As graph densifies, threshold lowers, allowing more nuanced
        merges to fire naturally.

        Early conversation  → sparse graph  → high threshold → only obvious merges
        Mid conversation    → denser graph  → lower threshold → nuanced merges
        Late conversation   → mature graph  → stable threshold → self-regulating

        Floor: 0.05  — never merge everything
        Ceiling: 0.30 — never require perfect overlap
        """
        if not self.nodes:
            return 0.30

        total_degree = sum(
            len(self._get_neighbors(n_id))
            for n_id in self.nodes
        )
        mean_degree = total_degree / len(self.nodes)

        # Inverse scaling — denser graph, lower threshold
        # 0.30 / (1 + mean_degree * 0.1) gives:
        #   mean_degree=0  → 0.30
        #   mean_degree=5  → 0.20
        #   mean_degree=10 → 0.15
        #   mean_degree=20 → 0.10
        #   mean_degree=50 → 0.06
        threshold = 0.30 / (1.0 + mean_degree * 0.1)
        return max(0.05, min(0.30, threshold))

    def __init__(self):
        self.nodes:        dict[str, Node]       = {}
        self.edges:        dict[str, Edge]       = {}  # key: "src::tgt::type"
        self.merge_events: list[MergeEvent]      = []
        self.state_hashes: list[str]             = []  # rolling window
        self.conv_pos:     int                   = 0   # current position
        self._write_count: int                   = 0   # writes this pass

    # -------------------------------------------------------------------
    # Write policy — the heart of the architecture
    # -------------------------------------------------------------------

    def write(self, label: str, node_type: NodeType,
              related_to: list[str] = None,
              edge_type: EdgeType = EdgeType.SEMANTIC) -> tuple[str, bool]:
        """
        Write a concept into the graph.

        Returns (node_id, is_new) where is_new is the novelty signal.
        If is_new is False, nothing changed — this is not new in current context.

        Write cases:
          1. No match found        → create provisional node
          2. Exact match           → reinforce, no new structure
          3. Partial match         → create delta edge capturing difference
          4. Maturity threshold    → stabilize identity, check for merges
        """
        novelty = False

        # Find best matching existing node
        match_id, similarity = self._find_match(label, node_type)

        if match_id is None:
            # Case 1: genuinely new concept
            node_id = self._create_node(label, node_type)
            novelty = True
        elif similarity >= self.SIMILARITY_THRESHOLD:
            # Case 2: existing concept — reinforce
            node_id = match_id
            self._reinforce_node(node_id)
        else:
            # Case 3: partial match — create delta
            node_id = self._create_node(label, node_type)
            self._add_edge(node_id, match_id, EdgeType.SEMANTIC, weight=similarity)
            novelty = True

        # Connect to related nodes
        if related_to:
            for rel_id in related_to:
                if rel_id in self.nodes:
                    created = self._add_edge(node_id, rel_id, edge_type)
                    if created:
                        novelty = True

        # Update maturity and check for stabilization
        self._update_maturity(node_id)
        if self.nodes[node_id].provisional:
            if self.nodes[node_id].maturity >= self.maturity_threshold:
                self._stabilize(node_id)

        if novelty:
            self._write_count += 1

        return node_id, novelty

    # -------------------------------------------------------------------
    # Dynamic looping interface
    # -------------------------------------------------------------------

    def begin_pass(self):
        """Mark start of a propagation pass."""
        self._write_count = 0

    def end_pass(self) -> dict:
        """
        Mark end of a propagation pass.
        Returns convergence status.

        No writes = convergence.
        Repeated state hash = loop detected.
        """
        h = self._state_hash()
        loop_detected = h in self.state_hashes

        # Maintain rolling window
        self.state_hashes.append(h)
        if len(self.state_hashes) > self.HASH_WINDOW:
            self.state_hashes.pop(0)

        return {
            "writes":         self._write_count,
            "converged":      self._write_count == 0,
            "loop_detected":  loop_detected,
            "state_hash":     h,
            # Loop is informative — expose it, don't just terminate
            "tension":        loop_detected and self._write_count > 0,
        }

    def advance(self):
        """Advance conversation position pointer."""
        self.conv_pos += 1

    # -------------------------------------------------------------------
    # Context assembly — graph → token budget
    # -------------------------------------------------------------------

    def get_context(self, query_nodes: list[str],
                    token_budget: int,
                    backing_store: list[str]) -> str:
        """
        Assemble context from graph for model consumption.

        Traverses relevant subgraph from query nodes outward,
        weighted by edge strength and node maturity.
        Dereferences backing store pointers only when needed.

        Returns a string fitting within token_budget.
        """
        visited   = set()
        context   = []
        tokens    = 0

        # Priority queue: (negative_weight, node_id)
        # Simple greedy traversal for now — BFS weighted by maturity
        frontier = [(-(self.nodes[n].maturity), n)
                    for n in query_nodes if n in self.nodes]
        frontier.sort()

        while frontier and tokens < token_budget:
            _, node_id = frontier.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            node = self.nodes[node_id]

            # Dereference backing store for this node's origin
            if node.conv_pos < len(backing_store):
                fragment = backing_store[node.conv_pos]
                fragment_tokens = len(fragment.split())
                if tokens + fragment_tokens <= token_budget:
                    context.append(fragment)
                    tokens += fragment_tokens

            # Expand neighbors, weighted by edge strength
            neighbors = self._get_neighbors(node_id)
            for neighbor_id, weight in neighbors:
                if neighbor_id not in visited:
                    frontier.append((-weight, neighbor_id))
            frontier.sort()

        return " ".join(context)

    # -------------------------------------------------------------------
    # Internal: node and edge operations
    # -------------------------------------------------------------------

    def _create_node(self, label: str, node_type: NodeType) -> str:
        node_id = f"{node_type.value}::{label}::{len(self.nodes)}"
        self.nodes[node_id] = Node(
            id        = node_id,
            node_type = node_type,
            label     = label,
            conv_pos  = self.conv_pos,
        )
        return node_id

    def _add_edge(self, source: str, target: str,
                  edge_type: EdgeType, weight: float = 1.0) -> bool:
        """Returns True if edge is new (novelty signal)."""
        key = f"{source}::{target}::{edge_type.value}"
        if key in self.edges:
            # Reinforce existing edge
            self.edges[key].weight = min(
                1.0,
                self.edges[key].weight + self.LEARNING_RATE
            )
            return False
        else:
            self.edges[key] = Edge(
                source     = source,
                target     = target,
                edge_type  = edge_type,
                weight     = weight,
                conv_pos   = self.conv_pos,
                maturity_at_creation = self.nodes[source].maturity,
            )
            return True

    def _reinforce_node(self, node_id: str):
        """Strengthen all edges connected to this node."""
        for edge in self.edges.values():
            if edge.source == node_id or edge.target == node_id:
                edge.weight = min(1.0, edge.weight + self.LEARNING_RATE)

    def _update_maturity(self, node_id: str):
        """
        maturity = (edge_count * mean_edge_weight)
                   * neighborhood_maturity_factor
        """
        connected = [e for e in self.edges.values()
                     if e.source == node_id or e.target == node_id]
        if not connected:
            return

        edge_count       = len(connected)
        mean_weight      = sum(e.weight for e in connected) / edge_count

        # Neighborhood factor: mean maturity of connected nodes
        neighbor_ids = set()
        for e in connected:
            neighbor_ids.add(e.target if e.source == node_id else e.source)

        if neighbor_ids:
            neighbor_maturity = sum(
                self.nodes[n].maturity
                for n in neighbor_ids if n in self.nodes
            ) / len(neighbor_ids)
            # Dampen neighborhood influence to avoid runaway growth
            neighborhood_factor = 1.0 + (neighbor_maturity * 0.1)
        else:
            neighborhood_factor = 1.0

        self.nodes[node_id].maturity = (
            edge_count * mean_weight * neighborhood_factor
        )

    def _stabilize(self, node_id: str):
        """
        Node has reached maturity threshold.
        Drop provisional flag, check for merge candidates.
        """
        self.nodes[node_id].provisional = False
        self._check_merges(node_id)

    # -------------------------------------------------------------------
    # Internal: merge — the aha moment
    # -------------------------------------------------------------------

    def _check_merges(self, node_id: str):
        """
        Check whether this newly stabilized node shares a relational
        signature with any other stable node. If so, merge.
        """
        node = self.nodes[node_id]
        candidates = [
            n for n in self.nodes.values()
            if not n.provisional
            and n.id != node_id
            and n.node_type == node.node_type
        ]

        for candidate in candidates:
            similarity = self._signature_similarity(node_id, candidate.id)

            if similarity >= self.merge_threshold:
                self._merge(node_id, candidate.id)
                break  # one merge at a time — cascades handled recursively

    def _merge(self, node_a_id: str, node_b_id: str):
        """
        Merge two nodes whose relational signatures have converged.
        This is an aha moment — record it as a first-class event.
        """
        # node_a absorbs node_b
        merged_id   = node_a_id
        absorbed_id = node_b_id

        # Collect edges that will become new after merge
        existing_edges = {
            (e.source, e.target) for e in self.edges.values()
        }
        new_edges = []

        # Redirect all edges from absorbed node to merged node
        edges_to_update = [
            (k, e) for k, e in self.edges.items()
            if e.source == absorbed_id or e.target == absorbed_id
        ]

        for key, edge in edges_to_update:
            del self.edges[key]
            new_source = merged_id if edge.source == absorbed_id else edge.source
            new_target = merged_id if edge.target == absorbed_id else edge.target

            if new_source == new_target:
                continue  # skip self-loops created by merge

            new_key = f"{new_source}::{new_target}::{edge.edge_type.value}"
            if (new_source, new_target) not in existing_edges:
                new_edges.append(new_key)

            if new_key not in self.edges:
                self.edges[new_key] = Edge(
                    source    = new_source,
                    target    = new_target,
                    edge_type = edge.edge_type,
                    weight    = edge.weight,
                    conv_pos  = self.conv_pos,
                )
            else:
                # Reinforce if edge already exists
                self.edges[new_key].weight = min(
                    1.0,
                    self.edges[new_key].weight + edge.weight
                )

        # Compute cascade depth — how many new relationships unlocked
        cascade_depth = len(new_edges)
        magnitude     = cascade_depth * self.nodes[merged_id].maturity

        # Record the aha moment
        event = MergeEvent(
            node_a        = node_a_id,
            node_b        = node_b_id,
            merged_into   = merged_id,
            conv_pos      = self.conv_pos,
            cascade_depth = cascade_depth,
            magnitude     = magnitude,
            new_edges     = new_edges,
        )
        self.merge_events.append(event)

        # Remove absorbed node
        del self.nodes[absorbed_id]

        # Update maturity of merged node
        self._update_maturity(merged_id)

        # Cascade — stabilization may trigger further merges
        self._check_merges(merged_id)

    # -------------------------------------------------------------------
    # Internal: similarity and matching
    # -------------------------------------------------------------------

    def _find_match(self, label: str,
                    node_type: NodeType) -> tuple[Optional[str], float]:
        """
        Find best matching existing node for incoming concept.
        Returns (node_id, similarity) or (None, 0.0).

        Current implementation: label similarity as proxy.
        POC will replace this with relational signature comparison
        as graph matures.
        """
        best_id    = None
        best_score = 0.0

        for node in self.nodes.values():
            if node.node_type != node_type:
                continue
            score = self._label_similarity(label, node.label)
            if score > best_score:
                best_score = score
                best_id    = node.id

        return best_id, best_score

    def _signature_similarity(self, node_a_id: str, node_b_id: str) -> float:
        """
        Compare relational signatures of two nodes.
        Signature = set of (neighbor_label, edge_type, weight_bucket)

        This is where primitive identity actually lives.
        Current implementation is intentionally simple —
        the POC will reveal what needs to be more sophisticated.
        """
        def signature(node_id):
            sig = set()
            for edge in self.edges.values():
                if edge.source == node_id:
                    neighbor = self.nodes.get(edge.target)
                    if neighbor:
                        # weight excluded from tuple —
                        # two nodes sharing a neighbor match
                        # regardless of edge strength
                        sig.add((neighbor.label, edge.edge_type.value))
                elif edge.target == node_id:
                    neighbor = self.nodes.get(edge.source)
                    if neighbor:
                        sig.add((neighbor.label, edge.edge_type.value))
            return sig

        sig_a = signature(node_a_id)
        sig_b = signature(node_b_id)

        if not sig_a and not sig_b:
            return 0.0
        if not sig_a or not sig_b:
            return 0.0

        # Jaccard similarity
        intersection = len(sig_a & sig_b)
        union        = len(sig_a | sig_b)
        return intersection / union if union > 0 else 0.0

    def _label_similarity(self, a: str, b: str) -> float:
        """
        Simple character-level similarity as bootstrap proxy.
        Will be superseded by relational signature as graph matures.
        """
        a, b = a.lower().strip(), b.lower().strip()
        if a == b:
            return 1.0
        # Jaccard on character bigrams
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        bg_a, bg_b = bigrams(a), bigrams(b)
        if not bg_a or not bg_b:
            return 0.0
        return len(bg_a & bg_b) / len(bg_a | bg_b)

    def _get_neighbors(self, node_id: str) -> list[tuple[str, float]]:
        """Return [(neighbor_id, edge_weight)] for a node."""
        neighbors = []
        for edge in self.edges.values():
            if edge.source == node_id:
                neighbors.append((edge.target, edge.weight))
            elif edge.target == node_id:
                neighbors.append((edge.source, edge.weight))
        return neighbors

    # -------------------------------------------------------------------
    # Internal: state hash for loop detection
    # -------------------------------------------------------------------

    def _state_hash(self) -> str:
        """
        Lightweight fingerprint of current graph state.
        Repeated hash = fixed point or limit cycle.
        """
        state = {
            "nodes": sorted([
                (n.id, round(n.maturity, 2), n.provisional)
                for n in self.nodes.values()
            ]),
            "edges": sorted([
                (e.source, e.target, round(e.weight, 2))
                for e in self.edges.values()
            ]),
        }
        serialized = json.dumps(state, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    # -------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------

    def summary(self) -> dict:
        """Human-readable graph summary for debugging and paper figures."""
        stable      = [n for n in self.nodes.values() if not n.provisional]
        provisional = [n for n in self.nodes.values() if n.provisional]
        return {
            "nodes_total":       len(self.nodes),
            "nodes_stable":      len(stable),
            "nodes_provisional": len(provisional),
            "edges_total":       len(self.edges),
            "merge_events":      len(self.merge_events),
            "mean_maturity":     (sum(n.maturity for n in self.nodes.values())
                                  / max(len(self.nodes), 1)),
            "aha_moments":       [
                {"magnitude": e.magnitude, "cascade": e.cascade_depth}
                for e in self.merge_events
            ],
        }