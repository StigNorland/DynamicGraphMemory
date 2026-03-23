"""
graph.py — Relational Concept Graph

The primary working memory substrate. Concepts are nodes defined
entirely by their relationships. Meaning is relational structure.

Performance note: _adjacency maintains a per-node neighbour index so
_get_neighbors() is O(degree) rather than O(|edges|). This gives
~150x speedup at FDT scale (300+ nodes) vs the linear scan.

Authors: Stig Norland, Claude (Anthropic)
"""

import math
import hashlib
import json
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


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
    label:       str
    provisional: bool        = True
    maturity:    float       = 0.0
    conv_pos:    int         = 0
    meta:        dict        = field(default_factory=dict)


@dataclass
class Edge:
    source:      str
    target:      str
    edge_type:   EdgeType
    weight:      float       = 1.0
    conv_pos:    int         = 0     # turn when edge was first created
    last_active: int         = 0     # turn of most recent reinforcement
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
    is the novelty signal.
    """

    MATURITY_THRESHOLD    = 5.0
    LEARNING_RATE         = 0.1
    HASH_WINDOW           = 8

    # Retrieval: label similarity required to treat an incoming concept as
    # the same node rather than a new one.  Pure surface-form comparison —
    # maturity is irrelevant here.  A match is a match.
    RETRIEVAL_THRESHOLD   = 0.85

    # Merge: base Jaccard threshold on relational *signature* similarity.
    # Only stable (mature) nodes are merge candidates — maturity gates
    # *eligibility*, not the similarity score itself.
    # The dynamic property merge_threshold scales this with graph density.

    @property
    def maturity_threshold(self) -> float:
        base = 5.0
        size_factor = max(1.0, len(self.nodes) / 60)
        return base * (1.0 / size_factor ** 0.3)

    @property
    def merge_threshold(self) -> float:
        if not self.nodes:
            return 0.30
        total_degree = sum(
            len(self._adjacency[n_id]) for n_id in self.nodes
        )
        mean_degree = total_degree / len(self.nodes)
        threshold = 0.30 / (1.0 + mean_degree * 0.1)
        return max(0.05, min(0.30, threshold))

    def __init__(self):
        self.nodes:        dict[str, Node]       = {}
        self.edges:        dict[str, Edge]       = {}
        self.merge_events: list[MergeEvent]      = []
        self.state_hashes: list[str]             = []
        self.conv_pos:     int                   = 0
        self._write_count: int                   = 0
        # --- Adjacency index: node_id -> {neighbour_id: edge_key} ---
        self._adjacency:   dict[str, dict[str, str]] = defaultdict(dict)

    # -------------------------------------------------------------------
    # Write policy
    # -------------------------------------------------------------------

    def write(self, label: str, node_type: NodeType,
              related_to: list[str] = None,
              edge_type: EdgeType = EdgeType.SEMANTIC) -> tuple[str, bool]:
        novelty = False

        match_id, similarity = self._find_match(label, node_type)

        if match_id is None:
            node_id = self._create_node(label, node_type)
            novelty = True
        elif similarity >= self.RETRIEVAL_THRESHOLD:
            # Same concept — reinforce regardless of maturity.
            # Maturity measures concept completeness, not retrieval confidence.
            node_id = match_id
            self._reinforce_node(node_id)
        else:
            # Partial match — create a new node with a delta edge to the
            # closest existing node, capturing the difference.
            node_id = self._create_node(label, node_type)
            self._add_edge(node_id, match_id, EdgeType.SEMANTIC, weight=similarity)
            novelty = True

        if related_to:
            for rel_id in related_to:
                if rel_id in self.nodes:
                    created = self._add_edge(node_id, rel_id, edge_type)
                    if created:
                        novelty = True

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
        self._write_count = 0

    def end_pass(self) -> dict:
        h = self._state_hash()
        loop_detected = h in self.state_hashes

        self.state_hashes.append(h)
        if len(self.state_hashes) > self.HASH_WINDOW:
            self.state_hashes.pop(0)

        return {
            "writes":        self._write_count,
            "converged":     self._write_count == 0,
            "loop_detected": loop_detected,
            "state_hash":    h,
            "tension":       loop_detected and self._write_count > 0,
        }

    def advance(self):
        self.conv_pos += 1

    # -------------------------------------------------------------------
    # Context assembly
    # -------------------------------------------------------------------

    def get_context(self, query_nodes: list[str],
                    token_budget: int,
                    backing_store: list[str],
                    recency_weight: float = 0.3) -> str:
        """
        Assemble context from graph for model consumption.

        recency_weight (0–1): how much to boost recently-active edges.
          0.0 = pure maturity (original behaviour)
          0.3 = mild recency bias (default)
          1.0 = pure recency
        """
        visited  = set()
        context  = []
        tokens   = 0

        total_turns = max(1, self.conv_pos)

        def priority(node_id: str, edge_weight: float, last_active: int) -> float:
            recency = last_active / total_turns   # 0..1, higher = more recent
            maturity = self.nodes[node_id].maturity if node_id in self.nodes else 0
            return -(edge_weight * (1 - recency_weight) + recency * recency_weight
                     + maturity * 0.05)

        frontier = [(-(self.nodes[n].maturity), n, 0)
                    for n in query_nodes if n in self.nodes]
        frontier.sort()

        while frontier and tokens < token_budget:
            _, node_id, _ = frontier.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            node = self.nodes[node_id]
            if node.conv_pos < len(backing_store):
                fragment = backing_store[node.conv_pos]
                fragment_tokens = len(fragment.split())
                if tokens + fragment_tokens <= token_budget:
                    context.append(fragment)
                    tokens += fragment_tokens

            for neighbour_id, edge_weight, last_active in self._get_neighbors(node_id):
                if neighbour_id not in visited:
                    p = priority(neighbour_id, edge_weight, last_active)
                    frontier.append((p, neighbour_id, last_active))
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
        # Ensure adjacency entry exists even for isolated nodes
        if node_id not in self._adjacency:
            self._adjacency[node_id] = {}
        return node_id

    def _add_edge(self, source: str, target: str,
                  edge_type: EdgeType, weight: float = 1.0) -> bool:
        key = f"{source}::{target}::{edge_type.value}"
        if key in self.edges:
            self.edges[key].weight = min(
                1.0, self.edges[key].weight + self.LEARNING_RATE
            )
            self.edges[key].last_active = self.conv_pos  # track recency
            return False
        else:
            self.edges[key] = Edge(
                source     = source,
                target     = target,
                edge_type  = edge_type,
                weight     = weight,
                conv_pos   = self.conv_pos,
                last_active = self.conv_pos,
                maturity_at_creation = self.nodes[source].maturity,
            )
            self._adjacency[source][target] = key
            self._adjacency[target][source] = key
            return True

    def _remove_edge(self, key: str):
        """Remove an edge and keep the adjacency index consistent."""
        edge = self.edges.pop(key, None)
        if edge:
            self._adjacency[edge.source].pop(edge.target, None)
            self._adjacency[edge.target].pop(edge.source, None)

    def _reinforce_node(self, node_id: str):
        for neighbour_id in list(self._adjacency.get(node_id, {})):
            edge_key = self._adjacency[node_id][neighbour_id]
            if edge_key in self.edges:
                self.edges[edge_key].weight = min(
                    1.0, self.edges[edge_key].weight + self.LEARNING_RATE
                )
                self.edges[edge_key].last_active = self.conv_pos  # track recency

    def _update_maturity(self, node_id: str):
        """
        Maturity = concept completeness in the current context.

        It measures how well-connected and reinforced a concept is —
        not how confident we are that the data is correct, and not
        a retrieval score.  Its only structural role is as the gate
        for merge eligibility: a node must reach the maturity threshold
        before it can be a merge candidate (_check_merges).

        Formula: edge_count × mean_edge_weight × neighbourhood_factor
        """
        neighbours = self._adjacency.get(node_id, {})
        if not neighbours:
            return

        edge_keys    = list(neighbours.values())
        connected    = [self.edges[k] for k in edge_keys if k in self.edges]
        if not connected:
            return

        edge_count   = len(connected)
        mean_weight  = sum(e.weight for e in connected) / edge_count

        neighbour_ids = list(neighbours.keys())
        if neighbour_ids:
            neighbour_maturity = sum(
                self.nodes[n].maturity
                for n in neighbour_ids if n in self.nodes
            ) / len(neighbour_ids)
            neighbourhood_factor = 1.0 + (neighbour_maturity * 0.1)
        else:
            neighbourhood_factor = 1.0

        self.nodes[node_id].maturity = (
                edge_count * mean_weight * neighbourhood_factor
        )

    def _stabilize(self, node_id: str):
        self.nodes[node_id].provisional = False
        self._check_merges(node_id)

    # -------------------------------------------------------------------
    # Internal: merge
    # -------------------------------------------------------------------

    def _check_merges(self, node_id: str):
        """
        Check whether this newly stabilised node shares a relational
        signature with any other stable node.

        Maturity gates *eligibility* — only stable (non-provisional) nodes
        participate in merges.  This is the only place maturity influences
        merge decisions.  The similarity score itself is purely structural
        (Jaccard on relational signature) and is independent of maturity.
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
                break

    def _merge(self, node_a_id: str, node_b_id: str):
        merged_id   = node_a_id
        absorbed_id = node_b_id

        existing_pairs = set(self._adjacency.get(merged_id, {}).keys())
        new_edges      = []

        # Collect edges to redirect
        edges_to_redirect = list(self._adjacency.get(absorbed_id, {}).values())

        for key in edges_to_redirect:
            edge = self.edges.get(key)
            if edge is None:
                continue

            new_source = merged_id if edge.source == absorbed_id else edge.source
            new_target = merged_id if edge.target == absorbed_id else edge.target

            self._remove_edge(key)

            if new_source == new_target:
                continue

            new_key = f"{new_source}::{new_target}::{edge.edge_type.value}"
            if new_target not in existing_pairs:
                new_edges.append(new_key)

            if new_key not in self.edges:
                self.edges[new_key] = Edge(
                    source    = new_source,
                    target    = new_target,
                    edge_type = edge.edge_type,
                    weight    = edge.weight,
                    conv_pos  = self.conv_pos,
                )
                self._adjacency[new_source][new_target] = new_key
                self._adjacency[new_target][new_source] = new_key
            else:
                self.edges[new_key].weight = min(
                    1.0, self.edges[new_key].weight + edge.weight
                )

        cascade_depth = len(new_edges)
        magnitude     = cascade_depth * self.nodes[merged_id].maturity

        self.merge_events.append(MergeEvent(
            node_a        = node_a_id,
            node_b        = node_b_id,
            merged_into   = merged_id,
            conv_pos      = self.conv_pos,
            cascade_depth = cascade_depth,
            magnitude     = magnitude,
            new_edges     = new_edges,
        ))

        # Clean up absorbed node's adjacency entry
        self._adjacency.pop(absorbed_id, None)
        del self.nodes[absorbed_id]

        self._update_maturity(merged_id)
        self._check_merges(merged_id)

    # -------------------------------------------------------------------
    # Internal: similarity and matching
    # -------------------------------------------------------------------

    def _find_match(self, label: str,
                    node_type: NodeType) -> tuple[Optional[str], float]:
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
        def signature(node_id):
            sig = set()
            for neighbour_id, edge_key in self._adjacency.get(node_id, {}).items():
                edge      = self.edges.get(edge_key)
                neighbour = self.nodes.get(neighbour_id)
                if edge and neighbour:
                    sig.add((neighbour.label, edge.edge_type.value))
            return sig

        sig_a = signature(node_a_id)
        sig_b = signature(node_b_id)

        if not sig_a or not sig_b:
            return 0.0

        intersection = len(sig_a & sig_b)
        union        = len(sig_a | sig_b)
        return intersection / union if union > 0 else 0.0

    def _label_similarity(self, a: str, b: str) -> float:
        a, b = a.lower().strip(), b.lower().strip()
        if a == b:
            return 1.0
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        bg_a, bg_b = bigrams(a), bigrams(b)
        if not bg_a or not bg_b:
            return 0.0
        return len(bg_a & bg_b) / len(bg_a | bg_b)

    def _get_neighbors(self, node_id: str) -> list[tuple[str, float, int]]:
        """
        O(degree) lookup via adjacency index.
        Returns [(neighbour_id, edge_weight, last_active_conv_pos)]
        Callers can use last_active to bias toward recently active edges.
        """
        result = []
        for neighbour_id, edge_key in self._adjacency.get(node_id, {}).items():
            edge = self.edges.get(edge_key)
            if edge and neighbour_id in self.nodes:
                result.append((neighbour_id, edge.weight, edge.last_active))
        return result

    # -------------------------------------------------------------------
    # Internal: state hash
    # -------------------------------------------------------------------

    def _state_hash(self) -> str:
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