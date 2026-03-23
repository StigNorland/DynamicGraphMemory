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

from .graph import ConceptGraph, NodeType, EdgeType

# ---------------------------------------------------------------------------
# Typed-triple extractor (spaCy — with stopword fallback)
# ---------------------------------------------------------------------------
try:
    from concept_extractor import extract_concepts as _spacy_extract
    _EXTRACTOR_AVAILABLE = True
except ImportError:
    _EXTRACTOR_AVAILABLE = False

# ---------------------------------------------------------------------------
# LLM concept extraction — loaded once at module level
# ---------------------------------------------------------------------------
import json as _json
import hashlib as _hashlib
from pathlib import Path as _Path

_CONCEPT_CACHE_DIR = _Path(".concept_cache")

_CONCEPT_PROMPT = """\
Extract the key concepts from this conversation turn as a JSON array of strings.

A concept is a domain term that:
- Could be a node in a knowledge graph or glossary entry
- Means the same thing regardless of which sentence it appears in
- Good examples: "retrieval_path", "big_five_traits", "sati", "holographic_memory",
  "context_window", "merge_event", "bioelectric_field", "ego_dissolution"
- Bad examples: "thing", "way", "the part that", "very", "interesting", "lot"

Rules:
- snake_case labels only
- 6-12 concepts maximum, prefer fewer high-quality over many low-quality
- Prefer specific over generic ("context_window" over "memory")
- Multi-word concepts are fine ("bioelectric_field", "merge_event")
- Output ONLY the JSON array, no explanation, no markdown

Turn ({speaker}):
{text}
"""


def _extract_concepts_llm(text: str, speaker: str, client,
                          model: str = "claude-haiku-4-5-20251001") -> list[str]:
    """
    Extract concepts using an LLM call.
    Results are cached on disk so repeat runs are free.
    Falls back to empty list on any error.
    """
    # Cache lookup
    _CONCEPT_CACHE_DIR.mkdir(exist_ok=True)
    cache_key  = _hashlib.md5(f"{speaker}:{text}".encode()).hexdigest()
    cache_file = _CONCEPT_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return _json.loads(cache_file.read_text())
        except Exception:
            pass

    # LLM call
    try:
        response = client.messages.create(
            model      = model,
            max_tokens = 256,
            messages   = [{
                "role":    "user",
                "content": _CONCEPT_PROMPT.format(
                    speaker = speaker,
                    text    = text[:1500],
                ),
            }],
        )
        raw      = response.content[0].text.strip()
        concepts = _json.loads(raw)
        if not isinstance(concepts, list):
            return []
        result = [str(c).lower().strip() for c in concepts[:12] if c]
        cache_file.write_text(_json.dumps(result))
        return result
    except Exception:
        return []

# ---------------------------------------------------------------------------
# LLM extractor and payload synthesiser (optional — enabled at runtime)
# Local synthesiser (regex + ollama) is preferred over cloud API.
# Swap order here to force cloud: import PayloadSynthesiser before Local.
# ---------------------------------------------------------------------------
try:
    from .llm_extractor import LLMExtractor
    from .payload_local import LocalPayloadSynthesiser as PayloadSynthesiser
    _LLM_MODULES_AVAILABLE = True
except ImportError:
    try:
        from .payload import PayloadSynthesiser  # cloud fallback
        from .llm_extractor import LLMExtractor
        _LLM_MODULES_AVAILABLE = True
    except ImportError:
        _LLM_MODULES_AVAILABLE = False

# ---------------------------------------------------------------------------
# Bioelectric field layer (Levin-inspired holographic memory)
# ---------------------------------------------------------------------------
try:
    from .field import BioelectricField
    _FIELD_AVAILABLE = True
except ImportError:
    _FIELD_AVAILABLE = False


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
      - GRAPH:      traverse relevant subgraph, dereference selectively
      - BASELINE:   raw recent conversation, standard KV-cache style
      - GRAPH_ONLY: relational structure + payloads, no backing store

    Optional upgrades (enabled by passing llm_client):
      - LLM-assisted triple extraction per turn (llm_extractor.py)
      - Rich payload synthesis per mature node (payload.py)
    """

    def __init__(self, graph: ConceptGraph, store: BackingStore,
                 tokenizer=None, llm_client=None,
                 use_llm_extraction: bool = False,
                 use_rich_payloads:  bool = False,
                 use_field:          bool = False,
                 ollama_model:       str  = "llama3.2"):
        self.graph     = graph
        self.store     = store
        self.tokenizer = tokenizer
        self._llm_client = llm_client  # stored for direct concept extraction

        # LLM-assisted extraction
        if _LLM_MODULES_AVAILABLE and llm_client and use_llm_extraction:
            self._llm_extractor = LLMExtractor(client=llm_client, enabled=True)
        else:
            self._llm_extractor = None

        # Payload synthesiser — fires once per node on stabilisation
        if _LLM_MODULES_AVAILABLE and use_rich_payloads:
            self._payload_synth = PayloadSynthesiser(
                ollama_model = ollama_model,
                use_ollama   = True,
            )
        else:
            self._payload_synth = None

        # Bioelectric field — holographic distributed memory
        if _FIELD_AVAILABLE and use_field:
            self._field = BioelectricField()
        else:
            self._field = None

    def assemble_graph(self, query: str, token_budget: int) -> str:
        query_labels   = self._extract_concepts(query, "query")
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
            total_turns = max(1, self.graph.conv_pos)
            for neighbor_id, edge_weight, last_active in neighbors:
                if neighbor_id not in visited and neighbor_id in self.graph.nodes:
                    neighbor_maturity = self.graph.nodes[neighbor_id].maturity
                    recency  = last_active / total_turns        # 0..1
                    priority = (edge_weight * 0.7               # base strength
                                + recency   * 0.2               # recency bias
                                + neighbor_maturity * 0.1)      # maturity bonus
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

        # Update bioelectric field with this turn (before graph write)
        if self._field is not None:
            self._field.ingest_turn(speaker, text, pos)

        concepts    = self._extract_concepts(text, speaker)
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

            if node_id in self.graph.nodes:
                node = self.graph.nodes[node_id]
                # Payload synthesis on stabilisation
                if self._payload_synth and not node.provisional:
                    if "payload" not in node.meta:
                        self._payload_synth.synthesise(node, self.graph, self.store)
                # Field projection on stabilisation
                if self._field is not None and not node.provisional:
                    if "field_projection" not in node.meta:
                        self._field.project_node(node, self.graph)

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
    # Concept extraction — LLM-assisted (with spaCy + stopword fallback)
    # -------------------------------------------------------------------

    def _extract_concepts(self, text: str, speaker: str = "unknown") -> list[str]:
        """
        Extract concept labels from text.

        Priority:
          1. LLM extractor via llm_client (Haiku — fast, cheap, cached)
             Returns clean snake_case concept labels directly.
          2. LLM-assisted triple extractor (llm_extractor.py)
             Full relation vocab + numerical facts.
          3. spaCy typed-triple extractor (concept_extractor.py)
          4. Stopword filter — original fallback

        All paths return a flat list of canonical concept labels.
        """
        # 1. Direct LLM concept extraction (preferred — simple and clean)
        if self._llm_client is not None:
            result = _extract_concepts_llm(text, speaker, self._llm_client)
            if result:
                return result

        # 2. LLM triple extractor
        if self._llm_extractor is not None:
            return self._extract_via_llm(text)

        # 3. spaCy
        if _EXTRACTOR_AVAILABLE:
            return self._extract_via_spacy(text)

        # 4. Stopword fallback
        return self._extract_stopword_fallback(text)

    def _extract_via_llm(self, text: str) -> list[str]:
        """Use LLM extractor (falls back to spaCy internally on failure)."""
        try:
            concept_dicts = self._llm_extractor.extract_concepts(text)
            return self._flatten_concept_dicts(concept_dicts)
        except Exception:
            return self._extract_via_spacy(text)

    def _extract_via_spacy(self, text: str) -> list[str]:
        """
        Use the typed-triple extractor to pull IS-A concept labels.
        Both subject and object of each triple are added as distinct
        concept nodes.
        """
        try:
            concept_dicts = _spacy_extract(text)
        except Exception:
            return self._extract_stopword_fallback(text)
        return self._flatten_concept_dicts(concept_dicts)

    def _flatten_concept_dicts(self, concept_dicts: list[dict]) -> list[str]:
        """Convert concept dicts (from LLM or spaCy) to flat label list."""
        labels = []
        seen   = set()
        for concept in concept_dicts:
            subj = concept["label"]
            if subj and subj not in seen:
                seen.add(subj)
                labels.append(subj)
            for rel in concept.get("relations", []):
                obj = rel.get("target", "")
                if obj and obj not in seen:
                    seen.add(obj)
                    labels.append(obj)
        return labels

    # -------------------------------------------------------------------
    # Graph-only assembly — relational structure + payloads
    # -------------------------------------------------------------------

    def assemble_graph_only(self, query: str, token_budget: int) -> str:
        """
        Assemble context from graph structure alone — no backing store.

        When rich payloads are present (use_rich_payloads=True), injects
        canonical_summaries and fact_dicts from the top-k mature nodes
        relevant to the query. This turns graph-only context from a pure
        relational skeleton into a self-contained factual generator.
        """
        query_labels   = self._extract_concepts(query, "query")
        query_node_ids = []
        for label in query_labels:
            node_id, score = self.graph._find_match(label, NodeType.CONCEPT)
            if node_id and score > 0.3:
                query_node_ids.append((node_id, score))

        # Stable nodes ranked by maturity
        stable = sorted(
            [n for n in self.graph.nodes.values() if not n.provisional],
            key=lambda n: -n.maturity
        )

        parts  = []
        tokens = 0

        # --- Payload injection (top-k relevant mature nodes) ---
        if any("payload" in n.meta for n in stable):
            payload_lines = []
            # Prioritise nodes that match the query
            query_ids = {nid for nid, _ in query_node_ids}
            ordered   = (
                    [n for n in stable if n.id in query_ids] +
                    [n for n in stable if n.id not in query_ids]
            )
            for node in ordered[:12]:
                p = node.meta.get("payload")
                if not p:
                    continue
                line = f"[{node.label}] {p['canonical_summary']}"
                facts = p.get("fact_dict", {})
                if facts:
                    fact_str = "; ".join(f"{k}={v}" for k, v in list(facts.items())[:4])
                    line += f" | facts: {fact_str}"
                payload_lines.append(line)
            if payload_lines:
                block = "Facts:\n" + "\n".join(payload_lines)
                t = self._count_tokens(block)
                if tokens + t <= token_budget:
                    parts.append((1000.0, block))
                    tokens += t

        # --- Relational structure (top nodes + neighbours) ---
        top_ids = {n.id for n in stable[:10]}
        rel_lines = []
        for node in stable[:10]:
            neighbours = self.graph._get_neighbors(node.id)
            nb_labels  = [
                self.graph.nodes[nb_id].label
                for nb_id, w, _ in neighbours
                if nb_id in self.graph.nodes and w > 0.5
            ][:4]
            if nb_labels:
                rel_lines.append(
                    f"{node.label}(mat={node.maturity:.1f}) → {', '.join(nb_labels)}"
                )
        if rel_lines:
            block = "Key relations:\n" + "\n".join(rel_lines)
            t = self._count_tokens(block)
            if tokens + t <= token_budget:
                parts.append((500.0, block))
                tokens += t

        # --- Query-relevant nodes ---
        q_lines = []
        for node_id, score in sorted(query_node_ids, key=lambda x: -x[1])[:6]:
            if node_id not in self.graph.nodes:
                continue
            node = self.graph.nodes[node_id]
            q_lines.append(
                f"{node.label}(deg={len(self.graph._adjacency.get(node_id, {}))}, "
                f"mat={node.maturity:.1f})"
            )
        if q_lines:
            block = "Query-relevant: " + ", ".join(q_lines)
            t = self._count_tokens(block)
            if tokens + t <= token_budget:
                parts.append((800.0, block))
                tokens += t

        # --- Merge events (aha moments) ---
        merge_ctx = self._merge_context(token_budget - tokens)
        if merge_ctx:
            parts.append((999.0, merge_ctx))

        # --- Bioelectric field reconstruction (holographic memory) ---
        # Reconstruct global field from visible node projections.
        # Even a small fragment of nodes can recover dominant conversation themes.
        if self._field is not None:
            visible_ids = [n.id for n in stable[:20]]
            field_summary = self._field.reconstruct(visible_ids, self.graph)
            if field_summary:
                t = self._count_tokens(field_summary)
                if tokens + t <= token_budget:
                    parts.append((750.0, field_summary))
                    tokens += t

        if not parts:
            return "(graph empty — no context available)"

        parts.sort(key=lambda x: -x[0])
        return "\n".join(text for _, text in parts).strip()

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