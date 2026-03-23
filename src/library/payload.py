"""
payload.py — Rich Payload Synthesis for Mature Concept Nodes

When a node stabilises (reaches maturity threshold), this module runs a
one-time LLM synthesis pass to attach structured factual content:

    node.meta["payload"] = {
        "canonical_summary": "one-sentence essence of the concept",
        "fact_dict":         {"key": "value", ...},   # extracted literals
        "key_literals":      ["2GM", "c²", "4.13 km"],
        "reconstruction_hint": "brief note on how to regenerate exact answer"
    }

This turns graph-only context from a pure relational structure into a
self-contained generator of missing facts — the primary driver of L1 weakness.

Design constraints:
  - One API call per mature node (50 tokens max, cheap)
  - Gated behind PayloadSynthesiser(enabled=True) — zero cost when disabled
  - Payloads stored in node.meta, no new node/edge types required
  - Context assembly in graph-only mode injects top-k canonical_summaries
    + fact_dicts alongside the relational structure

Authors: Stig Norland, Claude (Anthropic)
"""

from __future__ import annotations

import json
import re
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ConceptGraph, Node

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise fact extractor. Given a concept label and the "
    "conversation context it came from, output ONLY valid JSON with no "
    "markdown, no preamble, no explanation."
)

_USER_TMPL = """\
Concept label: "{label}"

Conversation context (the turns where this concept appeared):
{context}

Extract a rich payload for this concept. Output exactly this JSON structure:
{{
  "canonical_summary": "<one sentence: the essential meaning of this concept>",
  "fact_dict": {{
    "<fact_name>": "<value>",
    ...
  }},
  "key_literals": ["<literal1>", "<literal2>", ...],
  "reconstruction_hint": "<one sentence: what a reader needs to reconstruct an exact answer about this concept>"
}}

Rules:
- canonical_summary: max 20 words, no hedging, state the fact directly
- fact_dict: extract ALL numerical values, equations, definitions, named relationships
  (e.g. {{"Schwarzschild_radius": "2GM/c²", "value_km": "4.13", "letter_count": "3"}})
- key_literals: the 3-6 most irreplaceable tokens (numbers, symbols, proper names)
- reconstruction_hint: what fact_dict entry or literal is most likely to be asked about
- If the concept has no extractable facts, use empty dict/list but still give a summary
"""


# ---------------------------------------------------------------------------
# Synthesiser
# ---------------------------------------------------------------------------

class PayloadSynthesiser:
    """
    Attaches rich factual payloads to nodes when they reach maturity.

    Usage:
        synth = PayloadSynthesiser(client=anthropic_client, enabled=True)
        synth.synthesise(node, graph, backing_store)

    The synthesiser is a plug-in — ConceptGraph and BackingStore are
    unchanged. PayloadSynthesiser is wired in by the ContextAssembler
    (or directly by the caller) after _stabilize() fires.
    """

    MODEL        = "claude-haiku-4-5-20251001"   # cheapest, fast, sufficient
    MAX_TOKENS   = 120
    CONTEXT_TURNS = 3    # how many backing-store turns to include as context

    def __init__(self, client=None, enabled: bool = True):
        self.client  = client
        self.enabled = enabled
        self._calls  = 0
        self._errors = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesise(self, node: "Node",
                   graph: "ConceptGraph",
                   backing_store) -> bool:
        """
        Run payload synthesis for a single node.
        Returns True if a payload was attached, False otherwise.
        Idempotent — skips nodes that already have a payload.
        """
        if not self.enabled or self.client is None:
            return False
        if "payload" in node.meta:
            return False   # already synthesised

        context = self._gather_context(node, graph, backing_store)
        payload = self._call_llm(node.label, context)
        if payload:
            node.meta["payload"] = payload
            self._calls += 1
            return True
        return False

    def synthesise_batch(self, nodes: list["Node"],
                         graph: "ConceptGraph",
                         backing_store) -> int:
        """Synthesise payloads for a list of nodes. Returns count synthesised."""
        count = 0
        for node in nodes:
            if self.synthesise(node, graph, backing_store):
                count += 1
        return count

    @property
    def stats(self) -> dict:
        return {"calls": self._calls, "errors": self._errors}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _gather_context(self, node: "Node",
                        graph: "ConceptGraph",
                        backing_store) -> str:
        """
        Collect the backing-store turns most relevant to this node,
        plus its immediate neighbours in the graph.
        """
        parts = []

        # Backing store: turns where this node was active
        if backing_store is not None:
            relevant_turns = []
            for turn in backing_store.turns:
                if node.id in turn.nodes:
                    relevant_turns.append(turn)
            # Fall back to turns near the node's conv_pos
            if not relevant_turns:
                pos = node.conv_pos
                for turn in backing_store.turns:
                    if abs(turn.pos - pos) <= 1:
                        relevant_turns.append(turn)
            # Take the most recent CONTEXT_TURNS
            for turn in relevant_turns[-self.CONTEXT_TURNS:]:
                parts.append(f"{turn.speaker}: {turn.text[:400]}")

        # Graph neighbours: immediate relational context
        neighbours = []
        for nb_id, w, _ in graph._get_neighbors(node.id):
            if nb_id in graph.nodes:
                nb_label = graph.nodes[nb_id].label
                neighbours.append(f"{nb_label} (w={w:.2f})")
        if neighbours:
            parts.append("Related concepts: " + ", ".join(neighbours[:8]))

        return "\n\n".join(parts) if parts else "(no context available)"

    def _call_llm(self, label: str, context: str) -> dict | None:
        """Call the LLM and parse the JSON payload. Returns None on failure."""
        prompt = _USER_TMPL.format(label=label, context=context[:1200])
        try:
            response = self.client.messages.create(
                model      = self.MODEL,
                max_tokens = self.MAX_TOKENS,
                system     = _SYSTEM,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            payload = json.loads(raw)
            # Validate required keys
            required = {"canonical_summary", "fact_dict",
                        "key_literals", "reconstruction_hint"}
            if not required.issubset(payload.keys()):
                log.warning("Payload missing keys for %s: %s", label, payload.keys())
                return None
            return payload
        except json.JSONDecodeError as e:
            log.debug("Payload JSON parse error for %s: %s", label, e)
            self._errors += 1
            return None
        except Exception as e:
            log.debug("Payload synthesis error for %s: %s", label, e)
            self._errors += 1
            return None
