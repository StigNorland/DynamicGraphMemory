"""
llm_extractor.py — LLM-Assisted Triple + Property Extraction

Augments the spaCy typed-triple extractor with a one-shot LLM call per
conversation turn. Where spaCy extracts IS-A triples from syntax, the LLM
extracts the full relation vocabulary plus numerical/verbatim facts that
spaCy cannot see.

Relation types extracted:
  IS-A       — taxonomic membership (shared with spaCy)
  CAUSES     — causal direction
  PART-OF    — compositional
  HAS-PROP   — attribute / property
  EQUALS     — definitional identity or numerical value
  RELATES-TO — weak associative (fallback)

Each triple also carries any extracted literal value (for EQUALS / HAS-PROP),
enabling the graph to answer direct retrieval questions without backing-store
access.

Design:
  - One API call per turn (O(1) per turn, not per node)
  - Falls back to spaCy-only if LLM call fails or client not provided
  - Output is a superset of concept_extractor.extract_concepts() — same
    dict schema, extended with "literal_value" and richer edge types
  - Gated behind LLMExtractor(enabled=True)

Authors: Stig Norland, Claude (Anthropic)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise relation extractor. Output ONLY valid JSON. "
    "No markdown, no explanation, no preamble."
)

_USER_TMPL = """\
Extract ALL relations and facts from this conversation turn.

Turn text:
\"\"\"
{text}
\"\"\"

Output a JSON array of triples. Each triple must have:
  "subject":       string — the source concept (noun phrase, lower-case, underscores)
  "relation":      one of ["IS-A", "CAUSES", "PART-OF", "HAS-PROP", "EQUALS", "RELATES-TO"]
  "object":        string — the target concept or value
  "conf":          float 0.0-1.0 — extraction confidence
  "literal_value": string or null — if the triple encodes a concrete fact,
                   put the exact value here (e.g. "2GM/c²", "4.13 km", "3")

Rules:
- Extract numerical values, equations, definitions (use EQUALS or HAS-PROP)
- Use PART-OF for structural components (e.g. "quark PART-OF proton")
- Use CAUSES for mechanisms (e.g. "time_dilation CAUSES gravitational_redshift")
- Use IS-A for taxonomy (e.g. "electron IS-A vortex_ring")
- Use HAS-PROP for attributes (e.g. "Schwarzschild_radius HAS-PROP 4.13_km")
- Keep labels lower-case and underscore-joined: "event_horizon" not "Event Horizon"
- Minimum confidence 0.6 — omit uncertain triples
- Maximum 20 triples per turn
- If no clear relations exist, return []

Example output for "The Schwarzschild radius is 2GM/c², approximately 4.13 km":
[
  {{"subject": "Schwarzschild_radius", "relation": "EQUALS", "object": "2GM_over_c_squared",
    "conf": 0.98, "literal_value": "2GM/c²"}},
  {{"subject": "Schwarzschild_radius", "relation": "HAS-PROP", "object": "4.13_km",
    "conf": 0.95, "literal_value": "4.13 km"}},
  {{"subject": "Schwarzschild_radius", "relation": "IS-A", "object": "event_horizon_radius",
    "conf": 0.90, "literal_value": null}}
]
"""

# ---------------------------------------------------------------------------
# Edge type mapping from LLM relation strings to graph EdgeType values
# ---------------------------------------------------------------------------

RELATION_TO_EDGE = {
    "IS-A":       "semantic",
    "CAUSES":     "causal",
    "PART-OF":    "semantic",
    "HAS-PROP":   "semantic",
    "EQUALS":     "semantic",
    "RELATES-TO": "semantic",
}


# ---------------------------------------------------------------------------
# LLMExtractor
# ---------------------------------------------------------------------------

class LLMExtractor:
    """
    Wraps a spaCy extractor with an LLM augmentation pass.

    Usage:
        extractor = LLMExtractor(client=anthropic_client, enabled=True)
        concepts = extractor.extract_concepts(turn_text)

    Returns the same list[dict] schema as concept_extractor.extract_concepts(),
    extended with:
      - richer relation types (CAUSES, PART-OF, HAS-PROP, EQUALS)
      - "literal_value" on relations that encode concrete facts
    """

    MODEL      = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 400
    MAX_TEXT   = 1200   # truncate long turns before sending

    def __init__(self, client=None, enabled: bool = True):
        self.client  = client
        self.enabled = enabled
        self._calls  = 0
        self._errors = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_concepts(self, text: str) -> list[dict]:
        """
        Extract concepts and relations from a conversation turn.
        Falls back to spaCy-only if LLM is disabled or unavailable.
        """
        # Always run spaCy first as the fast baseline
        spacy_concepts = self._spacy_extract(text)

        if not self.enabled or self.client is None:
            return spacy_concepts

        # LLM augmentation
        llm_triples = self._call_llm(text)
        if not llm_triples:
            return spacy_concepts

        # Merge: LLM output takes priority for relation types; spaCy fills gaps
        merged = self._merge(spacy_concepts, llm_triples)
        self._calls += 1
        return merged

    @property
    def stats(self) -> dict:
        return {"calls": self._calls, "errors": self._errors}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spacy_extract(self, text: str) -> list[dict]:
        """Run the spaCy extractor, return empty list on failure."""
        try:
            from concept_extractor import extract_concepts
            return extract_concepts(text)
        except Exception as e:
            log.debug("spaCy extraction failed: %s", e)
            return []

    def _call_llm(self, text: str) -> list[dict] | None:
        """Call the LLM and return parsed triples, or None on failure."""
        prompt = _USER_TMPL.format(text=text[:self.MAX_TEXT])
        try:
            response = self.client.messages.create(
                model      = self.MODEL,
                max_tokens = self.MAX_TOKENS,
                system     = _SYSTEM,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            triples = json.loads(raw)
            if not isinstance(triples, list):
                return None
            # Validate and normalise
            valid = []
            for t in triples:
                if not all(k in t for k in ("subject", "relation", "object", "conf")):
                    continue
                if t["conf"] < 0.6:
                    continue
                t.setdefault("literal_value", None)
                t["relation"] = t["relation"].upper()
                if t["relation"] not in RELATION_TO_EDGE:
                    t["relation"] = "RELATES-TO"
                valid.append(t)
            return valid
        except json.JSONDecodeError as e:
            log.debug("LLM extractor JSON error: %s", e)
            self._errors += 1
            return None
        except Exception as e:
            log.debug("LLM extractor error: %s", e)
            self._errors += 1
            return None

    def _merge(self, spacy_concepts: list[dict],
               llm_triples: list[dict]) -> list[dict]:
        """
        Merge spaCy concepts (IS-A only) with LLM triples (full vocab).

        Strategy:
        1. Start with all LLM triples converted to concept dicts
        2. Add any spaCy concepts not already covered by LLM output
        3. Attach literal_value to relation dicts where present
        """
        concept_map: dict[str, dict] = {}

        def _get(label: str) -> dict:
            label = label.lower().replace(" ", "_")
            if label not in concept_map:
                concept_map[label] = {
                    "label":     label,
                    "type":      "Concept",
                    "relations": [],
                }
            return concept_map[label]

        # 1. LLM triples
        for t in llm_triples:
            subj = t["subject"].lower().replace(" ", "_")
            obj  = t["object"].lower().replace(" ", "_")
            _get(obj)  # ensure object node exists
            rel_entry = {
                "target":        obj,
                "edge_type":     RELATION_TO_EDGE.get(t["relation"], "semantic"),
                "conf":          t["conf"],
                "relation_type": t["relation"],
            }
            if t.get("literal_value"):
                rel_entry["literal_value"] = t["literal_value"]
            _get(subj)["relations"].append(rel_entry)

        # 2. spaCy concepts not already in map
        for sc in spacy_concepts:
            lbl = sc["label"].lower().replace(" ", "_")
            node = _get(lbl)
            for rel in sc.get("relations", []):
                # Only add if this (subject, target) pair not already present
                tgt = rel["target"].lower().replace(" ", "_")
                existing_targets = {r["target"] for r in node["relations"]}
                if tgt not in existing_targets:
                    _get(tgt)
                    node["relations"].append({
                        "target":        tgt,
                        "edge_type":     rel.get("edge_type", "semantic"),
                        "conf":          rel.get("conf", 0.75),
                        "relation_type": "IS-A",
                    })

        return list(concept_map.values())
