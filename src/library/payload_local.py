"""
payload_local.py — Local Rich Payload Synthesis

Replaces the cloud-API payload synthesiser with a fully local pipeline:

  Tier 1 — Fact extraction: regex + spaCy NER (deterministic, instant)
    Extracts numerical values, equations, named constants, units.
    No model needed. Falls back gracefully if spaCy unavailable.

  Tier 2 — Canonical summary: ollama (local LLM, free, offline)
    Uses whatever model is running in ollama (llama3.2, mistral, etc).
    Install: https://ollama.com  then  `ollama pull llama3.2`
    Falls back to a template summary if ollama is not running.

  Tier 3 — Template summary (always available)
    Generates a summary from the node's label + relational neighbours.
    No model, no API. Sufficient for graph-only structural context.

Usage in ContextAssembler:
    from .payload_local import LocalPayloadSynthesiser
    synth = LocalPayloadSynthesiser(
        ollama_model="llama3.2",   # or "mistral", "phi3", etc.
        use_ollama=True,           # set False to force template-only
    )
    synth.synthesise(node, graph, backing_store)

The payload schema is identical to payload.py so context.py works
unchanged — just swap PayloadSynthesiser for LocalPayloadSynthesiser.

Authors: Stig Norland, Claude (Anthropic)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ConceptGraph, Node

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns for fact extraction (Tier 1)
# ---------------------------------------------------------------------------

# Numerical value + unit  e.g. "4.13 km", "938 MeV", "1.2×10⁻¹⁰ m/s²"
# Numerical value + unit  e.g. "4.13 km", "938 MeV", "1.2e-10 m/s"
# Longer units listed before shorter to prevent partial matches (MeV before m)
_NUM_UNIT = re.compile(
    r"[-+]?\d+(?:[.,]\d+)?(?:\s*[xX]\s*10\s*[-]?\d+)?\s*"
    r"(kpc|Mpc|km/s|m/s|kHz|MHz|GHz|THz|keV|MeV|GeV|TeV"
    r"|km|cm|mm|nm|fm|ms|ns|Hz|eV|AU|ly|pc|kg|mg|%|m|g)",
    re.IGNORECASE,
)

# Equations and formulae  e.g. "2GM/c^2", "E=mc^2", "f = 1/M"
_EQUATION = re.compile(
    r"[A-Za-z_]\w*\s*[=~]\s*[^\s,;.]{3,30}"
    r"|[2-9]\s*[A-Z]+\s*/\s*[a-zA-Z23]+",
    re.UNICODE,
)

# Named constants  e.g. "alpha = 1/137", "H0 = 67.4"
_CONSTANT = re.compile(
    r"(?:alpha|beta|gamma|sigma|lambda|H_?0|hbar|k_B)"
    r"\s*[=~]\s*[\d\./\-\+]+[^\s,;.]{0,15}",
    re.IGNORECASE,
)

# Significant bare numbers: 4+ digit integers or precise decimals
_BARE_NUMBER = re.compile(
    r"\b(\d{4,}|\d+\.\d{2,})\b"
)


def extract_facts(text: str, label: str) -> tuple[dict, list]:
    """
    Extract numerical facts and key literals from text using regex.

    Returns:
        fact_dict    — {fact_name: value_string}
        key_literals — [important token strings]
    """
    fact_dict    = {}
    key_literals = []

    # 1. Numerical values with units
    for m in _NUM_UNIT.finditer(text):
        val  = m.group(0).strip()
        unit = m.group(1)
        key  = f"value_{unit.lower().replace('/', '_per_')}"
        # Avoid duplicate keys by appending index
        base = key
        i    = 1
        while key in fact_dict:
            key = f"{base}_{i}"
            i  += 1
        fact_dict[key] = val
        if val not in key_literals:
            key_literals.append(val)

    # 2. Equations
    for m in _EQUATION.finditer(text):
        eq = m.group(0).strip()
        if len(eq) > 3:
            fact_dict[f"equation"] = eq
            key_literals.append(eq)
            break  # take first equation only

    # 3. Named constants
    for m in _CONSTANT.finditer(text):
        const = m.group(0).strip()
        name  = const.split("=")[0].strip().lower()
        name  = re.sub(r"[^a-z0-9_]", "_", name)
        fact_dict[f"constant_{name}"] = const
        key_literals.append(const[:20])

    # 4. Significant bare numbers
    for m in _BARE_NUMBER.finditer(text):
        num = m.group(0)
        if num not in key_literals and len(key_literals) < 6:
            key_literals.append(num)

    # Deduplicate and limit
    key_literals = list(dict.fromkeys(key_literals))[:6]

    return fact_dict, key_literals


# ---------------------------------------------------------------------------
# spaCy NER for entity facts (optional enhancement)
# ---------------------------------------------------------------------------

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

def _spacy_facts(text: str) -> dict:
    """Extract named entities as facts using spaCy if available."""
    if _NLP is None:
        return {}
    try:
        doc = _NLP(text[:500])
        facts = {}
        for ent in doc.ents:
            if ent.label_ in {"CARDINAL", "QUANTITY", "PERCENT", "MONEY"}:
                key = f"entity_{ent.label_.lower()}"
                facts[key] = ent.text
        return facts
    except Exception:
        return {}
# ---------------------------------------------------------------------------
# Ollama summary (Tier 2)
# ---------------------------------------------------------------------------

_OLLAMA_URL    = "http://localhost:11434/api/generate"
_OLLAMA_PROMPT = """\
Summarise the concept "{label}" in one sentence (max 20 words).
Be direct and factual. Context: {context}
Output only the sentence, no preamble."""


def _ollama_summary(label: str, context: str,
                    model: str = "llama3.2",
                    timeout: int = 8) -> str | None:
    """
    Call ollama for a one-sentence canonical summary.
    Returns None if ollama is not running or call fails.
    """
    payload = json.dumps({
        "model":  model,
        "prompt": _OLLAMA_PROMPT.format(
            label   = label,
            context = context[:400],
        ),
        "stream": False,
        "options": {
            "num_predict": 40,   # max ~30 words
            "temperature": 0.1,
        }
    }).encode()

    try:
        req = urllib.request.Request(
            _OLLAMA_URL,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data     = json.loads(resp.read())
            summary  = data.get("response", "").strip()
            # Strip any accidental preamble
            summary  = re.sub(r"^(Sure|Certainly|Here|The concept)[^:]*:\s*",
                              "", summary, flags=re.IGNORECASE)
            return summary[:200] if summary else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None   # ollama not running — fall through to template
    except Exception as e:
        log.debug("ollama error for %s: %s", label, e)
        return None


# ---------------------------------------------------------------------------
# Template summary (Tier 3 — always available)
# ---------------------------------------------------------------------------

def _template_summary(label: str, graph: "ConceptGraph",
                      node: "Node") -> str:
    """
    Generate a summary from the node's label and graph neighbours.
    No model required.
    """
    neighbours = []
    for nb_id, w, _ in graph._get_neighbors(node.id):
        if nb_id in graph.nodes and w > 0.5:
            neighbours.append(graph.nodes[nb_id].label)

    label_readable = label.replace("_", " ")

    if neighbours:
        nb_str = ", ".join(neighbours[:4])
        return f"{label_readable} — concept related to {nb_str}."
    return f"{label_readable} — core concept in this conversation."


# ---------------------------------------------------------------------------
# LocalPayloadSynthesiser
# ---------------------------------------------------------------------------

class LocalPayloadSynthesiser:
    """
    Local drop-in replacement for PayloadSynthesiser.

    Fact extraction: regex + optional spaCy NER (deterministic, instant).
    Canonical summary: ollama if running, template fallback otherwise.

    Identical public API to PayloadSynthesiser — swap in context.py by
    changing the import:

        from .payload_local import LocalPayloadSynthesiser as PayloadSynthesiser
    """

    CONTEXT_TURNS = 3

    def __init__(self, ollama_model: str = "llama3.2",
                 use_ollama: bool = True):
        self.ollama_model = ollama_model
        self.use_ollama   = use_ollama
        self._calls       = 0
        self._ollama_hits = 0
        self._errors      = 0

        # Probe ollama on init so we know immediately whether it's running
        self._ollama_available = False
        if use_ollama:
            probe = _ollama_summary("test", "test", model=ollama_model, timeout=3)
            if probe is not None:
                self._ollama_available = True
                log.info("ollama available (%s)", ollama_model)
            else:
                log.info("ollama not available — using template summaries")

    # ------------------------------------------------------------------
    # Public API (identical to PayloadSynthesiser)
    # ------------------------------------------------------------------

    def synthesise(self, node: "Node",
                   graph: "ConceptGraph",
                   backing_store) -> bool:
        """Attach a rich payload to a node. Idempotent."""
        if "payload" in node.meta:
            return False

        context   = self._gather_context(node, graph, backing_store)
        fact_dict, key_literals = extract_facts(context, node.label)

        # Augment with spaCy NER if available
        spacy_facts = _spacy_facts(context)
        fact_dict.update(spacy_facts)

        # Canonical summary
        summary = None
        if self._ollama_available:
            summary = _ollama_summary(
                node.label, context, model=self.ollama_model
            )
            if summary:
                self._ollama_hits += 1

        if not summary:
            summary = _template_summary(node.label, graph, node)

        # Reconstruction hint
        if fact_dict:
            top_fact = list(fact_dict.items())[0]
            hint = f"Key fact: {top_fact[0]} = {top_fact[1]}"
        else:
            hint = f"Refer to context where '{node.label}' is discussed."

        node.meta["payload"] = {
            "canonical_summary":    summary,
            "fact_dict":            fact_dict,
            "key_literals":         key_literals,
            "reconstruction_hint":  hint,
            "source":               "ollama" if summary and self._ollama_hits > 0
            else "template",
        }

        self._calls += 1
        return True

    def synthesise_batch(self, nodes: list["Node"],
                         graph: "ConceptGraph",
                         backing_store) -> int:
        return sum(self.synthesise(n, graph, backing_store) for n in nodes)

    @property
    def stats(self) -> dict:
        return {
            "calls":         self._calls,
            "ollama_hits":   self._ollama_hits,
            "template_hits": self._calls - self._ollama_hits,
            "errors":        self._errors,
            "ollama_model":  self.ollama_model if self._ollama_available else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _gather_context(self, node: "Node",
                        graph: "ConceptGraph",
                        backing_store) -> str:
        parts = []

        if backing_store is not None:
            relevant = [t for t in backing_store.turns if node.id in t.nodes]
            if not relevant:
                pos = node.conv_pos
                relevant = [t for t in backing_store.turns
                            if abs(t.pos - pos) <= 1]
            for turn in relevant[-self.CONTEXT_TURNS:]:
                parts.append(f"{turn.speaker}: {turn.text[:400]}")

        neighbours = []
        for nb_id, w, _ in graph._get_neighbors(node.id):
            if nb_id in graph.nodes:
                neighbours.append(graph.nodes[nb_id].label)
        if neighbours:
            parts.append("Related: " + ", ".join(neighbours[:6]))

        return "\n\n".join(parts) if parts else node.label