"""
locomo.py — LoCoMo benchmark integration for RGM

Handles data parsing, conversation ingestion, and LLM-as-judge scoring
for the LoCoMo dataset (ACL 2024: "Evaluating Very Long-Term Conversational
Memory of LLM Agents").

LoCoMo QA categories:
  1 = direct retrieval
  2 = temporal reasoning
  3 = inference
  4 = multi-hop / cross-session
  5 = adversarial / unanswerable
"""

import json
from dataclasses import dataclass, field

import anthropic

from library.graph import ConceptGraph, NodeLevel
from library.convergence import ConvergenceMonitor


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LoCoMoTurn:
    speaker: str
    text:    str
    dia_id:  str


@dataclass
class LoCoMoSession:
    session_num:  int
    datetime_str: str
    turns:        list[LoCoMoTurn] = field(default_factory=list)


@dataclass
class LoCoMoSample:
    sample_id:  str
    speaker_a:  str
    speaker_b:  str
    sessions:   list[LoCoMoSession] = field(default_factory=list)
    qa:         list[dict]          = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_locomo_file(raw: list, sample_index: int) -> LoCoMoSample:
    """
    Parse one sample from the locomo10.json array.

    raw          — the full parsed JSON list (len=10)
    sample_index — 0-based index into raw
    """
    if not (0 <= sample_index < len(raw)):
        raise IndexError(f"sample_index {sample_index} out of range (0–{len(raw)-1})")

    data = raw[sample_index]
    conv = data["conversation"]

    # Discover session keys dynamically: session_N where value is a list
    session_nums = sorted(
        int(k[len("session_"):])
        for k in conv
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(conv[k], list)
    )

    sessions = []
    for n in session_nums:
        key      = f"session_{n}"
        dt_key   = f"session_{n}_date_time"
        dt_str   = conv.get(dt_key, "")
        raw_turns = conv[key]
        turns = [
            LoCoMoTurn(
                speaker = t["speaker"].strip(),
                text    = t["text"].strip(),
                dia_id  = t.get("dia_id", ""),
            )
            for t in raw_turns
            if t.get("text", "").strip()   # skip image-only turns with no text
        ]
        sessions.append(LoCoMoSession(session_num=n, datetime_str=dt_str, turns=turns))

    return LoCoMoSample(
        sample_id = str(data.get("sample_id", f"sample_{sample_index}")),
        speaker_a = conv.get("speaker_a", "Speaker A"),
        speaker_b = conv.get("speaker_b", "Speaker B"),
        sessions  = sessions,
        qa        = data.get("qa", []),
    )


def flatten_to_conversation(sample: LoCoMoSample) -> list[tuple[str, str]]:
    """Return all turns across all sessions as (speaker, text) pairs in order."""
    turns = []
    for session in sample.sessions:
        for t in session.turns:
            turns.append((t.speaker, t.text))
    return turns


def total_turns(sample: LoCoMoSample) -> int:
    return sum(len(s.turns) for s in sample.sessions)


# ---------------------------------------------------------------------------
# LLM-as-judge scorer
# ---------------------------------------------------------------------------

_CATEGORY_LABELS = {
    1: "direct retrieval (the answer should be a specific fact stated in the conversation)",
    2: "temporal reasoning (the answer requires reasoning about when events occurred)",
    3: "inference (the answer requires drawing a logical conclusion not stated explicitly)",
    4: "multi-hop / cross-session (the answer requires connecting facts from multiple sessions)",
    5: "adversarial / unanswerable (the question may not have a clear answer from the conversation)",
}


class LoCoMoScorer:
    """
    LLM-as-judge scorer using Claude Haiku.

    Scores whether a model answer captures the key information
    in a reference answer, calibrated per LoCoMo category.
    """

    MODEL      = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 150

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def score_answer(self, question: str, reference_answer: str,
                     model_answer: str, category: int) -> dict:
        """
        Returns {"score": float 0–1, "reasoning": str}.
        1.0 = model answer fully captures reference.
        0.5 = partial match.
        0.0 = wrong or missing.
        """
        cat_label = _CATEGORY_LABELS.get(category, f"category {category}")
        prompt = (
            f"You are evaluating an AI assistant's answer against a reference answer.\n\n"
            f"Question: {question}\n"
            f"Reference answer: {reference_answer}\n"
            f"Model answer: {model_answer}\n"
            f"Question category: {cat_label}\n\n"
            "Does the model answer correctly capture the key information in the reference answer? "
            "Give partial credit (0.5) if the answer is partially correct.\n\n"
            'Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
        )
        try:
            resp = self.client.messages.create(
                model       = self.MODEL,
                max_tokens  = self.MAX_TOKENS,
                temperature = 0,
                messages    = [{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            return {
                "score":     float(max(0.0, min(1.0, parsed["score"]))),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except Exception as e:
            return {"score": 0.0, "reasoning": f"parse error: {e}"}


# ---------------------------------------------------------------------------
# Graph propagation (mirrors evaluate.py — kept here to avoid circular import)
# ---------------------------------------------------------------------------

def propagation_pass(graph: ConceptGraph):
    """Single propagation pass: update maturity and stabilise nodes."""
    for node_id in list(graph.nodes.keys()):
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        graph._update_maturity(node_id)
        if node.provisional and node.maturity >= graph.maturity_threshold:
            graph._stabilize(node_id)
            graph._write_count += 1


def convergence_sweep(graph: ConceptGraph, max_passes: int = 20, window: int = 8):
    """Run propagation passes until convergence."""
    monitor = ConvergenceMonitor(max_passes=max_passes, window=window)
    while monitor.should_continue():
        graph.begin_pass()
        propagation_pass(graph)
        status = graph.end_pass()
        monitor.record(status)


def sleep_consolidation(graph: ConceptGraph, max_rounds: int = 10) -> dict:
    """
    Global consolidation pass — the 'sleep' phase.

    During sequential ingest, _check_merges only fires when a node is *newly*
    stabilised.  Nodes that stabilised early never re-check against nodes that
    stabilised later.  This pass visits every stable L1 concept node and
    re-runs the full merge check against the current graph, repeating until no
    new merges fire.

    Analogy: hippocampal replay during sleep — memories are compared globally,
    not just against whatever was active at encoding time.
    """
    total_merges = 0
    for rnd in range(max_rounds):
        # First ensure all pending maturity promotions are resolved
        convergence_sweep(graph, max_passes=5)

        before = len(graph.merge_events)

        # Touch every stable L1 concept node — same comparison a query would do
        candidates = [
            nid for nid, n in graph.nodes.items()
            if not n.provisional and n.level >= NodeLevel.CONCEPT
        ]
        for node_id in candidates:
            if node_id in graph.nodes:   # may have been merged away
                graph._check_merges(node_id)

        after = len(graph.merge_events)
        new_merges = after - before
        total_merges += new_merges

        if new_merges == 0:
            break   # converged

    return {
        "rounds":       rnd + 1,
        "total_merges": total_merges,
        "stable_nodes": sum(1 for n in graph.nodes.values() if not n.provisional),
        "total_nodes":  len(graph.nodes),
    }


# ---------------------------------------------------------------------------
# Summary helper (mirrors evaluate.py._summarise)
# ---------------------------------------------------------------------------

def summarise_results(results: list[dict], modes: list[str]) -> dict:
    """Compute per-category and overall average scores."""
    categories = sorted(set(r["category"] for r in results))
    by_cat = {
        str(cat): {
            mode: (
                sum(r["conditions"][mode]["score"]
                    for r in results if r["category"] == cat and mode in r["conditions"])
                / max(1, sum(1 for r in results if r["category"] == cat and mode in r["conditions"]))
            )
            for mode in modes
        }
        for cat in categories
    }
    n = len(results)
    overall = {
        mode: (
            sum(r["conditions"][mode]["score"] for r in results if mode in r["conditions"]) / max(1, n)
        )
        for mode in modes
    }
    return {"by_category": by_cat, "overall": overall}
