"""
evaluate_field_reconstruction.py

Tests the core claim of the bioelectric field layer:
  "A damaged graph with field reconstruction should perform
   significantly better than a damaged graph without it."

Methodology:
  1. Build the full graph from a conversation (with field enabled)
  2. Damage it at multiple levels: 25%, 50%, 75% nodes removed
  3. For each damage level, run QA in four conditions:
       - intact_graph_only   : undamaged graph, no field reconstruction
       - damaged_no_field    : damaged graph, no reconstruction
       - damaged_with_field  : damaged graph + field reconstruction
       - baseline            : raw recent turns (control — unaffected by damage)
  4. Compare scores across conditions and damage levels

The field "works" if:
  damaged_with_field >> damaged_no_field
  and the gap grows with damage level (more damage = more reconstruction benefit)

Authors: Stig Norland, Claude (Anthropic)
"""

import os
import json
import time
import random
import copy
from dataclasses   import dataclass
from datetime      import datetime
from pathlib       import Path
from collections   import defaultdict

import anthropic

from library.graph       import ConceptGraph, NodeType, EdgeType
from library.context     import ContextAssembler, BackingStore
from library.convergence import ConvergenceMonitor


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

TESTDATA_DIR   = Path(__file__).parent.parent / "testdata"
RESULTS_DIR    = Path(__file__).parent.parent / "results"
DAMAGE_LEVELS  = [0.25, 0.50, 0.75]   # fraction of nodes to remove
RANDOM_SEED    = 42                    # reproducible damage
TOKEN_BUDGET   = 600                   # graph-only budget (field context is compact)


# -------------------------------------------------------------------
# Data structures (reuse evaluate.py schema)
# -------------------------------------------------------------------

@dataclass
class Question:
    text:         str
    key_concepts: list[str]
    level:        int
    description:  str


@dataclass
class TestCase:
    name:         str
    description:  str
    conversation: list[tuple[str, str]]
    questions:    list[Question]
    token_budget: int = TOKEN_BUDGET


def load_testcase(path: Path) -> TestCase:
    with open(path) as f:
        data = json.load(f)
    conversation = [tuple(turn) for turn in data["conversation"]]
    questions = [
        Question(
            text         = q["text"],
            key_concepts = q["key_concepts"],
            level        = q["level"],
            description  = q["description"],
        )
        for q in data["questions"]
    ]
    return TestCase(
        name         = data["metadata"]["name"],
        description  = data["metadata"]["description"],
        conversation = conversation,
        questions    = questions,
        token_budget = data["metadata"].get("token_budget", TOKEN_BUDGET),
    )


# -------------------------------------------------------------------
# Graph damage utility
# -------------------------------------------------------------------

def damage_graph(graph: ConceptGraph, damage_fraction: float,
                 seed: int = RANDOM_SEED) -> ConceptGraph:
    """
    Return a NEW ConceptGraph with `damage_fraction` of nodes removed.
    Edges incident to removed nodes are also removed.
    The original graph is not modified.
    """
    rng = random.Random(seed)

    # Deep copy — we don't want to mutate the original
    damaged = copy.deepcopy(graph)

    all_ids    = list(damaged.nodes.keys())
    n_remove   = int(len(all_ids) * damage_fraction)
    remove_ids = set(rng.sample(all_ids, n_remove))

    for node_id in remove_ids:
        # Remove the node
        damaged.nodes.pop(node_id, None)
        # Remove from adjacency
        damaged._adjacency.pop(node_id, None)
        # Remove edges referencing this node
        for other_id in list(damaged._adjacency.keys()):
            damaged._adjacency[other_id].pop(node_id, None)
        # Remove from edge list
        damaged.edges = {
            eid: e for eid, e in damaged.edges.items()
            if e.source != node_id and e.target != node_id
        }

    return damaged, remove_ids


# -------------------------------------------------------------------
# Field reconstruction
# -------------------------------------------------------------------

def reconstruct_from_field(damaged_graph: ConceptGraph,
                            field,
                            assembler: ContextAssembler,
                            token_budget: int) -> str:
    """
    Use the field's holographic reconstruction to generate context
    from a damaged graph.

    The field was projected from the full graph — its distributed
    embedding encodes information about nodes that may no longer exist.
    Reconstruction sums the local projections of surviving nodes to
    recover the global pattern.
    """
    if field is None:
        return "(field not available)"

    surviving_ids = list(damaged_graph.nodes.keys())
    field_summary = field.reconstruct(surviving_ids, damaged_graph)
    print(f"    [field_raw] {repr(field_summary[:100])}", flush=True)

    if not field_summary:
        return "(field reconstruction returned empty)"

    # Also include whatever relational structure survives
    stable = sorted(
        [n for n in damaged_graph.nodes.values() if not n.provisional],
        key=lambda n: -n.maturity
    )[:8]

    rel_lines = []
    for node in stable:
        neighbours = damaged_graph._get_neighbors(node.id)
        nb_labels  = [
            damaged_graph.nodes[nb_id].label
            for nb_id, w, _ in neighbours
            if nb_id in damaged_graph.nodes and w > 0.5
        ][:3]
        if nb_labels:
            rel_lines.append(
                f"{node.label}(mat={node.maturity:.1f}) → {', '.join(nb_labels)}"
            )

    parts = [field_summary]
    if rel_lines:
        parts.append("Surviving relations:\n" + "\n".join(rel_lines))

    return "\n".join(parts)[:token_budget * 5]  # rough char limit


def graph_only_context_nodamage(assembler: ContextAssembler,
                                 query: str, token_budget: int) -> str:
    """Graph-only context from the intact graph."""
    return assembler.assemble_graph_only(query=query, token_budget=token_budget)


def graph_only_context_damaged(damaged_graph: ConceptGraph,
                                query: str, token_budget: int,
                                assembler: ContextAssembler) -> str:
    """
    Graph-only context from a damaged graph — no field reconstruction.
    Temporarily swap assembler's graph, assemble, then restore.
    """
    original_graph  = assembler.graph
    assembler.graph = damaged_graph
    try:
        ctx = assembler.assemble_graph_only(query=query, token_budget=token_budget)
    finally:
        assembler.graph = original_graph
    return ctx


# -------------------------------------------------------------------
# Evaluator
# -------------------------------------------------------------------

class FieldReconstructionEvaluator:

    MODEL = "claude-sonnet-4-20250514"

    def __init__(self, testcase: TestCase):
        self.testcase = testcase
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client  = anthropic.Anthropic(api_key=api_key)
        self.results = []

    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"FIELD RECONSTRUCTION TEST: {self.testcase.name}")
        print(f"  {self.testcase.description}")
        print(f"  {len(self.testcase.conversation)} turns  |  "
              f"{len(self.testcase.questions)} questions")
        print(f"  Damage levels: {[f'{int(d*100)}%' for d in DAMAGE_LEVELS]}")
        print(f"{'='*60}")

        # Build the full graph once with field enabled
        print("\nBuilding full graph (field enabled)...")
        graph, assembler, store = self._build_graph(use_field=True)
        field = getattr(assembler, '_field', None)

        if field is None:
            print("ERROR: field layer not available. "
                  "Install the field module and retry.")
            return {}

        print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
              f"{len(graph.merge_events)} merges")
        print(f"Field: vocab={field.stats()['vocab_size']}, "
              f"projected={field.stats()['n_projected']}")

        # Run questions at each damage level
        for damage_frac in DAMAGE_LEVELS:
            self._run_damage_level(
                damage_frac, graph, assembler, store, field
            )

        summary = self._summarise()
        self._save(summary)
        self._print_summary(summary)
        return summary

    def _run_damage_level(self, damage_frac: float, graph: ConceptGraph,
                          assembler: ContextAssembler, store: BackingStore,
                          field) -> None:
        pct = int(damage_frac * 100)
        print(f"\n{'─'*60}")
        print(f"  DAMAGE LEVEL: {pct}%  "
              f"({int(len(graph.nodes) * damage_frac)} nodes removed)")
        print(f"{'─'*60}")

        damaged_graph, removed_ids = damage_graph(graph, damage_frac)
        surviving = len(damaged_graph.nodes)
        print(f"  Surviving nodes: {surviving} / {len(graph.nodes)}")

        for qi, q in enumerate(self.testcase.questions, 1):
            print(f"\n  L{q.level}: {q.text[:70]}")

            conditions = {}

            # 1. Intact graph-only (baseline for this experiment)
            ctx_intact = graph_only_context_nodamage(
                assembler, q.text, TOKEN_BUDGET
            )
            ans_intact = self._ask(ctx_intact, q.text)
            conditions["intact_graph_only"] = self._record(
                ctx_intact, ans_intact, q.key_concepts
            )

            # 2. Damaged graph, no reconstruction
            ctx_damaged = graph_only_context_damaged(
                damaged_graph, q.text, TOKEN_BUDGET, assembler
            )
            ans_damaged = self._ask(ctx_damaged, q.text)
            conditions["damaged_no_field"] = self._record(
                ctx_damaged, ans_damaged, q.key_concepts
            )

            # 3. Damaged graph + field reconstruction
            ctx_field = reconstruct_from_field(
                damaged_graph, field, assembler, TOKEN_BUDGET
            )
            ans_field = self._ask(ctx_field, q.text)
            conditions["damaged_with_field"] = self._record(
                ctx_field, ans_field, q.key_concepts
            )

            # 4. Baseline (raw recent turns — unaffected by damage)
            ctx_baseline = assembler.assemble_baseline(
                token_budget=TOKEN_BUDGET
            )
            ans_baseline = self._ask(ctx_baseline, q.text)
            conditions["baseline"] = self._record(
                ctx_baseline, ans_baseline, q.key_concepts
            )

            # Print scores
            for cond, data in conditions.items():
                ind = "✓" if data["score"] >= 0.5 else "~" if data["score"] >= 0.25 else "✗"
                print(f"    {cond:22s} [{ind}] score={data['score']:.2f}  "
                      f"tokens={data['context_tokens']}")

            self.results.append({
                "damage_pct":  pct,
                "question":    q.text,
                "level":       q.level,
                "description": q.description,
                "conditions":  conditions,
            })

            time.sleep(0.3)

    def _build_graph(self, use_field: bool = True) -> tuple:
        graph     = ConceptGraph()
        store     = BackingStore()
        assembler = ContextAssembler(
            graph             = graph,
            store             = store,
            tokenizer         = None,
            llm_client        = self.client,
            use_llm_extraction= False,
            use_rich_payloads = True,
            use_field         = use_field,
        )

        monitor = ConvergenceMonitor(max_passes=20, window=8)
        for speaker, text in self.testcase.conversation:
            assembler.ingest(speaker, text)
            graph.begin_pass()
            self._propagation_pass(graph)
            status = graph.end_pass()
            monitor.record(status)

        while monitor.should_continue():
            graph.begin_pass()
            self._propagation_pass(graph)
            status = graph.end_pass()
            monitor.record(status)

        return graph, assembler, store

    def _propagation_pass(self, graph: ConceptGraph) -> None:
        for node_id in list(graph.nodes.keys()):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            if node.provisional and node.maturity >= graph.maturity_threshold:
                node.provisional = False

    def _ask(self, context: str, question: str) -> str:
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the context provided."
        ) if context.strip() else question
        try:
            response = self.client.messages.create(
                model      = self.MODEL,
                max_tokens = 400,
                temperature= 0,
                messages   = [{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"[ERROR: {e}]"

    def _record(self, ctx: str, answer: str,
                key_concepts: list[str]) -> dict:
        score = self._score(answer, key_concepts)
        return {
            "context_tokens": len(ctx.split()),
            "answer":         answer,
            "score":          score,
            "key_hits":       self._key_hits(answer, key_concepts),
        }

    def _score(self, answer: str, key_concepts: list[str]) -> float:
        hits = self._key_hits(answer, key_concepts)
        return hits / len(key_concepts) if key_concepts else 0.0

    def _key_hits(self, answer: str, key_concepts: list[str]) -> int:
        a = answer.lower()
        hits = 0
        for concept in key_concepts:
            c = concept.lower()
            if c in a:
                hits += 1; continue
            if c.replace("_", " ") in a:
                hits += 1; continue
            words = [w for w in c.split("_") if len(w) >= 4]
            if words and any(w in a for w in words):
                hits += 1
        return hits

    def _summarise(self) -> dict:
        conditions = ["intact_graph_only", "damaged_no_field",
                      "damaged_with_field", "baseline"]

        by_damage = defaultdict(lambda: defaultdict(list))
        for r in self.results:
            dpct = r["damage_pct"]
            for cond in conditions:
                score = r["conditions"][cond]["score"]
                by_damage[dpct][cond].append(score)

        summary = {}
        for dpct, cond_scores in sorted(by_damage.items()):
            summary[f"damage_{dpct}pct"] = {
                cond: round(sum(scores) / len(scores), 3)
                for cond, scores in cond_scores.items()
            }

        # Recovery ratio: how much of the intact performance does
        # field reconstruction recover vs no reconstruction?
        recovery = {}
        for dpct_key, scores in summary.items():
            intact  = scores["intact_graph_only"]
            no_rec  = scores["damaged_no_field"]
            with_rec= scores["damaged_with_field"]
            if intact > 0:
                gap_no_rec   = intact - no_rec
                gap_with_rec = intact - with_rec
                recovery[dpct_key] = round(
                    (gap_no_rec - gap_with_rec) / intact, 3
                ) if gap_no_rec > 0 else 0.0

        return {
            "metadata": {
                "name":           self.testcase.name,
                "timestamp":      datetime.now().strftime("%Y%m%d_%H%M%S"),
                "damage_levels":  DAMAGE_LEVELS,
                "token_budget":   TOKEN_BUDGET,
                "questions":      len(self.testcase.questions),
                "turns":          len(self.testcase.conversation),
            },
            "by_damage_level": summary,
            "field_recovery_ratio": recovery,
            "results": self.results,
        }

    def _print_summary(self, summary: dict) -> None:
        print(f"\n{'='*60}")
        print(f"FIELD RECONSTRUCTION RESULTS: {self.testcase.name}")
        print(f"{'='*60}")
        print(f"\n{'Damage':8s}  {'intact':10s}  {'no_field':10s}  "
              f"{'with_field':12s}  {'baseline':10s}  {'recovery':8s}")
        print("─" * 65)

        recovery = summary["field_recovery_ratio"]
        for dpct_key, scores in summary["by_damage_level"].items():
            pct = dpct_key.replace("damage_", "").replace("pct", "%")
            rec = recovery.get(dpct_key, 0.0)
            rec_str = f"+{rec:.3f}" if rec > 0 else f"{rec:.3f}"
            print(
                f"{pct:8s}  "
                f"{scores['intact_graph_only']:10.3f}  "
                f"{scores['damaged_no_field']:10.3f}  "
                f"{scores['damaged_with_field']:12.3f}  "
                f"{scores['baseline']:10.3f}  "
                f"{rec_str:8s}"
            )

        print()
        print("Recovery ratio = fraction of intact performance recovered by field")
        print("Positive = field helped, Negative = field hurt")

        # Verdict
        avg_recovery = sum(recovery.values()) / len(recovery) if recovery else 0
        if avg_recovery > 0.1:
            print(f"\n✓ FIELD RECONSTRUCTION WORKS  (avg recovery={avg_recovery:.3f})")
            print("  The field embedding recovers meaningful signal from damaged graphs.")
        elif avg_recovery > 0:
            print(f"\n~ FIELD RECONSTRUCTION MARGINAL  (avg recovery={avg_recovery:.3f})")
            print("  Weak positive effect — may need larger damage or denser graphs.")
        else:
            print(f"\n✗ FIELD RECONSTRUCTION NOT DEMONSTRATED  (avg recovery={avg_recovery:.3f})")
            print("  Field does not outperform damaged graph without reconstruction.")

    def _save(self, summary: dict) -> None:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts   = summary["metadata"]["timestamp"]
        name = summary["metadata"]["name"]
        path = RESULTS_DIR / f"field_reconstruction_{name}_{ts}.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nSaved → {path}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Test bioelectric field reconstruction on damaged graphs"
    )
    ap.add_argument(
        "testcase",
        nargs="?",
        default="how_i_remember",
        help="Name of test case in testdata/ (default: how_i_remember)"
    )
    args = ap.parse_args()

    path = TESTDATA_DIR / f"{args.testcase}.json"
    if not path.exists():
        available = [p.stem for p in TESTDATA_DIR.glob("*.json")]
        print(f"Test case '{args.testcase}' not found.")
        print(f"Available: {available}")
        raise SystemExit(1)

    tc  = load_testcase(path)
    ev  = FieldReconstructionEvaluator(tc)
    ev.run()