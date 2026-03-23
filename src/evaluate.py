"""
evaluate.py — Question Answering Evaluation Framework

Tests whether graph-compressed context preserves enough meaning
for a language model to answer questions requiring genuine
understanding of the conversation.

Three conditions, identical token budget:
  - GRAPH:      graph-derived context + backing store
  - BASELINE:   raw recent turns
  - GRAPH_ONLY: graph structure only, backing store discarded

Test data lives in testdata/*.json — add a new file there to add
a new experiment without touching this file.

Authors: Stig Norland, Claude (Anthropic)
"""

import os
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic

from library.graph       import ConceptGraph, NodeType, EdgeType
from library.context     import ContextAssembler, BackingStore
from library.convergence import ConvergenceMonitor


# -------------------------------------------------------------------
# Test data loading
# -------------------------------------------------------------------

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"


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
    token_budget: int = 200     # override per-file in metadata.token_budget


def load_testcase(path: Path) -> TestCase:
    """Load a single testdata JSON file into a TestCase."""
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
        token_budget = data["metadata"].get("token_budget", 200),
    )


def load_all_testcases(directory: Path = TESTDATA_DIR) -> list[TestCase]:
    """Load all *.json files from the testdata directory."""
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No JSON files found in {directory}. "
            "Add at least one testdata file."
        )
    cases = [load_testcase(p) for p in paths]
    print(f"Loaded {len(cases)} test case(s) from {directory}:")
    for c in cases:
        print(f"  {c.name}: {len(c.conversation)} turns, {len(c.questions)} questions")
    return cases


# -------------------------------------------------------------------
# Evaluator
# -------------------------------------------------------------------

class Evaluator:
    """
    Runs a single TestCase against three context conditions.
    Uses Claude Sonnet 4 (temperature 0) as the answering model.
    Token budget is read from the testcase (set in testdata JSON metadata).

    Optional upgrades:
      use_llm_extraction=True  — LLM-assisted triple + property extraction
                                  per turn (llm_extractor.py). Richer relation
                                  types + numerical facts in the graph.
      use_rich_payloads=True   — One-time LLM synthesis pass per mature node
                                  (payload.py). Injects canonical_summaries
                                  and fact_dicts into graph-only context.
                                  Expected to lift graph-only L1 scores
                                  significantly.
      use_field=True           — Bioelectric field layer (field.py).
                                  Levin-inspired holographic distributed memory.
                                  Each mature node stores a local projection of
                                  a global TF-IDF field. Graph-only assembly
                                  reconstructs dominant themes from any
                                  fragment of nodes.

    Both flags default to False for backwards-compatibility. When enabled,
    timing output reports extra API calls separately so users understand cost.
    """

    MODEL         = "claude-sonnet-4-20250514"
    ANSWER_TOKENS = 150
    TEMPERATURE   = 0

    def __init__(self, testcase: TestCase,
                 use_llm_extraction: bool = False,
                 use_rich_payloads:  bool = False,
                 use_field:          bool = False):
        self.testcase            = testcase
        self.TOKEN_BUDGET        = testcase.token_budget
        self.use_llm_extraction  = use_llm_extraction
        self.use_rich_payloads   = use_rich_payloads
        self.use_field           = use_field

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Set it with: export ANTHROPIC_API_KEY=your_key_here"
            )
        self.client  = anthropic.Anthropic(api_key=api_key)
        self.results = []

    def run(self) -> dict:
        upgrades = []
        if self.use_llm_extraction: upgrades.append("llm-extraction")
        if self.use_rich_payloads:  upgrades.append("rich-payloads")
        if self.use_field:          upgrades.append("field")
        upgrade_str = f"  upgrades: [{', '.join(upgrades)}]" if upgrades else ""

        print(f"\n{'='*60}")
        print(f"EVALUATION: {self.testcase.name}")
        print(f"  {self.testcase.description}")
        print(f"  {len(self.testcase.conversation)} turns  |  "
              f"{len(self.testcase.questions)} questions  |  "
              f"budget {self.TOKEN_BUDGET} tokens{upgrade_str}")
        print(f"{'='*60}")

        graph, assembler, store = self._build_graph()

        total_questions = len(self.testcase.questions)
        api_times = []

        for qi, q in enumerate(self.testcase.questions, 1):
            print(f"\n--- L{q.level}: {q.description} ---")
            print(f"Q: {q.text}")

            result = {
                "question":    q.text,
                "level":       q.level,
                "description": q.description,
                "conditions":  {},
            }

            for mode in ["graph", "baseline", "graph_only"]:
                ctx    = self._assemble_context(mode, assembler, store, graph, q.text)
                t0     = time.perf_counter()
                answer = self._ask(ctx, q.text)
                api_times.append(time.perf_counter() - t0)
                score  = self._score(answer, q.key_concepts)
                hits   = self._key_hits(answer, q.key_concepts)

                result["conditions"][mode] = {
                    "context_tokens": len(ctx.split()),
                    "answer":         answer,
                    "score":          score,
                    "key_hits":       hits,
                }

                ind = "✓" if score >= 0.5 else "~" if score >= 0.25 else "✗"
                print(f"  {mode:12s} [{ind}] score={score:.2f}  "
                      f"tokens={len(ctx.split())}  "
                      f"hits={hits}/{len(q.key_concepts)}  "
                      f"({api_times[-1]:.1f}s API)")

            self.results.append(result)
            time.sleep(0.5)

        # Timing breakdown — so users understand where time goes
        n_calls   = len(api_times)
        api_total = sum(api_times)
        print(f"\n{'─'*60}")
        print(f"Timing breakdown ({total_questions} questions × 3 conditions = {n_calls} API calls):")
        print(f"  Answer API calls:  {api_total:6.1f}s  ({api_total/n_calls:.1f}s avg per call)")
        print(f"  Sleep pauses:      {total_questions * 0.5:6.1f}s  (rate-limit courtesy)")
        if self.use_rich_payloads or self.use_llm_extraction or self.use_field:
            synth = getattr(assembler, '_payload_synth', None)
            extr  = getattr(assembler, '_llm_extractor', None)
            field = getattr(assembler, '_field', None)
            if synth:
                print(f"  Payload synthesis: {synth.stats['calls']} calls  "
                      f"({synth.stats['errors']} errors)")
            if extr:
                print(f"  LLM extraction:    {extr.stats['calls']} calls  "
                      f"({extr.stats['errors']} errors)")
            if field:
                fs = field.stats()
                print(f"  Field layer:       vocab={fs['vocab_size']}  "
                      f"projected={fs['n_projected']}  "
                      f"norm={fs['field_norm']}")
        print(f"  Note: slowness is API latency, not graph construction or Python.")

    # -------------------------------------------------------------------
    # Graph building
    # -------------------------------------------------------------------

    def _build_graph(self) -> tuple:
        graph     = ConceptGraph()
        store     = BackingStore()
        assembler = ContextAssembler(
            graph               = graph,
            store               = store,
            tokenizer           = None,
            llm_client          = self.client,
            use_llm_extraction  = self.use_llm_extraction,
            use_rich_payloads   = self.use_rich_payloads,
            use_field           = self.use_field,
        )

        t0 = time.perf_counter()
        monitor = ConvergenceMonitor(max_passes=20, window=8)  # once, outside

        for i, (speaker, text) in enumerate(self.testcase.conversation):
            assembler.ingest(speaker, text)
            # one propagation pass per turn, not 20
            graph.begin_pass()
            self._propagation_pass(graph)
            status = graph.end_pass()
            monitor.record(status)
            print(f"  turn {i+1:3d}  nodes={len(graph.nodes)}  edges={len(graph.edges)}  status={status}")

        # final convergence sweep after all turns ingested
        while monitor.should_continue():
            graph.begin_pass()
            self._propagation_pass(graph)
            status = graph.end_pass()
            monitor.record(status)

        n_payloads = sum(1 for n in graph.nodes.values() if "payload" in n.meta)
        print(f"\nGraph built in {graph_time:.2f}s: {len(graph.nodes)} nodes, "
              f"{len(graph.edges)} edges, "
              f"{len(graph.merge_events)} merges"
              + (f", {n_payloads} payloads" if self.use_rich_payloads else ""))
        return graph, assembler, store

    def _propagation_pass(self, graph: ConceptGraph):
        for node_id in list(graph.nodes.keys()):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            graph._update_maturity(node_id)
            if node.provisional and node.maturity >= graph.maturity_threshold:
                graph._stabilize(node_id)
                graph._write_count += 1

    # -------------------------------------------------------------------
    # Context assembly
    # -------------------------------------------------------------------

    def _assemble_context(self, mode: str, assembler: ContextAssembler,
                          store: BackingStore, graph: ConceptGraph,
                          query: str) -> str:
        if mode == "graph":
            return assembler.assemble_graph(query=query, token_budget=self.TOKEN_BUDGET)
        if mode == "baseline":
            return assembler.assemble_baseline(token_budget=self.TOKEN_BUDGET)
        if mode == "graph_only":
            # Use the upgraded assembler method when rich payloads are enabled
            if self.use_rich_payloads and hasattr(assembler, 'assemble_graph_only'):
                return assembler.assemble_graph_only(query=query,
                                                     token_budget=self.TOKEN_BUDGET)
            return self._graph_only_context(graph, query)
        return ""

    def _graph_only_context(self, graph: ConceptGraph, query: str) -> str:
        parts = []

        stable = sorted(
            [n for n in graph.nodes.values() if not n.provisional],
            key=lambda n: -n.maturity
        )[:10]

        if stable:
            parts.append("Core concepts: " + ", ".join(
                f"{n.label}(maturity={n.maturity:.1f})"
                for n in stable[:6]
            ))

        top_ids   = {n.id for n in stable[:8]}
        key_edges = sorted(
            [e for e in graph.edges.values()
             if e.source in top_ids and e.target in top_ids and e.weight > 0.5],
            key=lambda e: -e.weight
        )[:8]

        if key_edges:
            rel_lines = [
                f"{graph.nodes[e.source].label} -{e.edge_type.value}-> "
                f"{graph.nodes[e.target].label}(w={e.weight:.2f} t={e.last_active})"
                for e in key_edges
                if e.source in graph.nodes and e.target in graph.nodes
            ]
            if rel_lines:
                parts.append("Key relations: " + "; ".join(rel_lines))

        if graph.merge_events:
            top_merges = sorted(graph.merge_events, key=lambda e: -e.magnitude)[:3]
            merge_lines = [
                f"'{e.node_a}' unified with '{e.node_b}' (cascade={e.cascade_depth})"
                for e in top_merges
                if e.merged_into in graph.nodes
            ]
            if merge_lines:
                parts.append("Insights: " + "; ".join(merge_lines))

        query_words = query.lower().split()
        relevant = sorted(
            [n for n in graph.nodes.values()
             if any(w in n.label for w in query_words if len(w) > 3)],
            key=lambda n: -n.maturity
        )[:6]
        if relevant:
            parts.append("Query-relevant: " + ", ".join(
                f"{n.label}(deg={len(graph._get_neighbors(n.id))})"
                for n in relevant
            ))

        return "\n".join(parts)[:self.TOKEN_BUDGET * 6]

    # -------------------------------------------------------------------
    # Model call
    # -------------------------------------------------------------------

    def _ask(self, context: str, question: str) -> str:
        prompt = (
            "The following is context from a conversation:\n\n"
            f"{context}\n\n"
            "Based only on this context, answer concisely:\n"
            f"{question}"
        )
        try:
            response = self.client.messages.create(
                model       = self.MODEL,
                max_tokens  = self.ANSWER_TOKENS,
                temperature = self.TEMPERATURE,
                messages    = [{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"[ERROR: {e}]"

    # -------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------

    def _score(self, answer: str, key_concepts: list[str]) -> float:
        hits = self._key_hits(answer, key_concepts)
        return hits / len(key_concepts) if key_concepts else 0.0

    def _key_hits(self, answer: str, key_concepts: list[str]) -> int:
        a = answer.lower()
        return sum(1 for c in key_concepts if c.lower() in a)

    # -------------------------------------------------------------------
    # Reporting and persistence
    # -------------------------------------------------------------------

    def _summarise(self) -> dict:
        levels     = sorted(set(r["level"] for r in self.results))
        conditions = ["graph", "baseline", "graph_only"]
        by_level   = {
            str(lv): {
                c: (
                    sum(r["conditions"][c]["score"]
                        for r in self.results if r["level"] == lv)
                    / max(1, sum(1 for r in self.results if r["level"] == lv))
                )
                for c in conditions
            }
            for lv in levels
        }
        n = len(self.results)
        overall = {
            c: sum(r["conditions"][c]["score"] for r in self.results) / n
            for c in conditions
        }
        return {"by_level": by_level, "overall": overall}

    def _print_report(self, summary: dict):
        print(f"\n{'='*60}")
        print(f"SUMMARY: {self.testcase.name}")
        print(f"{'='*60}")
        print(f"\n{'Level':<8}", end="")
        for c in ["graph", "baseline", "graph_only"]:
            print(f"{c:>14}", end="")
        print()
        print("-" * 50)
        for lv, scores in sorted(summary["by_level"].items()):
            print(f"L{lv:<7}", end="")
            for c in ["graph", "baseline", "graph_only"]:
                s   = scores[c]
                ind = "✓" if s >= 0.5 else "~" if s >= 0.25 else "✗"
                print(f"{ind} {s:>10.2f}", end="")
            print()
        o = summary["overall"]
        print(f"\nOverall — graph: {o['graph']:.3f}  "
              f"baseline: {o['baseline']:.3f}  "
              f"graph_only: {o['graph_only']:.3f}")
        if o["graph_only"] > o["baseline"]:
            print("*** Graph-only BEATS baseline — meaning is in the structure.")
        elif o["graph_only"] >= o["baseline"] * 0.9:
            print("~ Graph-only matches baseline.")
        else:
            print("  Baseline leads over graph-only.")

    def save(self) -> dict:
        summary = self._summarise()
        self._print_report(summary)

        os.makedirs("experiments", exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"experiments/eval_{self.testcase.name}_{ts}"

        output = {
            "metadata": {
                "timestamp":           ts,
                "name":                self.testcase.name,
                "model":               self.MODEL,
                "temperature":         self.TEMPERATURE,
                "token_budget":        self.TOKEN_BUDGET,
                "questions":           len(self.testcase.questions),
                "turns":               len(self.testcase.conversation),
                "use_llm_extraction":  self.use_llm_extraction,
                "use_rich_payloads":   self.use_rich_payloads,
                "use_field":           self.use_field,
            },
            "results": self.results,
            "summary": summary,
        }

        json_path = f"{base}.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results  → {json_path}")

        tex_path = f"{base}_table.tex"
        self._write_latex_table(summary, tex_path)
        print(f"LaTeX    → {tex_path}")

        return output

    def _write_latex_table(self, summary: dict, path: str):
        level_labels = {
            "1": "L1 (direct retrieval)",
            "2": "L2 (synthesis)",
            "3": "L3 (inference)",
            "4": "L4 (cross-domain)",
        }
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            rf"\caption{{QA evaluation — {self.testcase.name}}}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Level & Graph + store & Baseline & Graph only \\",
            r"\midrule",
        ]
        for lv, label in level_labels.items():
            if lv in summary["by_level"]:
                d = summary["by_level"][lv]
                lines.append(
                    rf"{label} & {d.get('graph',0):.2f} & "
                    rf"{d.get('baseline',0):.2f} & "
                    rf"{d.get('graph_only',0):.2f} \\"
                )
            else:
                lines.append(rf"{label} & --- & --- & --- \\")
        o = summary["overall"]
        lines += [
            r"\midrule",
            rf"Overall & {o['graph']:.2f} & {o['baseline']:.2f} & {o['graph_only']:.2f} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RGM QA Evaluation — runs all testdata/*.json files"
    )
    parser.add_argument(
        "--rich", action="store_true",
        help="Enable rich payload synthesis (use_rich_payloads=True). "
             "Runs a one-shot LLM synthesis call per mature node. "
             "Expected to lift graph-only L1 scores significantly."
    )
    parser.add_argument(
        "--field", action="store_true",
        help="Enable bioelectric field layer (use_field=True). "
             "Levin-inspired holographic memory: each mature node stores a "
             "local projection of a global TF-IDF field. Graph-only assembly "
             "reconstructs dominant themes from any fragment of nodes. "
             "Pure stdlib — no API calls, no external dependencies."
    )
    parser.add_argument(
        "--llm-extract", action="store_true",
        help="Enable LLM-assisted triple extraction (use_llm_extraction=True). "
             "Augments spaCy with a per-turn LLM call for richer relation types "
             "and numerical facts."
    )
    parser.add_argument(
        "--testcase", type=str, default=None,
        help="Run only a specific testcase by name (e.g. 'memory'). "
             "Default: run all."
    )
    args = parser.parse_args()

    testcases = load_all_testcases()
    if args.testcase:
        testcases = [tc for tc in testcases if tc.name == args.testcase]
        if not testcases:
            print(f"No testcase named '{args.testcase}' found.")
            raise SystemExit(1)

    for tc in testcases:
        ev = Evaluator(
            tc,
            use_llm_extraction = args.llm_extract,
            use_rich_payloads  = args.rich,
            use_field          = args.field,
        )
        ev.run()
        ev.save()
        print()

    print(f"Done. Results saved to experiments/")
