"""
gpt2_evaluator.py — GPT-2 Row of the 2×2 Experiment

Runs the same three context conditions (graph + store, baseline, graph-only)
as evaluate.py, but uses GPT-2 for both:
  1. Concept extraction  — spaCy noun phrases (GPT-2 tokenizer for budget
                           counting, matching the tiktoken gpt2 encoding
                           used in the rest of the codebase)
  2. Answering           — GPT-2 completion (not instruction-following;
                           prompt is restructured as a cloze template)

The answering model is deliberately weak. The scientific question is:
  Does graph-structured context help GPT-2 more than it helps Sonnet?
  Can memory architecture compensate for low model capacity?

Output schema is identical to evaluate.py so results can be merged into
the same summary table for the 2×2 comparison.

Usage:
    python gpt2_evaluator.py
    python gpt2_evaluator.py --testcase memory
    python gpt2_evaluator.py --rich --field

Dependencies (beyond the existing project requirements):
    pip install torch transformers

Authors: Stig Norland, Claude (Anthropic)
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass
from datetime    import datetime
from pathlib     import Path

# ---------------------------------------------------------------------------
# Optional heavy imports — fail loudly with a helpful message
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    _GPT2_AVAILABLE = True
except ImportError:
    _GPT2_AVAILABLE = False

try:
    import spacy
    _SPACY_NLP = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_NLP = None

# Project imports — same as evaluate.py
sys.path.insert(0, str(Path(__file__).parent / "src"))
from library.graph       import ConceptGraph, NodeType, EdgeType
from library.context     import ContextAssembler, BackingStore
from library.convergence import ConvergenceMonitor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TESTDATA_DIR  = Path(__file__).parent.parent / "testdata"
ANSWER_TOKENS = 60    # GPT-2 is slow; keep completions short
MAX_NEW_TOKENS = 60
GPT2_MODEL    = "gpt2"          # 117M — the weakest, most honest baseline
TEMPERATURE   = 1.0             # greedy via do_sample=False


# ---------------------------------------------------------------------------
# Data classes (mirror evaluate.py)
# ---------------------------------------------------------------------------

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
    token_budget: int = 200


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
        token_budget = data["metadata"].get("token_budget", 200),
    )


def load_all_testcases(directory: Path = TESTDATA_DIR) -> list[TestCase]:
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files in {directory}")
    cases = [load_testcase(p) for p in paths]
    print(f"Loaded {len(cases)} test case(s):")
    for c in cases:
        print(f"  {c.name}: {len(c.conversation)} turns, {len(c.questions)} questions")
    return cases


# ---------------------------------------------------------------------------
# GPT-2 concept extractor
# ---------------------------------------------------------------------------

def extract_concepts_spacy(text: str) -> list[str]:
    """
    Extract noun-phrase concepts using spaCy.
    This is the extraction backbone regardless of the answering model.
    GPT-2's tokenizer is used for token-budget counting only (elsewhere).

    Falls back to a simple stopword filter if spaCy is unavailable.
    """
    if _SPACY_NLP is not None:
        try:
            doc = _SPACY_NLP(text[:1000])
            seen   = set()
            labels = []
            # Prefer noun chunks (multi-word concepts preserved)
            for chunk in doc.noun_chunks:
                label = chunk.root.lemma_.lower().strip()
                label = label.replace(" ", "_")
                if label and len(label) > 2 and label not in seen:
                    seen.add(label)
                    labels.append(label)
            # Also add named entities
            for ent in doc.ents:
                label = ent.text.lower().replace(" ", "_")
                if label not in seen and len(label) > 2:
                    seen.add(label)
                    labels.append(label)
            return labels[:12]
        except Exception:
            pass

    # Stopword fallback
    import re
    STOPS = {
        'the','a','an','is','are','was','were','be','been','being',
        'have','has','had','do','does','did','will','would','could',
        'should','may','might','can','to','of','in','for','on','with',
        'at','by','from','i','you','he','she','it','we','they','what',
        'which','who','that','this','and','but','or','not','so','as',
        'if','then','because','while','just','very',
    }
    words = re.findall(r"[a-z]{3,}", text.lower())
    seen, out = set(), []
    for w in words:
        if w not in STOPS and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:12]


# ---------------------------------------------------------------------------
# GPT-2 answering
# ---------------------------------------------------------------------------

class GPT2Answerer:
    """
    Wraps GPT-2 for completion-style QA.

    Because GPT-2 is not instruction-tuned, we use a cloze prompt:

        Context: {context}

        Question: {question}
        The answer is:

    GPT-2 continues the completion. We take everything up to the first
    newline or sentence boundary as the answer.
    """

    def __init__(self, model_name: str = GPT2_MODEL):
        if not _GPT2_AVAILABLE:
            raise ImportError(
                "torch and transformers are required for GPT2Evaluator.\n"
                "Install with: pip install torch transformers"
            )
        print(f"Loading GPT-2 ({model_name})...", end=" ", flush=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model     = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"loaded on {self.device}.")

    def ask(self, context: str, question: str,
            max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """
        Generate a completion answer for the given context + question.
        Trims context to fit within GPT-2's 1024-token window.
        """
        # Build cloze prompt
        prompt = self._build_prompt(context, question)

        # Encode — respect GPT-2's 1024 token limit
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=900
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens     = max_new_tokens,
                do_sample          = False,       # greedy — reproducible
                repetition_penalty = 1.3,         # prevents "X is X is X" loops
                pad_token_id       = self.tokenizer.eos_token_id,
                eos_token_id       = self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._trim_answer(raw)

    def count_tokens(self, text: str) -> int:
        """Token count using GPT-2 tokenizer (matches tiktoken gpt2)."""
        return len(self.tokenizer.encode(text))

    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(context: str, question: str) -> str:
        """
        Cloze-style prompt for a non-instruction-tuned model.

        The hard separator (===) is critical: without it, GPT-2 treats
        graph-only notation (node labels, edge arrows) as text to continue
        rather than as context to reason from. The separator creates a
        clean boundary that steers the completion toward prose.
        """
        ctx = context.strip()
        q   = question.strip().rstrip("?")
        if ctx:
            return (
                f"Conversation summary:\n\n"
                f"{ctx}\n\n"
                f"===\n\n"
                f"Based on the conversation above, {q}. The answer is:"
            )
        else:
            return f"{q}. The answer is:"

    @staticmethod
    def _trim_answer(raw: str) -> str:
        """
        Extract the first meaningful sentence from GPT-2's completion.
        GPT-2 often runs on — we want just the answer fragment.
        """
        import re
        # Stop at first newline
        first_line = raw.split("\n")[0].strip()
        # Stop at first sentence boundary (period, ?, !)
        sentence   = re.split(r"(?<=[.?!])\s", first_line)[0].strip()
        # If nothing useful, return what we have (up to 100 chars)
        return sentence[:150] if sentence else raw[:150]


# ---------------------------------------------------------------------------
# GPT2 Context Assembler (thin wrapper that patches extraction)
# ---------------------------------------------------------------------------

class GPT2ContextAssembler(ContextAssembler):
    """
    Subclass of ContextAssembler that overrides concept extraction
    to use spaCy noun phrases instead of Claude Haiku.

    Token budget counting uses GPT-2's tokenizer to stay consistent
    with the answering model.
    """

    def __init__(self, graph, store, gpt2_answerer: GPT2Answerer,
                 use_rich_payloads: bool = False,
                 use_field: bool = False):
        super().__init__(
            graph              = graph,
            store              = store,
            tokenizer          = None,    # we override _count_tokens below
            llm_client         = None,    # no Haiku calls
            use_llm_extraction = False,
            use_rich_payloads  = use_rich_payloads,
            use_field          = use_field,
        )
        self._gpt2 = gpt2_answerer

    def _extract_concepts(self, text: str, speaker: str = "unknown") -> list[str]:
        """Override: use spaCy only, no LLM calls."""
        return extract_concepts_spacy(text)

    def _count_tokens(self, text: str) -> int:
        """Override: use GPT-2 tokenizer for budget accounting."""
        return self._gpt2.count_tokens(text)


# ---------------------------------------------------------------------------
# GPT2Evaluator
# ---------------------------------------------------------------------------

class GPT2Evaluator:
    """
    Runs a TestCase through the full RGM pipeline using GPT-2 for both
    concept extraction (via spaCy) and answer generation.

    Drop-in complement to evaluate.py's Evaluator — produces the same
    output schema so results merge cleanly into the 2×2 comparison table.

    The answering model is the key difference: instead of Claude Sonnet 4
    (instruction-tuned, 175B+), we use GPT-2 (117M, completion-only).
    This directly tests whether graph memory architecture adds value
    independently of model capability.
    """

    ANSWERING_MODEL = f"gpt2-117M"    # label for output metadata

    def __init__(self, testcase: TestCase,
                 use_rich_payloads: bool = False,
                 use_field:         bool = False):
        self.testcase          = testcase
        self.TOKEN_BUDGET      = testcase.token_budget
        self.use_rich_payloads = use_rich_payloads
        self.use_field         = use_field
        self.results           = []
        self._inference_times  = []   # per-question timing for speed report

        self.gpt2 = GPT2Answerer(GPT2_MODEL)

    # ------------------------------------------------------------------
    # Public API (mirrors evaluate.py Evaluator)
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"\n{'='*60}")
        print(f"GPT-2 EVALUATION: {self.testcase.name}")
        print(f"  {self.testcase.description}")
        print(f"  {len(self.testcase.conversation)} turns  |  "
              f"{len(self.testcase.questions)} questions  |  "
              f"budget {self.TOKEN_BUDGET} tokens")
        print(f"  answering model: {self.ANSWERING_MODEL}  "
              f"(spaCy extraction, completion-style QA)")
        print(f"{'='*60}")

        graph, assembler, store = self._build_graph()

        for qi, q in enumerate(self.testcase.questions, 1):
            print(f"\n--- L{q.level}: {q.description} ---")
            print(f"Q: {q.text}")

            result = {
                "question":    q.text,
                "level":       q.level,
                "description": q.description,
                "conditions":  {},
            }

            inference_times = {}
            for mode in ["graph", "baseline", "graph_only"]:
                ctx    = self._assemble_context(mode, assembler, store, graph, q.text)
                t0     = time.perf_counter()
                answer = self.gpt2.ask(ctx, q.text)
                elapsed = time.perf_counter() - t0
                inference_times[mode] = elapsed
                score  = self._score(answer, q.key_concepts)
                hits   = self._key_hits(answer, q.key_concepts)

                result["conditions"][mode] = {
                    "context_tokens":  self.gpt2.count_tokens(ctx),
                    "answer":          answer,
                    "score":           score,
                    "key_hits":        hits,
                    "inference_time_s": round(elapsed, 3),
                }

                ind = "✓" if score >= 0.5 else "~" if score >= 0.25 else "✗"
                print(f"  {mode:12s} [{ind}] score={score:.2f}  "
                      f"tokens={self.gpt2.count_tokens(ctx)}  "
                      f"hits={hits}/{len(q.key_concepts)}  "
                      f"({elapsed:.2f}s local)")

            self.results.append(result)
            self._inference_times.append(inference_times)

        # Timing summary — the speed finding
        all_times = [t for q_times in self._inference_times
                     for t in q_times.values()]
        n_calls   = len(all_times)
        total_s   = sum(all_times)
        print(f"\n{'─'*60}")
        print(f"Timing ({len(self.testcase.questions)} questions × 3 conditions = {n_calls} calls):")
        print(f"  Total inference: {total_s:.1f}s  ({total_s/n_calls:.2f}s avg per call)")
        print(f"  All local — zero API latency, zero rate limiting, zero cost.")

    def save(self) -> dict:
        summary = self._summarise()
        self._print_report(summary)

        os.makedirs("experiments", exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"experiments/eval_gpt2_{self.testcase.name}_{ts}"

        all_times = [t for q_times in self._inference_times
                     for t in q_times.values()]
        output = {
            "metadata": {
                "timestamp":            ts,
                "name":                 self.testcase.name,
                "model":                self.ANSWERING_MODEL,
                "extraction":           "spacy_noun_phrases",
                "temperature":          0,
                "repetition_penalty":   1.3,
                "token_budget":         self.TOKEN_BUDGET,
                "questions":            len(self.testcase.questions),
                "turns":                len(self.testcase.conversation),
                "use_rich_payloads":    self.use_rich_payloads,
                "use_field":            self.use_field,
                "experiment_role":      "gpt2_row_of_2x2",
                "total_inference_s":    round(sum(all_times), 2),
                "avg_inference_s":      round(sum(all_times)/max(len(all_times),1), 3),
                "note_speed":           "All inference local — no API calls, no network latency.",
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

    # ------------------------------------------------------------------
    # Graph building
    # ------------------------------------------------------------------

    def _build_graph(self) -> tuple:
        graph     = ConceptGraph()
        store     = BackingStore()
        assembler = GPT2ContextAssembler(
            graph              = graph,
            store              = store,
            gpt2_answerer      = self.gpt2,
            use_rich_payloads  = self.use_rich_payloads,
            use_field          = self.use_field,
        )

        t0      = time.perf_counter()
        monitor = ConvergenceMonitor(max_passes=20, window=8)

        for i, (speaker, text) in enumerate(self.testcase.conversation):
            assembler.ingest(speaker, text)
            graph.begin_pass()
            self._propagation_pass(graph)
            status = graph.end_pass()
            monitor.record(status)
            print(f"  turn {i+1:3d}  nodes={len(graph.nodes)}  "
                  f"edges={len(graph.edges)}  status={status}")

        while monitor.should_continue():
            graph.begin_pass()
            self._propagation_pass(graph)
            status = graph.end_pass()
            monitor.record(status)

        elapsed = time.perf_counter() - t0
        print(f"\nGraph built in {elapsed:.2f}s: "
              f"{len(graph.nodes)} nodes, "
              f"{len(graph.edges)} edges, "
              f"{len(graph.merge_events)} merges")
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

    # ------------------------------------------------------------------
    # Context assembly (delegates to assembler, same as evaluate.py)
    # ------------------------------------------------------------------

    def _assemble_context(self, mode: str, assembler: GPT2ContextAssembler,
                          store: BackingStore, graph: ConceptGraph,
                          query: str) -> str:
        if mode == "graph":
            ctx = assembler.assemble_graph(query=query, token_budget=self.TOKEN_BUDGET)
            if not ctx.strip():
                ctx = assembler.assemble_baseline(token_budget=self.TOKEN_BUDGET)
            return ctx
        if mode == "baseline":
            return assembler.assemble_baseline(token_budget=self.TOKEN_BUDGET)
        if mode == "graph_only":
            return assembler.assemble_graph_only(
                query=query, token_budget=self.TOKEN_BUDGET
            )
        return ""

    # ------------------------------------------------------------------
    # Scoring (identical to evaluate.py)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Reporting (mirrors evaluate.py)
    # ------------------------------------------------------------------

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
        print(f"SUMMARY (GPT-2): {self.testcase.name}")
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
            rf"\caption{{GPT-2 QA evaluation — {self.testcase.name}}}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Level & Graph + store & Baseline & Graph only \\",
            r"\midrule",
        ]
        for lv, label in level_labels.items():
            if lv in summary["by_level"]:
                d = summary["by_level"][lv]
                lines.append(
                    rf"{label} & {d.get('graph', 0):.2f} & "
                    rf"{d.get('baseline', 0):.2f} & "
                    rf"{d.get('graph_only', 0):.2f} \\"
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _GPT2_AVAILABLE:
        print("ERROR: torch and transformers not installed.")
        print("Run: pip install torch transformers")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(
        description="RGM 2×2 Experiment — GPT-2 row"
    )
    parser.add_argument(
        "--rich", action="store_true",
        help="Enable rich payload synthesis (local, no API calls)."
    )
    parser.add_argument(
        "--field", action="store_true",
        help="Enable bioelectric field layer."
    )
    parser.add_argument(
        "--testcase", type=str, default=None,
        help="Run only a specific testcase by name. Default: run all."
    )
    args = parser.parse_args()

    testcases = load_all_testcases()
    if args.testcase:
        testcases = [tc for tc in testcases if tc.name == args.testcase]
        if not testcases:
            print(f"No testcase named '{args.testcase}' found.")
            raise SystemExit(1)

    for tc in testcases:
        ev = GPT2Evaluator(
            tc,
            use_rich_payloads = args.rich,
            use_field         = args.field,
        )
        ev.run()
        ev.save()
        print()

    print("Done. GPT-2 results saved to experiments/eval_gpt2_*.json")
    print("Compare against experiments/eval_<name>_*.json for the 2×2 table.")