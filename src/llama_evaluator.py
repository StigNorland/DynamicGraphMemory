"""
qwen_evaluator.py — Local Small Model Row of the 2×2 Experiment

Runs the same three context conditions as evaluate.py (graph + store,
baseline, graph-only) but uses a local Qwen3-1.7B model via ollama
for both concept extraction and answering.

Scientific question:
  Does RGM graph memory add value for a small, fully local model?
  Can the architecture deliver reasoning improvement without any cloud
  dependency — zero API cost, zero latency, complete privacy?

Model tiers in the full 2×2:
  GPT-2 117M   — floor: completion-only, cannot read context (score: 0.00)
  Qwen3 1.7B   — this file: instruction-tuned, local, private, free
  Sonnet 4     — ceiling: frontier cloud model (evaluate.py)

Qwen3-1.7B differences from Sonnet that matter scientifically:
  - 1.7B vs ~200B+ parameters
  - Local CPU/GPU inference vs cloud API
  - No API cost, no rate limits, no data leaving the machine
  - Uses /no_think mode: fast responses without chain-of-thought overhead

Architecture is identical to evaluate.py — only the answering model
and concept extractor differ. Graph construction, propagation,
convergence monitoring, and context assembly are unchanged.

Usage:
    # Make sure ollama is running and qwen3:1.7b is pulled:
    #   ollama pull qwen3:1.7b
    python qwen_evaluator.py
    python qwen_evaluator.py --testcase memory
    python qwen_evaluator.py --testcase memory --rich

Authors: Stig Norland, Claude (Anthropic)
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime    import datetime
from pathlib     import Path

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

# Optional: attention-derived edge extraction
# Requires a local transformer model (e.g. GPT-2 via transformers).
# Set use_attention_edges=True in QwenEvaluator to enable.
try:
    from extractors.attention_edges import AttentionEdgeExtractor
    _ATTENTION_EDGES_AVAILABLE = True
except ImportError:
    _ATTENTION_EDGES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TESTDATA_DIR   = Path(__file__).parent.parent / "testdata"
OLLAMA_URL     = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL   = "llama3.2:3b"
ANSWER_TOKENS  = 120    # enough for 2-3 sentences
OLLAMA_TIMEOUT = 60     # seconds — local inference can be slow on CPU


# ---------------------------------------------------------------------------
# Data classes (mirror evaluate.py exactly)
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
        print(f"  {c.name}: {len(c.conversation)} turns, "
              f"{len(c.questions)} questions")
    return cases


# ---------------------------------------------------------------------------
# Ollama answerer
# ---------------------------------------------------------------------------

class OllamaAnswerer:
    """
    Calls a local ollama model for instruction-following QA.

    Uses the same HTTP pattern as payload_local.py — proven working
    in the existing codebase. Extended here for full QA answers
    rather than one-sentence summaries.

    Qwen3-specific: prepends /no_think to suppress chain-of-thought
    output, keeping answers concise and the timing fair vs Sonnet.
    """

    def __init__(self, model: str = OLLAMA_MODEL,
                 timeout: int = OLLAMA_TIMEOUT):
        self.model   = model
        self.timeout = timeout
        self._calls  = 0
        self._errors = 0
        self._total_s = 0.0

        # Probe on init — fail fast if ollama not running
        print(f"Connecting to ollama ({model})...", end=" ", flush=True)
        probe = self._call("Say ok", "", max_tokens=10)
        if probe is None:
            raise RuntimeError(
                f"\nollama not responding at {OLLAMA_URL}\n"
                f"On Windows: open a terminal and run 'ollama serve', "
                f"leave it running, then retry.\n"
                f"Model check: ollama list  (should show {model})"
            )
        print(f"ready. (probe: '{probe[:30]}')")

    def ask(self, context: str, question: str) -> str:
        """Answer a question given assembled context."""
        t0     = time.perf_counter()
        answer = self._call(question, context, max_tokens=ANSWER_TOKENS)
        elapsed = time.perf_counter() - t0
        self._total_s += elapsed
        if answer is None:
            self._errors += 1
            return "[ERROR: ollama call failed]"
        return answer

    def ask_timed(self, context: str, question: str) -> tuple[str, float]:
        """Returns (answer, elapsed_seconds)."""
        t0      = time.perf_counter()
        answer  = self.ask(context, question)
        elapsed = time.perf_counter() - t0
        return answer, elapsed

    def count_tokens(self, text: str) -> int:
        """Word-count approximation — no tokenizer needed for ollama."""
        return len(text.split())

    @property
    def stats(self) -> dict:
        return {
            "calls":   self._calls,
            "errors":  self._errors,
            "total_s": round(self._total_s, 2),
            "avg_s":   round(self._total_s / max(self._calls, 1), 3),
        }

    # ------------------------------------------------------------------

    def _call(self, question: str, context: str,
              max_tokens: int = ANSWER_TOKENS) -> str | None:
        """
        Single ollama chat API call.

        Uses /api/chat with messages format — required for instruction-tuned
        models in current ollama versions. /api/generate returns 405.

        Qwen3-specific: /no_think in the user message suppresses
        <think>...</think> chain-of-thought blocks.
        """
        if context.strip():
            user_content = (
                f"Context:\n{context.strip()}\n\n"
                f"Question: {question}\n\n"
                "Answer in 1-3 sentences using only the context above."
            )
        else:
            user_content = f"Answer in 1-3 sentences: {question}"

        payload = json.dumps({
            "model":  self.model,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.0,
                "top_p":       1.0,
            },
            "messages": [
                {
                    "role":    "system",
                    "content": (
                        "You answer questions based solely on the provided "
                        "context. Do not use outside knowledge. Be concise."
                    ),
                },
                {
                    "role":    "user",
                    "content": user_content,
                },
            ],
        }).encode()

        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data   = json.loads(resp.read())
                # /api/chat returns message.content, not response
                answer = (
                        data.get("message", {}).get("content", "") or
                        data.get("response", "")
                ).strip()
                self._calls += 1
                return answer if answer else None
        except (urllib.error.URLError, TimeoutError):
            return None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Qwen3 Context Assembler — spaCy extraction, word-count budget
# ---------------------------------------------------------------------------

def extract_concepts_spacy(text: str) -> list[str]:
    """
    spaCy noun-phrase extractor — same as gpt2_evaluator.py.
    Falls back to stopword filter if spaCy unavailable.
    """
    if _SPACY_NLP is not None:
        try:
            doc    = _SPACY_NLP(text[:1000])
            seen   = set()
            labels = []
            for chunk in doc.noun_chunks:
                label = chunk.root.lemma_.lower().replace(" ", "_")
                if label and len(label) > 2 and label not in seen:
                    seen.add(label)
                    labels.append(label)
            for ent in doc.ents:
                label = ent.text.lower().replace(" ", "_")
                if label not in seen and len(label) > 2:
                    seen.add(label)
                    labels.append(label)
            return labels[:12]
        except Exception:
            pass

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


class QwenContextAssembler(ContextAssembler):
    """
    Thin subclass of ContextAssembler: overrides extraction (spaCy)
    and token counting (word count) only. Everything else — graph
    traversal, backing store, field layer — is identical.

    Optional: pass use_attention_edges=True together with a HuggingFace
    model and tokenizer to enrich the graph with attention-derived edges
    on every ingest() call. The model runs a forward pass over each turn
    and the extractor maps high-weight attention pairs back to concept
    nodes, injecting or reinforcing edges in the graph.
    """

    def __init__(self, graph, store,
                 use_rich_payloads:   bool = False,
                 use_field:           bool = False,
                 use_attention_edges: bool = False,
                 attn_model          = None,
                 attn_tokenizer      = None):
        super().__init__(
            graph              = graph,
            store              = store,
            tokenizer          = None,
            llm_client         = None,
            use_llm_extraction = False,
            use_rich_payloads  = use_rich_payloads,
            use_field          = use_field,
        )

        # Attention-derived edge extractor (optional)
        self._attention_extractor = None
        self._attn_tokenizer      = None
        if (use_attention_edges
                and _ATTENTION_EDGES_AVAILABLE
                and attn_model is not None
                and attn_tokenizer is not None):
            self._attention_extractor = AttentionEdgeExtractor(
                attn_model, attn_tokenizer, graph
            )
            self._attn_tokenizer = attn_tokenizer
            print("  [attention_edges] extractor ready.")
        elif use_attention_edges:
            print("  [attention_edges] skipped — "
                  "module or model not available.")

    def _extract_concepts(self, text: str,
                          speaker: str = "unknown") -> list[str]:
        return extract_concepts_spacy(text)

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def ingest(self, speaker: str, text: str) -> dict:
        """
        Ingest a turn. If attention_extractor is active, runs a GPT-2
        forward pass over the turn text and injects attention-derived
        edges after the standard graph write.
        """
        status = super().ingest(speaker, text)

        if self._attention_extractor is not None:
            try:
                import torch
                input_ids = self._attn_tokenizer.encode(
                    text, return_tensors="pt"
                )
                with self._attention_extractor.hooked():
                    with torch.no_grad():
                        self._attention_extractor._model(
                            input_ids, output_attentions=True
                        )
                token_node_map = self._attention_extractor.build_token_node_map(
                    input_ids, self.graph
                )
                candidates = self._attention_extractor.get_edges(token_node_map)
                n = self._attention_extractor.inject(
                    candidates, self.graph, self.graph.conv_pos
                )
                if n:
                    status["attention_edges_injected"] = n

                # Surface model-internal tension: pairs the model attends to
                # but the graph has no edge for yet.
                gaps = self._attention_extractor.tension_candidates(
                    candidates, self.graph
                )
                if gaps:
                    status["attention_tension"] = gaps[:5]

            except Exception as e:
                # Never let extractor failure break the main eval loop
                status["attention_edges_error"] = str(e)

        return status


# ---------------------------------------------------------------------------
# QwenEvaluator
# ---------------------------------------------------------------------------

class QwenEvaluator:
    """
    Runs a TestCase through RGM using Qwen3-1.7B (local ollama) for
    both concept extraction and QA answering.

    The 2×2 table position:
      Rows: model tier (Qwen3-1.7B local  vs  Sonnet 4 cloud)
      Cols: context condition (graph+store, baseline, graph-only)

    Output schema identical to evaluate.py — results merge directly
    into the same comparison table.
    """

    ANSWERING_MODEL = "llama3.2-3b-local"

    def __init__(self, testcase: TestCase,
                 use_rich_payloads:   bool = False,
                 use_field:           bool = False,
                 use_attention_edges: bool = False,
                 ollama_model:        str  = OLLAMA_MODEL):
        self.testcase              = testcase
        self.TOKEN_BUDGET          = testcase.token_budget
        self.use_rich_payloads     = use_rich_payloads
        self.use_field             = use_field
        self.use_attention_edges   = use_attention_edges
        self.results               = []
        self._inference_times      = []

        self.ollama = OllamaAnswerer(model=ollama_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"\n{'='*60}")
        print(f"LLAMA3.2-3B LOCAL EVALUATION: {self.testcase.name}")
        print(f"  {self.testcase.description}")
        print(f"  {len(self.testcase.conversation)} turns  |  "
              f"{len(self.testcase.questions)} questions  |  "
              f"budget {self.TOKEN_BUDGET} tokens")
        print(f"  model: {OLLAMA_MODEL}  extraction: spaCy")
        print(f"  fully local — no API calls, no network, no cost")
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
                ctx     = self._assemble_context(
                    mode, assembler, store, graph, q.text
                )
                t0      = time.perf_counter()
                answer  = self.ollama.ask(ctx, q.text)
                elapsed = time.perf_counter() - t0
                inference_times[mode] = round(elapsed, 3)

                score = self._score(answer, q.key_concepts)
                hits  = self._key_hits(answer, q.key_concepts)

                result["conditions"][mode] = {
                    "context_tokens":   self.ollama.count_tokens(ctx),
                    "answer":           answer,
                    "score":            score,
                    "key_hits":         hits,
                    "inference_time_s": round(elapsed, 3),
                }

                ind = "✓" if score >= 0.5 else "~" if score >= 0.25 else "✗"
                print(f"  {mode:12s} [{ind}] score={score:.2f}  "
                      f"tokens={self.ollama.count_tokens(ctx)}  "
                      f"hits={hits}/{len(q.key_concepts)}  "
                      f"({elapsed:.2f}s local)")

            self.results.append(result)
            self._inference_times.append(inference_times)

        # Timing summary
        all_times = [t for qt in self._inference_times
                     for t in qt.values()]
        n  = len(all_times)
        total = sum(all_times)
        print(f"\n{'─'*60}")
        print(f"Timing ({len(self.testcase.questions)} questions × 3 = "
              f"{n} calls):")
        print(f"  Total:   {total:.1f}s  |  Avg: {total/n:.2f}s/call")
        print(f"  Local inference — zero API cost, zero latency variance.")

    def save(self) -> dict:
        summary = self._summarise()
        self._print_report(summary)

        os.makedirs("experiments", exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"experiments/eval_qwen_{self.testcase.name}_{ts}"

        all_times = [t for qt in self._inference_times
                     for t in qt.values()]

        output = {
            "metadata": {
                "timestamp":          ts,
                "name":               self.testcase.name,
                "model":              self.ANSWERING_MODEL,
                "ollama_model":       OLLAMA_MODEL,
                "extraction":         "spacy_noun_phrases",
                "temperature":        0,
                "token_budget":       self.TOKEN_BUDGET,
                "questions":          len(self.testcase.questions),
                "turns":              len(self.testcase.conversation),
                "use_rich_payloads":    self.use_rich_payloads,
                "use_field":            self.use_field,
                "use_attention_edges":  self.use_attention_edges,
                "experiment_role":      "local_small_model_row_of_2x2",
                "total_inference_s":  round(sum(all_times), 2),
                "avg_inference_s":    round(
                    sum(all_times) / max(len(all_times), 1), 3
                ),
                "note":               (
                    "Fully local — ollama/llama3.2:3b. "
                    "No API calls, no network latency, no data egress."
                ),
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
    # Graph building — identical to evaluate.py
    # ------------------------------------------------------------------

    def _build_graph(self) -> tuple:
        graph     = ConceptGraph()
        store     = BackingStore()

        # Load GPT-2 for attention-edge extraction if requested.
        # GPT-2 is used as a frozen oracle — weights never updated.
        attn_model, attn_tokenizer = None, None
        if self.use_attention_edges and _ATTENTION_EDGES_AVAILABLE:
            try:
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                print("  [attention_edges] loading GPT-2...", end=" ", flush=True)
                attn_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                attn_model     = GPT2LMHeadModel.from_pretrained("gpt2")
                attn_model.eval()
                print("ready.")
            except Exception as e:
                print(f"failed ({e}) — continuing without attention edges.")

        assembler = QwenContextAssembler(
            graph               = graph,
            store               = store,
            use_rich_payloads   = self.use_rich_payloads,
            use_field           = self.use_field,
            use_attention_edges = self.use_attention_edges,
            attn_model          = attn_model,
            attn_tokenizer      = attn_tokenizer,
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
                  f"edges={len(graph.edges)}")

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
            if (node.provisional
                    and node.maturity >= graph.maturity_threshold):
                graph._stabilize(node_id)
                graph._write_count += 1

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _assemble_context(self, mode: str,
                          assembler: QwenContextAssembler,
                          store: BackingStore,
                          graph: ConceptGraph,
                          query: str) -> str:
        if mode == "graph":
            ctx = assembler.assemble_graph(
                query=query, token_budget=self.TOKEN_BUDGET
            )
            if not ctx.strip():
                ctx = assembler.assemble_baseline(
                    token_budget=self.TOKEN_BUDGET
                )
            return ctx
        if mode == "baseline":
            return assembler.assemble_baseline(
                token_budget=self.TOKEN_BUDGET
            )
        if mode == "graph_only":
            return assembler.assemble_graph_only(
                query=query, token_budget=self.TOKEN_BUDGET
            )
        return ""

    # ------------------------------------------------------------------
    # Scoring — identical to evaluate.py
    # ------------------------------------------------------------------

    def _score(self, answer: str, key_concepts: list[str]) -> float:
        hits = self._key_hits(answer, key_concepts)
        return hits / len(key_concepts) if key_concepts else 0.0

    def _key_hits(self, answer: str, key_concepts: list[str]) -> int:
        import re
        a       = answer.lower()
        a_clean = re.sub(r"[^\w\s]", " ", a)
        a_words = set(a_clean.split())
        hits    = 0
        for concept in key_concepts:
            c = concept.lower()
            if c in a:
                hits += 1; continue
            if c.replace("_", " ") in a:
                hits += 1; continue
            words = [w for w in re.split(r"[_\s]", c) if len(w) >= 4]
            if words and any(w in a_words for w in words):
                hits += 1
        return hits

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _summarise(self) -> dict:
        levels     = sorted(set(r["level"] for r in self.results))
        conditions = ["graph", "baseline", "graph_only"]
        by_level   = {
            str(lv): {
                c: (
                        sum(r["conditions"][c]["score"]
                            for r in self.results if r["level"] == lv)
                        / max(1, sum(
                    1 for r in self.results if r["level"] == lv
                ))
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
        print(f"SUMMARY (Llama3.2-3B local): {self.testcase.name}")
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
        # Compare graph vs baseline delta
        delta = o["graph"] - o["baseline"]
        if delta > 0.05:
            print(f"Graph advantage: +{delta:.3f} over baseline")
        elif delta < -0.05:
            print(f"Baseline leads by: {-delta:.3f}")
        else:
            print("Graph and baseline within rounding.")

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
            rf"\caption{{Llama3.2-3B local QA — {self.testcase.name}}}",
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
            rf"Overall & {o['graph']:.2f} & {o['baseline']:.2f} & "
            rf"{o['graph_only']:.2f} \\",
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
    parser = argparse.ArgumentParser(
        description="RGM 2×2 Experiment — Qwen3-1.7B local row"
    )
    parser.add_argument(
        "--rich", action="store_true",
        help="Enable rich payload synthesis (local ollama, no cloud API)."
    )
    parser.add_argument(
        "--field", action="store_true",
        help="Enable bioelectric field layer."
    )
    parser.add_argument(
        "--attention", action="store_true",
        help="Enable attention-derived edge extraction (loads GPT-2 locally)."
    )
    parser.add_argument(
        "--testcase", type=str, default=None,
        help="Run only a specific testcase by name. Default: run all."
    )
    parser.add_argument(
        "--model", type=str, default=OLLAMA_MODEL,
        help=f"Ollama model name (default: {OLLAMA_MODEL})."
    )
    args = parser.parse_args()

    testcases = load_all_testcases()
    if args.testcase:
        testcases = [tc for tc in testcases if tc.name == args.testcase]
        if not testcases:
            print(f"No testcase named '{args.testcase}' found.")
            raise SystemExit(1)

    for tc in testcases:
        ev = QwenEvaluator(
            tc,
            use_rich_payloads   = args.rich,
            use_field           = args.field,
            use_attention_edges = args.attention,
            ollama_model        = args.model,
        )
        ev.run()
        ev.save()
        print()

    print("Done. Results saved to experiments/eval_qwen_*.json")
    print("Compare with experiments/eval_<name>_*.json for the 2×2 table.")