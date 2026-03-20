"""
evaluate.py — Question Answering Evaluation Framework

Tests whether graph-compressed context preserves enough meaning
for a language model to answer questions requiring genuine
understanding of the conversation.

Three conditions, identical token budget:
  - GRAPH:      graph-derived context + backing store
  - BASELINE:   raw recent turns
  - GRAPH_ONLY: graph structure only, backing store discarded

If GRAPH_ONLY beats BASELINE on synthesis and inference questions,
meaning is in the relational structure, not the raw text.

Authors: Stig Norland, Claude (Anthropic)
"""

import os
import json
import time
from dataclasses import dataclass
from datetime import datetime

import anthropic

from graph_memory.graph       import ConceptGraph, N
from graph_memory.context     import ContextAssembler, BackingStore
from graph_memory.convergence import ConvergenceMonitor


# -------------------------------------------------------------------
# Question sets — four levels × two conversations
# -------------------------------------------------------------------

@dataclass
class Question:
    text:         str
    key_concepts: list[str]
    level:        int          # 1=retrieval 2=synthesis 3=inference 4=cross-domain
    description:  str


MEMORY_QUESTIONS = [
    # Level 1 — direct retrieval
    Question(
        text         = "What causes forgetting according to our conversation?",
        key_concepts = ["retrieval", "path", "weak"],
        level        = 1,
        description  = "Direct retrieval — answer stated explicitly",
    ),
    Question(
        text         = "Where does memory consolidation take place?",
        key_concepts = ["sleep", "hippocampus"],
        level        = 1,
        description  = "Direct retrieval — stated in single turn",
    ),
    # Level 2 — synthesis
    Question(
        text         = "What do muscle memory and forgetting your keys have in common?",
        key_concepts = ["retrieval", "mechanism", "same", "path"],
        level        = 2,
        description  = "Synthesis — requires connecting turns 1 and 9",
    ),
    Question(
        text         = "How does spaced repetition relate to retrieval paths?",
        key_concepts = ["repetition", "strengthen", "path", "retrieval"],
        level        = 2,
        description  = "Synthesis — connects repetition and path strengthening",
    ),
    # Level 3 — inference
    Question(
        text         = "Would practicing recalling something help you forget it less? Why?",
        key_concepts = ["repetition", "strengthen", "path", "retrieval"],
        level        = 3,
        description  = "Inference — requires reasoning from relational structure",
    ),
    Question(
        text         = "If retrieval failure differs from storage failure, what does that imply about forgetting?",
        key_concepts = ["path", "block", "exist", "access"],
        level        = 3,
        description  = "Inference — forgetting as blockage not erasure",
    ),
    # Level 4 — cross-domain
    Question(
        text         = "What is the single most fundamental concept in everything we discussed?",
        key_concepts = ["retrieval"],
        level        = 4,
        description  = "Primitive identification — tests whether graph found the anchor",
    ),
    Question(
        text         = "The conversation describes a structure for memory. What is the closest analogy to a graph node in this model?",
        key_concepts = ["memory", "concept", "retriev", "encod"],
        level        = 4,
        description  = "Cross-domain — maps memory model onto graph structure",
    ),
]

CROSS_DOMAIN_QUESTIONS = [
    # Level 1 — direct retrieval
    Question(
        text         = "What controls the size of each weight adjustment in neural network training?",
        key_concepts = ["learning rate", "rate"],
        level        = 1,
        description  = "Direct retrieval — stated in turn 4",
    ),
    Question(
        text         = "What happens when a population cannot adapt fast enough to environmental change?",
        key_concepts = ["extinct", "extinction"],
        level        = 1,
        description  = "Direct retrieval — stated in turn 21",
    ),
    # Level 2 — synthesis
    Question(
        text         = "What do learning rate and mutation rate have in common?",
        key_concepts = ["rate", "change", "control", "variation"],
        level        = 2,
        description  = "Synthesis — requires connecting ML and evolution domains",
    ),
    Question(
        text         = "How does overfitting in neural networks relate to evolutionary traps?",
        key_concepts = ["stuck", "trap", "memoris", "suboptimal", "local"],
        level        = 2,
        description  = "Synthesis — cross-domain structural equivalence",
    ),
    # Level 3 — inference
    Question(
        text         = "Why might a neural network and a species both get permanently stuck?",
        key_concepts = ["minimum", "trap", "environment", "adapt", "solution"],
        level        = 3,
        description  = "Inference — requires cross-domain structural equivalence",
    ),
    Question(
        text         = "If the training environment defines the task and the environment defines fitness, what is the evolutionary equivalent of a loss function?",
        key_concepts = ["fitness", "select", "environment", "survival", "adapt"],
        level        = 3,
        description  = "Inference — loss function ≈ inverse fitness",
    ),
    # Level 4 — cross-domain
    Question(
        text         = "Is evolution a kind of learning? Make the case.",
        key_concepts = ["same", "analog", "equivalent", "both", "process"],
        level        = 4,
        description  = "Cross-domain synthesis — the core aha moment question",
    ),
    Question(
        text         = "What concept plays the same structural role in both topics we discussed?",
        key_concepts = ["environment", "pressure", "selection", "error", "signal"],
        level        = 4,
        description  = "Cross-domain primitive — tests whether environment was found",
    ),
]


# -------------------------------------------------------------------
# Evaluator
# -------------------------------------------------------------------

class Evaluator:
    """
    Runs questions against three context conditions and scores answers.
    Uses Claude claude-sonnet-4-20250514 as the answering model.
    """

    MODEL        = "claude-sonnet-4-20250514"
    TOKEN_BUDGET = 200
    ANSWER_TOKENS = 150

    def __init__(self, conversation: list[tuple[str, str]],
                 questions: list[Question],
                 label: str = "experiment"):
        self.conversation = conversation
        self.questions    = questions
        self.label        = label
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Set it with: export ANTHROPIC_API_KEY=your_key_here"
            )
        self.client  = anthropic.Anthropic(api_key=api_key)
        self.results = []

    def run(self):
        print(f"\n{'='*60}")
        print(f"EVALUATION: {self.label}")
        print(f"{'='*60}")
        print(f"Questions: {len(self.questions)}")
        print(f"Token budget per condition: {self.TOKEN_BUDGET}")

        graph, assembler, store = self._build_graph()

        for q in self.questions:
            print(f"\n--- L{q.level}: {q.description} ---")
            print(f"Q: {q.text}")

            result = {
                "question":    q.text,
                "level":       q.level,
                "description": q.description,
                "conditions":  {},
            }

            for mode in ["graph", "baseline", "graph_only"]:
                context = self._assemble_context(mode, assembler, store, graph, q.text)
                answer  = self._ask(context, q.text)
                score   = self._score(answer, q.key_concepts)

                result["conditions"][mode] = {
                    "context_tokens": len(context.split()),
                    "answer":         answer,
                    "score":          score,
                    "key_hits":       self._key_hits(answer, q.key_concepts),
                }

                indicator = "✓" if score >= 0.5 else "~" if score >= 0.25 else "✗"
                print(f"  {mode:12s} [{indicator}] score={score:.2f}  "
                      f"tokens={len(context.split())}  "
                      f"hits={self._key_hits(answer, q.key_concepts)}/{len(q.key_concepts)}")

            self.results.append(result)
            time.sleep(0.5)

        self._final_report()
        self._save()

    # -------------------------------------------------------------------
    # Graph building
    # -------------------------------------------------------------------

    def _build_graph(self) -> tuple:
        graph     = ConceptGraph()
        store     = BackingStore()
        assembler = ContextAssembler(graph=graph, store=store, tokenizer=None)

        for speaker, text in self.conversation:
            monitor = ConvergenceMonitor(max_passes=20, window=8)
            assembler.ingest(speaker, text)
            while monitor.should_continue():
                graph.begin_pass()
                self._propagation_pass(graph)
                status = graph.end_pass()
                monitor.record(status)

        print(f"\nGraph built: {len(graph.nodes)} nodes, "
              f"{len(graph.edges)} edges, "
              f"{len(graph.merge_events)} merges")
        return graph, assembler, store

    def _propagation_pass(self, graph: ConceptGraph):
        for node_id in list(graph.nodes.keys()):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            graph._update_maturity(node_id)
            if node.provisional:
                if node.maturity >= graph.maturity_threshold:
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
        elif mode == "baseline":
            return assembler.assemble_baseline(token_budget=self.TOKEN_BUDGET)
        elif mode == "graph_only":
            return self._assemble_graph_only(graph, query)
        return ""

    def _assemble_graph_only(self, graph: ConceptGraph, query: str) -> str:
        """
        Relational structure alone — no backing store.
        The core thesis test: if meaning is in the edges, this should work.
        """
        parts  = []

        stable = sorted(
            [n for n in graph.nodes.values() if not n.provisional],
            key=lambda n: -n.maturity
        )[:10]

        if stable:
            parts.append("Core concepts: " + ", ".join(
                f"{n.label}(maturity={n.maturity:.1f})"
                for n in stable[:6]
            ))

        top_ids  = {n.id for n in stable[:8]}
        key_edges = sorted(
            [e for e in graph.edges.values()
             if e.source in top_ids and e.target in top_ids and e.weight > 0.5],
            key=lambda e: -e.weight
        )

        if key_edges:
            rel_lines = []
            for edge in key_edges[:8]:
                src = graph.nodes.get(edge.source)
                tgt = graph.nodes.get(edge.target)
                if src and tgt:
                    rel_lines.append(
                        f"{src.label} -{edge.edge_type.value}-> {tgt.label}"
                        f"(w={edge.weight:.2f})"
                    )
            if rel_lines:
                parts.append("Key relations: " + "; ".join(rel_lines))

        if graph.merge_events:
            significant = sorted(graph.merge_events, key=lambda e: -e.magnitude)[:3]
            merge_lines = []
            for event in significant:
                node = graph.nodes.get(event.merged_into)
                if node:
                    merge_lines.append(
                        f"'{event.node_a}' unified with '{event.node_b}' "
                        f"(cascade={event.cascade_depth})"
                    )
            if merge_lines:
                parts.append("Insights: " + "; ".join(merge_lines))

        query_words = query.lower().split()
        relevant = sorted(
            [n for n in graph.nodes.values()
             if any(w in n.label for w in query_words if len(w) > 3)],
            key=lambda n: -n.maturity
        )
        if relevant:
            parts.append("Query-relevant: " + ", ".join(
                f"{n.label}(deg={len(graph._get_neighbors(n.id))})"
                for n in relevant[:6]
            ))

        return "\n".join(parts)[:self.TOKEN_BUDGET * 6]

    # -------------------------------------------------------------------
    # Model call
    # -------------------------------------------------------------------

    def _ask(self, context: str, question: str) -> str:
        prompt = (
            f"The following is context from a conversation:\n\n"
            f"{context}\n\n"
            f"Based only on this context, answer concisely:\n"
            f"{question}"
        )
        try:
            response = self.client.messages.create(
                model      = self.MODEL,
                max_tokens = self.ANSWER_TOKENS,
                messages   = [{"role": "user", "content": prompt}],
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
        answer_lower = answer.lower()
        return sum(1 for c in key_concepts if c.lower() in answer_lower)

    # -------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------

    def _final_report(self):
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY: {self.label}")
        print(f"{'='*60}")

        levels     = sorted(set(r["level"] for r in self.results))
        conditions = ["graph", "baseline", "graph_only"]

        print(f"\n{'Level':<8}", end="")
        for c in conditions:
            print(f"{c:>14}", end="")
        print()
        print("-" * 50)

        for level in levels:
            level_results = [r for r in self.results if r["level"] == level]
            print(f"L{level:<7}", end="")
            for c in conditions:
                scores = [r["conditions"][c]["score"]
                          for r in level_results if c in r["conditions"]]
                mean = sum(scores) / len(scores) if scores else 0
                ind  = "✓" if mean >= 0.5 else "~" if mean >= 0.25 else "✗"
                print(f"{ind} {mean:>10.2f}", end="")
            print()

        n = len(self.results)
        totals = {c: sum(r["conditions"][c]["score"] for r in self.results)
                  for c in conditions}

        print(f"\nMean scores across all questions:")
        for c in conditions:
            print(f"  {c:14s}: {totals[c]/n:.3f}")

        if totals["graph_only"] > totals["baseline"]:
            print(f"\n*** Graph-only BEATS baseline "
                  f"({totals['graph_only']/n:.3f} vs {totals['baseline']/n:.3f}) ***")
            print("    Meaning is in the relational structure.")
        elif totals["graph_only"] >= totals["baseline"] * 0.9:
            print(f"\n~ Graph-only matches baseline "
                  f"({totals['graph_only']/n:.3f} vs {totals['baseline']/n:.3f})")
        else:
            print(f"\n  Baseline leads "
                  f"({totals['baseline']/n:.3f} vs {totals['graph_only']/n:.3f})")

    def _save(self):
        os.makedirs("experiments", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = f"experiments/eval_{self.label}_{timestamp}.json"

        levels     = sorted(set(r["level"] for r in self.results))
        conditions = ["graph", "baseline", "graph_only"]

        by_level = {
            str(lv): {
                c: sum(r["conditions"][c]["score"]
                       for r in self.results if r["level"] == lv
                       and c in r["conditions"])
                   / max(1, sum(1 for r in self.results if r["level"] == lv))
                for c in conditions
            }
            for lv in levels
        }

        output = {
            "metadata": {
                "timestamp":    timestamp,
                "label":        self.label,
                "model":        self.MODEL,
                "token_budget": self.TOKEN_BUDGET,
                "questions":    len(self.questions),
            },
            "results": self.results,
            "summary": {"by_level": by_level},
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {path}")

        # Also write LaTeX table
        self._write_latex_table(by_level, path.replace(".json", "_table.tex"))

    def _write_latex_table(self, by_level: dict, path: str):
        """
        Writes Table 4 from the paper directly as a LaTeX snippet.
        Copy-paste into Section 5.4.
        """
        level_labels = {
            "1": "L1 (direct retrieval)",
            "2": "L2 (synthesis)",
            "3": "L3 (inference)",
            "4": "L4 (cross-domain)",
        }
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            rf"\caption{{QA evaluation scores by level and context condition — {self.label}}}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Level & Graph + store & Baseline & Graph only \\",
            r"\midrule",
        ]
        for lv, label in level_labels.items():
            if lv in by_level:
                g  = by_level[lv].get("graph", 0)
                b  = by_level[lv].get("baseline", 0)
                go = by_level[lv].get("graph_only", 0)
                lines.append(rf"{label} & {g:.2f} & {b:.2f} & {go:.2f} \\")
            else:
                lines.append(rf"{label} & --- & --- & --- \\")

        # Overall row
        all_levels = list(by_level.values())
        if all_levels:
            og  = sum(d.get("graph", 0)      for d in all_levels) / len(all_levels)
            ob  = sum(d.get("baseline", 0)   for d in all_levels) / len(all_levels)
            ogo = sum(d.get("graph_only", 0) for d in all_levels) / len(all_levels)
            lines += [
                r"\midrule",
                rf"Overall & {og:.2f} & {ob:.2f} & {ogo:.2f} \\",
            ]

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"LaTeX table written to {path}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    from experiment import CONVERSATION, CONVERSATION_CROSS_DOMAIN

    eval_memory = Evaluator(
        conversation = CONVERSATION,
        questions    = MEMORY_QUESTIONS,
        label        = "memory",
    )
    eval_memory.run()

    print("\n" + "=" * 60 + "\n")

    eval_cross = Evaluator(
        conversation = CONVERSATION_CROSS_DOMAIN,
        questions    = CROSS_DOMAIN_QUESTIONS,
        label        = "cross_domain",
    )
    eval_cross.run()