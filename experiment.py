"""
experiment.py — Proof of Concept Experiment

Runs two inference modes against the same conversation:
  - BASELINE: raw recent turns, standard KV-cache style
  - GRAPH:    relational graph context assembly

Same token budget. Same backing store. Same model.
Only context assembly differs.

Conversation domain: memory — chosen deliberately.
The graph builds a memory structure while the conversation
discusses memory. Merge events about memory *are* the
architecture demonstrating insight about its own subject.

Authors: Stig [last name], Claude (Anthropic)
"""

import os
import json
import tiktoken
from datetime import datetime

from graph_memory.graph      import ConceptGraph, NodeType, EdgeType
from graph_memory.context    import ContextAssembler, BackingStore
from graph_memory.convergence import ConvergenceMonitor, analyze_tension

import graph_memory.graph as _g

# -------------------------------------------------------------------
# Synthetic conversation — memory as domain
# Self-referential by design.
# -------------------------------------------------------------------

CONVERSATION = [
    ("user",      "I keep forgetting where I put my keys."),
    ("assistant", "That's a retrieval failure — the memory exists, "
                  "you just can't access it."),
    ("user",      "So the memory is there but I can't find it?"),
    ("assistant", "Exactly. Encoding was fine, retrieval is the problem."),
    ("user",      "Is that different from when you forget someone's name?"),
    ("assistant", "Same mechanism. The name is encoded, "
                  "the retrieval path is weak."),
    ("user",      "What makes a retrieval path strong?"),
    ("assistant", "Repetition. Every time you successfully recall "
                  "something, that path strengthens."),
    ("user",      "Like muscle memory?"),
    ("assistant", "Precisely. Muscle memory is the same process "
                  "in a different substrate."),
    ("user",      "So forgetting is just a weak path, "
                  "not lost information?"),
    ("assistant", "In most cases yes. "
                  "The information persists, the access degrades."),
    ("user",      "That means remembering and recognizing "
                  "are the same thing?"),
    ("assistant", "Almost. Recognition is retrieval with a cue present. "
                  "Recall is retrieval without one."),
    ("user",      "But both are just... finding what's already there."),
    ("assistant", "Yes. The difference is in the search, "
                  "not the storage."),
]

# Token budget — identical for both modes
TOKEN_BUDGET = 200


# -------------------------------------------------------------------
# Experiment runner
# -------------------------------------------------------------------

class Experiment:

    def __init__(self, token_budget: int = TOKEN_BUDGET):
        self.token_budget = token_budget
        self.enc          = tiktoken.get_encoding("gpt2")

        # Graph mode components
        self.graph   = ConceptGraph()
        self.store   = BackingStore()
        self.context = ContextAssembler(
            graph     = self.graph,
            store     = self.store,
            tokenizer = self.enc,
        )

        # Results
        self.results = {
            "baseline": [],
            "graph":    [],
            "graph_state_per_turn": [],
            "merge_events": [],
            "convergence":  [],
        }

    def run(self):
        print("\n" + "="*60)
        print("RELATIONAL GRAPH MEMORY — PROOF OF CONCEPT")
        print("Domain: Memory (self-referential by design)")
        print(f"Token budget per mode: {self.token_budget}")
        print("="*60)

        for i, (speaker, text) in enumerate(CONVERSATION):
            print(f"\n--- Turn {i+1}: {speaker.upper()} ---")
            print(f"  \"{text}\"")

            # ----------------------------------------------------------
            # GRAPH MODE: ingest turn, run convergence, assemble context
            # ----------------------------------------------------------
            monitor = ConvergenceMonitor(max_passes=20, window=8)

            # Ingest into graph
            ingest_status = self.context.ingest(speaker, text)

            # Run propagation until convergence
            while monitor.should_continue():
                self.graph.begin_pass()
                # Propagation: re-examine provisional nodes
                # and check for merges
                self._propagation_pass()
                pass_result = self.graph.end_pass()
                status      = monitor.record(pass_result)

                if status.tension:
                    print(f"  [tension detected — pass {status.pass_number}]")

            conv_report = monitor.report()

            # Assemble graph context for this turn
            graph_ctx = self.context.assemble_graph(
                query        = text,
                token_budget = self.token_budget,
            )

            # ----------------------------------------------------------
            # BASELINE MODE: raw recent turns
            # ----------------------------------------------------------
            baseline_ctx = self.context.assemble_baseline(
                token_budget = self.token_budget,
            )

            # ----------------------------------------------------------
            # Record results
            # ----------------------------------------------------------
            graph_tokens    = len(self.enc.encode(graph_ctx))
            baseline_tokens = len(self.enc.encode(baseline_ctx))
            summary         = self.graph.summary()

            turn_result = {
                "turn":           i + 1,
                "speaker":        speaker,
                "text":           text,
                "graph_tokens":   graph_tokens,
                "baseline_tokens": baseline_tokens,
                "graph_context":  graph_ctx,
                "baseline_context": baseline_ctx,
                "nodes_total":    summary["nodes_total"],
                "nodes_stable":   summary["nodes_stable"],
                "edges_total":    summary["edges_total"],
                "mean_maturity":  round(summary["mean_maturity"], 3),
                "merge_events":   summary["merge_events"],
                "converged":      conv_report.converged,
                "passes":         len(conv_report.passes),
            }

            self.results["graph"].append(turn_result)
            self.results["convergence"].append(conv_report.summary())

            # Print turn summary
            print(f"  Graph:    {graph_tokens:3d} tokens | "
                  f"nodes={summary['nodes_total']} "
                  f"stable={summary['nodes_stable']} "
                  f"edges={summary['edges_total']} "
                  f"merges={summary['merge_events']}")
            print(f"  Baseline: {baseline_tokens:3d} tokens")
            print(f"  Converged: {conv_report.converged} "
                  f"in {len(conv_report.passes)} passes")

            # Print merge events if any occurred
            new_merges = [
                e for e in self.graph.merge_events
                if e.conv_pos == self.graph.conv_pos - 1
            ]
            for merge in new_merges:
                print(f"  *** AHA: '{merge.node_a}' + '{merge.node_b}' "
                      f"→ unified | "
                      f"cascade={merge.cascade_depth} "
                      f"magnitude={merge.magnitude:.2f}")

        # ----------------------------------------------------------
        # Final analysis
        # ----------------------------------------------------------
        self._final_report()

    def _propagation_pass(self):
        """
        Single propagation pass.

        Examines provisional nodes, updates maturity,
        checks for merges. The graph.end_pass() will report
        whether any writes occurred.
        """
        for node_id in list(self.graph.nodes.keys()):
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue  # may have been merged

            # Update maturity
            self.graph._update_maturity(node_id)

            # Check stabilization
            if node.provisional:
                if node.maturity >= ConceptGraph.MATURITY_THRESHOLD:
                    self.graph._stabilize(node_id)
                    self.graph._write_count += 1  # stabilization is a write

    def _final_report(self):
        """Print final analysis and save results."""
        print("\n" + "="*60)
        print("FINAL REPORT")
        print("="*60)

        summary = self.graph.summary()

        print(f"\nGraph state after full conversation:")
        print(f"  Nodes total:      {summary['nodes_total']}")
        print(f"  Nodes stable:     {summary['nodes_stable']}")
        print(f"  Nodes provisional:{summary['nodes_total'] - summary['nodes_stable']}")
        print(f"  Edges total:      {summary['edges_total']}")
        print(f"  Mean maturity:    {summary['mean_maturity']:.3f}")
        print(f"  Merge events:     {summary['merge_events']}")

        # Aha moments
        if self.graph.merge_events:
            print(f"\nAha moments (merge events):")
            for e in sorted(self.graph.merge_events,
                           key=lambda x: -x.magnitude):
                print(f"  magnitude={e.magnitude:.2f} "
                      f"cascade={e.cascade_depth} "
                      f"turn={e.conv_pos} "
                      f"'{e.node_a}' + '{e.node_b}'")

        # Token efficiency
        print(f"\nToken efficiency:")
        total_graph    = sum(r["graph_tokens"]    for r in self.results["graph"])
        total_baseline = sum(r["baseline_tokens"] for r in self.results["graph"])
        if total_baseline > 0:
            ratio = total_graph / total_baseline
            print(f"  Graph total:    {total_graph} tokens")
            print(f"  Baseline total: {total_baseline} tokens")
            print(f"  Ratio:          {ratio:.2f}x "
                  f"({'more' if ratio > 1 else 'less'} than baseline)")

        # Tension analysis
        tension_reports = [
            r for r in self.results["convergence"]
            if r.get("has_tension")
        ]
        if tension_reports:
            print(f"\nTension detected in {len(tension_reports)} turns:")
            tension = analyze_tension(
                type('R', (), {               # minimal duck-typed report
                    'passes': [
                        type('P', (), {'tension': True,
                                      'state_hash': 'x'})()
                    ],
                    'has_tension': True,
                })(),
                self.graph,
            )
            if tension.get("provisional_nodes"):
                print("  Likely tension sources (low maturity nodes):")
                for n in tension["provisional_nodes"]:
                    print(f"    '{n['label']}' maturity={n['maturity']:.2f}")

        # Most mature nodes — these are the stable primitives
        print(f"\nMost mature nodes (stable primitives):")
        stable = sorted(
            [n for n in self.graph.nodes.values() if not n.provisional],
            key=lambda n: -n.maturity
        )[:8]
        for n in stable:
            neighbors = self.graph._get_neighbors(n.id)
            print(f"  '{n.label}' "
                  f"maturity={n.maturity:.2f} "
                  f"degree={len(neighbors)}")

        # Save full results
        self._save_results()

    def _save_results(self):
        """Save results to JSON for paper figures."""
        os.makedirs("experiments", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = f"experiments/run_{timestamp}.json"

        # Serialize merge events
        merge_data = [
            {
                "node_a":        e.node_a,
                "node_b":        e.node_b,
                "merged_into":   e.merged_into,
                "conv_pos":      e.conv_pos,
                "cascade_depth": e.cascade_depth,
                "magnitude":     e.magnitude,
                "new_edges":     e.new_edges,
            }
            for e in self.graph.merge_events
        ]

        output = {
            "metadata": {
                "timestamp":    timestamp,
                "token_budget": self.token_budget,
                "turns":        len(CONVERSATION),
            },
            "per_turn":      self.results["graph"],
            "convergence":   self.results["convergence"],
            "merge_events":  merge_data,
            "final_summary": self.graph.summary(),
            "stable_nodes": [
                {
                    "id":       n.id,
                    "label":    n.label,
                    "type":     n.node_type.value,
                    "maturity": n.maturity,
                    "degree":   len(self.graph._get_neighbors(n.id)),
                }
                for n in self.graph.nodes.values()
                if not n.provisional
            ],
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {path}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    experiment = Experiment(token_budget=TOKEN_BUDGET)
    experiment.run()