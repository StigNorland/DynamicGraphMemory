"""
convergence.py — Loop Detection and Convergence Monitoring

Standalone module for monitoring graph propagation convergence.
Separated from graph.py so it can be tested independently
and reasoned about clearly.

Three conditions we monitor:
  1. Convergence    — no writes, graph stable, stop iterating
  2. Fixed point    — writes happening but state not changing
  3. Limit cycle    — state oscillating between known configurations

Limit cycles are informative, not just failures.
They signal genuine tension in the relational structure.

Authors: Stig [last name], Claude (Anthropic)
"""

from dataclasses import dataclass, field
from collections import Counter


# -------------------------------------------------------------------
# Status
# -------------------------------------------------------------------

@dataclass
class PassStatus:
    """Result of a single propagation pass."""
    pass_number:   int
    writes:        int
    state_hash:    str
    converged:     bool
    fixed_point:   bool   # writes > 0 but hash unchanged
    loop_detected: bool   # hash seen before in window
    tension:       bool   # loop detected AND writes > 0
    message:       str    = ""


@dataclass
class ConvergenceReport:
    """Full convergence history for a reasoning session."""
    passes:          list[PassStatus] = field(default_factory=list)
    total_writes:    int              = 0
    converged_at:    int              = -1   # pass number, -1 if not converged
    loop_at:         int              = -1   # pass number of first loop
    tension_nodes:   list[str]        = field(default_factory=list)
    terminated_by:   str              = ""   # "convergence" / "loop" / "limit"

    @property
    def converged(self) -> bool:
        return self.converged_at >= 0

    @property
    def has_tension(self) -> bool:
        return any(p.tension for p in self.passes)

    def summary(self) -> dict:
        return {
            "total_passes":    len(self.passes),
            "total_writes":    self.total_writes,
            "converged":       self.converged,
            "converged_at":    self.converged_at,
            "loop_detected":   self.loop_at >= 0,
            "loop_at":         self.loop_at,
            "has_tension":     self.has_tension,
            "terminated_by":   self.terminated_by,
            "pass_writes":     [p.writes for p in self.passes],
        }


# -------------------------------------------------------------------
# Monitor
# -------------------------------------------------------------------

class ConvergenceMonitor:
    """
    Monitors graph propagation passes and detects convergence,
    fixed points, and limit cycles.

    Usage:
        monitor = ConvergenceMonitor(max_passes=20, window=8)
        while monitor.should_continue():
            graph.begin_pass()
            # ... propagation ...
            status = graph.end_pass()
            monitor.record(status)
        report = monitor.report()
    """

    def __init__(self, max_passes: int = 20, window: int = 8):
        self.max_passes  = max_passes
        self.window      = window        # hash history window size
        self._passes:    list[PassStatus]  = []
        self._hashes:    list[str]         = []
        self._done:      bool              = False
        self._reason:    str               = ""

    def record(self, pass_result: dict) -> PassStatus:
        """
        Record the result of one propagation pass.
        pass_result is the dict returned by graph.end_pass().
        """
        pass_number = len(self._passes)
        h           = pass_result["state_hash"]
        writes      = pass_result["writes"]

        # Detect conditions
        converged    = pass_result["converged"]           # no writes
        loop         = h in self._hashes                  # hash seen before
        fixed_point  = (writes > 0) and loop              # writing but stuck
        tension      = pass_result.get("tension", False)  # loop + writes

        # Build message
        if converged:
            message = "Converged — graph stable."
        elif fixed_point:
            message = "Fixed point — writes occurring but state unchanged."
        elif tension:
            message = ("Tension detected — limit cycle with active writes. "
                       "Genuine relational conflict in graph.")
        elif loop:
            message = "Limit cycle — state oscillating."
        else:
            message = f"Pass {pass_number} — {writes} writes."

        status = PassStatus(
            pass_number   = pass_number,
            writes        = writes,
            state_hash    = h,
            converged     = converged,
            fixed_point   = fixed_point,
            loop_detected = loop,
            tension       = tension,
            message       = message,
        )
        self._passes.append(status)

        # Update rolling hash window
        self._hashes.append(h)
        if len(self._hashes) > self.window:
            self._hashes.pop(0)

        # Determine if we should stop
        if converged:
            self._done   = True
            self._reason = "convergence"
        elif tension:
            # Don't stop immediately on tension —
            # give it a few more passes to resolve
            # Stop if tension persists for 3 consecutive passes
            recent_tension = sum(
                1 for p in self._passes[-3:] if p.tension
            )
            if recent_tension >= 3:
                self._done   = True
                self._reason = "tension"
        elif loop and not tension:
            self._done   = True
            self._reason = "loop"
        elif pass_number >= self.max_passes - 1:
            self._done   = True
            self._reason = "limit"

        return status

    def should_continue(self) -> bool:
        """Main loop condition."""
        return not self._done

    def report(self) -> ConvergenceReport:
        """Full convergence report for logging and paper figures."""
        converged_at = -1
        loop_at      = -1

        for p in self._passes:
            if p.converged and converged_at < 0:
                converged_at = p.pass_number
            if p.loop_detected and loop_at < 0:
                loop_at = p.pass_number

        return ConvergenceReport(
            passes        = self._passes,
            total_writes  = sum(p.writes for p in self._passes),
            converged_at  = converged_at,
            loop_at       = loop_at,
            terminated_by = self._reason,
        )

    def reset(self):
        """Reset for a new reasoning session."""
        self._passes  = []
        self._hashes  = []
        self._done    = False
        self._reason  = ""


# -------------------------------------------------------------------
# Tension analysis
# -------------------------------------------------------------------

def analyze_tension(report: ConvergenceReport,
                    graph) -> dict:
    """
    When a limit cycle with tension is detected, analyze which
    nodes are involved in the conflict.

    Tension means the graph has a genuine relational contradiction
    it cannot resolve internally. This is worth surfacing —
    it may indicate:
      - A concept that needs to be split into two distinct nodes
      - A causal loop that reflects real-world circularity
      - A missing node that would resolve the tension

    Returns analysis dict for logging and inspection.
    """
    if not report.has_tension:
        return {"tension": False}

    # Find passes where tension occurred
    tension_passes = [p for p in report.passes if p.tension]

    # Hash frequency in tension window — most repeated = stuck nodes
    hash_counts = Counter(p.state_hash for p in tension_passes)
    most_common = hash_counts.most_common(3)

    # Identify low-maturity provisional nodes —
    # often the source of unresolved tension
    provisional = [
        {"id": n.id, "label": n.label, "maturity": n.maturity}
        for n in graph.nodes.values()
        if n.provisional
    ]
    provisional.sort(key=lambda x: x["maturity"])

    return {
        "tension":              True,
        "tension_passes":       len(tension_passes),
        "total_passes":         len(report.passes),
        "most_repeated_states": most_common,
        "provisional_nodes":    provisional[:5],  # most likely culprits
        "interpretation": (
            "Graph contains relational conflict that cannot resolve "
            "internally. Candidate causes listed in provisional_nodes. "
            "Consider: concept split, missing mediating node, or "
            "genuine circular causality in the domain."
        ),
    }