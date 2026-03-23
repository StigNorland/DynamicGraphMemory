"""
debug_field.py  —  drop this into your project and import where needed.

Usage:
    from debug_field import timed, field_debug_ingest

    # Wrap your existing ingest call:
    field_debug_ingest(assembler, speaker, text, turn_num)
"""

import time
import functools
from contextlib import contextmanager


# ── Simple timer context manager ──────────────────────────────────────────────

@contextmanager
def timed(label: str, warn_threshold_s: float = 1.0):
    """
    Context manager that prints elapsed time for any block.
    Prints a WARNING if it exceeds warn_threshold_s.

    Usage:
        with timed("spacy_facts"):
            facts = _spacy_facts(text)
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    flag = "  ⚠ SLOW" if elapsed > warn_threshold_s else ""
    print(f"    [{label}] {elapsed:.3f}s{flag}")


# ── Turn-level ingest wrapper ─────────────────────────────────────────────────

def field_debug_ingest(assembler, speaker: str, text: str, turn_num: int):
    """
    Drop-in replacement for assembler.ingest(speaker, text).
    Prints per-turn timing and graph stats after each ingest.

    Replace:
        assembler.ingest(speaker, text)
    With:
        field_debug_ingest(assembler, speaker, text, i)
    """
    g = assembler.graph

    nodes_before = g.number_of_nodes()
    edges_before = g.number_of_edges()
    t0 = time.perf_counter()

    assembler.ingest(speaker, text)

    elapsed = time.perf_counter() - t0
    nodes_after = g.number_of_nodes()
    edges_after = g.number_of_edges()

    flag = "  ⚠ SLOW" if elapsed > 2.0 else ""
    print(
        f"  turn {turn_num:3d} [{speaker[:4]}]  "
        f"{elapsed:6.2f}s{flag}  |  "
        f"nodes {nodes_before}→{nodes_after} (+{nodes_after - nodes_before})  "
        f"edges {edges_before}→{edges_after} (+{edges_after - edges_before})"
    )


# ── Function-level timing decorator ──────────────────────────────────────────

def timeit(warn_threshold_s: float = 1.0):
    """
    Decorator that prints timing for any function call.

    Usage:
        @timeit(warn_threshold_s=0.5)
        def _spacy_facts(text):
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            flag = "  ⚠ SLOW" if elapsed > warn_threshold_s else ""
            print(f"    [{fn.__name__}] {elapsed:.3f}s{flag}")
            return result
        return wrapper
    return decorator


# ── Graph build summary ───────────────────────────────────────────────────────

def build_summary(assembler, total_elapsed: float):
    """Print a summary after the full graph build."""
    g = assembler.graph
    print()
    print("─" * 60)
    print(f"  Graph build complete in {total_elapsed:.1f}s")
    print(f"  Nodes : {g.number_of_nodes()}")
    print(f"  Edges : {g.number_of_edges()}")
    try:
        merges = getattr(assembler, 'merge_events', [])
        print(f"  Merges: {len(merges)}")
    except Exception:
        pass
    try:
        payloads = sum(
            1 for _, d in g.nodes(data=True) if d.get('payload')
        )
        print(f"  Nodes with payload: {payloads}")
    except Exception:
        pass
    print("─" * 60)
    print()