# Design Notes: Relational Graph Memory Architecture

*Working document — decisions, rationale, and open questions*
*Authors: Stig [last name], Claude (Anthropic)*

---

## Core Thesis

Meaning is relational structure, not symbol properties. A concept 
is defined entirely by its relationships to other concepts — 
there is no intrinsic meaning, only relational position. This has 
a direct architectural consequence: the right substrate for language 
model memory is a relational graph, not a token sequence.

Current KV cache architecture re-reads the full conversation on 
every forward pass. It is expensive, uncompressed, and operates 
at the wrong granularity. This project replaces it with a concept 
graph as first-class working memory, with the raw conversation 
retained as a lazy-loaded backing store indexed by graph pointers.

---

## Graph Schema

### Node Types

Three node types, coexisting in a single unified graph. Type is 
a node attribute, not a structural partition.

**Concept nodes**
The primary semantic units. Language-agnostic — not words, but 
the things words point at. "trust", "causality", "recursion" are 
concept nodes. The same concept expressed in Norwegian and English 
maps to the same node.

**Temporal nodes**
Sequence anchors. Encode when something occurred in the 
conversation. Provide the time dimension that allows the graph 
to represent narrative and progression without flattening 
everything into an unordered semantic cloud.

**Causal nodes**
Directed relationship carriers. Not just "A relates to B" but 
"A produces B", "A prevents B", "A is precondition for B". 
Causality is explicit structure, not inferred from proximity.

### Edge Types

Edges cross node type boundaries freely. A concept can have 
temporal position. A causal relationship can be anchored in 
time. A temporal sequence can carry causal weight.

Edge attributes:
- **weight** — strength of relationship, updated by traversal
- **type** — semantic / temporal / causal
- **pointer** — index back to raw conversation position where 
  this edge was first established
- **maturity_at_creation** — maturity of both nodes when edge 
  was created, useful for later analysis

---

## Concept Identity: Reduction to Primitive

### The Problem

When new input arrives, what determines whether it maps to an 
existing concept node or creates a new one? String matching is 
too brittle. Embedding similarity requires tuning an arbitrary 
threshold. Neither captures what we actually want: structural 
identity.

### The Solution: Relational Signature

A concept's identity is its **primitive** — its most irreducible 
relational signature. Not the word that expresses it, not its 
vector embedding, but the pattern of relationships that defines 
it. Two concepts with the same relational signature are the same 
node, regardless of the words used to express them.

This is where language-agnosticism emerges naturally — not as 
a design choice but as a consequence of structural identity.

### Maturity

Every node carries a maturity score:
```python
maturity = (edge_count * mean_edge_weight) * neighborhood_maturity_factor
```

- **edge_count** — number of established relationships
- **mean_edge_weight** — average strength of those relationships  
- **neighborhood_maturity_factor** — whether connected nodes 
  are themselves mature

Maturity is local and computable without global graph traversal.

New nodes enter as **provisional** — low maturity, weak identity, 
tentative relational signature. As the graph grows around them, 
maturity increases. At a threshold (to be determined empirically), 
identity stabilizes and the provisional flag drops.

This solves the bootstrapping problem: early in a conversation 
the graph is sparse, so concepts enter provisionally and stabilize 
as context accumulates. No external lookup required. Identity 
emerges from the same process that builds meaning.

---

## Write Policy

### The Core Principle

**The update operation is the novelty signal.**

If incoming input modifies the graph — creates a node, adds an 
edge, updates a weight — it is new in the current context by 
definition. No separate novelty detector required. The write 
operation itself is the signal.

### Write Cases

1. **New concept, no match** → create provisional node, 
   establish edges to active context nodes, set maturity = 0

2. **Existing concept, exact primitive match** → strengthen 
   existing edge weights, update temporal pointer, 
   no new structure created

3. **Existing concept, partial match** → create delta — 
   new edge or modified weight capturing the difference. 
   The delta *is* the meaning of the difference.

4. **Provisional node reaches maturity threshold** → identity 
   stabilizes, check for merge candidates

### Hebbian Reinforcement

Repeated traversal of relational paths strengthens edge weights:
```python
edge.weight += learning_rate * activation_strength
```

Recurring themes compress naturally — the graph doesn't accumulate 
copies of the same structure, it deepens existing structure. 
Episodic detail in raw conversation is preserved via pointers; 
the graph holds the schema.

---

## Merge Events: Aha Moments

### What a Merge Is

When two provisionally separate nodes mature and their relational 
signatures converge to the same primitive, they merge. This is 
not a cleanup operation — it is a **semantic event**.

Two concepts that developed independently, accumulated their own 
edge networks, matured separately — the graph discovers they are 
the same structure viewed from different angles.

### Why It Matters

When two mature nodes merge, their full edge networks unify. 
Every concept connected to A is now connected to B's neighborhood 
and vice versa. This is a cascade — new relationships become 
visible that weren't visible before, not because new information 
arrived but because existing information reorganized.

This is the computational correlate of insight. No new data — 
sudden new meaning. The magnitude of the insight is the depth 
of the cascade.

### MergeEvent as First-Class Structure

Merges are stored as causal nodes in the graph itself:
```python
class MergeEvent:
    node_a            # first node
    node_b            # second node
    timestamp         # conversation position pointer
    cascade_depth     # how many new relationships unlocked
    magnitude         # weighted measure of insight
    new_edges         # list of relationships that became 
                      # visible post-merge
```

The insight becomes part of the relational structure, available 
for future reasoning. A graph that records its own aha moments 
can reason about its own understanding.

---

## Dynamic Looping and Convergence

### The Problem with Static Unrolling

Current transformers are statically unrolled — fixed number of 
layers, each applied once. Reasoning depth is baked into 
architecture size, not responsive to problem complexity.

### Graph Propagation as Dynamic Reasoning

Queries propagate through the graph, activating related nodes, 
updating edge weights, potentially triggering merges. This 
iterates until convergence. Reasoning depth becomes a function 
of problem complexity, not model size.

### Convergence Criterion

**No graph updates = convergence signal.**

If a propagation pass produces no writes — no new nodes, no edge 
weight changes, no merges — the relevant subgraph has stabilized. 
This follows directly from the write-as-novelty principle.

### Loop Detection

Two failure modes:

**Fixed point** — same state, same output, no progress.  
**Limit cycle** — A→B→A→B, oscillating without settling.

Both are caught by a rolling hash of recent graph states:
```python
state_hash = hash(frozen_graph_signature)
if state_hash in recent_hashes:
    flag_as_loop()
```

Critically — a detected limit cycle is **informative**, not just 
a failure. It means the query has exposed a genuine tension in 
the relational structure that cannot be resolved internally. 
This is worth surfacing explicitly rather than just terminating.

### Self-Reference Without Infinite Regress

A self-referencing node is an edge that points back to itself — 
explicit structure, not recursive evaluation. Loops of arbitrary 
length are representable because the structure is explicit rather 
than re-evaluated each traversal. The loop detection mechanism 
handles pathological cases without eliminating legitimate 
self-reference.

---

## Lazy Backing Store

Raw conversation is retained in full. Every graph edge carries 
a pointer to the conversation position where it was established. 
Full fidelity is available on demand — dereference the pointer 
when precision matters.

Most queries resolve at graph level without touching the backing 
store. Inference cost scales with relevant subgraph size, not 
full conversation length. Very long conversations become tractable 
because the graph grows sublinearly — new input mostly reinforces 
or modifies existing structure rather than adding entirely new nodes.

---

## Open Questions

These are genuine unknowns — the POC exists to discover them.

1. **Maturity threshold** — what value triggers identity 
   stabilization? Probably determined empirically.

2. **Primitive resolution** — how deep does reduction go before 
   bottoming out? Is there a universal primitive set or is it 
   corpus-dependent?

3. **Edge type interaction** — how do semantic, temporal, and 
   causal edges interact during propagation? Do they have 
   different learning rates?

4. **Merge threshold** — how similar do two relational signatures 
   need to be to trigger a merge? Exact match is too strict, 
   fuzzy match needs calibration.

5. **Cascade depth limit** — should merge cascades be bounded 
   to prevent runaway reorganization? Or is unbounded cascade 
   the point?

6. **Provisonal node lifetime** — how long does a node stay 
   provisional before being either stabilized or pruned?

---

## What the POC Will Test

Single falsifiable claim:

**Graph-compressed context preserves sufficient meaning for 
coherent language model generation at a fraction of the token 
budget required by raw conversation context.**

Setup: nanoGPT + pretrained GPT-2 weights. Two inference modes — 
standard KV context vs graph-derived context, identical token 
budget. Synthetic conversation dataset with known relational 
structure, designed so ground truth is verifiable by inspection.

The interesting result is not just whether it works — it is 
what the graph looks like when it does, and what the merge 
events reveal about the relational structure of the test 
conversations.

---

*This document is a living design record. Decisions will be 
updated as the POC reveals surprises — and it will.*
