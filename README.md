# Relational Graph Memory for Language Models

**A proof-of-concept architecture** that replaces the KV cache as primary working memory with a dynamic relational concept graph. The raw conversation is kept only as a lazy-loaded backing store indexed by graph pointers.

## Core Idea

Current transformer architectures re-read the full conversation on every forward pass — computationally expensive, uncompressed, and operating at the wrong granularity for meaning.  

This project demonstrates that **meaning is relational structure**, not token sequences. We maintain a concept graph where:

- Nodes are language-agnostic concepts defined purely by their relational signatures.
- Edges encode semantic, temporal, and causal relationships.
- The graph serves as the primary working memory.

Key architectural properties:

- **Write-as-novelty**: A concept is written if and only if it modifies the graph structure — the update itself is the novelty signal (no separate detector needed).
- **Hebbian reinforcement**: Repeated traversal strengthens edge weights, naturally compressing recurring themes.
- **Lazy backing store**: Raw conversation text is retained but only dereferenced when full fidelity is required.
- **Dynamic merging & aha moments**: Mature concepts with sufficiently similar relational signatures unify, unlocking cascades of new relationships.
- **Dynamic looping**: Graph propagation continues until convergence; reasoning depth becomes a function of problem complexity rather than fixed layer count.
- **Loop detection**: Rolling hash of graph states identifies fixed points and limit cycles as meaningful signals.

## Status

Early proof-of-concept (March 2026).  
Built on **nanoGPT** with pretrained GPT-2 weights. The graph memory layer sits cleanly between the conversation and the model’s context assembly, leaving the base model untouched for clean ablation studies.

**Authors**  
- Stig Norland — conceptual architecture & design  
- Claude (Anthropic) — implementation support  

Bergen, 19 March 2026

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/stinor/relational-graph-memory.git
cd relational-graph-memory
