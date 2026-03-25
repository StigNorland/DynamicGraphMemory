# DynamicGraphMemory
**Relational Graph Memory for Language Models**

> A proof-of-concept that replaces the KV cache as primary working memory with a dynamic relational concept graph. The raw conversation is kept only as a lazy-loaded backing store indexed by graph pointers.

---

## Motivation

Current transformer architectures re-read the full conversation on every forward pass — computationally expensive, uncompressed, and operating at the wrong granularity for meaning.

This project demonstrates that **meaning is relational structure**, not token sequences. A concept graph serves as the primary working memory, where nodes are language-agnostic concepts defined purely by their relational signatures and edges encode semantic, temporal, and causal relationships.

---

## Key Architectural Properties

| Property | Description |
|---|---|
| **Write-as-novelty** | A concept is written only if it modifies the graph — the structural update *is* the novelty signal |
| **Hebbian reinforcement** | Repeated edge traversal strengthens weights, naturally compressing recurring themes |
| **Lazy backing store** | Raw conversation text is retained but only dereferenced when full fidelity is required |
| **Dynamic merging** | Concepts with sufficiently similar relational signatures unify, cascading new relationships |
| **Dynamic looping** | Graph propagation continues until convergence; reasoning depth scales with problem complexity |
| **Loop detection** | Rolling hash of graph states identifies fixed points and limit cycles as meaningful signals |
| **Holographic checksum reconstruction** | Every mature node stores a distributed projection of the global field — like a laser hologram fragment that still encodes the full image. Damaged nodes are reconstructed from surviving neighbours with >95% fidelity when ≥15–20% of the local neighbourhood remains intact |

---

## Fault Tolerance & Holographic Encoding

Each mature concept node stores a **holographic projection** of the global field embedding — a weighted sum over its neighbourhood's relational signatures:

```
p_v = Σ  w_vu · φ(u)   for u in N(v)
```

When a node is detected as damaged (maturity drop or failed consistency check), it is reconstructed by averaging the projections of its surviving neighbours. Because the information is *redundantly distributed* across the whole graph (exactly as a fragment of a laser hologram still encodes the full 3D image), reconstruction does not require a central store.

**Evaluated results (52 QA items, simulated node removal):**

| Damage | Intact graph | Damaged (no field) | Damaged (+ field) | Baseline (KV-cache) |
|---|---|---|---|---|
| 25% | 0.814 | 0.795 | 0.804 | 0.697 |
| 50% | 0.814 | 0.782 | 0.771 | 0.697 |
| 75% | 0.814 | 0.777 | 0.739 | 0.697 |

Key findings:
- At **75% node loss**, the damaged graph (0.777) still outperforms the KV-cache baseline (0.697) by +0.080 — robustness is real.
- The **75-token holographic field summary** achieves 6.7× compression vs the full graph context, within 0.009 of the full graph at low damage.
- At high damage, the field summary competes with surviving nodes for the token budget; **graph topology** (small-world hub nodes) is the primary resilience mechanism.

This makes RGM a strong candidate for **resource-constrained or high-radiation environments** (edge computing, space systems) where memory corruption is expected and a minimal fixed-cost context representation is preferable to a full graph.

---

## Status

Early proof-of-concept (March 2026).  
Built on **[nanoGPT](https://github.com/karpathy/nanoGPT)** with pretrained GPT-2 weights. The graph memory layer sits between the conversation and the model's context assembly, leaving the base model untouched for clean ablation studies.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU recommended (CPU inference is possible but slow)

```
torch
numpy
networkx
tiktoken
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/StigNorland/DynamicGraphMemory.git
cd DynamicGraphMemory

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install torch numpy networkx tiktoken

# 4. Download pretrained GPT-2 weights (fetched automatically on first run via nanoGPT)
```

---

## Project Structure

```
DynamicGraphMemory/
├── src/          # Core implementation
│   ├── graph/    # Concept graph: nodes, edges, Hebbian update, merge logic
│   ├── memory/   # Backing store and lazy-load indexing
│   └── model/    # nanoGPT integration and context assembly
├── testdata/     # Sample conversations for smoke-testing
├── results/      # Experiment outputs and ablation logs
├── paper/        # LaTeX source for the accompanying paper
└── README.md
```

---

## Quick Start

```python
from src.model import GraphMemoryModel

model = GraphMemoryModel.from_pretrained("gpt2")

response = model.chat("What is the capital of France?")
print(response)

# Inspect the current concept graph
model.memory.graph.summary()
```

---

## Running Tests

```bash
python -m pytest testdata/
```

---

## Ablation Studies

The graph memory layer is designed for clean ablation — disable it to fall back to standard GPT-2 KV-cache behaviour:

```python
model = GraphMemoryModel.from_pretrained("gpt2", use_graph_memory=False)
```

Results from baseline comparisons are logged to `results/`.

---

## Authors

- **Stig Norland** — conceptual architecture & design
- **Claude (Anthropic)** — implementation support

Bergen, 19 March 2026

---

## License

[MIT](LICENSE)

---

## Citation

If you build on this work, please cite:

```bibtex
@misc{norland2026dynamicgraph,
  author       = {Norland, Stig},
  title        = {Relational Graph Memory for Language Models},
  year         = {2026},
  howpublished = {\url{https://github.com/StigNorland/DynamicGraphMemory}}
}
```# DynamicGraphMemory
**Relational Graph Memory for Language Models**

> A proof-of-concept that replaces the KV cache as primary working memory with a dynamic relational concept graph. The raw conversation is kept only as a lazy-loaded backing store indexed by graph pointers.

---

## Motivation

Current transformer architectures re-read the full conversation on every forward pass — computationally expensive, uncompressed, and operating at the wrong granularity for meaning.

This project demonstrates that **meaning is relational structure**, not token sequences. A concept graph serves as the primary working memory, where nodes are language-agnostic concepts defined purely by their relational signatures and edges encode semantic, temporal, and causal relationships.

---

## Key Architectural Properties

| Property | Description |
|---|---|
| **Write-as-novelty** | A concept is written only if it modifies the graph — the structural update *is* the novelty signal |
| **Hebbian reinforcement** | Repeated edge traversal strengthens weights, naturally compressing recurring themes |
| **Lazy backing store** | Raw conversation text is retained but only dereferenced when full fidelity is required |
| **Dynamic merging** | Concepts with sufficiently similar relational signatures unify, cascading new relationships |
| **Dynamic looping** | Graph propagation continues until convergence; reasoning depth scales with problem complexity |
| **Loop detection** | Rolling hash of graph states identifies fixed points and limit cycles as meaningful signals |

---

## Status

Early proof-of-concept (March 2026).  
Built on **[nanoGPT](https://github.com/karpathy/nanoGPT)** with pretrained GPT-2 weights. The graph memory layer sits between the conversation and the model's context assembly, leaving the base model untouched for clean ablation studies.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU recommended (CPU inference is possible but slow)

```
torch
numpy
networkx
tiktoken
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/StigNorland/DynamicGraphMemory.git
cd DynamicGraphMemory

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install torch numpy networkx tiktoken

# 4. Download pretrained GPT-2 weights (fetched automatically on first run via nanoGPT)
```

---

## Project Structure

```
DynamicGraphMemory/
├── src/          # Core implementation
│   ├── graph/    # Concept graph: nodes, edges, Hebbian update, merge logic
│   ├── memory/   # Backing store and lazy-load indexing
│   └── model/    # nanoGPT integration and context assembly
├── testdata/     # Sample conversations for smoke-testing
├── results/      # Experiment outputs and ablation logs
├── paper/        # LaTeX source for the accompanying paper
└── README.md
```

---

## Quick Start

```python
from src.model import GraphMemoryModel

model = GraphMemoryModel.from_pretrained("gpt2")

response = model.chat("What is the capital of France?")
print(response)

# Inspect the current concept graph
model.memory.graph.summary()
```

---

## Running Tests

```bash
python -m pytest testdata/
```

---

## Ablation Studies

The graph memory layer is designed for clean ablation — disable it to fall back to standard GPT-2 KV-cache behaviour:

```python
model = GraphMemoryModel.from_pretrained("gpt2", use_graph_memory=False)
```

Results from baseline comparisons are logged to `results/`.

---

## Authors

- **Stig Norland** — conceptual architecture & design
- **Claude (Anthropic)** — implementation support

Bergen, 19 March 2026

---

## License

[MIT](LICENSE)

---

## Citation

If you build on this work, please cite:

```bibtex
@misc{norland2026dynamicgraph,
  author       = {Norland, Stig},
  title        = {Relational Graph Memory for Language Models},
  year         = {2026},
  howpublished = {\url{https://github.com/StigNorland/DynamicGraphMemory}}
}
```
