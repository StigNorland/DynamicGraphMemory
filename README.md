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
