# Relational Graph Memory for Language Models

A proof-of-concept architecture replacing the KV cache as primary 
working memory with a relational concept graph, where the raw 
conversation serves as a lazy-loaded backing store indexed by 
graph pointers.

## Core Idea

Current transformer architectures re-read the full conversation 
on every forward pass — expensive, uncompressed, and the wrong 
granularity for meaning. This project proposes that meaning is 
relational structure, not token sequences. A concept graph where 
nodes are language-agnostic concepts and edges encode semantic, 
temporal, and causal relationships is both more compressible and 
more semantically addressable than a KV cache.

Key properties of the architecture:

- **Write-as-novelty**: a concept update only occurs when incoming 
  input modifies the graph — the update operation itself is the 
  novelty signal, requiring no separate comparator
- **Hebbian reinforcement**: repeated traversal of relational paths 
  strengthens edge weights, compressing recurring structure naturally
- **Lazy backing store**: raw conversation is retained but only 
  dereferenced when full fidelity is needed — most queries resolve 
  at graph level
- **Dynamic looping**: graph propagation iterates until convergence, 
  making reasoning depth a function of problem complexity rather 
  than architecture size
- **Loop detection**: rolling hash of graph states catches fixed 
  points and limit cycles — a repeating hash is a signal, not just 
  a termination condition

## Status

Early proof-of-concept. Built on nanoGPT + pretrained GPT-2 weights. 
The graph memory layer sits between conversation and model context — 
everything else is untouched, giving clean experimental control.

## Authors

Stig Norland — conceptual architecture  
Claude (Anthropic) — implementation support
Bergen, 19 March 2026
