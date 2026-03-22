# feature/typed-triple-extractor

Replaces the stopword-filter concept extractor with a spaCy-based pipeline
that emits typed `(subject, IS-A, object)` triples.

## Setup

```bash
pip install spacy
python -m spacy download en_core_web_sm   # minimum
python -m spacy download en_core_web_lg   # recommended
```

## Smoke-test

```bash
python concept_extractor.py
```

Expected output (en_core_web_sm):

```
(proton)  --[IS-A conf=0.95]-->  (breather)
(electron)  --[IS-A conf=0.95]-->  (ring)
(neutron)  --[IS-A conf=0.85]-->  (mechanism)
(dark_matter)  --[IS-A conf=0.95]-->  (vortex)
(pion)  --[IS-A conf=0.95]-->  (tube)
```

## Plugging into the graph engine

In `graph_engine.py` (or wherever concepts are currently extracted), replace:

```python
# OLD — stopword filter
concepts = [w for w in text.split() if w not in STOPWORDS]
```

with:

```python
# NEW — typed triple extractor
from concept_extractor import extract_concepts
concepts = extract_concepts(turn_text)
```

The returned list has the same shape the graph's write policy already expects:
```json
[
  {
    "label": "proton",
    "type": "Concept",
    "relations": [{"target": "breather", "edge_type": "IS-A", "conf": 0.95}]
  }
]
```

## Extraction strategies (layered)

| Strategy | Pattern detected | Confidence |
|---|---|---|
| Copula IS-A | "X is a Y" / "X are Y" | 0.70 – 0.95 |
| NP apposition | "X, a Y, ..." | 0.85 |
| NER typing | Named entity → spaCy label | 0.75 |
| Noun compound | "dark matter" → IS-A "matter" | 0.60 |

## Adding more relation types later

Each strategy is an isolated generator function (`_extract_*`).
To add CAUSES or PART-OF, add a new generator and register it in `extract()`.

## Known limitations

- Extraction quality scales with model size (`en_core_web_lg` >> `sm`)
- Non-English text requires a different model (`de_core_news_sm` etc.)
- Confidence values are heuristic; no calibration dataset yet
