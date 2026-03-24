#!/usr/bin/env python3
"""
generate_test_v2.py

Converts a Grok/Claude PDF export into a test JSON matching
the memory.json schema used in the RGM paper.

Turn detection: user turns are blue headings (size=14, RGB ~0.184/0.329/0.588).
Assistant turns are regular black body text (size=11, grayscale 0.0).
This is 100% reliable — no regex heuristics needed.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python generate_test_v2.py How_I_remember.pdf --out test.json
    python generate_test_v2.py How_I_remember.pdf --out test.json --name "how_i_remember"

Requirements:
    pip install anthropic pdfplumber
"""

import json, re, sys, time, argparse
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic"); sys.exit(1)

try:
    import pdfplumber
except ImportError:
    print("ERROR: pip install pdfplumber"); sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL         = "claude-sonnet-4-20250514"
MAX_TOKENS    = 4096
TURNS_PER_WIN = 6     # exchange pairs per window
OVERLAP_TURNS = 2     # overlap between windows
SLEEP         = 1.5   # seconds between API calls

# PDF font signature for user turns (blue headings)
USER_SIZE  = 14.0
USER_COLOR = (0.184, 0.329, 0.588)

# ── PDF Parser ────────────────────────────────────────────────────────────────

def _is_user_char(c: dict) -> bool:
    return (round(c['size'], 1) == USER_SIZE and
            c.get('non_stroking_color') == USER_COLOR)


def parse_pdf(path: str) -> list[tuple[str, str]]:
    """
    Parse a Grok/Claude PDF export into (role, text) pairs.

    User turns  = blue heading text  (size 14, color ~0.184/0.329/0.588)
    Asst turns  = black body text    (size 11, color 0.0)

    Lines on the same page at the same vertical position are joined.
    Consecutive lines with the same role are merged into one turn.
    Metadata lines ('N sources', 'Thought for Ns', 'Fast', 'Expert')
    are stripped.
    """
    META = re.compile(
        r'^(\d+\s*sources?'
        r'|\d+(\.\d+)?s(Fast|Expert)?$'
        r'|Thought for \d+s'
        r'|Fast|Expert'
        r'|2\s*/\s*2'
        r')$',
        re.IGNORECASE
    )

    turns = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # Group chars by vertical position (one bucket per text line)
            line_buckets: dict[float, list] = {}
            for c in page.chars:
                y = round(c['top'], 1)
                line_buckets.setdefault(y, []).append(c)

            for y in sorted(line_buckets):
                chars = sorted(line_buckets[y], key=lambda c: c['x0'])
                text  = ''.join(c['text'] for c in chars).strip()

                if not text or META.match(text):
                    continue

                role = 'user' if _is_user_char(chars[0]) else 'assistant'

                # Merge consecutive lines of the same role
                if turns and turns[-1][0] == role:
                    turns[-1] = (role, turns[-1][1] + '\n' + text)
                else:
                    turns.append([role, text])

    # Convert to immutable tuples and clean up whitespace
    return [(role, text.strip()) for role, text in turns]


def pair_turns(turns: list[tuple]) -> list[tuple[str, str]]:
    """Group (user, assistant) into exchange pairs."""
    pairs, i = [], 0
    while i < len(turns):
        role, text = turns[i]
        if role == 'user':
            asst = turns[i+1][1] if i+1 < len(turns) and turns[i+1][0] == 'assistant' else ''
            pairs.append((text, asst))
            i += 2 if asst else 1
        else:
            pairs.append(('', text))   # orphan assistant
            i += 1
    return pairs


# ── Windowing ─────────────────────────────────────────────────────────────────

def windows(pairs, size, overlap):
    step = max(1, size - overlap)
    i = 0
    while i < len(pairs):
        yield i, pairs[i : i + size]
        if i + size >= len(pairs): break
        i += step


def fmt_window(pairs, start):
    parts = []
    for j, (u, a) in enumerate(pairs):
        n = start + j + 1
        if u: parts.append(f"[Turn {n} — User]\n{u}")
        if a: parts.append(f"[Turn {n} — Assistant]\n{a}")
    return '\n\n'.join(parts)


# ── Prompts ───────────────────────────────────────────────────────────────────

WINDOW_SYSTEM = """
You are building a test dataset for the Relational Graph Memory (RGM) architecture.
RGM replaces flat token context with a concept graph. The test checks whether
graph-only context can answer questions as well as raw text context.

Output ONLY valid JSON — no markdown fences, no preamble.

Schema:
{
  "concepts": ["snake_case_label", ...],
  "questions": [
    {
      "text": "the question",
      "key_concepts": ["concept1", "concept2"],
      "level": 1,
      "description": "one sentence: why this level"
    }
  ]
}

Level definitions:
  1 — answer stated explicitly in a single turn (direct retrieval)
  2 — requires connecting 2+ non-adjacent turns (synthesis)
  3 — requires reasoning FROM the relational structure, not just finding it
  4 — requires seeing the same structure across two different domains

Per window: 2 L1, 2 L2, 1-2 L3. Add L4 only if a genuine cross-domain
equivalence is clearly visible within this specific segment.

key_concepts: 2-5 short snake_case labels directly relevant to the answer.
"""

WINDOW_USER = "Segment (turns {s}–{e} of {t} total):\n\n{text}\n\nExtract JSON."

L4_SYSTEM = """
You are writing L4 cross-domain test questions for the RGM architecture paper.
L4 tests whether a system recognises the SAME structural pattern across domains.
These are the hardest and most important questions for distinguishing RGM from
flat-context baselines.

Output ONLY valid JSON — no markdown, no preamble.

Schema:
{
  "questions": [
    {
      "text": "question naming both domains explicitly",
      "key_concepts": ["concept1", "concept2", "concept3"],
      "level": 4,
      "description": "domain A + domain B: the structural property they share"
    }
  ]
}

Rules:
- Name both domains in the question text
- Answer must require holding both domains simultaneously
- 5-8 questions total
- Be structurally specific: not 'how does X relate to Y'
  but 'what structural property do X and Y share'
"""

L4_USER = """Conversation: {name} ({turns} exchange pairs)

Domains covered: AI memory mechanics, Big Five personality psychology,
LLM scaling limits, RGM architecture, holographic bioelectric memory
(Levin/planaria), Hameroff-Penrose Orch OR, classical laser holography,
Buddhist philosophy (sati, anatta, enlightenment, nibbana).

All concepts extracted from the full conversation:
{concepts}

Write L4 cross-domain questions.
"""

DEDUP_SYSTEM = """
Deduplicate this RGM test question list.
- Remove near-duplicates (keep the sharper version)
- Keep ALL level-4 questions without exception
- Preserve level distribution
Output ONLY valid JSON: {"questions": [...]}
"""

# ── API ───────────────────────────────────────────────────────────────────────

def call(client, system, user):
    r = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return r.content[0].text.strip()


def parse_json(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
    try:    return json.loads(text)
    except: return None


# ── Main ──────────────────────────────────────────────────────────────────────

def build(pdf_path, out_path, name=None):
    print(f"Parsing {pdf_path} ...")
    raw   = parse_pdf(pdf_path)
    pairs = pair_turns(raw)
    total = len(pairs)
    name  = name or Path(pdf_path).stem

    print(f"  {len(raw)} turns → {total} exchange pairs")
    print()
    # Print turn map for verification
    for i, (u, a) in enumerate(pairs):
        print(f"  [{i+1:2d}] U: {u[:70]!r}")
        if a: print(f"       A: {a[:70]!r}")

    client        = anthropic.Anthropic()
    all_concepts  = set()
    all_questions = []

    # ── Window pass ───────────────────────────────────────────────────────
    print(f"\nWindows (size={TURNS_PER_WIN}, overlap={OVERLAP_TURNS}):")
    for start, chunk in windows(pairs, TURNS_PER_WIN, OVERLAP_TURNS):
        end = start + len(chunk) - 1
        print(f"  [{start+1}–{end+1}] ({len(chunk)} pairs) ... ", end='', flush=True)
        p = parse_json(call(
            client, WINDOW_SYSTEM,
            WINDOW_USER.format(s=start+1, e=end+1, t=total,
                               text=fmt_window(chunk, start))
        ))
        if p:
            all_concepts.update(p.get('concepts', []))
            qs = p.get('questions', [])
            all_questions.extend(qs)
            print(f"{len(qs)}q  {len(p.get('concepts',[]))}c")
        else:
            print("SKIP (bad JSON)")
        time.sleep(SLEEP)

    # ── L4 pass ───────────────────────────────────────────────────────────
    print(f"\nL4 pass ({len(all_concepts)} concepts) ... ", end='', flush=True)
    p = parse_json(call(
        client, L4_SYSTEM,
        L4_USER.format(name=name, turns=total,
                       concepts='\n'.join(sorted(all_concepts)))
    ))
    if p:
        l4 = p.get('questions', [])
        all_questions.extend(l4)
        print(f"{len(l4)} added")
    else:
        print("SKIP")
    time.sleep(SLEEP)

    # ── Dedup ─────────────────────────────────────────────────────────────
    print(f"Deduplicating {len(all_questions)} questions ... ", end='', flush=True)
    deduped, BATCH = [], 60
    for i in range(0, len(all_questions), BATCH):
        p = parse_json(call(client, DEDUP_SYSTEM,
                            json.dumps({"questions": all_questions[i:i+BATCH]}, indent=2)))
        deduped.extend(p.get('questions', all_questions[i:i+BATCH]) if p
                       else all_questions[i:i+BATCH])
        time.sleep(SLEEP)

    from collections import Counter
    dist = dict(sorted(Counter(q.get('level') for q in deduped).items()))
    print(f"{len(deduped)} remaining  levels={dist}")

    # ── Output (memory.json schema) ───────────────────────────────────────
    sample = []
    for u, a in pairs[:4]:
        if u: sample.append(["user",      u[:300]])
        if a: sample.append(["assistant", a[:300]])

    # Full conversation for graph builder
    full_conv = []
    for u, a in pairs:
        if u: full_conv.append(["user",      u])
        if a: full_conv.append(["assistant", a])

    out = {
        "metadata": {
            "name":           name,
            "description": (
                "Multi-domain conversation: AI memory mechanics, personality psychology "
                "(Big Five / Gerlach 2018), LLM scaling limits, RGM architecture, "
                "holographic/bioelectric memory (Levin), Hameroff-Penrose Orch OR, "
                "classical laser holography, Buddhist philosophy of mind (sati, anatta, "
                "nibbana). Tests cross-domain synthesis, long-range recall, concept unification."
            ),
            "turns":          total,
            "token_budget":   200,
            "domain":         "multi-domain",
            "source_file":    pdf_path,
            "model_used":     MODEL,
            "concepts_found": len(all_concepts),
        },
        "conversation": full_conv,   # full conversation for graph builder
        "conversation_sample": sample,
        "questions":    deduped,
    }

    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n✓  {out_path}")
    print(f"   {len(deduped)} questions  |  {len(all_concepts)} concepts  |  {total} turn-pairs")
    print(f"   {len(full_conv)} entries in conversation field (full)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input",  help="PDF file from Grok/Claude export")
    ap.add_argument("--out",  default="test.json")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    build(args.input, args.out, args.name)