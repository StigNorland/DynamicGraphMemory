"""
server.py — FastAPI backend for the Python-backed RGM chat client.

Wraps the Python library (library/context.py + library/graph.py) and
exposes a streaming SSE endpoint so the browser frontend can drive
Claude while the graph memory lives server-side.

Run from the src/ directory:
    uvicorn server:app --port 8000 --reload
"""

import json
import os
import sys
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure the library package is importable (same pattern as experiment.py)
sys.path.insert(0, os.path.dirname(__file__))

import anthropic
from library.context import ContextAssembler, BackingStore
from library.graph import ConceptGraph
from library.convergence import ConvergenceMonitor
from locomo import (
    LoCoMoSample, parse_locomo_file, flatten_to_conversation, total_turns,
    LoCoMoScorer, propagation_pass, convergence_sweep, summarise_results,
)

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*50}", flush=True)
    print(f"  🔄  RGM server (re)started at {ts}", flush=True)
    print(f"{'='*50}\n", flush=True)


# ---------------------------------------------------------------------------
# Server-level state (single-user, local use)
# ---------------------------------------------------------------------------
_client: anthropic.Anthropic | None = None
_assembler: ContextAssembler | None = None
_cost_log: list[dict] = []

# LoCoMo benchmark state
_locomo_sample: LoCoMoSample | None = None
_locomo_eval_results: list[dict] = []
_locomo_state: str = "idle"        # idle|loaded|ingesting|ready|evaluating|done|error
_locomo_progress: dict = {"done": 0, "total": 0}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class InitRequest(BaseModel):
    api_key: str


class ChatRequest(BaseModel):
    message: str
    mode: str = "baseline"  # "graph" | "baseline"


class LoCoMoLoadRequest(BaseModel):
    data: list           # full raw JSON array from the file
    sample_index: int = 0


class LoCoMoRunRequest(BaseModel):
    token_budget: int = 400
    modes: list[str] = ["graph", "baseline", "graph_only"]


class IngestRequest(BaseModel):
    turns: list[dict]  # [{"speaker": "user"|"assistant", "text": "..."}]


# ---------------------------------------------------------------------------
# Graph state serialisation
# ---------------------------------------------------------------------------
def build_graph_state() -> dict:
    """Serialise the current graph into JSON the frontend can consume."""
    if _assembler is None:
        return {
            "nodes": [], "edges": [], "merges": [], "stats": {},
            "mergeThr": 0.3, "matThr": 5.0,
        }

    g = _assembler.graph

    nodes = [
        {
            "id":          n.id,
            "label":       n.label,
            "prov":        n.provisional,
            "maturity":    n.maturity,
            "entity_type": n.meta.get("entity_type"),   # None for concepts
        }
        for n in g.nodes.values()
    ]
    edges = [
        {"src": e.source, "tgt": e.target, "type": e.edge_type.value, "w": e.weight}
        for e in g.edges.values()
    ]
    merges = [
        {
            "nodeA":      m.node_a,
            "nodeB":      m.node_b,
            "mergedInto": m.merged_into,
            "cascade":    m.cascade_depth,
            "mag":        m.magnitude,
            "turn":       m.conv_pos,
        }
        for m in g.merge_events
    ]
    stats = g.summary()

    return {
        "nodes":    nodes,
        "edges":    edges,
        "merges":   merges,
        "stats":    stats,
        "mergeThr": g.merge_threshold,
        "matThr":   g.maturity_threshold,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/init")
async def api_init(req: InitRequest):
    global _client, _assembler, _cost_log
    _client    = anthropic.Anthropic(api_key=req.api_key)
    _assembler = ContextAssembler(
        graph      = ConceptGraph(),
        store      = BackingStore(),
        llm_client = _client,
    )
    _cost_log  = []
    return {"ok": True}


@app.post("/api/reset")
async def api_reset():
    global _assembler, _cost_log
    if _client is None:
        return JSONResponse({"ok": False, "error": "not initialized"}, status_code=400)
    _assembler = ContextAssembler(
        graph      = ConceptGraph(),
        store      = BackingStore(),
        llm_client = _client,
    )
    _cost_log = []
    return {"ok": True}


@app.post("/api/ingest")
async def api_ingest(req: IngestRequest):
    """Replay a list of turns into the graph without calling Claude."""
    if _assembler is None or _client is None:
        return JSONResponse({"error": "not initialized"}, status_code=400)

    async def generate():
        total = len(req.turns)
        ingested = 0
        for i, turn in enumerate(req.turns):
            speaker = str(turn.get("speaker", "user")).lower()
            text    = str(turn.get("text", "")).strip()
            if not text:
                continue
            _assembler.ingest(speaker, text)
            ingested += 1
            preview = text[:60] + ("…" if len(text) > 60 else "")
            yield f"data: {json.dumps({'type':'progress','done':i+1,'total':total,'speaker':speaker,'preview':preview})}\n\n"

        state = build_graph_state()
        state["type"]     = "graph_state"
        state["cost_log"] = _cost_log
        yield f"data: {json.dumps(state)}\n\n"
        yield f"data: {json.dumps({'type':'done','ingested':ingested})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/graph-state")
async def api_graph_state():
    state = build_graph_state()
    state["cost_log"] = _cost_log
    return state


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if _assembler is None or _client is None:
        return JSONResponse({"error": "not initialized"}, status_code=400)

    async def generate():
        prev_merge_count = len(_assembler.graph.merge_events)

        # --- Ingest user turn ---
        _assembler.ingest("user", req.message)

        # Emit aha events from user ingest
        for m in _assembler.graph.merge_events[prev_merge_count:]:
            yield f"data: {json.dumps({'type':'aha','nodeA':m.node_a,'nodeB':m.node_b,'mergedInto':m.merged_into,'cascade':m.cascade_depth,'mag':m.magnitude})}\n\n"

        prev_merge_count = len(_assembler.graph.merge_events)

        # --- Assemble context ---
        token_budget = 300
        if req.mode == "graph":
            ctx = _assembler.assemble_graph(req.message, token_budget)
        else:
            ctx = _assembler.assemble_baseline(token_budget)

        def _est_tokens(text: str) -> int:
            """Rough token estimate: words × 1.33 (good enough for comparison)."""
            return round(len(text.split()) * 1.33) if text else 0

        baseline_ctx    = _assembler.assemble_baseline(token_budget)
        graph_ctx_tok   = _est_tokens(ctx)
        baseline_ctx_tok = _est_tokens(baseline_ctx)

        # --- Build system message ---
        if req.mode == "graph":
            sys_prompt = (
                "You are a helpful assistant. Context assembled by Relational Graph Memory "
                "(most structurally relevant turns, not necessarily most recent):\n\n"
                f"{ctx}\n\nRespond concisely."
            )
        else:
            sys_prompt = (
                f"You are a helpful assistant. Recent conversation:\n\n{ctx}\n\nRespond concisely."
            )

        # --- Stream Claude response ---
        full_reply = ""
        with _client.messages.stream(
            model      = "claude-sonnet-4-6",
            max_tokens = 4096,
            system     = sys_prompt,
            messages   = [{"role": "user", "content": req.message}],
        ) as stream:
            for text in stream.text_stream:
                full_reply += text
                yield f"data: {json.dumps({'type':'token','text':text})}\n\n"
            usage = stream.get_final_message().usage

        _cost_log.append({
            "turn":    len(_assembler.store.turns),
            "mode":    req.mode,
            "g":       graph_ctx_tok,       # estimated context tokens (graph)
            "b":       baseline_ctx_tok,    # estimated context tokens (baseline)
            "inp":     usage.input_tokens,  # actual billed input tokens
            "out":     usage.output_tokens, # actual billed output tokens
        })

        # --- Ingest assistant reply ---
        _assembler.ingest("assistant", full_reply)

        # Emit aha events from assistant ingest
        for m in _assembler.graph.merge_events[prev_merge_count:]:
            yield f"data: {json.dumps({'type':'aha','nodeA':m.node_a,'nodeB':m.node_b,'mergedInto':m.merged_into,'cascade':m.cascade_depth,'mag':m.magnitude})}\n\n"

        # --- Emit final graph state ---
        state = build_graph_state()
        state["type"]     = "graph_state"
        state["cost_log"] = _cost_log
        yield f"data: {json.dumps(state)}\n\n"
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# LoCoMo benchmark endpoints
# ---------------------------------------------------------------------------

def _reset_assembler():
    """Create a fresh ContextAssembler, keeping the existing client."""
    global _assembler, _cost_log
    _assembler = ContextAssembler(
        graph      = ConceptGraph(),
        store      = BackingStore(),
        llm_client = _client,
    )
    _cost_log = []


def _graph_only_context(graph: ConceptGraph, query: str, token_budget: int) -> str:
    """Mirrors evaluate.py Evaluator._graph_only_context."""
    parts = []
    stable = sorted(
        [n for n in graph.nodes.values() if not n.provisional],
        key=lambda n: -n.maturity,
    )[:10]
    if stable:
        parts.append("Core concepts: " + ", ".join(
            f"{n.label}(maturity={n.maturity:.1f})" for n in stable[:6]
        ))
    top_ids   = {n.id for n in stable[:8]}
    key_edges = sorted(
        [e for e in graph.edges.values()
         if e.source in top_ids and e.target in top_ids and e.weight > 0.5],
        key=lambda e: -e.weight,
    )[:8]
    if key_edges:
        rel_lines = [
            f"{graph.nodes[e.source].label} -{e.edge_type.value}-> "
            f"{graph.nodes[e.target].label}(w={e.weight:.2f})"
            for e in key_edges
            if e.source in graph.nodes and e.target in graph.nodes
        ]
        if rel_lines:
            parts.append("Key relations: " + "; ".join(rel_lines))
    if graph.merge_events:
        top_merges = sorted(graph.merge_events, key=lambda e: -e.magnitude)[:3]
        merge_lines = [
            f"'{e.node_a}' unified with '{e.node_b}'"
            for e in top_merges if e.merged_into in graph.nodes
        ]
        if merge_lines:
            parts.append("Insights: " + "; ".join(merge_lines))
    query_words = query.lower().split()
    relevant = sorted(
        [n for n in graph.nodes.values()
         if any(w in n.label for w in query_words if len(w) > 3)],
        key=lambda n: -n.maturity,
    )[:6]
    if relevant:
        parts.append("Query-relevant: " + ", ".join(
            f"{n.label}(deg={len(graph._get_neighbors(n.id))})" for n in relevant
        ))
    return "\n".join(parts)[:token_budget * 6]


def _assemble_context_for_eval(mode: str, query: str, token_budget: int) -> str:
    """Assemble context using the current assembler."""
    if mode == "graph":
        ctx = _assembler.assemble_graph(query=query, token_budget=token_budget)
        if not ctx.strip():
            ctx = _assembler.assemble_baseline(token_budget=token_budget)
        return ctx
    if mode == "baseline":
        return _assembler.assemble_baseline(token_budget=token_budget)
    if mode == "graph_only":
        return _graph_only_context(_assembler.graph, query, token_budget)
    return ""


@app.post("/api/locomo/load")
async def api_locomo_load(req: LoCoMoLoadRequest):
    global _locomo_sample, _locomo_eval_results, _locomo_state, _locomo_progress

    if _client is None:
        return JSONResponse({"ok": False, "error": "not initialized"}, status_code=400)

    try:
        sample = parse_locomo_file(req.data, req.sample_index)
    except (IndexError, KeyError, ValueError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    _locomo_sample        = sample
    _locomo_eval_results  = []
    _locomo_state         = "loaded"
    _locomo_progress      = {"done": 0, "total": 0}
    _reset_assembler()

    turns = total_turns(sample)
    return {
        "ok":         True,
        "sample_id":  sample.sample_id,
        "sessions":   len(sample.sessions),
        "total_turns": turns,
        "qa_count":   len(sample.qa),
        "speaker_a":  sample.speaker_a,
        "speaker_b":  sample.speaker_b,
    }


@app.post("/api/locomo/ingest")
async def api_locomo_ingest():
    global _locomo_state, _locomo_progress

    if _client is None or _assembler is None:
        return JSONResponse({"error": "not initialized"}, status_code=400)
    if _locomo_sample is None:
        return JSONResponse({"error": "no sample loaded"}, status_code=400)

    async def generate():
        global _locomo_state, _locomo_progress

        _reset_assembler()
        _locomo_state = "ingesting"

        turns     = flatten_to_conversation(_locomo_sample)
        n_total   = len(turns)
        _locomo_progress = {"done": 0, "total": n_total}

        last_keepalive = time.time()

        for i, (speaker, text) in enumerate(turns):
            _assembler.ingest(speaker, text)
            # Lightweight propagation pass per turn (same as evaluate.py)
            _assembler.graph.begin_pass()
            propagation_pass(_assembler.graph)
            _assembler.graph.end_pass()

            _locomo_progress["done"] = i + 1
            yield f"data: {json.dumps({'type':'progress','done':i+1,'total':n_total,'speaker':speaker})}\n\n"

            # Keepalive comment to prevent proxy/browser timeout on long ingestions
            now = time.time()
            if now - last_keepalive > 10:
                yield ": keepalive\n\n"
                last_keepalive = now

        # Final convergence sweep
        convergence_sweep(_assembler.graph)

        _locomo_state = "ready"

        # Emit graph state so the existing graph tabs update
        state          = build_graph_state()
        state["type"]  = "graph_state"
        state["cost_log"] = _cost_log
        yield f"data: {json.dumps(state)}\n\n"
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/locomo/run-eval")
async def api_locomo_run_eval(req: LoCoMoRunRequest):
    global _locomo_state, _locomo_eval_results

    if _client is None or _assembler is None:
        return JSONResponse({"error": "not initialized"}, status_code=400)
    if _locomo_sample is None:
        return JSONResponse({"error": "no sample loaded"}, status_code=400)
    if _locomo_state not in ("ready", "done"):
        return JSONResponse({"error": f"ingestion not complete (state={_locomo_state})"}, status_code=400)

    async def generate():
        global _locomo_state, _locomo_eval_results

        _locomo_state        = "evaluating"
        _locomo_eval_results = []

        scorer  = LoCoMoScorer(_client)
        qa_list = _locomo_sample.qa
        n_total = len(qa_list)

        for qi, q in enumerate(qa_list):
            yield f"data: {json.dumps({'type':'question_start','index':qi,'total':n_total,'question':q['question'],'category':q.get('category',0)})}\n\n"

            conditions = {}
            for mode in req.modes:
                ctx    = _assemble_context_for_eval(mode, q["question"], req.token_budget)
                prompt = (
                    "The following is context from a long-term conversation:\n\n"
                    f"{ctx}\n\n"
                    "Based only on this context, answer concisely:\n"
                    f"{q['question']}"
                )
                try:
                    resp = _client.messages.create(
                        model       = "claude-sonnet-4-6",
                        max_tokens  = 200,
                        temperature = 0,
                        messages    = [{"role": "user", "content": prompt}],
                    )
                    model_answer = resp.content[0].text.strip()
                except Exception as e:
                    model_answer = f"[ERROR: {e}]"

                scored = scorer.score_answer(
                    question         = q["question"],
                    reference_answer = str(q.get("answer", "")),
                    model_answer     = model_answer,
                    category         = q.get("category", 1),
                )
                conditions[mode] = {
                    "context_tokens": len(ctx.split()),
                    "answer":         model_answer,
                    "score":          scored["score"],
                    "reasoning":      scored["reasoning"],
                }

            result = {
                "question":  q["question"],
                "reference": str(q.get("answer", "")),
                "evidence":  q.get("evidence", []),
                "category":  q.get("category", 0),
                "conditions": conditions,
            }
            _locomo_eval_results.append(result)

            yield f"data: {json.dumps({'type':'question_result','index':qi,'question':q['question'],'category':q.get('category',0),'reference':str(q.get('answer','')),'conditions':conditions})}\n\n"

        summary = summarise_results(_locomo_eval_results, req.modes)
        _locomo_state = "done"
        yield f"data: {json.dumps({'type':'eval_complete','summary':summary})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/locomo/status")
async def api_locomo_status():
    sample_info = None
    if _locomo_sample is not None:
        sample_info = {
            "sample_id":   _locomo_sample.sample_id,
            "sessions":    len(_locomo_sample.sessions),
            "total_turns": total_turns(_locomo_sample),
            "qa_count":    len(_locomo_sample.qa),
            "speaker_a":   _locomo_sample.speaker_a,
            "speaker_b":   _locomo_sample.speaker_b,
        }
    return {
        "state":         _locomo_state,
        "progress":      _locomo_progress,
        "results_count": len(_locomo_eval_results),
        "has_sample":    _locomo_sample is not None,
        "sample_info":   sample_info,
    }


# ---------------------------------------------------------------------------
# Static file serving (must come last so API routes take priority)
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "chat.html"))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)
