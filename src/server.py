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

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure the library package is importable (same pattern as experiment.py)
sys.path.insert(0, os.path.dirname(__file__))

import anthropic
from library.context import ContextAssembler, BackingStore
from library.graph import ConceptGraph

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


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class InitRequest(BaseModel):
    api_key: str


class ChatRequest(BaseModel):
    message: str
    mode: str = "baseline"  # "graph" | "baseline"


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
        {"id": n.id, "label": n.label, "prov": n.provisional, "maturity": n.maturity}
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

        ctx_words      = len(ctx.split()) if ctx else 0
        baseline_ctx   = _assembler.assemble_baseline(token_budget)
        baseline_words = len(baseline_ctx.split()) if baseline_ctx else 0
        _cost_log.append({
            "turn": len(_assembler.store.turns),
            "g":    ctx_words,
            "b":    baseline_words,
            "mode": req.mode,
        })

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
