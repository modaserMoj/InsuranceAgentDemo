from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import run_turn
from .data import PII_FIELDS
from .engine import PHASES, new_session
from .intents import index as intent_index
from .intents import retrieval_mode
from .providers import LLMError, Provider, resolve
from .simulator import PERSONAS, next_caller_message

INDEX = Path(__file__).resolve().parent.parent / "public" / "index.html"
MAX_SESSIONS = 500
GREETING = (
    "Hi, thanks for calling Claims Support. I'm Alex. I can help with questions about your policy and claims. "
    "To get started, could you tell me who I'm speaking with and what you're calling about?"
)

app = FastAPI(title="SOP-guided insurance claims agent")
intent_index()  # load the embedding model at startup rather than on the first caller's message
sessions: "OrderedDict[str, dict]" = OrderedDict()
session_providers: dict[str, Provider | None] = {}


class NewSession(BaseModel):
    api_key: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=200)
    consent_scenario: str = "default"
    offline: bool = False
    today: str | None = None


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def public_state(s: dict) -> dict:
    ident = s["identity"]
    return {
        "session_id": s["id"],
        "today": s["today"],
        "phase": s["phase"],
        "phases": PHASES,
        "llm": s["llm"],
        "identity": {
            "verified": ident["verified"],
            "party_id": ident["party_id"],
            "caller_role": ident["caller_role"],
            "rep_name": ident["rep_name"],
            "rep_relationship": ident["rep_relationship"],
            "fields_provided": [f for f in PII_FIELDS if ident["claimed"][f]],
            "policy_number_provided": bool(ident["claimed"]["policy_number"]),
            "matched_fields": ident["matched_fields"],
            "failed_attempts": ident["failed_attempts"],
        },
        **{k: s[k] for k in ("consent", "memory", "case", "counters", "emotion", "pending_offer", "discussed", "followups", "email", "handoff", "outbox", "trace")},
    }


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {"server_llm": resolve(None).public(), "intent_retrieval": retrieval_mode()}


@app.post("/api/sessions")
def create_session(body: NewSession):
    while len(sessions) >= MAX_SESSIONS:
        old, _ = sessions.popitem(last=False)
        session_providers.pop(old, None)
    s = new_session(consent_scenario="timeout" if body.consent_scenario == "timeout" else "default", today=body.today)
    s["transcript"].append({"role": "assistant", "text": GREETING})

    if body.offline:
        provider, llm = None, {"status": "offline", "message": "Offline mode selected (regex understanding + SOP templates).", "provider": None, "model": None}
    else:
        model = (body.model or "").strip() or None
        res = resolve((body.api_key or "").strip() or None, model)
        if res.status == "invalid" and (body.api_key or "").strip():
            server = resolve(None, model)
            if server.status == "ok":
                server.message = f"The key typed in the UI was rejected, so the server's key is used instead. {server.message}"
                res = server
        provider, llm = res.provider, res.public()
    s["llm"] = llm
    sessions[s["id"]] = s
    session_providers[s["id"]] = provider
    return {"reply": GREETING, "llm": llm, "state": public_state(s)}


def _get(session_id: str) -> dict:
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    return s


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    s = _get(session_id)
    return {"state": public_state(s), "transcript": s["transcript"]}


class Simulate(BaseModel):
    persona: str = Field(min_length=1, max_length=64)


@app.get("/api/personas")
def personas():
    return {"personas": [{"id": k, "label": v["label"]} for k, v in PERSONAS.items()]}


@app.post("/api/sessions/{session_id}/simulate")
async def simulate(session_id: str, body: Simulate):
    """Next line for an LLM-simulated caller (the UI's "Watch a demo"). It never touches the engine directly:
    the UI sends the returned text through the normal /messages endpoint like a real caller."""
    s = _get(session_id)
    provider = session_providers.get(session_id)
    if provider is None:
        raise HTTPException(400, "The demo caller needs an LLM key.")
    if body.persona not in PERSONAS:
        raise HTTPException(404, "unknown persona")
    if s["phase"] in ("ENDED", "HANDOFF"):
        return {"text": "", "done": True}
    try:
        text, done = await run_in_threadpool(next_caller_message, provider, body.persona, s["transcript"])
    except LLMError as e:
        raise HTTPException(502, f"simulator failed: {e}")
    return {"text": text, "done": done}


@app.post("/api/sessions/{session_id}/messages")
async def post_message(session_id: str, body: Message):
    s = _get(session_id)
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "text is required")
    out = await run_in_threadpool(run_turn, s, text, session_providers.get(session_id))
    return {**out, "state": public_state(s)}
