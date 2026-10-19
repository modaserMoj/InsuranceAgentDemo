"""Integration: the HTTP API end to end (FastAPI TestClient, offline mode, no API key needed)."""
from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def new_chat(**kw):
    r = client.post("/api/sessions", json={"offline": True, "today": "2026-03-05", **kw})
    assert r.status_code == 200
    return r.json()


def say(sid, text):
    r = client.post(f"/api/sessions/{sid}/messages", json={"text": text})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_and_ui_are_served():
    assert client.get("/healthz").json() == {"ok": True}
    page = client.get("/")
    assert page.status_code == 200 and "Alex" in page.text


def test_full_conversation_over_http():
    chat = new_chat()
    sid = chat["state"]["session_id"]
    assert chat["llm"]["status"] == "offline" and chat["state"]["phase"] == "VERIFY_ID"
    out = say(sid, "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
                   "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.")
    assert out["state"]["phase"] == "PROCESS_CASE" and "CL-2048" in out["reply"]
    say(sid, "that's all")
    out = say(sid, "yes")
    assert out["state"]["phase"] == "ENDED" and len(out["state"]["outbox"]) == 1


def test_public_state_never_exposes_raw_identity_values():
    sid = new_chat()["state"]["session_id"]
    out = say(sid, "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472")
    body = str(out["state"])
    assert "4472" not in body and "1985-03-15" not in body, "only field names are exposed, never the values"


def test_input_validation():
    sid = new_chat()["state"]["session_id"]
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": ""}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "x" * 2001}).status_code == 422
    assert client.post("/api/sessions/does-not-exist/messages", json={"text": "hi"}).status_code == 404


def test_demo_caller_requires_an_llm():
    sid = new_chat()["state"]["session_id"]
    assert client.get("/api/personas").json()["personas"]
    assert client.post(f"/api/sessions/{sid}/simulate", json={"persona": "margaret_happy_path"}).status_code == 400


def test_bad_ui_key_falls_back_cleanly():
    chat = client.post("/api/sessions", json={"api_key": "not-a-real-key", "today": "2026-03-05"}).json()
    assert chat["llm"]["status"] in ("ok", "invalid"), "either the server key took over or the UI says the key was rejected"
