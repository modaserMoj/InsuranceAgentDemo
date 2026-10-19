"""PROCESS_CASE tightening: multiple intents per message, unanswerable-question escalation, document checklist."""
from app import agent
from app.agent import run_turn
from app.engine import new_session
from app.extract import regex_extract

OPEN = "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472. Calling about my denied healthcare claim from January."


def in_case():
    s = new_session(today="2026-03-05")
    run_turn(s, OPEN)
    assert s["phase"] == "PROCESS_CASE" and s["case"]["selected_case_id"] == "CL-2048"
    return s


def acts(s):
    return s["trace"][-1]["acts"]


def test_two_questions_in_one_message_are_both_answered():
    s = in_case()
    reply = run_turn(s, "how much did they pay? also what happens next")["reply"]
    assert "allowed maximum" in reply, "payment answered"
    assert "within a week" in reply or "less than a week" in reply, "next steps answered too"


def test_two_unanswerable_questions_in_a_row_offer_a_human():
    s = in_case()
    run_turn(s, "no")
    r1 = run_turn(s, "will the claim reviewer be the same person as before?")["reply"]
    assert "don't have a specific answer" in r1 and s["pending_offer"] != "human"
    r2 = run_turn(s, "is the claim reviewer a doctor?")["reply"]
    assert "offer_human_unanswered" in acts(s)
    assert s["pending_offer"] == "human" and "claims representative can look into it" in r2


def test_answered_question_resets_the_unanswerable_streak():
    s = in_case()
    run_turn(s, "no")
    run_turn(s, "will the claim reviewer be the same person as before?")
    run_turn(s, "how much did they pay?")
    assert s["counters"]["unanswered_streak"] == 0


def test_llm_no_answer_marker_counts_and_is_hidden(monkeypatch):
    calls = {"n": 0}

    def fake_phrase(provider, payload):
        calls["n"] += 1
        return "[[NO_ANSWER]] I'm sorry, I don't have that information."

    s = in_case()
    monkeypatch.setattr(agent, "interpret", lambda message, context, provider=None: (regex_extract(message), None))
    monkeypatch.setattr(agent, "phrase", fake_phrase)
    out1 = agent.run_turn(s, "how much did they pay?", provider=object())
    assert "[[NO_ANSWER]]" not in out1["reply"] and s["counters"]["unanswered_streak"] == 1
    out2 = agent.run_turn(s, "and the allowed amount?", provider=object())
    assert s["pending_offer"] == "human" and "claims representative can look into it" in out2["reply"]


def test_document_checklist_tracks_each_document():
    s = in_case()
    reply = run_turn(s, "I have the office note but I can't get the pathology report from the lab")["reply"]
    assert acts(s)[0] == "document_update"
    checklist = s["case"]["documents"]["CL-2048"]
    assert checklist == {"pathology report": "cannot_get", "office note": "have"}
    assert "upload it" in reply and "replacement copy" in reply
    assert s["pending_offer"] != "human", "one document is still obtainable"


def test_all_documents_unavailable_triggers_human_review_offer():
    s = in_case()
    run_turn(s, "I lost the office note and the lab won't give me the pathology report")
    assert set(s["case"]["documents"]["CL-2048"].values()) == {"cannot_get"}
    assert s["pending_offer"] == "human"
    run_turn(s, "yes")
    assert s["phase"] == "HANDOFF"


def test_email_includes_document_checklist():
    s = in_case()
    run_turn(s, "I already uploaded the office note")
    run_turn(s, "that's all")
    run_turn(s, "yes")
    body = s["outbox"][0]["body"]
    assert "Document checklist:" in body
    assert "CL-2048 / office note: you said it was already sent" in body and "CL-2048 / pathology report: still needed" in body
