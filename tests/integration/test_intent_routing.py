"""Integration: catch-all intents and intent-driven claim resolution in full conversations."""
from app.agent import run_turn
from app.engine import new_session

VERIFY = "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472"


def _verified():
    s = new_session(today="2026-03-05")
    run_turn(s, VERIFY)
    assert s["phase"] == "RESOLVE_INTENT"
    return s


def test_unsupported_request_is_redirected_with_human_offer():
    s = _verified()
    reply = run_turn(s, "I want to update my address")["reply"]
    assert s["trace"][-1]["acts"] == ["redirect_unsupported"]
    assert "can't help with that" in reply and "human representative" in reply
    assert s["pending_offer"] == "human"
    run_turn(s, "yes")
    assert s["phase"] == "HANDOFF"


def test_general_question_gets_general_answer_directive():
    s = _verified()
    out = run_turn(s, "what is a deductible?")
    assert s["trace"][-1]["acts"] == ["general_answer", "reprompt"]
    assert s["phase"] == "RESOLVE_INTENT"
    assert "general insurance question" in out["reply"]


def test_unsupported_request_during_verification_keeps_form_going():
    s = new_session(today="2026-03-05")
    run_turn(s, "Margaret Chen")
    reply = run_turn(s, "can I change my email on file")["reply"]
    acts = s["trace"][-1]["acts"]
    assert "note_unsupported" in acts and acts[-1] in ("ask_identity", "form_wrong_field")
    assert "date of birth" in reply
    assert s["phase"] == "VERIFY_ID"


def test_catch_all_mid_case_returns_to_claim():
    s = new_session(today="2026-03-05")
    run_turn(s, VERIFY + ". Calling about my denied healthcare claim from January.")
    assert s["phase"] == "PROCESS_CASE"
    reply = run_turn(s, "does my plan cover physical therapy?")["reply"]
    assert s["trace"][-1]["acts"] == ["redirect_unsupported"]
    assert s["phase"] == "PROCESS_CASE"
    assert "human representative" in reply


def test_intent_narrows_ambiguous_claim_match():
    s = _verified()
    run_turn(s, "they turned down my healthcare claim from january, why?")
    assert s["case"]["selected_case_id"] == "CL-2048", "denial intent picks the denied January claim over the closed one"


def test_offering_an_id_detail_is_not_an_unsupported_request():
    from app.engine import step
    from app.extract import regex_extract
    s = new_session(today="2026-03-05")
    run_turn(s, "Margaret Chen")
    nlu = {**regex_extract("I can give you her SSN last four if that helps verify"), "intent": "other_request"}
    step(s, "I can give you her SSN last four if that helps verify", nlu)
    assert "note_unsupported" not in s["trace"][-1]["acts"]
