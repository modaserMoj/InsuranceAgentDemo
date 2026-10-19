from app.llm import LLMError
from app.nlu import interpret


def llm_out(**over):
    base = {
        "identity": {k: None for k in ["full_name", "dob", "phone", "email", "id_last4", "policy_number"]},
        "caller_role": None,
        "rep_name": None,
        "rep_relationship": None,
        "intent": None,
        "case_hints": {"case_id": None, "case_type": None, "status": None, "month": None, "year": None},
        "selected_case_id": None,
        "followup_topics": [],
        "dialog_act": "provide_info",
        "in_scope": True,
        "emotion": "neutral",
        "emotion_intensity": 0,
        "refuses_verification": False,
        "wants_human": False,
        "affirm": None,
        "wants_to_end": False,
        "question": None,
    }
    return base | over


def test_ungrounded_llm_identity_values_are_dropped():
    fake = lambda *_: llm_out(identity={**llm_out()["identity"], "full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"})
    nlu, err = interpret("Hi it's Margaret Chen, born March 15th 1985", {}, provider=object(), understand_fn=fake)
    assert nlu["identity"]["full_name"] == "Margaret Chen"
    assert nlu["identity"]["dob"] == "1985-03-15"
    assert nlu["identity"]["id_last4"] is None, "hallucinated SSN must not reach the verifier"
    assert "id_last4" in err


def test_grounded_llm_semantics_are_used():
    fake = lambda *_: llm_out(
        identity={**llm_out()["identity"], "email": "margaret@email.com"},
        intent="denial_question",
        case_hints={"case_id": None, "case_type": None, "status": "denied", "month": 1, "year": None},
    )
    nlu, _ = interpret("my email is margaret at email dot com and it's about the claim they rejected back in jan", {}, provider=object(), understand_fn=fake)
    assert nlu["identity"]["email"] == "margaret@email.com"
    assert nlu["intent"] == "denial_question"
    assert nlu["case_hints"]["month"] == 1


def test_llm_failure_falls_back_to_regex():
    def boom(*_):
        raise LLMError("authentication failed")

    nlu, err = interpret("DOB 1985-03-15", {}, provider=object(), understand_fn=boom)
    assert nlu["source"] == "regex"
    assert nlu["identity"]["dob"] == "1985-03-15"
    assert "fell back" in err
