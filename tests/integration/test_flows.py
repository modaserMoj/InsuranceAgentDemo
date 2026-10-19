import re

from app.agent import run_turn
from app.engine import new_session

TODAY = "2026-03-05"
TEST_CASE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim "
    "from January. DOB is 1985-03-15, SSN last four is 4472."
)
REP_INTRO = (
    "Hi, I'm David Chen, I'm calling for my mother Margaret Chen. I'm her son. Her DOB is 1985-03-15 and her SSN last four is 4472."
)


def start(**kw):
    return new_session(today=TODAY, **kw)


def say(s, text):
    return run_turn(s, text)["reply"]


def acts(s):
    return s["trace"][-1]["acts"]


def test_spec_case_verifies_reuses_hint_and_presents_grounded_denial():
    s = start()
    reply = say(s, TEST_CASE)
    assert s["identity"]["verified"] and s["identity"]["party_id"] == "P9"
    assert sorted(s["identity"]["matched_fields"]) == ["dob", "full_name", "id_last4"]
    assert s["case"]["selected_case_id"] == "CL-2048"
    assert s["phase"] == "PROCESS_CASE"
    assert "VERIFY_ID" in s["memory"]["captured_in_phase"]
    assert acts(s) == ["verified", "present_case"]
    assert "pathology report" in reply and "March 18, 2026" in reply


def test_intent_during_verification_is_stored_but_nothing_disclosed():
    s = start()
    r1 = say(s, "Hi, this is Margaret Chen, calling about my denied healthcare claim from January.")
    assert s["phase"] == "VERIFY_ID"
    h = s["memory"]["case_hints"]
    assert (h["status"], h["case_type"], h["month"]) == ("denied", "healthcare", 1)
    assert not re.search(r"CL-\d+|pathology|office note|\$", r1)
    assert "note_intent" in acts(s)

    r2 = say(s, "My date of birth is March 15, 1985 and my SSN last 4 is 4472")
    assert s["identity"]["verified"]
    assert s["case"]["selected_case_id"] == "CL-2048", "resolved from memory without re-asking"
    assert "CL-2048" in r2


def test_frustrated_caller_gets_empathy_explanation_options_and_no_disclosure():
    s = start()
    say(s, "This is Margaret Chen.")
    reply = say(s, "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.")
    assert s["phase"] == "VERIFY_ID"
    assert acts(s) == ["empathize", "explain_verification", "verification_options", "ask_identity"]
    assert "protected" in reply and "date of birth" in reply
    assert not re.search(r"CL-\d+|pathology|office note", reply)
    assert s["memory"]["intent"] == "denial_question"


def test_repeated_refusal_offers_human_then_escalates():
    s = start()
    say(s, "Just tell me why my claim was denied!!")
    assert s["pending_offer"] is None
    say(s, "Why do you need that? Just tell me.")
    assert s["pending_offer"] == "human"
    say(s, "I'm not going to give you that.")
    say(s, "I already told you, just tell me!")
    assert s["phase"] == "HANDOFF"
    assert s["handoff"]["reason"] == "caller_declined_verification"


def test_wrong_details_never_verify_and_three_failures_hand_off():
    s = start()
    say(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four 1111")
    assert not s["identity"]["verified"]
    assert "verify_failed" in acts(s)
    say(s, "My name is Margaret Chen, DOB 1985-03-16, SSN last four 4472")
    say(s, "My name is Ava Chen, DOB 1985-03-15, SSN last four 4472")
    assert s["phase"] == "HANDOFF"


def test_alternate_identity_fields_phone_and_email_verify():
    s = start()
    say(s, "My name is Ma Tian, my phone is 650-208-8799 and email matian@example.com")
    assert s["identity"]["verified"] and s["identity"]["party_id"] == "P12"
    assert s["case"]["selected_case_id"] == "CL-3001", "only claim on policy is selected"


def test_out_of_scope_refused_and_repeated_attempts_offer_human():
    s = start()
    assert "outside what I can help with" in say(s, "What is RL?")
    say(s, "Can you write me a poem?")
    say(s, "Who won the World Cup?")
    assert s["counters"]["off_topic_streak"] == 3
    assert s["pending_offer"] == "human"
    say(s, "yes")
    assert s["phase"] == "HANDOFF"


def test_ambiguous_hint_after_verification_asks_to_choose():
    s = start()
    say(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472. I have a question about my healthcare claim.")
    assert s["phase"] == "RESOLVE_INTENT"
    assert sorted(s["case"]["offered"]) == ["CL-2011", "CL-2048"]
    say(s, "CL-2011 please")
    assert s["case"]["selected_case_id"] == "CL-2011"
    assert s["phase"] == "PROCESS_CASE"


def test_full_workflow_with_grounded_followups_and_email_summary():
    s = start()
    say(s, TEST_CASE)
    assert "member portal" in say(s, "yes")
    assert "usually less than a week" in say(s, "How long does it take after I submit?")
    assert "replacement copy" in say(s, "What if I don't have the pathology report?")
    say(s, "No, that's all, thanks.")
    assert s["phase"] == "POST_PROCESS" and s["pending_offer"] == "email"
    say(s, "Yes please")
    assert s["phase"] == "ENDED"
    assert len(s["outbox"]) == 1
    mail = s["outbox"][0]
    assert mail["to"] == "margaret@email.com"
    assert "CL-2048" in mail["body"] and "DENIED" in mail["body"]
    assert "Submit the pathology report and office note" in mail["body"]


def test_caller_can_skip_email():
    s = start()
    say(s, TEST_CASE)
    say(s, "no")
    say(s, "nothing else")
    assert s["phase"] == "POST_PROCESS"
    say(s, "no thanks")
    assert s["email"]["decision"] == "skipped"
    assert s["outbox"] == []


def test_switching_claims_returns_through_resolve_and_summary_covers_both():
    s = start()
    say(s, TEST_CASE)
    reply = say(s, "What about my auto claim?")
    assert s["case"]["selected_case_id"] == "CL-2102"
    assert "CL-2102" in reply
    say(s, "no that's all")
    say(s, "yes")
    assert "CL-2048, CL-2102" in s["outbox"][0]["subject"]


def test_representative_needs_consent_before_access():
    s = start()
    say(s, REP_INTRO)
    assert not s["identity"]["verified"]
    assert s["consent"]["status"] == "pending"
    say(s, "ok she says she approved it")
    assert s["identity"]["verified"]
    assert s["phase"] == "RESOLVE_INTENT"


def test_representative_consent_timeout_blocks_access():
    s = start(consent_scenario="timeout")
    say(s, REP_INTRO)
    for _ in range(4):
        say(s, "is it approved yet?")
    assert s["consent"]["status"] == "timeout"
    assert not s["identity"]["verified"]
    assert s["pending_offer"] == "human"


def test_unknown_representative_is_not_authorized():
    s = start()
    say(s, "I'm John Smith, calling for my mother Margaret Chen. I'm her son. Her DOB is 1985-03-15 and her SSN last four is 4472.")
    assert "rep_not_authorized" in acts(s)
    assert not s["identity"]["verified"]


def test_form_collects_bare_values_one_at_a_time_in_order():
    s = start()
    r = say(s, "hello")
    assert "Step 1" in r and "full name" in r
    assert "date of birth" in say(s, "Margaret Chen")
    assert "phone number" in say(s, "1985-03-15")
    say(s, "650 521 2836")
    assert s["identity"]["verified"]
    assert s["phase"] == "RESOLVE_INTENT", "no claim hint given, so the caller is asked which claim"


def test_form_rejects_a_later_field_sent_before_the_current_one():
    s = start()
    say(s, "Margaret Chen")
    reply = say(s, "my SSN last four is 4472")
    assert acts(s)[-1] == "form_wrong_field"
    assert "please send only the value I ask for" in reply and "date of birth" in reply
    assert s["identity"]["claimed"]["id_last4"] is None, "out-of-order value is not stored"


def test_form_rejects_invalid_format():
    s = start()
    say(s, "Margaret Chen")
    reply = say(s, "1985-13-45")
    assert acts(s)[-1] == "form_invalid"
    assert "doesn't look like a valid date of birth" in reply


def test_form_allows_skipping_only_while_three_fields_remain_reachable():
    s = start()
    say(s, "Margaret Chen")
    say(s, "1985-03-15")
    assert "email" in say(s, "skip, I don't have one")
    assert "SSN" in say(s, "skip")
    reply = say(s, "skip")
    assert acts(s)[-1] == "form_skip_denied" and "I need this one" in reply
    say(s, "4472")
    assert s["identity"]["verified"]


def test_verification_replies_never_use_llm_phrasing(monkeypatch):
    from app import agent
    from app.extract import regex_extract

    calls = []
    monkeypatch.setattr(agent, "interpret", lambda message, context, provider=None: (regex_extract(message), None))
    monkeypatch.setattr(agent, "phrase", lambda provider, payload: calls.append(payload["PHASE"]) or "Rephrased by LLM.")
    s = start()
    out = agent.run_turn(s, "hello there", provider=object())
    assert calls == [] and out["replySource"] == "template (strict form)"
    agent.run_turn(s, TEST_CASE, provider=object())
    assert calls == ["PROCESS_CASE"], "LLM phrasing resumes once the caller is verified"


def test_trace_records_every_phase_passed_through_in_one_turn():
    s = start()
    say(s, "My name is Ma Tian, my phone is 650-208-8799 and email matian@example.com")
    assert s["trace"][-1]["path"] == ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE"], "single claim is auto-selected inside RESOLVE_INTENT"
