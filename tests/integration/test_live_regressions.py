"""Regressions from the first hands-on test session (offline mode)."""
import pytest

from app.agent import run_turn
from app.engine import new_session


def start():
    return new_session(today="2026-03-05")


def acts(s):
    return s["trace"][-1]["acts"]


def test_name_at_start_of_bundle_is_recognized():
    s = start()
    run_turn(s, "Margaret Chen born 1985-03-15 with phone number 650-521-2836")
    assert s["identity"]["verified"]


def test_malformed_phone_is_explained_not_silently_reasked():
    s = start()
    run_turn(s, "Margaret Chen")
    run_turn(s, "1985-03-15")
    reply = run_turn(s, "50-521-2836")["reply"]
    assert acts(s) == ["form_invalid"]
    assert "doesn't look like a valid phone number" in reply and "10 digits" in reply
    assert "Thank you" not in reply


def test_weak_intent_guess_is_not_remembered():
    s = start()
    for t in ["Margaret Chen", "1985-03-15", "skip"]:
        run_turn(s, t)
    run_turn(s, "mailto:margaret@email.com")
    assert s["identity"]["verified"]
    assert s["memory"]["intent"] is None, "a 0.40 retrieval guess from an email address must not become the caller's intent"


@pytest.mark.parametrize("pick, expected", [
    ("2048", "CL-2048"), ("cl 2048", "CL-2048"), ("the second one", "CL-2011"), ("1", "CL-2048"),
    ("the dental one", "CL-1899"), ("the car claim", "CL-2102"), ("the denied one", "CL-2048"),
])
def test_offered_claim_can_be_picked_naturally(pick, expected):
    s = start()
    run_turn(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472")
    assert s["case"]["offered"] == ["CL-2048", "CL-2011", "CL-1899", "CL-2102"]
    run_turn(s, pick)
    assert s["case"]["selected_case_id"] == expected


def test_ambiguous_pick_still_asks():
    s = start()
    run_turn(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472")
    run_turn(s, "the healthcare one")
    assert s["phase"] == "RESOLVE_INTENT", "two healthcare claims on file, so the agent must ask again"


def test_opening_a_denied_claim_always_states_the_denial():
    s = start()
    run_turn(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472. How do I upload the documents for my denied healthcare claim from January?")
    reply = s["transcript"][-1]["text"]
    assert "was denied because" in reply, "status first"
    assert "member portal" in reply, "then the requested document guidance"


@pytest.mark.parametrize("text, intent", [
    ("How do I upload the documents for my denied healthcare claim from January?", "document_submission"),
    ("I'm calling about my denied claim, how do I appeal?", "appeal"),
    ("I'm calling about my denied healthcare claim from January", "denial_question"),
    ("what's the status of my auto claim", "status_inquiry"),
    ("Just tell me why my claim was denied", "denial_question"),
])
def test_claim_description_does_not_hide_the_request(text, intent):
    from app.extract import regex_extract
    r = regex_extract(text)
    assert intent in [r["intent"], *r["additional_intents"]], r["intent"]


def test_switching_to_another_claim_without_naming_it():
    s = start()
    run_turn(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472")
    run_turn(s, "the dental one")
    assert s["case"]["selected_case_id"] == "CL-1899"
    run_turn(s, "why not they pay me max?")
    reply = run_turn(s, "can you check on my other claim")["reply"]
    assert s["phase"] == "RESOLVE_INTENT"
    assert s["case"]["offered"] == ["CL-2048", "CL-2011", "CL-2102"], "current claim is left out of the list"
    assert "offer_human_unanswered" not in s["trace"][-1]["acts"] and "CL-2102" in reply
    run_turn(s, "the car one")
    assert s["case"]["selected_case_id"] == "CL-2102" and s["phase"] == "PROCESS_CASE"


def test_switch_when_it_is_the_only_claim_says_so():
    s = start()
    run_turn(s, "My name is Ma Tian, my phone is 650-208-8799 and email matian@example.com")
    assert s["case"]["selected_case_id"] == "CL-3001"
    reply = run_turn(s, "what other claims do I have?")["reply"]
    assert "only claim on your policy" in reply


def test_thanks_after_the_call_ended_does_not_reopen_or_reoffer_email():
    s = start()
    run_turn(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472. Calling about my denied healthcare claim from January.")
    run_turn(s, "that's all")
    run_turn(s, "yes")
    assert s["phase"] == "ENDED" and len(s["outbox"]) == 1
    run_turn(s, "Thanks so much, have a good day!")
    assert s["phase"] == "ENDED" and len(s["outbox"]) == 1
    assert "offer_email" not in s["trace"][-1]["acts"]


def test_typing_the_on_file_email_counts_as_yes():
    s = start()
    run_turn(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472. Calling about my denied healthcare claim from January.")
    run_turn(s, "that's all")
    run_turn(s, "please send it to margaret@email.com")
    assert s["email"]["decision"] == "sent"


def test_fishing_for_the_failed_field_is_explained_not_answered():
    s = start()
    run_turn(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four 1111")
    reply = run_turn(s, "Which one was wrong, the DOB or the SSN?")["reply"]
    assert "can't tell you which detail didn't match" in reply


def test_reason_is_noted_only_once_during_verification():
    s = start()
    run_turn(s, "Hi, I'm calling about my denied healthcare claim from January")
    run_turn(s, "Margaret Chen, it's about my denied claim from January")
    assert "note_intent" not in s["trace"][-1]["acts"]


def test_declining_the_email_while_wrapping_up_is_respected():
    s = start()
    run_turn(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472. Calling about my denied healthcare claim from January.")
    run_turn(s, "Fine, I'm done with this. Don't send me an email either.")
    assert s["phase"] == "ENDED" and s["email"]["decision"] == "skipped"
    assert "offer_email" not in s["trace"][-1]["acts"]


def test_asking_for_the_summary_while_wrapping_up_sends_it():
    s = start()
    run_turn(s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472. Calling about my denied healthcare claim from January.")
    run_turn(s, "That's all, please email me a summary.")
    assert s["email"]["decision"] == "sent"
