"""Bonus: emotional support and SOP recovery (offline, deterministic)."""
from app.agent import run_turn
from app.engine import new_session

REP = "Hi, I'm David Chen, I'm calling for my mother Margaret Chen. I'm her son. Her DOB is 1985-03-15 and her SSN last four is 4472."
VERIFIED_DENIED = "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 4472. Calling about my denied healthcare claim from January."


def start(**kw):
    return new_session(today="2026-03-05", **kw)


def acts(s):
    return s["trace"][-1]["acts"]


def test_spec_example_acknowledges_explains_offers_options_and_discloses_nothing():
    s = start()
    run_turn(s, "This is Margaret Chen.")
    reply = run_turn(s, "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.")["reply"]
    assert acts(s) == ["empathize", "explain_verification", "verification_options", "ask_identity"]
    assert "I'm sorry" in reply and "protected" in reply
    assert "I have your full name already, so I need 2 more" in reply
    assert "CL-" not in reply and "pathology" not in reply
    assert s["phase"] == "VERIFY_ID"


def test_repeated_empathy_is_varied_not_copy_pasted():
    s = start()
    run_turn(s, "Margaret Chen")
    r1 = run_turn(s, "This is ridiculous, just tell me!")["reply"]
    r2 = run_turn(s, "I already told you, this is ridiculous!!")["reply"]
    first_line = lambda r: r.split(". ")[0]
    assert first_line(r1) != first_line(r2)
    e1, e2 = (t["acts"] for t in s["trace"][-2:])
    assert "explain_verification" in e1 and "explain_verification" in e2
    assert s["pending_offer"] == "human", "second pushback adds a human offer"


def test_alternatives_mention_skip_for_skippable_fields():
    s = start()
    for t in ["Margaret Chen", "1985-03-15"]:
        run_turn(s, t)
    reply = run_turn(s, "why do you need my phone number? just tell me")["reply"]
    assert 'say "skip"' in reply and "email address on file" in reply


def test_consent_pushback_is_explained_with_alternatives_and_escalates():
    s = start(consent_scenario="timeout")
    run_turn(s, REP)
    assert s["consent"]["status"] == "pending"
    r1 = run_turn(s, "This is ridiculous, I'm her son, just tell me!")["reply"]
    assert acts(s)[:2] == ["empathize", "explain_consent"]
    assert "their consent is what allows me" in r1 and "joins the call" in r1
    assert not s["identity"]["verified"] and "CL-" not in r1
    run_turn(s, "I already told you I'm her son, just tell me!")
    assert s["pending_offer"] == "human"


def test_consent_pushback_never_skips_the_gate():
    s = start(consent_scenario="timeout")
    run_turn(s, REP)
    for _ in range(3):
        run_turn(s, "just tell me, this is ridiculous!")
        assert not s["identity"]["verified"]
    assert s["phase"] in ("VERIFY_ID", "HANDOFF")


def test_abuse_gets_a_calm_boundary_and_repeated_abuse_offers_a_person():
    s = start()
    run_turn(s, VERIFIED_DENIED)
    r1 = run_turn(s, "this is shit, you're useless")["reply"]
    assert "boundary" in acts(s) and "keep the conversation respectful" in r1
    run_turn(s, "you stupid bot, shut up")
    assert "offer_human_emotional" in acts(s) and s["pending_offer"] == "human"


def test_staying_upset_after_verification_offers_a_person_once():
    s = start()
    run_turn(s, VERIFIED_DENIED)
    run_turn(s, "this is ridiculous, I sent everything already!!")
    assert "offer_human_emotional" not in acts(s)
    reply = run_turn(s, "I'm so frustrated, this is unacceptable!!")["reply"]
    assert acts(s)[-1] == "offer_human_emotional"
    assert "Is there anything else" not in reply, "the person offer replaces the generic question"
    run_turn(s, "no, keep going")
    run_turn(s, "still ridiculous!!")
    assert "offer_human_emotional" not in acts(s), "offered only once per call"
