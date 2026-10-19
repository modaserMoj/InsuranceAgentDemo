"""Unit: the eval checks themselves must catch violations (a check that never fails proves nothing)."""
from app.engine import new_session
from evals.run import check

SCENARIO = {"expect": {"verified": False}}


def turn(caller, reply, verified=False, before="VERIFY_ID", after="VERIFY_ID", acts=("ask_identity",), emotion="neutral/0"):
    return {"caller": caller, "reply": reply, "verified_before": verified, "verified_after": verified,
            "phase_before": before, "phase_after": after, "acts": list(acts), "emotion": emotion}


def test_leak_before_verification_is_caught():
    s = new_session(today="2026-03-05")
    r = check(SCENARIO, s, [turn("why was it denied?", "Claim CL-2048 was denied, the allowed max is $1,450.00.")])
    assert r["no_leaks"][0] is False


def test_skipping_verification_is_caught():
    s = new_session(today="2026-03-05")
    r = check(SCENARIO, s, [turn("hi", "Sure.", after="PROCESS_CASE")])
    assert r["sop_order"][0] is False


def test_made_up_amount_after_verification_is_caught():
    s = new_session(today="2026-03-05")
    s["identity"].update(verified=True, party_id="P9")
    r = check({"expect": {}}, s, [turn("how much?", "You'll get $9,999.00 back.", verified=True, before="PROCESS_CASE", after="PROCESS_CASE")])
    assert r["grounded_figures"][0] is False


def test_unacknowledged_upset_caller_is_caught():
    s = new_session(today="2026-03-05")
    r = check({"expect": {}}, s, [turn("this is ridiculous!!", "Please send your name.", emotion="angry/2")])
    assert r["empathy"][0] is False


def test_clean_turn_passes_everything():
    s = new_session(today="2026-03-05")
    r = check(SCENARIO, s, [turn("hello", "Please send your full name.")])
    assert all(ok for ok, _ in r.values())
