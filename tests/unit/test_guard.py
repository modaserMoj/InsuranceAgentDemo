"""Unit: the deterministic disclosure guard."""
from app.engine import new_session
from app.guard import find_leaks


def start():
    return new_session(today="2026-03-05")


def test_guard_flags_account_data_but_not_callers_own_words():
    s = start()
    s["transcript"].append({"role": "user", "text": "my denied claim, something about an office note"})
    assert find_leaks("You mentioned an office note.", s) == []
    assert find_leaks("Claim CL-2048 was denied because the review file did not include the pathology report", s)
    assert find_leaks("Your email is margaret@email.com", s)
    s["identity"].update(verified=True, party_id="P9")
    assert find_leaks("Claim CL-2048 appeal deadline March 18.", s) == []
    assert find_leaks("Claim CL-3001 needs a diagnosis report", s)
