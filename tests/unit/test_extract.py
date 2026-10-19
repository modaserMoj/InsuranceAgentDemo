"""Unit: deterministic extraction of identity values, case hints and conversational signals."""
import pytest

from app.extract import extract_bare_value, leading_name, looks_like_attempt, regex_extract


@pytest.mark.parametrize("text, dob", [
    ("DOB is 1985-03-15", "1985-03-15"), ("born 03/15/1985", "1985-03-15"),
    ("March 15, 1985", "1985-03-15"), ("15th of March 1985", "1985-03-15"),
])
def test_date_of_birth_formats(text, dob):
    assert regex_extract(text)["identity"]["dob"] == dob


def test_spec_message_extracts_everything_without_mixing_up_dates():
    r = regex_extract("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
                      "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.")
    assert r["identity"] == {"full_name": "Margaret Chen", "dob": "1985-03-15", "phone": None, "email": None,
                             "id_last4": "4472", "policy_number": "POL-9921"}
    assert r["case_hints"]["status"] == "denied" and r["case_hints"]["case_type"] == "healthcare"
    assert r["case_hints"]["month"] == 1, "January, not the March from the DOB"
    assert r["caller_role"] == "policyholder"


def test_representative_name_is_kept_separate_from_policyholder():
    r = regex_extract("Hi, I'm David Chen, I'm calling for my mother Margaret Chen. I'm her son.")
    assert r["caller_role"] == "representative" and r["rep_name"] == "David Chen"
    assert r["rep_relationship"] == "son" and r["identity"]["full_name"] == "Margaret Chen"


@pytest.mark.parametrize("field, text, value", [
    ("full_name", "margaret chen", "Margaret Chen"), ("phone", "(650) 521-2836", "+16505212836"),
    ("id_last4", "4472", "4472"), ("email", "margaret@email.com", "margaret@email.com"), ("dob", "1985-03-15", "1985-03-15"),
])
def test_bare_values_for_the_requested_field(field, text, value):
    assert extract_bare_value(field, text) == value


def test_bare_name_rejects_greetings():
    assert extract_bare_value("full_name", "hello there") is None


def test_leading_name_and_malformed_attempts():
    assert leading_name("Margaret Chen born 1985-03-15") == "Margaret Chen"
    assert looks_like_attempt("phone", "50-521-2836") and not looks_like_attempt("phone", "what?")


@pytest.mark.parametrize("text, key", [
    ("Can I talk to a real person?", "wants_human"), ("I already told you, just tell me!", "refuses_verification"),
    ("no that's all", "wants_to_end"), ("can you check on my other claim", "switch_claim"), ("you stupid bot", "abusive"),
])
def test_conversational_signals(text, key):
    assert regex_extract(text)[key]
