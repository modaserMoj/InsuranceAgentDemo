"""Unit: identity matching against the policyholder fixtures."""
from app.data import verify_identity


def claimed(**kw):
    base = {"full_name": None, "dob": None, "phone": None, "email": None, "id_last4": None, "policy_number": None}
    return base | kw


def test_three_matching_fields_verify():
    r = verify_identity(claimed(full_name="Margaret Chen", dob="1985-03-15", id_last4="4472"))
    assert r["verified"] and r["party_id"] == "P9"


def test_two_fields_are_not_enough():
    assert not verify_identity(claimed(full_name="Margaret Chen", dob="1985-03-15"))["verified"]


def test_any_mismatch_blocks_even_with_three_matches():
    r = verify_identity(claimed(full_name="Margaret Chen", dob="1985-03-15", id_last4="4472", phone="+16500000000"))
    assert not r["verified"] and r["mismatched"] == ["phone"]


def test_policy_number_narrows_lookup_but_never_counts():
    assert not verify_identity(claimed(full_name="Margaret Chen", dob="1985-03-15", policy_number="POL-9921"))["verified"]
    assert not verify_identity(claimed(full_name="Margaret Chen", dob="1985-03-15", id_last4="4472", policy_number="POL-1044"))["verified"]


def test_aliases_match_speech_to_text_variants():
    r = verify_identity(claimed(full_name="Yaven Li", email="yawen.li@example.com", id_last4="5317"))
    assert r["verified"] and r["party_id"] == "P13"


def test_phone_formats_are_normalised():
    r = verify_identity(claimed(full_name="Ma Tian", phone="(650) 208-8799", email="MATIAN@example.com"))
    assert r["verified"] and r["party_id"] == "P12"


def test_fields_from_different_people_do_not_combine():
    r = verify_identity(claimed(full_name="Margaret Chen", dob="1964-09-10", id_last4="9180"))
    assert not r["verified"]
