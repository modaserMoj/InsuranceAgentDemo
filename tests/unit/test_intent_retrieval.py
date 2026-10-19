"""Unit: hybrid intent retrieval quality on phrasings that are NOT in the example bank."""
from app.extract import regex_extract
from app.intents import CONFIDENT, MIN_SCORE, classify, retrieval_mode

HELD_OUT = [
    ("can you look up how my claim is doing", "status_inquiry"),
    ("is the car claim still being worked on", "status_inquiry"),
    ("hey, any news on where my claim stands?", "status_inquiry"),
    ("why on earth was this refused", "denial_question"),
    ("what caused my claim to be rejected", "denial_question"),
    ("how come you rejected it", "denial_question"),
    ("do I need to send the original or is a copy fine", "document_submission"),
    ("where do I send the lab report", "document_submission"),
    ("can I just take a picture of the office note with my phone", "document_submission"),
    ("what am I supposed to do from here", "next_steps"),
    ("how long until the review is done", "next_steps"),
    ("so what happens after I upload everything", "next_steps"),
    ("what amount did the insurer actually pay", "payment_question"),
    ("why did they only pay part of it", "payment_question"),
    ("I want to formally dispute the outcome", "appeal"),
    ("am I still in time to appeal", "appeal"),
    ("what does copay mean", "general_insurance_question"),
    ("explain out-of-pocket max to me", "general_insurance_question"),
    ("I got a new email address, can you update it", "other_request"),
    ("is acupuncture covered on my plan", "other_request"),
    ("I want to cancel my insurance", "other_request"),
    ("I moved, need to change my mailing address", "other_request"),
]
NOISE = ["What is RL?", "write me a poem about cats", "who won the world cup", "yes", "no thanks", "hello", "ok", "4472"]


def test_retrieval_uses_embeddings():
    assert retrieval_mode() == "hybrid (BM25 + embeddings)"


def test_held_out_accuracy_is_high():
    correct = [classify(text)["intent"] == want for text, want in HELD_OUT]
    assert sum(correct) / len(correct) >= 0.85, [t for (t, _), ok in zip(HELD_OUT, correct) if not ok]


def test_noise_and_bare_values_match_no_intent():
    for text in NOISE:
        r = classify(text)
        assert r["intent"] is None and r["confidence"] < MIN_SCORE, (text, r)


def test_identity_statements_are_not_mistaken_for_update_requests():
    assert regex_extract("My name is Ma Tian, my phone is 650-208-8799 and email matian@example.com")["intent"] is None
    assert regex_extract("can you change my phone number")["intent"] == "other_request"


def test_threshold_ordering():
    assert MIN_SCORE < CONFIDENT
