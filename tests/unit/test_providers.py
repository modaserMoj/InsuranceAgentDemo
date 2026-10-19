from app.llm import normalize_nlu
from app.providers import _candidates, pick_model


def test_prefix_routing_tries_the_likely_provider_first():
    assert _candidates("sk-ant-abc", None)[0] == "anthropic"
    assert _candidates("sk-or-v1-abc", None)[0] == "openrouter"
    assert _candidates("gsk_abc", None)[0] == "groq"
    assert _candidates("AIzaSyabc", None)[0] == "gemini"
    assert _candidates("sk-proj-abc", None)[0] == "openai"
    assert set(_candidates("sk-abc", None)[:2]) == {"openai", "deepseek"}
    assert "mistral" in _candidates("some-unknown-format", None), "unprefixed keys go to vendors without a prefix"
    assert "anthropic" not in _candidates("sk-proj-abc", None), "a recognizable key is never sent to other vendors"


def test_model_picking_prefers_newest_small_chat_model():
    ids = ["gpt-4o-mini", "gpt-5-mini", "gpt-5-mini-2025-08-07", "text-embedding-3-large", "gpt-5.1-mini", "whisper-1"]
    assert pick_model("openai", ids) == "gpt-5.1-mini"
    assert pick_model("gemini", ["models/gemini-2.0-flash", "models/gemini-2.5-flash", "models/text-embedding-004"]) == "gemini-2.5-flash"
    assert pick_model("anthropic", ["claude-opus-5-5", "claude-haiku-4-5", "claude-haiku-5-5"]) == "claude-haiku-5-5"


def test_normalizer_repairs_loose_json_from_weaker_models():
    out = normalize_nlu({
        "identity": {"full_name": "Margaret Chen", "dob": "null"},
        "intent": "why_denied",
        "case_hints": {"month": "1", "case_type": "Healthcare"},
        "emotion": "furious",
        "emotion_intensity": "7",
        "affirm": "yes",
    })
    assert out["identity"]["full_name"] == "Margaret Chen" and out["identity"]["dob"] is None
    assert out["intent"] is None, "unknown enum values are dropped, not passed to the engine"
    assert out["case_hints"]["month"] == 1
    assert out["emotion"] == "neutral" and out["emotion_intensity"] == 3
    assert out["affirm"] is None and out["in_scope"] is True
