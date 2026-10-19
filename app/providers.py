"""Provider-agnostic LLM access: detect which service a key belongs to, validate it with a free
model-list call, pick a sensible model, and send chat requests (Anthropic SDK or OpenAI-compatible HTTP)."""
import json
import os
import re
import time
from dataclasses import dataclass, field

import anthropic
import httpx

EFFORT = os.environ.get("LLM_EFFORT", "low")
PROBE_TIMEOUT = 8.0
CALL_TIMEOUT = 60.0


class LLMError(Exception):
    pass


# kind "anthropic" uses the official SDK; kind "openai" speaks the OpenAI chat-completions format.
CATALOG = {
    "anthropic": {"kind": "anthropic", "label": "Anthropic", "default_model": "claude-haiku-5-5",
                  "prefs": [r"^claude-haiku-(\d+(?:-\d+)?)$", r"^claude-sonnet-(\d+(?:-\d+)?)$", r"^claude-"]},
    "openrouter": {"kind": "openai", "label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
                   "validate_path": "/key", "default_model": "anthropic/claude-haiku-5.5"},
    "openai": {"kind": "openai", "label": "OpenAI", "base_url": "https://api.openai.com/v1",
               "prefs": [r"^gpt-(\d+(?:\.\d+)?)-mini$", r"^gpt-4o-mini$", r"^gpt-(\d+(?:\.\d+)?)$"]},
    "gemini": {"kind": "openai", "label": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "prefs": [r"^(?:models/)?gemini-(\d+(?:\.\d+)?)-flash$", r"gemini-[\d.]+-flash", r"gemini"]},
    "groq": {"kind": "openai", "label": "Groq", "base_url": "https://api.groq.com/openai/v1",
             "prefs": [r"^openai/gpt-oss-120b$", r"^llama-(\d+(?:\.\d+)?)-70b-versatile$", r"70b", r"llama"]},
    "xai": {"kind": "openai", "label": "xAI", "base_url": "https://api.x.ai/v1",
            "prefs": [r"^grok-(\d+(?:\.\d+)?)-mini$", r"^grok-(\d+(?:\.\d+)?)$", r"grok"]},
    "mistral": {"kind": "openai", "label": "Mistral", "base_url": "https://api.mistral.ai/v1",
                "prefs": [r"^mistral-small-latest$", r"^mistral-medium-latest$", r"^mistral-large-latest$"]},
    "deepseek": {"kind": "openai", "label": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "prefs": [r"^deepseek-chat$"]},
    "together": {"kind": "openai", "label": "Together AI", "base_url": "https://api.together.xyz/v1",
                 "prefs": [r"Llama-3\.3-70B-Instruct-Turbo$", r"Instruct-Turbo$", r"Instruct"]},
}

# Prefix -> providers to try first. Anything unrecognized is probed against every provider.
PREFIX_HINTS = [
    ("sk-ant-", ["anthropic"]),
    ("sk-or-", ["openrouter"]),
    ("gsk_", ["groq"]),
    ("xai-", ["xai"]),
    ("AIza", ["gemini"]),
    ("tgp_", ["together"]),
    ("sk-proj-", ["openai"]),
    ("sk-svcacct-", ["openai"]),
    ("sk-", ["openai", "deepseek"]),
]

ENV_KEYS = [
    ("LLM_API_KEY", None),
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("OPENROUTER_API_KEY", "openrouter"),
    ("OPENAI_API_KEY", "openai"),
    ("GEMINI_API_KEY", "gemini"),
    ("GOOGLE_API_KEY", "gemini"),
    ("GROQ_API_KEY", "groq"),
    ("XAI_API_KEY", "xai"),
    ("MISTRAL_API_KEY", "mistral"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("TOGETHER_API_KEY", "together"),
]

NON_CHAT = re.compile(r"embed|whisper|tts|audio|image|dall-e|moderation|realtime|transcribe|search|guard|vision-preview|rerank|ocr|codestral-embed|imagen|veo|aqa", re.I)


@dataclass
class Provider:
    name: str
    label: str
    kind: str
    key: str
    model: str
    base_url: str | None = None
    auth_token: bool = False
    json_mode: str = "json_schema"  # downgraded to json_object / prompt if the endpoint rejects schemas
    client: anthropic.Anthropic | None = field(default=None, repr=False)

    def describe(self) -> str:
        return f"{self.model} via {self.label}"


@dataclass
class Resolution:
    provider: Provider | None
    status: str  # "ok" | "invalid" | "none"
    message: str

    def public(self) -> dict:
        p = self.provider
        return {"status": self.status, "message": self.message,
                "provider": p.label if p else None, "model": p.model if p else None}


def _version_key(model_id: str):
    return tuple(int(n) for n in re.findall(r"\d+", model_id)[:3]) or (0,)


def pick_model(provider: str, ids: list[str]) -> str | None:
    ids = [i for i in ids if not NON_CHAT.search(i)]
    for pattern in CATALOG[provider].get("prefs", []):
        hits = [i for i in ids if re.search(pattern, i)]
        if hits:
            return max(hits, key=_version_key).removeprefix("models/")
    return ids[0].removeprefix("models/") if ids else None


def _probe_openai_compatible(name: str, base_url: str, key: str, validate_path: str = "/models"):
    """Returns (ok, model_ids_or_None, error). The model list call is free on every provider."""
    try:
        r = httpx.get(base_url.rstrip("/") + validate_path, headers={"Authorization": f"Bearer {key}"}, timeout=PROBE_TIMEOUT)
    except httpx.HTTPError as e:
        return False, None, f"could not reach {name}: {e.__class__.__name__}"
    if r.status_code in (401, 403):
        return False, None, f"{name} rejected the key (HTTP {r.status_code})"
    if r.status_code >= 400:
        return False, None, f"{name} returned HTTP {r.status_code}"
    if validate_path != "/models":
        return True, None, None
    try:
        return True, [m["id"] for m in r.json().get("data", [])], None
    except (ValueError, KeyError, TypeError):
        return True, [], None


def _probe_anthropic(key: str, auth_token: bool):
    client = anthropic.Anthropic(**({"auth_token": key} if auth_token else {"api_key": key}), timeout=PROBE_TIMEOUT, max_retries=0)
    try:
        ids = [m.id for m in client.models.list(limit=100)]
        return True, ids, None
    except anthropic.AuthenticationError:
        return False, None, "Anthropic rejected the key (HTTP 401)"
    except anthropic.PermissionDeniedError:
        return False, None, "Anthropic key lacks permission (HTTP 403)"
    except anthropic.APIStatusError as e:
        return False, None, f"Anthropic returned HTTP {e.status_code}"
    except anthropic.APIConnectionError:
        return False, None, "could not reach Anthropic"


def _candidates(key: str, hinted: str | None) -> list[str]:
    forced = os.environ.get("LLM_PROVIDER", "").lower()
    if forced in CATALOG:
        return [forced]
    if hinted:
        return [hinted]
    # A recognizable key is only ever sent to its own vendor; only unknown formats are probed broadly.
    for prefix, names in PREFIX_HINTS:
        if key.startswith(prefix):
            return names
    return [n for n in CATALOG if n not in ("anthropic", "openrouter", "openai", "groq", "xai", "gemini")] + ["openai"]


_cache: dict[tuple, Resolution] = {}


def resolve(key: str | None, model_override: str | None = None) -> Resolution:
    """Figure out which provider a key belongs to, validate it, and choose a model. Cached per key."""
    hinted, auth_token = None, False
    if not key:
        for env, prov in ENV_KEYS:
            if os.environ.get(env):
                key, hinted = os.environ[env].strip(), prov
                break
        else:
            if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
                key, hinted, auth_token = os.environ["ANTHROPIC_AUTH_TOKEN"].strip(), "anthropic", True
    if not key:
        return Resolution(None, "none", "No API key configured. Running offline (regex understanding + SOP templates).")

    model_override = model_override or os.environ.get("LLM_MODEL") or None
    cache_key = (key, model_override, os.environ.get("LLM_BASE_URL"))
    if cache_key in _cache:
        return _cache[cache_key]

    errors = []
    custom_base = os.environ.get("LLM_BASE_URL")
    if custom_base:
        ok, ids, err = _probe_openai_compatible("custom endpoint", custom_base, key)
        if ok:
            model = model_override or (ids[0] if ids else None)
            if model:
                res = Resolution(Provider("custom", "custom endpoint", "openai", key, model, base_url=custom_base), "ok", "")
                res.message = f"Key accepted: {res.provider.describe()}"
                _cache[cache_key] = res
                return res
            err = "custom endpoint has no models; set LLM_MODEL"
        errors.append(err)

    for name in _candidates(key, hinted):
        spec = CATALOG[name]
        if spec["kind"] == "anthropic":
            ok, ids, err = _probe_anthropic(key, auth_token)
        else:
            ok, ids, err = _probe_openai_compatible(spec["label"], spec["base_url"], key, spec.get("validate_path", "/models"))
        if not ok:
            errors.append(err)
            continue
        model = model_override or spec.get("default_model") or pick_model(name, ids or [])
        if not model_override and ids and model not in ids:
            model = pick_model(name, ids)
        if not model:
            errors.append(f"{spec['label']} accepted the key but no chat model was found; set LLM_MODEL")
            continue
        prov = Provider(name, spec["label"], spec["kind"], key, model, base_url=spec.get("base_url"), auth_token=auth_token)
        if prov.kind == "anthropic":
            prov.client = anthropic.Anthropic(**({"auth_token": key} if auth_token else {"api_key": key}), timeout=CALL_TIMEOUT, max_retries=2)
        res = Resolution(prov, "ok", f"Key accepted: {prov.describe()}")
        _cache[cache_key] = res
        return res

    res = Resolution(None, "invalid", "The API key was not accepted: " + "; ".join(dict.fromkeys(e for e in errors if e)))
    _cache[cache_key] = res
    return res


# ------------------------------------------------------------------ calls

def chat(p: Provider, system: str, user: str, max_tokens: int, schema: dict | None = None) -> str:
    if p.kind == "anthropic":
        return _chat_anthropic(p, system, user, max_tokens, schema)
    return _chat_openai(p, system, user, max_tokens, schema)


def _chat_anthropic(p: Provider, system, user, max_tokens, schema) -> str:
    params = {
        "model": p.model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": EFFORT, **({"format": {"type": "json_schema", "schema": schema}} if schema else {})},
    }
    if re.match(r"claude-(opus-5|fable-5|sonnet-5-5)", p.model):
        params |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    try:
        res = p.client.beta.messages.create(**params)
    except anthropic.AuthenticationError as e:
        raise LLMError("authentication failed (check the API key)") from e
    except anthropic.RateLimitError as e:
        raise LLMError("rate limited") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"connection error: {e}") from e
    if res.stop_reason == "refusal":
        raise LLMError(f"model refusal ({res.stop_details.category if res.stop_details else 'unknown'})")
    if res.stop_reason == "max_tokens":
        raise LLMError("model output truncated")
    return "".join(b.text for b in res.content if b.type == "text").strip()


def _openai_body(p: Provider, system, user, max_tokens, schema) -> dict:
    sys_text = system
    if schema and p.json_mode != "json_schema":
        sys_text += "\n\nRespond with ONLY a single JSON object (no prose, no code fences) that matches this JSON Schema:\n" + json.dumps(schema)
    body = {"model": p.model, "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]}
    reasoning_model = re.match(r"^(gpt-5|o\d)", p.model)
    body["max_completion_tokens" if p.name == "openai" else "max_tokens"] = max_tokens
    if p.name == "openai" and reasoning_model:
        body["reasoning_effort"] = EFFORT
    if p.name == "openrouter":
        body["reasoning"] = {"effort": EFFORT}
    if schema:
        if p.json_mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "caller_understanding", "strict": True, "schema": schema}}
            if p.name == "openrouter":
                body["provider"] = {"require_parameters": True}
        elif p.json_mode == "json_object":
            body["response_format"] = {"type": "json_object"}
    return body


def _post_with_retries(p: Provider, body: dict) -> httpx.Response:
    headers = {"Authorization": f"Bearer {p.key}", "X-Title": "SOP Claims Agent"}
    url = p.base_url.rstrip("/") + "/chat/completions"
    for attempt in range(3):
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=CALL_TIMEOUT)
        except httpx.HTTPError as e:
            if attempt == 2:
                raise LLMError(f"connection error: {e.__class__.__name__}") from e
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(1.5 * (attempt + 1))
            continue
        return r
    return r


def _chat_openai(p: Provider, system, user, max_tokens, schema) -> str:
    while True:
        r = _post_with_retries(p, _openai_body(p, system, user, max_tokens, schema))
        text = r.text[:300]
        # Not every OpenAI-compatible endpoint supports strict JSON schemas: step down and retry.
        if schema and r.status_code in (400, 404, 422) and p.json_mode != "prompt" and re.search(r"response_format|json|schema|structured", text, re.I):
            p.json_mode = {"json_schema": "json_object", "json_object": "prompt"}[p.json_mode]
            continue
        break
    if r.status_code == 401:
        raise LLMError(f"authentication failed ({p.label} rejected the key)")
    if r.status_code == 402:
        raise LLMError(f"{p.label} account has insufficient credits")
    try:
        data = r.json()
    except ValueError as e:
        raise LLMError(f"{p.label} returned non-JSON (HTTP {r.status_code})") from e
    if r.status_code >= 400 or (isinstance(data, dict) and data.get("error")):
        err = data.get("error") if isinstance(data, dict) else None
        msg = err.get("message") if isinstance(err, dict) else (err or text)
        raise LLMError(f"{p.label} error (HTTP {r.status_code}): {msg}")
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    if msg.get("refusal"):
        raise LLMError(f"model refusal: {msg['refusal']}")
    if choice.get("finish_reason") == "length":
        raise LLMError("model output truncated")
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content.strip()
