import re

from .data import norm_name
from .extract import regex_extract
from .intents import CONFIDENT
from .llm import LLMError, understand


def _digits(s) -> str:
    return re.sub(r"\D", "", str(s or ""))


def grounded_in_message(field: str, value, message: str) -> bool:
    """The LLM may normalize ("March 15th" -> 1985-03-15) but must not invent: every identity
    value has to be traceable to the caller's own words."""
    if not value:
        return False
    msg = message.lower()
    msg_digits = _digits(message)
    value = str(value)
    if field == "full_name":
        words = norm_name(message).split()
        return all(t in words for t in norm_name(value).split())
    if field == "dob":
        return value[:4] in msg
    if field == "phone":
        return _digits(value)[-7:] in msg_digits
    if field == "email":
        spoken = re.sub(r"\s+dot\s+", ".", re.sub(r"\s+at\s+", "@", msg))
        return value.split("@")[0].lower() in spoken
    if field == "id_last4":
        return bool(re.fullmatch(r"\d{4}", value)) and value in msg_digits
    if field == "policy_number":
        return bool(_digits(value)) and _digits(value) in msg_digits
    return False


def interpret(message: str, context: dict, provider=None, understand_fn=understand):
    rx = regex_extract(message)
    if provider is None:
        return rx, None
    retrieval = rx["intent_retrieval"]
    context = {**context, "INTENT_CANDIDATES": [{"intent": c["intent"], "score": c["score"], "closest_example": c["closest_example"]}
                                                for c in retrieval["candidates"]]}
    try:
        llm = understand_fn(provider, message, context)
    except (LLMError, ValueError) as e:
        return rx, f"NLU fell back to regex: {e}"

    llm_id = llm.get("identity") or {}
    identity, dropped = {}, []
    for f in rx["identity"]:
        if rx["identity"][f]:
            identity[f] = rx["identity"][f]
        elif llm_id.get(f) and grounded_in_message(f, llm_id[f], message):
            identity[f] = llm_id[f]
        else:
            identity[f] = None
            if llm_id.get(f):
                dropped.append(f)
    # A regex name grab is crude; prefer the LLM's reading of who is who when it is grounded.
    if llm_id.get("full_name") and grounded_in_message("full_name", llm_id["full_name"], message):
        identity["full_name"] = llm_id["full_name"]
    elif llm.get("caller_role") == "representative" and llm.get("rep_name"):
        if identity["full_name"] and norm_name(identity["full_name"]) == norm_name(llm["rep_name"]):
            identity["full_name"] = None

    # The LLM decides the intent; retrieval fills a gap only when it is confident, and disagreements are logged.
    intent = llm.get("intent")
    retrieved, confidence = rx["intent"], retrieval["confidence"]
    if intent is None and retrieved and confidence >= CONFIDENT:
        intent = retrieved
    retrieval = {**retrieval, "retrieved": retrieved, "llm": llm.get("intent"),
                 "agree": retrieved is None or llm.get("intent") is None or retrieved == llm.get("intent")}

    hints = llm.get("case_hints") or {}
    nlu = {
        **llm,
        "identity": identity,
        "rep_name": llm.get("rep_name") or rx["rep_name"],
        "case_hints": {
            "case_id": hints.get("case_id") or rx["case_hints"]["case_id"],
            "case_type": hints.get("case_type"),
            "status": hints.get("status"),
            "month": hints.get("month"),
            "year": hints.get("year"),
        },
        "intent": intent,
        "intent_retrieval": retrieval,
        "wants_human": bool(llm.get("wants_human") or rx["wants_human"]),
        "switch_claim": bool(llm.get("switch_claim") or rx["switch_claim"]),
        "abusive": bool(llm.get("abusive") or rx["abusive"]),
        "refuses_verification": bool(llm.get("refuses_verification") or rx["refuses_verification"]),
        "source": "llm+regex",
    }
    return nlu, (f"dropped ungrounded LLM identity fields: {', '.join(dropped)}" if dropped else None)
