import json

from .data import FOLLOWUP_TOPICS
from .extract import CASE_TYPES, DIALOG_ACTS, EMOTIONS, STATUSES
from .intents import INTENT_DESCRIPTIONS, INTENTS
from .providers import LLMError, Provider, chat

__all__ = ["LLMError", "normalize_nlu", "phrase", "understand"]

DOC_STATUSES = ["have", "cannot_get", "submitted"]
IDENTITY_FIELDS = ["full_name", "dob", "phone", "email", "id_last4", "policy_number"]


def _nullable(t: str) -> dict:
    return {"type": [t, "null"]}


def _nullable_enum(values: list) -> dict:
    return {"type": ["string", "null"], "enum": [*values, None]}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


NLU_SCHEMA = _obj({
    "identity": _obj({f: _nullable("string") for f in IDENTITY_FIELDS}),
    "caller_role": _nullable_enum(["policyholder", "representative"]),
    "rep_name": _nullable("string"),
    "rep_relationship": _nullable("string"),
    "intent": _nullable_enum(INTENTS),
    "additional_intents": {"type": "array", "items": {"type": "string", "enum": INTENTS}},
    "document_updates": {"type": "array", "items": _obj({
        "document": {"type": "string"},
        "status": {"type": "string", "enum": DOC_STATUSES},
    })},
    "case_hints": _obj({
        "case_id": _nullable("string"),
        "case_type": _nullable_enum(CASE_TYPES),
        "status": _nullable_enum(STATUSES),
        "month": _nullable("integer"),
        "year": _nullable("integer"),
    }),
    "selected_case_id": _nullable("string"),
    "followup_topics": {"type": "array", "items": {"type": "string", "enum": FOLLOWUP_TOPICS}},
    "dialog_act": {"type": "string", "enum": DIALOG_ACTS},
    "in_scope": {"type": "boolean"},
    "emotion": {"type": "string", "enum": EMOTIONS},
    "emotion_intensity": {"type": "integer"},
    "refuses_verification": {"type": "boolean"},
    "wants_human": {"type": "boolean"},
    "affirm": _nullable("boolean"),
    "wants_to_end": {"type": "boolean"},
    "switch_claim": {"type": "boolean"},
    "abusive": {"type": "boolean"},
    "question": _nullable("string"),
})

INTENT_LINES = "\n".join(f"    - {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items())

NLU_SYSTEM = f"""You are the language-understanding module of an insurance claims support phone agent.
You never talk to the caller. You convert the caller's latest message into JSON. A separate deterministic SOP engine decides what happens next.

Extraction rules:
- identity: only values the caller actually stated in THIS message, about the POLICYHOLDER. Normalize dob to YYYY-MM-DD, phone to +1XXXXXXXXXX, policy number to POL-NNNN, id_last4 to 4 digits (SSN or national ID last four). Never infer or invent values. If the caller is a representative, their own name goes in rep_name, not identity.full_name.
- caller_role: "representative" if they call on behalf of someone else (son, daughter, spouse, caregiver...); "policyholder" if they say it is their own policy; else null.
- intent: what the caller wants, one of:
{INTENT_LINES}
  Capture it even during identity verification. "Denied claim" alone implies denial_question. A question about whether THEIR plan covers something is other_request (we have no coverage data), while a question about what an insurance term means is general_insurance_question. Use null only for messages with no request at all (greetings, yes/no, identity values).
  INTENT_CANDIDATES in CONTEXT are the closest matches from a retrieval index over example phrasings, with similarity scores. Treat them as evidence, not as the answer.
- additional_intents: if the message contains MORE than one request ("why was it denied and how do I appeal?"), the other ones in the order asked, excluding the main intent. Usually empty.
- document_updates: when the caller says something about a requested document, one entry per document using the document name from CONTEXT.current_case_documents: status "have" (they have it / can get it), "cannot_get" (they don't have it, lost it, the provider won't give it), or "submitted" (already sent/uploaded). Empty otherwise.
- case_hints: claim details the caller mentions (e.g. "my denied healthcare claim from January" -> case_type healthcare, status denied, month 1).
- selected_case_id: if OFFERED_CASES is given and the caller picks one ("the first one", "the car one", "CL-2102"), return its case_id; otherwise null. Only ids from OFFERED_CASES.
- followup_topics: which of these the caller is asking about right now: {", ".join(FOLLOWUP_TOPICS)}. Use missing_required_material_alternatives when they cannot get a requested document.
- in_scope: true for anything related to their insurance policy, claims, documents, payments, identity verification, consent, the call itself (greetings, thanks, yes/no answers, complaints about the process, asking for a human). false for unrelated topics (general knowledge, coding, trivia, other companies, jokes, ML terms like "what is RL").
- emotion + emotion_intensity (0 none, 1 mild, 2 clear, 3 severe).
- refuses_verification: caller pushes back on giving identity info or demands details without verifying.
- wants_human: caller asks for a human / real person / supervisor / transfer.
- affirm: if the agent's last message asked a yes/no question, true for yes, false for no; otherwise null.
- wants_to_end: caller indicates they have no more questions / wants to wrap up.
- abusive: the caller insults, curses at, or threatens the agent (not just strong frustration about the situation).
- switch_claim: caller wants to look at a different claim than CONTEXT.current_case_id, or to see their list of claims ("my other claim", "what other claims do I have", "go back to the list"), even without saying which one.
- question: the caller's question restated in one short sentence, or null."""


def _enum(value, allowed, default=None):
    if isinstance(value, str):
        value = value.strip().lower()
    return value if value in allowed else default


def _opt_str(value):
    if isinstance(value, str) and value.strip() and value.strip().lower() not in ("null", "none", "n/a"):
        return value.strip()
    return None


def _opt_int(value):
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _bool(value, default=False):
    return value if isinstance(value, bool) else default


def normalize_nlu(raw: dict) -> dict:
    """Coerce whatever a model returned into the exact shape the engine expects. Strict-schema
    providers already comply; this keeps weaker JSON modes and other vendors from breaking the SOP."""
    raw = raw if isinstance(raw, dict) else {}
    ident = raw.get("identity") if isinstance(raw.get("identity"), dict) else {}
    hints = raw.get("case_hints") if isinstance(raw.get("case_hints"), dict) else {}
    month = _opt_int(hints.get("month"))
    affirm = raw.get("affirm")
    return {
        "identity": {f: _opt_str(ident.get(f)) for f in IDENTITY_FIELDS},
        "caller_role": _enum(raw.get("caller_role"), ["policyholder", "representative"]),
        "rep_name": _opt_str(raw.get("rep_name")),
        "rep_relationship": _opt_str(raw.get("rep_relationship")),
        "intent": _enum(raw.get("intent"), INTENTS),
        "additional_intents": list(dict.fromkeys(i for i in (raw.get("additional_intents") or []) if i in INTENTS)),
        "document_updates": [
            {"document": d["document"].strip(), "status": d["status"]}
            for d in (raw.get("document_updates") or [])
            if isinstance(d, dict) and isinstance(d.get("document"), str) and d.get("status") in DOC_STATUSES
        ],
        "case_hints": {
            "case_id": _opt_str(hints.get("case_id")),
            "case_type": _enum(hints.get("case_type"), CASE_TYPES),
            "status": _enum(hints.get("status"), STATUSES),
            "month": month if month and 1 <= month <= 12 else None,
            "year": _opt_int(hints.get("year")),
        },
        "selected_case_id": _opt_str(raw.get("selected_case_id")),
        "followup_topics": [t for t in (raw.get("followup_topics") or []) if t in FOLLOWUP_TOPICS],
        "dialog_act": _enum(raw.get("dialog_act"), DIALOG_ACTS, "other"),
        "in_scope": _bool(raw.get("in_scope"), True),
        "emotion": _enum(raw.get("emotion"), EMOTIONS, "neutral"),
        "emotion_intensity": max(0, min(3, _opt_int(raw.get("emotion_intensity")) or 0)),
        "refuses_verification": _bool(raw.get("refuses_verification")),
        "wants_human": _bool(raw.get("wants_human")),
        "affirm": affirm if isinstance(affirm, bool) else None,
        "wants_to_end": _bool(raw.get("wants_to_end")),
        "switch_claim": _bool(raw.get("switch_claim")),
        "abusive": _bool(raw.get("abusive")),
        "question": _opt_str(raw.get("question")),
    }


def understand(provider: Provider, message: str, context: dict) -> dict:
    user = f"CONTEXT:\n{json.dumps(context, indent=2)}\n\nCALLER MESSAGE:\n{message}"
    text = chat(provider, NLU_SYSTEM, user, 4000, NLU_SCHEMA)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise LLMError("understanding step returned no JSON")
    try:
        raw = json.loads(text[start : end + 1])
    except ValueError as e:
        raise LLMError("understanding step returned invalid JSON") from e
    return {**normalize_nlu(raw), "source": "llm"}


PHRASE_SYSTEM = """You are "Alex", a warm, professional customer-service representative for an insurance company, speaking with a caller in a chat.
A deterministic SOP engine has already decided WHAT you must do this turn. You decide only HOW to say it.

Hard rules:
1. Accomplish every item in DIRECTIVES, in order, and nothing else. Do not skip any required question or offer listed there.
2. State facts ONLY if they appear in FACTS. Never invent claim details, amounts, dates, reasons, policies, phone numbers, or promises. If FACTS lacks an answer, say you don't have that detail and offer what the directives allow.
3. Never reveal claim or account information unless FACTS contains it. Never confirm or deny which identity field was wrong.
4. If EMOTION is not neutral, open with one short, genuine acknowledgement of the feeling before moving on. Don't over-apologize or be syrupy.
5. Sound natural and human: plain language, contractions, no markdown headers, no JSON. Short paragraphs; a short list only when giving options or several steps. Usually 2-6 sentences.
6. Don't greet again mid-conversation and don't repeat information already said in RECENT_TURNS unless asked.
   Never reuse a sentence you already said in RECENT_TURNS word for word; vary closing questions ("Anything else on this claim?", "What else can I help with?").
DRAFT is a safe, correct template answer; you may rephrase it, but keep all of its facts and its questions.
If the caller asked something that neither FACTS nor DRAFT answers (for example a prediction, a policy rule, or a detail not in the file), say plainly that you don't have that information, and start your reply with the marker [[NO_ANSWER]] (it is removed before the caller sees it). Never guess.
Exception: a general_answer directive explicitly allows general insurance knowledge (never claims about the caller's own plan); for that directive the DRAFT is only an offline placeholder, so replace it with the actual explanation."""


def phrase(provider: Provider, payload: dict) -> str:
    return chat(provider, PHRASE_SYSTEM, json.dumps(payload, indent=2), 3000)
