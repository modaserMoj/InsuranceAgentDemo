"""Deterministic SOP engine: phases, gates, cross-phase memory, case matching, grounded answers, email.

Session state is a plain dict so it serializes straight to the inspector UI.
"""
import os
import re
import uuid
from datetime import date, datetime, timezone

from .data import (
    FOLLOWUP_TOPICS,
    PII_FIELDS,
    check_consent,
    consent_sequence,
    document_guidance,
    field_definitions,
    find_representative,
    followup_fallback,
    followup_guidance,
    get_claim,
    get_policyholder,
    list_claims,
    mask_email,
    norm_name,
    verify_identity,
)
from .extract import extract_bare_value, leading_name, looks_like_attempt
from .intents import CONFIDENT

PHASES = ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"]
FORM_ORDER = ["full_name", "dob", "phone", "email", "id_last4"]
VERIFY_FIELDS_REQUIRED = 3
FIELD_PROMPT = {
    "full_name": "please send {who} full name (first and last name).",
    "dob": "please send {who} date of birth in YYYY-MM-DD format (for example 1970-01-31).",
    "phone": "please send {who} phone number on file (10 digits).",
    "email": "please send {who} email address on file.",
    "id_last4": "please send the last four digits of {who} SSN or national ID.",
}
SKIPPABLE_HINT = {"phone", "email", "id_last4"}
FIELD_FORMAT_HINT = {
    "full_name": "It should be your first and last name.",
    "dob": "It should be a real date in YYYY-MM-DD format.",
    "phone": "It should have 10 digits, including the area code.",
    "email": "It should look like name@example.com.",
    "id_last4": "It should be exactly 4 digits.",
}
CATCH_ALL = {"other_request", "general_insurance_question"}
UNANSWERED_HANDOFF_STREAK = 2
ID_TALK_RE = re.compile(r"\b(ssn|social security|last (?:4|four)|national id|date of birth|dob|birthday|phone|e-?mail|verify|verification|identity)\b", re.I)
EMAIL_DECLINE_RE = re.compile(r"\b(?:no|don'?t|do not|skip|without)\b[^.?!]{0,25}\b(?:e-?mail|summary)", re.I)
EMAIL_ACCEPT_RE = re.compile(r"\b(?:e-?mail|send) me (?:a |the )?(?:summary|recap|copy)|\bsend (?:me )?(?:a |the )?summary\b", re.I)
WHICH_FIELD_RE = re.compile(r"\bwhich\b.{0,30}\b(wrong|incorrect|match|failed|mistake)|\bwhat (?:was|did i get) wrong\b", re.I)
HUMAN_FOR_UNANSWERED = ("Since I don't have those details in your file, a claims representative can look into it directly. "
                        "Would you like me to connect you?")
SUPPORTED_SUMMARY = ("checking a claim's status, explaining why a claim was denied, the documents needed and how to send them, "
                     "payment amounts, appeals, and next steps")
GENERAL_ANSWER_INSTRUCTION = (
    "Answer the caller's general insurance question (FACTS.general_question) in 2-4 plain sentences using general, "
    "widely accepted insurance knowledge. Say it is general information. Do NOT say anything about the caller's own "
    "plan, coverage, benefits, or costs beyond FACTS; add that a representative can confirm how it works on their plan.")
SKIP_RE = re.compile(r"\b(skip|don'?t have (?:one|it|a|an)|do not have|no (?:phone|email|ssn)|don'?t know|can'?t remember|not sure|rather not)\b", re.I)
MAX_VERIFY_ATTEMPTS = 3
OFF_TOPIC_HANDOFF_STREAK = 3
PERSUASION_OFFER_HUMAN = 2
PERSUASION_ESCALATE = 4
CONSENT_EXPLANATIONS = [
    "Because you're calling on someone else's behalf, their consent is what allows me to share their health and claim "
    "information with you. It's their legal protection, the same protection you'd want for your own records.",
    "I know it's hard to wait when you're trying to help. I can't skip the policyholder's consent, but it usually only "
    "takes them a moment to approve.",
    "I understand, and I'm sorry this is slowing you down. The consent step is required for every representative, so "
    "I can't make an exception here.",
]
CONSENT_ALTERNATIVES = ("Your options are: the policyholder approves the request on their phone, the policyholder "
                        "joins the call and verifies themselves, or I connect you with a human representative.")
VERIFY_EXPLANATIONS = [
    "Claim details are protected health and financial information, so I have to confirm I'm speaking with the "
    "policyholder before I can share anything. It protects you from someone else getting into your claim. It only "
    "takes a moment.",
    "I really do want to get you that answer. The only thing between us is this quick check, which protects your "
    "account. Once it's done, I'll go straight to your claim.",
    "I understand if you'd prefer not to do this here. I can't share claim details without verification, but I can "
    "connect you with a person who can help another way.",
]

FIELD_LABEL = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number on file",
    "email": "email address on file",
    "id_last4": "last four digits of your SSN or national ID",
}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
TOPIC_LABEL = {
    "denial": "Why the claim was denied and what is needed to reconsider it",
    "status": "The current claim status",
    "payment": "The claim's payment amounts and what they mean",
    "documents": "How to prepare and submit the requested documents",
    "appeal": "The appeal deadline and how to appeal",
    "missing_required_material_alternatives": "What to do if a requested document isn't available",
    "submission_timing": "When to submit the missing documents",
    "processing_time_after_submission": "How long review takes after submission",
    "submission_method": "Where and how to submit documents",
    "file_format_requirements": "File format and quality requirements",
    "receipt_confirmation": "How to confirm the documents were received",
    "general": "General questions about the claim",
}
# Each emotion has escalating variants: a repeat acknowledgement never sounds copy-pasted, and gets more conciliatory.
EMPATHY = {
    "frustrated": [
        "I understand this is frustrating, and I'm sorry for the hassle.",
        "I know this is taking longer than you'd like, and I appreciate your patience with me.",
        "I'm really sorry this has been so frustrating. You shouldn't have to fight this hard to get an answer.",
    ],
    "angry": [
        "I hear you, and I'm sorry this has been such an aggravating experience.",
        "I understand why you're upset, and I'm not trying to make this harder for you.",
        "You have every right to be upset, and I'm sorry. I want to get this resolved for you.",
    ],
    "anxious": [
        "I can tell this is stressful, and I want to help you get it sorted out.",
        "I know waiting on this is worrying. Let's take it one step at a time.",
        "I understand how much is riding on this for you. I'll make sure you know exactly what to do next.",
    ],
    "confused": [
        "No problem, let me make this clearer.",
        "Let me put that a different way.",
        "Sorry, I haven't explained that well. Here it is simply.",
    ],
    "sad": [
        "I'm sorry you're dealing with this. I'll do what I can to help.",
        "That sounds really hard, and I'm sorry.",
        "I'm so sorry. Let's see what we can do together.",
    ],
}
BOUNDARY = ("I want to help you get this sorted out, and I'm going to keep working on it with you. "
            "I do need us to keep the conversation respectful.")
EMOTIONAL_HUMAN_OFFER = ("If it would help to talk this through with a person, I can connect you with a claims "
                         "representative right now. Would you like that?")
UPSET_OFFER_AFTER = 2
DOC_HELP_QUESTION = " Would you like me to walk you through how to submit those documents?"


def _empty_hints() -> dict:
    return {"case_id": None, "case_type": None, "status": None, "month": None, "year": None}


def new_session(consent_scenario: str = "default", today: str | None = None) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "today": today or os.environ.get("DEMO_TODAY") or date.today().isoformat(),
        "phase": "VERIFY_ID",
        "turn": 0,
        "identity": {
            "claimed": {f: None for f in [*PII_FIELDS, "policy_number"]},
            "caller_role": None,
            "rep_name": None,
            "rep_relationship": None,
            "verified": False,
            "party_id": None,
            "pending_party": None,
            "matched_fields": [],
            "skipped": [],
            "failed_attempts": 0,
        },
        "consent": {"required": False, "scenario": consent_scenario, "checks": 0, "status": None},
        "memory": {"intent": None, "case_hints": _empty_hints(), "next_hints": None, "pending_question": None, "captured_in_phase": {}},
        "case": {"selected_case_id": None, "offered": [], "touched": [], "documents": {}},
        "counters": {"off_topic_streak": 0, "persuasion": 0, "frustration": 0, "unanswered_streak": 0,
                     "consent_pushback": 0, "upset": 0, "abuse": 0, "emotional_offer_made": False, "empathy": {}},
        "emotion": {"label": "neutral", "intensity": 0},
        "pending_offer": None,
        "discussed": [],
        "followups": [],
        "email": {"offered": False, "decision": None, "sent_to": None},
        "handoff": None,
        "transcript": [],
        "trace": [],
        "outbox": [],
    }


def _goto(s, phase):
    """Change phase and record it, so a turn that passes through several phases shows the whole path."""
    s["phase"] = phase
    s.setdefault("_path", []).append(phase)


# ------------------------------------------------------------------ helpers

def _add(items: list, item):
    if item and item not in items:
        items.append(item)


def _discuss(s, claim, topic):
    _add(s["case"]["touched"], claim["case_id"])
    _add(s["discussed"], f"{claim['case_id']}: {TOPIC_LABEL.get(topic, TOPIC_LABEL['general'])}")


def _fmt_date(iso: str) -> str:
    y, m, d = (int(x) for x in iso.split("-"))
    return f"{MONTH_NAMES[m - 1]} {d}, {y}"


def _days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _money(v) -> str:
    return f"${float(v):,.2f}"


def _first_name(s) -> str:
    h = get_policyholder(s["identity"]["party_id"])
    return h["name"].split()[0] if h else ""


def _describe_hints(h: dict) -> str:
    bits = [h.get("status"), h.get("case_type"), "claim",
            f"from {MONTH_NAMES[h['month'] - 1]}" if h.get("month") else None,
            f"({h['case_id']})" if h.get("case_id") else None]
    return " ".join(b for b in bits if b)


def _hints_empty(h: dict) -> bool:
    return not any(h.get(k) for k in ("case_id", "case_type", "status", "month", "year"))


def _part(act, say, facts=None) -> dict:
    return {"act": act, "say": say, "facts": facts or {}}


# ------------------------------------------------------------------ memory

def _intent_is_confident(nlu) -> bool:
    """LLM intents are trusted; retrieval-only intents must clear the confidence bar before they are remembered."""
    if nlu.get("source") != "regex":
        return True
    return (nlu.get("intent_retrieval") or {}).get("confidence", 0) >= CONFIDENT


def _absorb(s, nlu) -> dict:
    ident = s["identity"]
    if nlu.get("caller_role") and not ident["verified"]:
        ident["caller_role"] = nlu["caller_role"]
    if nlu.get("rep_name"):
        ident["rep_name"] = nlu["rep_name"]
    if nlu.get("rep_relationship"):
        ident["rep_relationship"] = nlu["rep_relationship"]

    mem = s["memory"]
    new_hints = {k: v for k, v in (nlu.get("case_hints") or {}).items() if v not in (None, "")}
    got_hints = bool(new_hints)
    if got_hints or nlu.get("intent"):
        if s["phase"] in ("PROCESS_CASE", "POST_PROCESS", "ENDED"):
            mem["next_hints"] = new_hints
        else:
            mem["case_hints"].update(new_hints)
        if nlu.get("intent") and _intent_is_confident(nlu):
            mem["intent"] = nlu["intent"]
        if not ident["verified"]:
            mem["captured_in_phase"][s["phase"]] = {"intent": mem["intent"], "case_hints": dict(mem["case_hints"])}
    if not ident["verified"] and nlu.get("question") and nlu.get("in_scope") and nlu.get("intent"):
        mem["pending_question"] = nlu["question"]

    s["emotion"] = {"label": nlu.get("emotion") or "neutral", "intensity": nlu.get("emotion_intensity") or 0}
    if s["emotion"]["label"] in ("angry", "frustrated"):
        s["counters"]["frustration"] += 1
    if ident["verified"] and s["emotion"]["label"] in ("angry", "frustrated", "anxious", "sad") and s["emotion"]["intensity"] >= 2:
        s["counters"]["upset"] += 1
    return {"got_hints": got_hints, "got_intent": bool(nlu.get("intent"))}


# ------------------------------------------------------------------ shared parts

def _empathy_part(s):
    label, intensity = s["emotion"]["label"], s["emotion"]["intensity"]
    if label == "neutral" or intensity < 2 or label not in EMPATHY:
        return None
    used = s["counters"]["empathy"].get(label, 0)
    s["counters"]["empathy"][label] = used + 1
    variants = EMPATHY[label]
    say = variants[-1] if intensity >= 3 else variants[min(used, len(variants) - 1)]
    return _part("empathize", say, {"caller_emotion": label, "intensity": intensity})


def _emotional_escalation(s, nlu, parts):
    """De-escalation and escalation independent of phase: set a respectful boundary on abuse, and offer a person
    (once) when a caller stays strongly upset, is severely upset, or is abusive."""
    if s["phase"] in ("HANDOFF", "ENDED") or s["counters"]["emotional_offer_made"]:
        return
    severe = s["emotion"]["intensity"] >= 3
    if nlu.get("abusive"):
        s["counters"]["abuse"] += 1
        parts.append(_part("boundary", BOUNDARY))
    if not (s["counters"]["upset"] >= UPSET_OFFER_AFTER or severe or s["counters"]["abuse"] >= 2):
        return
    if s["pending_offer"] == "human":
        return
    if s["phase"] == "VERIFY_ID" and not (severe or s["counters"]["abuse"] >= 2):
        return  # during verification the persuasion ladder already decides when to offer a person
    parts[:] = [p for p in parts if p["act"] != "ask_more"]
    for p in parts:
        p["say"] = p["say"].replace(DOC_HELP_QUESTION, "").rstrip()
    s["pending_offer"] = "human"
    s["counters"]["emotional_offer_made"] = True
    parts.append(_part("offer_human_emotional", EMOTIONAL_HUMAN_OFFER))


def _collected(s) -> list[str]:
    return [f for f in FORM_ORDER if s["identity"]["claimed"][f]]


def _current_field(s):
    ident = s["identity"]
    return next((f for f in FORM_ORDER if not ident["claimed"][f] and f not in ident["skipped"]), None)


def _can_skip(s) -> bool:
    """A field may be skipped only while enough fields remain to reach the 3-field threshold."""
    remaining = [f for f in FORM_ORDER if not s["identity"]["claimed"][f] and f not in s["identity"]["skipped"]]
    return len(_collected(s)) + len(remaining) - 1 >= VERIFY_FIELDS_REQUIRED


def _field_prompt(s, field) -> str:
    who = "the policyholder's" if s["identity"]["caller_role"] == "representative" else "your"
    p = FIELD_PROMPT[field].format(who=who)
    return p[0].upper() + p[1:]


def _ask_identity_part(s, first=False, thank=True):
    cur = _current_field(s)
    n = len(_collected(s))
    skip_hint = ' If you don\'t have it, reply "skip".' if _can_skip(s) and cur in SKIPPABLE_HINT else ""
    if first or n == 0:
        say = (f"Before I can access any claim details, I need to verify your identity. I'll ask for {VERIFY_FIELDS_REQUIRED} details, "
               f"one at a time. Step 1: {_field_prompt(s, cur)}{skip_hint}")
    else:
        say = f"{'Thank you. ' if thank else ''}Step {n + 1} of {VERIFY_FIELDS_REQUIRED}: {_field_prompt(s, cur)}{skip_hint}"
    return _part("ask_identity", say, {"identity_field_requested": cur, "identity_fields_received": _collected(s)})


def _reprompt_for(s):
    phase = s["phase"]
    if phase == "VERIFY_ID":
        return _ask_identity_part(s)
    if phase == "RESOLVE_INTENT":
        return _part("reprompt", "What can I help you with on your policy or claims today?")
    if phase == "PROCESS_CASE":
        return _part("reprompt", f"Is there anything else about claim {s['case']['selected_case_id']} I can help with?")
    if phase == "POST_PROCESS":
        return _part("reprompt", "Would you like me to email you a summary of today's conversation? Just say yes or no.")
    return _part("reprompt", "Is there anything else about your policy or claims I can help with?")


def _handoff(s, reason, parts):
    s["handoff"] = {
        "reason": reason,
        "at_phase": s["phase"],
        "verified": s["identity"]["verified"],
        "party_id": s["identity"]["party_id"],
        "caller_role": s["identity"]["caller_role"],
        "intent": s["memory"]["intent"],
        "case_hints": s["memory"]["case_hints"],
        "selected_case_id": s["case"]["selected_case_id"],
        "emotion": s["emotion"]["label"],
    }
    _goto(s, "HANDOFF")
    parts.append(_part(
        "handoff",
        "I'm connecting you with a human claims representative now. I've passed along a note about what we've covered so you won't have to start over. For your security, they may need to confirm your identity again.",
        {"handoff_reason": reason},
    ))


# ------------------------------------------------------------------ VERIFY_ID

def _mark_verified(s, party_id, parts):
    ident = s["identity"]
    ident.update(verified=True, party_id=party_id, pending_party=None)
    s["counters"]["persuasion"] = 0
    _goto(s, "RESOLVE_INTENT")
    who = (ident["rep_name"] or "") if ident["caller_role"] == "representative" else _first_name(s)
    parts.append(_part("verified", f"Thank you{', ' + who.split()[0] if who else ''}. You're verified.",
                       {"verified": True, "policyholder_first_name": _first_name(s)}))


def _consent_approved(s, parts):
    parts.append(_part("consent_approved", "The policyholder's consent came through.", {"consent_status": "approved"}))
    _mark_verified(s, s["identity"]["pending_party"], parts)


def _representative_gate(s, parts):
    ident = s["identity"]
    if not ident["rep_name"] or not ident["rep_relationship"]:
        parts.append(_part("ask_rep_details", "Since you're calling on someone else's behalf, could you tell me your full name and your relationship to the policyholder?"))
        return
    rep = find_representative(ident["rep_name"], ident["pending_party"])
    if not rep or norm_name(rep["relationship"]) != norm_name(ident["rep_relationship"]):
        ident["pending_party"] = None
        s["pending_offer"] = "human"
        parts.append(_part(
            "rep_not_authorized",
            "I'm sorry, but I don't see you listed as an authorized representative on this policy, so I can't share claim details. The policyholder can join the call to verify, or I can connect you with a human representative who can explain how to get authorized. Would you like me to transfer you?",
        ))
        return
    s["consent"]["required"] = True
    status = check_consent(s["consent"])
    if status == "approved":
        return _consent_approved(s, parts)
    parts.append(_part(
        "consent_requested",
        "Thanks. You're listed as an authorized representative. For the policyholder's protection, I've just sent a consent request to the phone number on file. Once it's approved I can go over the claim with you. Let me know when it's done.",
        {"consent_status": status},
    ))


def _handle_consent_pending(s, parts) -> bool:
    status = check_consent(s["consent"])
    if status == "approved":
        _consent_approved(s, parts)
        return True
    if status != "pending" or s["consent"]["checks"] >= len(consent_sequence(s["consent"]["scenario"])):
        s["consent"]["status"] = "timeout"
        s["identity"]["pending_party"] = None
        s["pending_offer"] = "human"
        parts.append(_part(
            "consent_timeout",
            "I still haven't received the policyholder's consent, and the request has now timed out. Without it I can't share claim details. The policyholder can call us directly, or I can connect you with a human representative. Would you like a transfer?",
            {"consent_status": "timeout"},
        ))
        return False
    parts.append(_part(
        "consent_pending",
        "I'm still waiting on the policyholder's consent. It's pending on their phone. Once they tap approve, just let me know and I'll check again.",
        {"consent_status": "pending"},
    ))
    return False


def _verify_phase(s, nlu, parts, ctx):
    ident = s["identity"]
    if ident["pending_party"] and s["consent"]["status"] == "pending":
        pushback = nlu.get("refuses_verification") or (s["emotion"]["label"] in ("angry", "frustrated") and s["emotion"]["intensity"] >= 2)
        if pushback:
            s["counters"]["consent_pushback"] += 1
            n = s["counters"]["consent_pushback"]
            parts.append(_part("explain_consent", CONSENT_EXPLANATIONS[min(n, len(CONSENT_EXPLANATIONS)) - 1]))
        if _handle_consent_pending(s, parts):
            _resolve_phase(s, nlu, parts)
            return
        if pushback and s["consent"]["status"] == "pending":
            if s["counters"]["consent_pushback"] >= PERSUASION_ESCALATE:
                return _handoff(s, "representative_declined_consent_wait", parts)
            alt = CONSENT_ALTERNATIVES
            if s["counters"]["consent_pushback"] >= PERSUASION_OFFER_HUMAN:
                s["pending_offer"] = "human"
                alt += " Would you like me to transfer you to a representative now?"
            parts.append(_part("consent_alternatives", "\n\n" + alt))
        return
    if ident["pending_party"]:
        _representative_gate(s, parts)
        if ident["verified"]:
            _resolve_phase(s, nlu, parts)
        return

    intake = _form_intake(s, nlu)
    if intake in ("invalid", "wrong_field", "skip_denied"):
        _note_intent_part(s, ctx, parts)
        parts.append(s.pop("_form_reply"))
        return
    if intake == "skipped":
        nxt = _current_field(s)
        parts.append(_part("form_skip", f"No problem, we'll use your {FIELD_LABEL[nxt]} instead." if nxt else "No problem."))

    if len(_collected(s)) >= VERIFY_FIELDS_REQUIRED:
        result = verify_identity(ident["claimed"])
        ident["matched_fields"] = result["matched"]
        if result["verified"]:
            if ident["caller_role"] == "representative":
                ident["pending_party"] = result["party_id"]
                _representative_gate(s, parts)
                if ident["verified"]:
                    _resolve_phase(s, nlu, parts)
                return
            _mark_verified(s, result["party_id"], parts)
            _resolve_phase(s, nlu, parts)
            return
        ident["failed_attempts"] += 1
        ident["claimed"] = {f: None for f in ident["claimed"]}
        ident["skipped"] = []
        if ident["failed_attempts"] >= MAX_VERIFY_ATTEMPTS:
            parts.append(_part("verify_locked", "I'm sorry, I wasn't able to verify your identity with the information provided."))
            return _handoff(s, "identity_verification_failed", parts)
        parts.append(_part(
            "verify_failed",
            f"I'm sorry, I wasn't able to verify your identity with those details. For your security, I can't say which "
            f"detail didn't match. Let's start over and check each one carefully. "
            f"Step 1: {_field_prompt(s, FORM_ORDER[0])}",
            {"attempts_remaining": MAX_VERIFY_ATTEMPTS - ident["failed_attempts"]},
        ))
        return

    if intake in ("accepted", "skipped"):
        _note_intent_part(s, ctx, parts)
        parts.append(_ask_identity_part(s))
        return

    if intake == "exhausted":
        s["pending_offer"] = "human"
        parts.append(_part(
            "form_exhausted",
            "I need at least three of these details to verify you, so I can't continue without them. I can connect you with a human representative who can help another way. Would you like me to do that?",
        ))
        return

    if nlu.get("wants_to_end") and not ctx["new_identity"]:
        _goto(s, "ENDED")
        parts.append(_part("closing", "No problem. If you need help with your claim later, just reach out. Take care!"))
        return

    if WHICH_FIELD_RE.search(s["_message"]):
        parts.append(_part("explain_no_field_hint", "For your security, I can't tell you which detail didn't match, since "
                           "that would help someone guess their way into an account."))
        parts.append(_ask_identity_part(s, thank=False))
        return

    pushed_back = bool(nlu.get("refuses_verification"))
    if pushed_back:
        s["counters"]["persuasion"] += 1
        if s["counters"]["persuasion"] >= PERSUASION_ESCALATE:
            return _handoff(s, "caller_declined_verification", parts)
        n = s["counters"]["persuasion"]
        parts.append(_part("explain_verification", VERIFY_EXPLANATIONS[min(n, len(VERIFY_EXPLANATIONS)) - 1]))
        parts.append(_options_part(s))
        opts = _ask_identity_part(s, thank=False)
        opts["say"] = "\n\n" + opts["say"]
        if s["counters"]["persuasion"] >= PERSUASION_OFFER_HUMAN:
            s["pending_offer"] = "human"
            opts["say"] += " If you'd rather not do this with me, I can also connect you with a human representative. Would you prefer that?"
        parts.append(opts)
        return

    _note_intent_part(s, ctx, parts)
    if s["turn"] == 1 or not (ctx["got_hints"] or ctx["got_intent"]):
        parts.append(_ask_identity_part(s, first=s["turn"] == 1))
    else:
        parts.append(_part("ask_identity", f"To continue, please send only this: {_field_prompt(s, _current_field(s))}",
                           {"identity_field_requested": _current_field(s)}))


def _options_part(s):
    """Spell out the allowed verification alternatives so the caller isn't stuck on one field."""
    have = _collected(s)
    cur = _current_field(s)
    alternatives = [f for f in FORM_ORDER if f != cur and not s["identity"]["claimed"][f] and f not in s["identity"]["skipped"]]
    need = VERIFY_FIELDS_REQUIRED - len(have)
    lead = f"I have your {' and '.join(FIELD_LABEL[f] for f in have)} already, so I need {need} more." if have else f"Any {VERIFY_FIELDS_REQUIRED} of these work: {', '.join(FIELD_LABEL[f] for f in FORM_ORDER)}."
    tail = ""
    if cur and _can_skip(s) and alternatives:
        tail = f" If you don't have your {FIELD_LABEL[cur]} handy, just say \"skip\" and we can use your {' or '.join(FIELD_LABEL[f] for f in alternatives[:2])} instead."
    return _part("verification_options", "\n\n" + lead + tail, {"verification_fields_accepted": [FIELD_LABEL[f] for f in FORM_ORDER]})


def _note_intent_part(s, ctx, parts):
    if not (ctx["got_hints"] or ctx["got_intent"]):
        return
    if (s["memory"]["intent"] == "other_request" and ID_TALK_RE.search(s["_message"])
            and not re.search(r"\b(change|update|new|reset|correct|fix|replace|lost)\b", s["_message"], re.I)):
        s["memory"]["intent"] = None
        return
    if s["memory"]["intent"] == "other_request":
        s["memory"]["intent"] = None
        parts.append(_part("note_unsupported", f"Just so you know, that's not something I can handle in this chat. I help with {SUPPORTED_SUMMARY}. You can ask for a human representative at any time."))
        return
    if s["memory"]["intent"] == "general_insurance_question" and _hints_empty(s["memory"]["case_hints"]):
        parts.append(_part("note_intent", "Good question. I'll answer it as soon as you're verified."))
        return
    h = s["memory"]["case_hints"]
    noted = f"{s['memory']['intent']}|{_describe_hints(h)}"
    if s["memory"].get("noted") == noted:
        return
    s["memory"]["noted"] = noted
    say = ("Got it. I've noted what you're calling about, and I'll pick that up as soon as you're verified."
           if _hints_empty(h) else
           f"Got it. I've noted that you're calling about your {_describe_hints(h)}, and I'll pull that up as soon as you're verified.")
    parts.append(_part("note_intent", say, {"caller_stated_hint": _describe_hints(h)}))


def _form_intake(s, nlu) -> str:
    """Strict form semantics. A message is accepted only if it contains the field currently being asked
    for; other valid values in that same message are kept too (so a full one-message answer works).
    A message that offers other fields without the current one is rejected as a whole."""
    ident = s["identity"]
    cur = _current_field(s)
    if cur is None:
        return "exhausted" if len(_collected(s)) < VERIFY_FIELDS_REQUIRED else "none"
    message = s["_message"]
    offered = {f: v for f, v in (nlu.get("identity") or {}).items() if v and f in ident["claimed"]}
    if cur not in offered:
        bare = extract_bare_value(cur, message) or (leading_name(message) if cur == "full_name" else None)
        if bare:
            offered[cur] = bare
    if cur not in offered and looks_like_attempt(cur, message):
        s["_form_reply"] = _part("form_invalid", f"That doesn't look like a valid {FIELD_LABEL[cur]}. {FIELD_FORMAT_HINT[cur]} "
                                 f"{_field_prompt(s, cur)}", {"identity_field_requested": cur})
        return "invalid"

    if cur in offered:
        bad = [f for f, v in offered.items() if f != "policy_number" and not _valid_value(f, v)]
        if cur in bad:
            s["_form_reply"] = _part("form_invalid", f"That doesn't look like a valid {FIELD_LABEL[cur]}. {FIELD_FORMAT_HINT[cur]} "
                                     f"{_field_prompt(s, cur)}", {"identity_field_requested": cur})
            return "invalid"
        for f, v in offered.items():
            if f not in bad and not ident["claimed"][f]:
                ident["claimed"][f] = v
                if f in ident["skipped"]:
                    ident["skipped"].remove(f)
        return "accepted"

    if SKIP_RE.search(message):
        if not _can_skip(s):
            s["_form_reply"] = _part("form_skip_denied",
                                     f"I'm sorry, I need this one to have enough details to verify you. {_field_prompt(s, cur)}",
                                     {"identity_field_requested": cur})
            return "skip_denied"
        ident["skipped"].append(cur)
        return "skipped" if _current_field(s) else "exhausted"

    if any(f != "policy_number" for f in offered):
        s["_form_reply"] = _part(
            "form_wrong_field",
            f"Let's go one step at a time, and please send only the value I ask for. {_field_prompt(s, cur)}",
            {"identity_field_requested": cur},
        )
        return "wrong_field"
    if offered.get("policy_number") and not ident["claimed"]["policy_number"]:
        ident["claimed"]["policy_number"] = offered["policy_number"]
    return "none"


def _valid_value(field, value) -> bool:
    value = str(value).strip()
    if field == "full_name":
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){1,3}", value))
    if field == "dob":
        try:
            d = date.fromisoformat(value)
        except ValueError:
            return False
        return 1900 <= d.year and d <= date.today()
    if field == "phone":
        return len(re.sub(r"\D", "", value)) in (10, 11)
    if field == "email":
        return bool(re.fullmatch(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", value))
    if field == "id_last4":
        return bool(re.fullmatch(r"\d{4}", value))
    return True


def _catch_all_part(s, nlu, intent):
    """Requests outside the claim workflow: general concepts get a general answer, everything else is redirected."""
    q = nlu.get("question") or s["memory"].get("pending_question") or s["_message"]
    if intent == "general_insurance_question":
        _add(s["discussed"], f"General insurance question: {q}")
        return {**_part(
            "general_answer",
            "That's a general insurance question, and I can't explain it in detail in this mode. A representative can walk you "
            "through it and confirm how it applies to your plan.",
            {"general_question": q},
        ), "instruction": GENERAL_ANSWER_INSTRUCTION}
    _add(s["discussed"], f"Request outside this chat: {q}")
    _add(s["followups"], f"Contact a human representative for: {q}")
    s["pending_offer"] = "human"
    return _part(
        "redirect_unsupported",
        f"I'm sorry, I can't help with that in this chat. Here I can help with {SUPPORTED_SUMMARY}. "
        "Would you like help with one of those, or should I connect you with a human representative?",
        {"supported_requests": SUPPORTED_SUMMARY},
    )


# ------------------------------------------------------------------ RESOLVE_INTENT

def _claim_line(c) -> str:
    return f"{c['case_id']}: {c['case_type']} claim filed {_fmt_date(c['created_at'])}, {c['status']}"


def _match_claims(claims, h):
    out = []
    for c in claims:
        y, m, _ = (int(x) for x in c["created_at"].split("-"))
        if ((not h.get("case_type") or c["case_type"] == h["case_type"])
                and (not h.get("status") or c["status"] == h["status"])
                and (not h.get("month") or m == int(h["month"]))
                and (not h.get("year") or y == int(h["year"]))):
            out.append(c)
    return out


def _offer_cases(s, cases, parts, lead):
    s["case"]["offered"] = [c["case_id"] for c in cases]
    lines = "\n".join(f"- {_claim_line(c)}" for c in cases)
    parts.append(_part(
        "offer_cases",
        f"{lead}\n{lines}\nWhich one would you like to go over?",
        {"claims_on_policy": [{k: c[k] for k in ("case_id", "case_type", "created_at", "status", "summary")} for c in cases]},
    ))


TYPE_WORDS = {"healthcare": r"health\w*|medical|hospital|doctor", "dental": r"dental|teeth|tooth|dentist", "auto": r"auto|car|vehicle|accident"}
STATUS_WORDS = {"denied": r"denied|denial|rejected|declined", "open": r"open|in progress|pending", "closed": r"closed|settled|paid|finished"}
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "last": -1}


def _pick_offered(s, claims):
    """Deterministic pick from the list just shown: '2048', 'the second one', 'the dental one'."""
    offered = [c for c in claims if c["case_id"] in s["case"]["offered"]]
    if not offered:
        return None
    text = s["_message"].lower()
    for c in offered:
        if re.search(rf"(?<!\d){re.escape(c['case_id'].split('-')[-1])}(?!\d)", text):
            return c
    for word, idx in ORDINALS.items():
        if re.search(rf"\b{word}\b", text):
            return offered[idx] if -len(offered) <= idx < len(offered) else None
    if re.fullmatch(r"\s*#?\s*([1-9])\s*\.?\s*", text):
        idx = int(re.findall(r"[1-9]", text)[0]) - 1
        return offered[idx] if idx < len(offered) else None
    pool = [c for c in offered if re.search(rf"\b(?:{TYPE_WORDS[c['case_type']]})", text)] or offered
    by_status = [c for c in pool if re.search(rf"\b(?:{STATUS_WORDS[c['status']]})\b", text)]
    pool = by_status or pool
    return pool[0] if len(pool) == 1 else None


def _resolve_phase(s, nlu, parts):
    claims = list_claims(s["identity"]["party_id"])

    def by_id(cid):
        return next((c for c in claims if cid and c["case_id"].upper() == str(cid).upper()), None)

    if nlu.get("wants_to_end") and not nlu.get("selected_case_id"):
        _goto(s, "POST_PROCESS")
        return _post_phase(s, nlu, parts, entering=True)

    h = s["memory"]["case_hints"]
    intent = s["memory"]["intent"]
    if intent in CATCH_ALL and _hints_empty(h) and not nlu.get("selected_case_id"):
        s["memory"]["intent"] = None
        parts.append(_catch_all_part(s, nlu, intent))
        if intent == "general_insurance_question":
            parts.append(_part("reprompt", "Is there a claim I can help you with today?"))
        return
    if (intent is None and nlu.get("question") and _hints_empty(h) and not nlu.get("selected_case_id")
            and s["_last_offer"] is None and s["case"]["offered"]):
        parts.append(_part("clarify_request", f"Sorry, I'm not sure I understood. I can help with {SUPPORTED_SUMMARY}. Which claim and which of those would you like help with?"))
        return
    chosen = by_id(nlu.get("selected_case_id")) or by_id(h.get("case_id")) or _pick_offered(s, claims)
    if not chosen and not _hints_empty(h):
        hits = _match_claims(claims, h)
        if len(hits) > 1 and not h.get("status"):
            # The intent itself narrows ties: a denial or appeal question is about a denied claim.
            implied = {"denial_question": "denied", "appeal": "denied"}.get(s["memory"]["intent"])
            narrowed = [c for c in hits if c["status"] == implied]
            if narrowed:
                hits = narrowed
        if len(hits) == 1:
            chosen = hits[0]
        elif len(hits) > 1:
            return _offer_cases(s, hits, parts, f"I found {len(hits)} claims that could match your {_describe_hints(h)}:")
        else:
            desc = _describe_hints(h)
            s["memory"]["case_hints"] = _empty_hints()
            return _offer_cases(s, claims, parts, f"I couldn't find a {desc} on your policy. Here's what I do see:")
    if not chosen and s["memory"]["intent"] in ("denial_question", "appeal"):
        denied = [c for c in claims if c["status"] == "denied"]
        if len(denied) == 1:
            chosen = denied[0]
    if not chosen:
        if not claims:
            parts.append(_part("no_claims", "I don't see any claims on this policy. Is there something else about your policy I can help with?", {"claims_on_policy": []}))
            return
        if len(claims) == 1:
            chosen = claims[0]
        else:
            return _offer_cases(s, claims, parts, "Which claim are you calling about? Here's what I see on your policy:")
    s["case"]["selected_case_id"] = chosen["case_id"]
    s["case"]["offered"] = []
    _goto(s, "PROCESS_CASE")
    _process_phase(s, nlu, parts, fresh=True)


# ------------------------------------------------------------------ PROCESS_CASE

def _case_facts(s, claim) -> dict:
    facts = {
        "claim": {
            "case_id": claim["case_id"],
            "case_type": claim["case_type"],
            "filed": _fmt_date(claim["created_at"]),
            "status": claim["status"],
            "summary": claim["summary"],
            **{k: _money(claim[k]) for k in ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee")},
        },
        "amount_field_definitions": field_definitions(),
        "today": _fmt_date(s["today"]),
    }
    if claim.get("denial_reason"):
        facts["claim"]["denial_reason"] = claim["denial_reason"]
    if claim.get("appeal_deadline"):
        left = _days_between(s["today"], claim["appeal_deadline"])
        facts["claim"]["appeal_deadline"] = _fmt_date(claim["appeal_deadline"])
        facts["claim"]["appeal_deadline_status"] = f"{left} days remaining" if left >= 0 else f"passed {-left} days ago"
    if claim.get("documents_needed"):
        facts["claim"]["documents_needed"] = claim["documents_needed"]
        facts["document_guidance"] = document_guidance(claim)
        facts["followup_guidance_all"] = followup_guidance(claim, "", None, FOLLOWUP_TOPICS)
    if claim.get("documents_needed"):
        facts["caller_reported_document_status"] = {
            "note": "What the CALLER told us during this call. The claim file itself still lists every one of these documents as missing; never describe these as our records.",
            "documents": s["case"]["documents"].get(claim["case_id"], {d: "not discussed yet" for d in claim["documents_needed"]}),
        }
    facts["followup_fallback"] = followup_fallback()
    facts["human_handoff_available"] = True
    return facts


def _deadline_sentence(claim, s) -> str:
    if not claim.get("appeal_deadline"):
        return ""
    left = _days_between(s["today"], claim["appeal_deadline"])
    if left >= 0:
        return f" The deadline to appeal is {_fmt_date(claim['appeal_deadline'])}, which is {left} days from today."
    return (f" The appeal deadline was {_fmt_date(claim['appeal_deadline'])}, which has already passed, so a human "
            "claims representative would need to review whether a late appeal is possible.")


def _denial_answer(claim, s) -> str:
    _discuss(s, claim, "denial")
    docs = claim.get("documents_needed") or []
    if docs:
        _add(s["followups"], f"Submit the {' and '.join(docs)} for claim {claim['case_id']}")
    if claim.get("appeal_deadline"):
        _add(s["followups"], f"Appeal deadline for claim {claim['case_id']}: {_fmt_date(claim['appeal_deadline'])}")
    s["pending_offer"] = "doc_help" if docs else "more_help"
    text = (f"I'm sorry to say your {claim['case_type']} claim {claim['case_id']}, filed {_fmt_date(claim['created_at'])}, "
            f"was denied because {claim['denial_reason']}. The good news is it can be reconsidered.")
    if docs:
        text += f" To have it reconsidered, we'll need the {' and '.join(docs)}."
    text += _deadline_sentence(claim, s)
    if docs:
        text += DOC_HELP_QUESTION
    return text


def _status_answer(claim, s) -> str:
    _discuss(s, claim, "status")
    if claim["status"] == "denied":
        return _denial_answer(claim, s)
    if claim["status"] == "open":
        tail = f" It's still being processed. The expected reimbursement is {_money(claim['expected_reimbursement_amount'])}, and nothing has been paid yet."
    elif float(claim["net_pay"]) > 0:
        tail = f" It's settled, and the insurer paid {_money(claim['net_pay'])}."
    else:
        tail = ""
    return f"Your {claim['case_type']} claim {claim['case_id']}, filed {_fmt_date(claim['created_at'])}, is {claim['status']}: {claim['summary'].lower()}.{tail}"


def _payment_answer(claim, s) -> str:
    _discuss(s, claim, "payment")
    text = (f"For claim {claim['case_id']}: the allowed maximum (the most an in-network insurer pays for this service) is "
            f"{_money(claim['allowed_max_amount'])}, the expected reimbursement is {_money(claim['expected_reimbursement_amount'])}, "
            f"and the net amount actually paid so far is {_money(claim['net_pay'])}.")
    if claim["status"] == "denied":
        text += " Because the claim was denied, nothing has been paid yet."
    return text


def _document_answer(claim, s) -> str:
    docs = claim.get("documents_needed") or []
    if not docs:
        return f"Claim {claim['case_id']} doesn't have any outstanding document requests right now."
    _discuss(s, claim, "documents")
    g = document_guidance(claim)
    _add(s["followups"], f"Upload the {' and '.join(docs)} through the member portal or claim upload link")
    per_doc = "\n".join(
        f"- {d['document']}: {d['requirements'] or 'make sure it is complete and legible, with the claim details visible.'}" for d in g["documents"]
    )
    case_type = f"{g['case_type']} " if g["case_type"] else ""
    return (f"Here's what you need for claim {claim['case_id']}:\n{per_doc}\n{case_type}The easiest way to send them is the member "
            "portal or the claim upload link. If online upload doesn't work for you, support can arrange fax or mail.")


def _appeal_answer(claim, s) -> str:
    _discuss(s, claim, "appeal")
    docs = " and ".join(claim.get("documents_needed") or []) or "supporting documents"
    return f"To appeal claim {claim['case_id']}, send in the requested {docs}. Once they're received, the claim goes back into review.{_deadline_sentence(claim, s)}"


def _switch_claim(s, nlu, current, parts):
    """'Can you check my other claim?' -> back to RESOLVE_INTENT with the caller's remaining claims."""
    others = [c for c in list_claims(s["identity"]["party_id"]) if c["case_id"] != current["case_id"]]
    if not others:
        s["pending_offer"] = "more_help"
        parts.append(_part("only_claim", f"Claim {current['case_id']} is the only claim on your policy. Is there anything else I can help you with on it?"))
        return
    s["memory"]["case_hints"] = _empty_hints()
    s["memory"]["intent"] = nlu.get("intent") if nlu.get("intent") not in CATCH_ALL and _intent_is_confident(nlu) else None
    _goto(s, "RESOLVE_INTENT")
    if len(others) == 1:
        s["case"]["selected_case_id"] = others[0]["case_id"]
        _goto(s, "PROCESS_CASE")
        return _process_phase(s, nlu, parts, fresh=True)
    _offer_cases(s, others, parts, "Sure. Here are your other claims:")


def _process_phase(s, nlu, parts, fresh=False):
    claim = get_claim(s["identity"]["party_id"], s["case"]["selected_case_id"])
    facts = _case_facts(s, claim)
    last_offer = s["_last_offer"]

    if not fresh:
        nxt = s["memory"]["next_hints"] or {}
        s["memory"]["next_hints"] = None
        other = (nxt.get("case_id") and nxt["case_id"].upper() != claim["case_id"]) or (nxt.get("case_type") and nxt["case_type"] != claim["case_type"])
        if other:
            s["memory"]["case_hints"] = {**_empty_hints(), **nxt}
            _goto(s, "RESOLVE_INTENT")
            return _resolve_phase(s, {**nlu, "selected_case_id": nxt.get("case_id")}, parts)
        if nlu.get("switch_claim"):
            return _switch_claim(s, nlu, claim, parts)
        declined_more =last_offer == "more_help" and nlu.get("affirm") is False and not nlu.get("question")
        if nlu.get("wants_to_end") or nlu.get("dialog_act") == "goodbye" or declined_more:
            _goto(s, "POST_PROCESS")
            return _post_phase(s, nlu, parts, entering=True)
        if last_offer == "doc_help" and nlu.get("affirm") is False and not nlu.get("question"):
            s["pending_offer"] = "more_help"
            parts.append(_part("ask_more", "No problem. Is there anything else I can help you with on this claim?"))
            return

    intent = s["memory"]["intent"] if fresh else nlu.get("intent")
    doc_part = None if fresh else _apply_document_updates(s, claim, nlu)
    if doc_part:
        parts.append(doc_part)
    if not fresh and intent in CATCH_ALL:
        parts.append(_catch_all_part(s, nlu, intent))
        if not s["pending_offer"]:
            s["pending_offer"] = "more_help"
            parts.append(_part("ask_more", f"Is there anything else I can help you with on claim {claim['case_id']}?"))
        return
    extra = [i for i in (nlu.get("additional_intents") or []) if i != intent and i not in CATCH_ALL][:2]
    if doc_part and intent in (None, "document_submission", "general_claim_question") and not extra:
        s["_turn_answered"] = True
        if not s["pending_offer"]:
            s["pending_offer"] = "more_help"
            parts.append(_part("ask_more", "Is there anything else I can help you with on this claim?"))
        return
    followups = followup_guidance(claim, f"{s['_message']} {nlu.get('question') or ''}", intent, nlu.get("followup_topics") or [])
    topics = [f["topic"] for f in followups]
    answered = True
    if fresh:
        say = _status_answer(claim, s)
        if intent in ("payment_question", "document_submission", "appeal", "next_steps"):
            more = _intent_answer(claim, s, intent)
            if more:
                say += "\n\n" + more
    elif last_offer == "doc_help" and nlu.get("affirm") is True:
        say = _document_answer(claim, s)
    elif followups and not fresh:
        for t in topics:
            _discuss(s, claim, t)
        if "missing_required_material_alternatives" in topics:
            g = document_guidance(claim)
            lead = next(f["guidance"] for f in followups if f["topic"] == "missing_required_material_alternatives")
            say = lead + "\n" + "\n".join(f"- {d['document']}: {d['if_unavailable']}" for d in g["documents"])
            _add(s["followups"], "If a requested document can't be obtained, ask for a human claims review of alternatives")
        else:
            say = " ".join(f["guidance"] for f in followups)
        if "processing_time_after_submission" in topics:
            _add(s["followups"], f"Expect review to take {facts['document_guidance']['average_processing_time']} after the documents are received")
        if "submission_timing" in topics:
            _add(s["followups"], "Submit the missing documents within a week")
    elif intent == "payment_question":
        say = _payment_answer(claim, s)
    elif intent == "document_submission":
        say = _document_answer(claim, s)
    elif intent == "appeal":
        say = _appeal_answer(claim, s)
    elif intent == "denial_question" or (fresh and claim["status"] == "denied"):
        say = _denial_answer(claim, s) if claim["status"] == "denied" else _status_answer(claim, s)
    elif intent == "status_inquiry" or fresh:
        say = _status_answer(claim, s)
    elif nlu.get("question"):
        _discuss(s, claim, "general")
        answered = False
        say = f"I don't have a specific answer to that in your claim file. {followup_fallback()}"
    else:
        say = _status_answer(claim, s)
    for extra_intent in extra:
        more = _intent_answer(claim, s, extra_intent)
        if more:
            say += "\n\n" + more

    if fresh and s["memory"]["pending_question"]:
        facts["caller_question_from_before_verification"] = s["memory"]["pending_question"]
    if fresh:
        s["memory"]["pending_question"] = None

    parts.append(_part("present_case" if fresh else "answer", say, facts))
    s["_turn_answered"] = answered
    if not answered:
        s["counters"]["unanswered_streak"] += 1
        s["_streak_counted"] = True
        if s["counters"]["unanswered_streak"] >= UNANSWERED_HANDOFF_STREAK and s["pending_offer"] != "human":
            s["pending_offer"] = "human"
            parts.append(_part("offer_human_unanswered", HUMAN_FOR_UNANSWERED))
    if not s["pending_offer"]:
        s["pending_offer"] = "more_help"
        parts.append(_part("ask_more", "Is there anything else I can help you with on this claim?"))



def settle_unanswered(s, llm_had_no_answer: bool, reply: str) -> str:
    """Called once per turn after phrasing. Two questions in a row that the claim file can't answer
    (judged by the engine's answer path or flagged by the LLM) -> stop improvising and offer a human,
    replacing the usual "anything else?" question. Returns the final reply."""
    answered = s.pop("_turn_answered", None)
    counted = s.pop("_streak_counted", False)
    if answered is None:
        return reply
    if answered and not llm_had_no_answer:
        s["counters"]["unanswered_streak"] = 0
        return reply
    if not counted:
        s["counters"]["unanswered_streak"] += 1
    if s["counters"]["unanswered_streak"] < UNANSWERED_HANDOFF_STREAK or s["pending_offer"] == "human":
        return reply
    s["pending_offer"] = "human"
    s["trace"][-1]["acts"].append("offer_human_unanswered")
    if re.search(r"connect you|representative", reply[-250:], re.I) and reply.rstrip().endswith("?"):
        return reply
    trimmed = re.sub(r"\s*(?:Is there|Anything)[^.?!]*\banything else\b[^?]*\?\s*$", "", reply, flags=re.I)
    return f"{trimmed}\n\n{HUMAN_FOR_UNANSWERED}"


def _intent_answer(claim, s, intent):
    if intent == "payment_question":
        return _payment_answer(claim, s)
    if intent == "document_submission":
        return _document_answer(claim, s)
    if intent == "appeal":
        return _appeal_answer(claim, s)
    if intent == "denial_question":
        return _denial_answer(claim, s) if claim["status"] == "denied" else _status_answer(claim, s)
    if intent == "status_inquiry":
        return _status_answer(claim, s)
    if intent == "next_steps":
        steps = followup_guidance(claim, "", "next_steps", ["submission_timing", "processing_time_after_submission"])
        for f in steps:
            _discuss(s, claim, f["topic"])
        return " ".join(f["guidance"] for f in steps) or None
    return None


# ------------------------------------------------------------------ document checklist

GENERIC_DOC_WORDS = {"the", "a", "original", "report", "note", "treating", "provider", "supplemental", "copy", "of"}
DOC_STATUS_LABEL = {
    "needed": "still needed",
    "have": "you have it: upload it through the member portal or claim upload link",
    "submitted": "you said it was already sent, but the file still lists it as missing: re-upload it and ask support to check the earlier submission",
    "cannot_get": "not available: request a replacement or a substitute record; ask for a human review if none can be obtained",
}


def _doc_terms(doc: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z]+", doc.lower()) if w not in GENERIC_DOC_WORDS]
    return words or re.findall(r"[a-z]+", doc.lower())


def _match_doc(name: str, docs: list[str]):
    n = name.lower()
    return next((d for d in docs if d.lower() in n or n in d.lower() or all(t in n for t in _doc_terms(d))), None)


def _offline_doc_updates(message: str, docs: list[str]) -> list[dict]:
    out = []
    for clause in re.split(r"[,;.!?]|\bbut\b|\bwhile\b", message, flags=re.I):
        c = clause.lower()
        mentioned = [d for d in docs if all(t in c for t in _doc_terms(d))]
        if not mentioned:
            continue
        if re.search(r"\b(can'?t|cannot|unable|don'?t have|do not have|lost|won'?t|no longer|not available|missing|never got)\b", c):
            status = "cannot_get"
        elif re.search(r"\b(sent|submitted|uploaded|faxed|mailed|already gave)\b", c):
            status = "submitted"
        elif re.search(r"\b(have|got|ready|can get|i'?ll get|on hand)\b", c):
            status = "have"
        else:
            continue
        out += [{"document": d, "status": status} for d in mentioned]
    return out


def _apply_document_updates(s, claim, nlu):
    docs = claim.get("documents_needed") or []
    if not docs:
        return None
    updates = [{"document": _match_doc(u["document"], docs), "status": u["status"]} for u in nlu.get("document_updates") or []]
    updates = [u for u in updates if u["document"]] or _offline_doc_updates(s["_message"], docs)
    if not updates:
        return None
    checklist = s["case"]["documents"].setdefault(claim["case_id"], {d: "needed" for d in docs})
    guidance = {d["document"]: d for d in document_guidance(claim)["documents"]}
    lines = []
    for u in updates:
        doc, status = u["document"], u["status"]
        checklist[doc] = status
        if status == "have":
            lines.append(f"Great, since you have the {doc}, you can upload it through the member portal or the claim upload link.")
            _add(s["followups"], f"Upload the {doc} for claim {claim['case_id']}")
        elif status == "submitted":
            lines.append(f"Thanks for letting me know about the {doc}. Our file for claim {claim['case_id']} still lists it as missing, "
                         "so it may not have been attached to this claim. Re-uploading it through the member portal or claim upload link "
                         "is the quickest fix, and support can check whether your earlier submission was received.")
            _add(s["followups"], f"Re-upload the {doc} for claim {claim['case_id']} and ask support to check whether the earlier submission arrived")
        else:
            lines.append(f"For the {doc}: {guidance[doc]['if_unavailable']}")
            _add(s["followups"], f"Request a replacement {doc} (or a substitute record) for claim {claim['case_id']}")
    _discuss(s, claim, "documents")
    if all(v == "cannot_get" for v in checklist.values()):
        s["pending_offer"] = "human"
        lines.append(f"{document_guidance(claim)['human_review_rule']} Would you like me to connect you with a claims representative to review other options?")
        _add(s["followups"], f"Ask a human claims representative to review alternatives for claim {claim['case_id']}")
    return _part("document_update", " ".join(lines), {"caller_reported_document_status": dict(checklist),
                                                       "claim_file_lists_as_missing": docs})


# ------------------------------------------------------------------ POST_PROCESS

def build_summary_email(s) -> dict:
    holder = get_policyholder(s["identity"]["party_id"])
    claims = [get_claim(s["identity"]["party_id"], cid) for cid in s["case"]["touched"]]
    subject = "Summary of your call"
    if claims:
        subject += f" about claim{'s' if len(claims) > 1 else ''} {', '.join(c['case_id'] for c in claims)}"
    lines = [
        f"To: {holder['email']}",
        f"Subject: {subject}",
        "",
        f"Hi {holder['name'].split()[0]},",
        "",
        "Thank you for contacting us today. Here is a summary of our conversation.",
        "",
        "What we discussed:",
        *([f"  - {d}" for d in s["discussed"]] or ["  - General questions about your policy"]),
    ]
    if claims:
        lines += ["", "Claim status:"]
        for c in claims:
            lines.append(f"  - {c['case_id']} ({c['case_type']}, filed {_fmt_date(c['created_at'])}): {c['status'].upper()}. {c['summary']}.")
            if c.get("denial_reason"):
                lines.append(f"    Denial reason: {c['denial_reason']}.")
    checklists = [(cid, docs) for cid, docs in s["case"]["documents"].items() if cid in s["case"]["touched"]]
    for c in claims:
        if c.get("documents_needed") and c["case_id"] not in s["case"]["documents"]:
            checklists.append((c["case_id"], {d: "needed" for d in c["documents_needed"]}))
    if checklists:
        lines += ["", "Document checklist:"]
        for cid, docs in checklists:
            lines += [f"  - {cid} / {doc}: {DOC_STATUS_LABEL[status]}" for doc, status in docs.items()]
    lines += ["", "Next steps and follow-up items:", *([f"  - {f}" for f in s["followups"]] or ["  - No action is needed from you right now."])]
    lines += ["", "If you have questions, reply to this email or call us anytime.", "", "Insurance Claims Support"]
    return {"to": holder["email"], "subject": subject, "body": "\n".join(lines)}


def _post_phase(s, nlu, parts, entering=False):
    holder = get_policyholder(s["identity"]["party_id"])
    masked = mask_email(holder["email"])
    if entering and not s["email"]["decision"]:
        # The caller may already have answered the email question in the same breath ("I'm done, don't email me").
        msg = s["_message"]
        if EMAIL_DECLINE_RE.search(msg):
            nlu = {**nlu, "affirm": False}
            s["email"]["offered"] = True
            entering = False
        elif EMAIL_ACCEPT_RE.search(msg):
            nlu = {**nlu, "affirm": True}
            s["email"]["offered"] = True
            entering = False
    if s["email"]["decision"] in ("sent", "skipped") and entering:
        _goto(s, "ENDED")
        done = "The summary is already on its way to your email on file. " if s["email"]["decision"] == "sent" else ""
        parts.append(_part("closing", f"{done}Thanks for calling, and take care!"))
        return
    if entering or not s["email"]["offered"]:
        s["email"]["offered"] = True
        s["pending_offer"] = "email"
        parts.append(_part(
            "offer_email",
            f"Before you go, would you like me to email you a summary of today's conversation, including what we discussed, the claim status, and your next steps? It would go to the email on file, {masked}.",
            {"email_on_file_masked": masked},
        ))
        return
    given = (nlu.get("identity") or {}).get("email")
    if given and given.lower() == holder["email"].lower() and nlu.get("affirm") is None:
        nlu = {**nlu, "affirm": True}
    if given and given.lower() != holder["email"].lower():
        s["pending_offer"] = "email"
        parts.append(_part("email_on_file_only", f"For your security, I can only send the summary to the email already on file ({masked}). Would you like me to send it there?"))
        return
    if nlu.get("affirm") is True:
        mail = build_summary_email(s)
        s["outbox"].append({**mail, "sent_at": datetime.now(timezone.utc).isoformat()})
        s["email"] = {"offered": True, "decision": "sent", "sent_to": mail["to"]}
        _goto(s, "ENDED")
        parts.append(_part("email_sent", f"Done. I've sent the summary to {masked}. Thanks for calling, and take care!", {"email_sent_to_masked": masked}))
        return
    if nlu.get("affirm") is False:
        s["email"] = {"offered": True, "decision": "skipped", "sent_to": None}
        _goto(s, "ENDED")
        parts.append(_part("email_skipped", "No problem, I won't send an email. Thanks for calling, and take care!"))
        return
    if nlu.get("question") or nlu.get("intent"):
        _goto(s, "PROCESS_CASE")
        s["email"]["offered"] = False
        return _process_phase(s, nlu, parts)
    s["pending_offer"] = "email"
    parts.append(_part("reask_email", "Sorry, I didn't catch that. Should I email you the summary? Just say yes or no."))


# ------------------------------------------------------------------ main step

def step(s: dict, message: str, nlu: dict) -> dict:
    s["turn"] += 1
    s["_message"] = message
    s["_last_offer"] = s["pending_offer"]
    s["pending_offer"] = None
    phase_before = s["phase"]
    s["_path"] = [phase_before]
    parts: list[dict] = []

    new_identity = any((nlu.get("identity") or {}).values()) or bool(nlu.get("rep_name"))
    ctx = {**_absorb(s, nlu), "new_identity": new_identity}

    if s["phase"] == "HANDOFF":
        parts.append(_part("handoff_wait", "A human representative will be with you shortly. Thanks for your patience."))
        return _finish(s, phase_before, parts, nlu)

    if nlu.get("wants_human") or (s["_last_offer"] == "human" and nlu.get("affirm") is True):
        e = _empathy_part(s)
        if e:
            parts.append(e)
        _handoff(s, "caller_requested_human" if nlu.get("wants_human") else "accepted_offer", parts)
        return _finish(s, phase_before, parts, nlu)

    if not nlu.get("in_scope", True):
        s["counters"]["off_topic_streak"] += 1
        parts.append(_part("out_of_scope", "I'm sorry, that's outside what I can help with. I can only assist with your insurance policy and claims."))
        if s["counters"]["off_topic_streak"] >= OFF_TOPIC_HANDOFF_STREAK:
            s["pending_offer"] = "human"
            parts.append(_part("offer_human", "If you need help with something else, I can connect you with a human representative. Would you like me to do that?"))
        else:
            parts.append(_reprompt_for(s))
        return _finish(s, phase_before, parts, nlu)
    s["counters"]["off_topic_streak"] = 0

    e = _empathy_part(s)
    if e:
        parts.append(e)
    if s["_last_offer"] == "human" and nlu.get("affirm") is False and not nlu.get("wants_to_end"):
        parts.append(_part("continue", "Okay, let's keep going together."))

    phase = s["phase"]
    if phase == "VERIFY_ID":
        _verify_phase(s, nlu, parts, ctx)
    elif phase == "RESOLVE_INTENT":
        _resolve_phase(s, nlu, parts)
    elif phase == "PROCESS_CASE":
        _process_phase(s, nlu, parts)
    elif phase == "POST_PROCESS":
        _post_phase(s, nlu, parts)
    elif phase == "ENDED":
        closing_words = nlu.get("wants_to_end") or nlu.get("dialog_act") in ("goodbye", "affirm", "deny", "other") or not nlu.get("question")
        if not closing_words and (nlu.get("question") or nlu.get("intent")):
            _goto(s, "PROCESS_CASE")
            s["email"]["offered"] = False
            _process_phase(s, nlu, parts)
        else:
            parts.append(_part("closing", "Thanks again for calling. Have a great day!"))
    _emotional_escalation(s, nlu, parts)
    return _finish(s, phase_before, parts, nlu)


def _finish(s, phase_before, parts, nlu) -> dict:
    facts: dict = {}
    for p in parts:
        facts.update(p["facts"])
    draft = " ".join(p["say"] for p in parts).replace(" \n", "\n")
    s["trace"].append({
        "turn": s["turn"],
        "phase_before": phase_before,
        "phase_after": s["phase"],
        "path": [p for i, p in enumerate(s.get("_path", [phase_before])) if i == 0 or p != s["_path"][i - 1]],
        "acts": [p["act"] for p in parts],
        "nlu": {
            "source": nlu.get("source"),
            "intent": nlu.get("intent"),
            "dialog_act": nlu.get("dialog_act"),
            "in_scope": nlu.get("in_scope"),
            "emotion": f"{nlu.get('emotion')}/{nlu.get('emotion_intensity')}",
            "identity_fields": [k for k, v in (nlu.get("identity") or {}).items() if v],
            "case_hints": {k: v for k, v in (nlu.get("case_hints") or {}).items() if v is not None},
            "intent_retrieval": nlu.get("intent_retrieval"),
        },
    })
    s.pop("_message", None)
    s.pop("_path", None)
    s.pop("_last_offer", None)
    return {"parts": parts, "facts": facts, "draft": draft}


def nlu_context(s: dict) -> dict:
    """Context handed to the NLU model. Before verification it contains no account data at all."""
    last = next((t["text"] for t in reversed(s["transcript"]) if t["role"] == "assistant"), None)
    ctx = {
        "phase": s["phase"],
        "identity_verified": s["identity"]["verified"],
        "agent_last_message": last,
        "agent_asked_yes_no": bool(s["pending_offer"]),
    }
    if s["identity"]["verified"] and s["case"]["offered"]:
        ctx["OFFERED_CASES"] = [
            {k: c[k] for k in ("case_id", "case_type", "created_at", "status")}
            for c in (get_claim(s["identity"]["party_id"], cid) for cid in s["case"]["offered"])
        ]
    if s["case"]["selected_case_id"]:
        ctx["current_case_id"] = s["case"]["selected_case_id"]
        c = get_claim(s["identity"]["party_id"], s["case"]["selected_case_id"])
        ctx["current_case_documents"] = c.get("documents_needed") or []
    return ctx
