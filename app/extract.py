"""Deterministic understanding. Always runs: it is the offline NLU and a cross-check on the LLM.
Identity values and case hints come from regex; intent comes from hybrid retrieval (app/intents.py)."""
import re

from .intents import CONFIDENT, INTENTS, classify
CASE_TYPES = ["healthcare", "dental", "auto"]
STATUSES = ["denied", "open", "closed"]
EMOTIONS = ["neutral", "frustrated", "angry", "anxious", "confused", "sad"]
DIALOG_ACTS = ["provide_info", "ask_question", "affirm", "deny", "refuse", "request_human", "greeting", "goodbye", "other"]

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
_MONTH = r"\b(" + "|".join(MONTHS) + "|" + "|".join(m[:3] for m in MONTHS) + r")\b\.?"
I = re.IGNORECASE

NAME_STOP = {
    "the", "a", "an", "calling", "her", "his", "my", "policyholder", "representative", "son", "daughter", "here",
    "not", "just", "trying", "looking", "so", "very", "really", "sorry", "frustrated", "confused", "worried",
    "upset", "angry", "fine", "good", "okay", "ok",
}


def _month_index(s: str) -> int:
    t = s.lower()[:3]
    return next(i + 1 for i, m in enumerate(MONTHS) if m.startswith(t))


def _extract_dob(text: str):
    m = re.search(r"\b(19\d{2}|20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", text)
    if m:
        return f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}", m[0]
    m = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](19\d{2}|20\d{2})\b", text)
    if m:
        return f"{m[3]}-{int(m[1]):02d}-{int(m[2]):02d}", m[0]
    m = re.search(_MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(19\d{2}|20\d{2})", text, I)
    if m:
        return f"{m[3]}-{_month_index(m[1]):02d}-{int(m[2]):02d}", m[0]
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH + r",?\s+(19\d{2}|20\d{2})", text, I)
    if m:
        return f"{m[3]}-{_month_index(m[2]):02d}-{int(m[1]):02d}", m[0]
    return None


def _extract_name(text: str):
    pattern = r"\b(?:my name is|name's|this is|i am|i'm|im|name:)\s+([A-Za-z][A-Za-z'-]+(?:\s+[A-Za-z][A-Za-z'-]+){1,2})"
    for m in re.finditer(pattern, text, I):
        kept = []
        for w in m[1].split():
            if w.lower() in NAME_STOP or re.fullmatch(r"and|but|from|calling|with|policy|dob|ssn|born", w, I):
                break
            if not w[0].isupper():
                break
            kept.append(w)
        if len(kept) >= 2:
            return " ".join(kept)
    return None


def extract_bare_value(field: str, raw: str):
    """When the form has asked for one specific field, accept a reply that is just the value
    ("4472", "Margaret Chen", "650 521 2836") without any lead-in words."""
    text = str(raw or "").strip().strip(".!")
    if field == "full_name":
        words = text.split()
        if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", w) for w in words) \
                and not any(w.lower() in NAME_STOP or w.lower() in {"hi", "hello", "hey", "yes", "no", "skip", "thanks"} for w in words):
            return " ".join(w[0].upper() + w[1:] for w in words)
        return None
    if field == "dob":
        found = _extract_dob(text)
        return found[0] if found else None
    if field == "phone":
        digits = re.sub(r"\D", "", text)
        if re.fullmatch(r"[\d\s()+.-]+", text) and len(digits) in (10, 11):
            return "+1" + digits[-10:]
        return None
    if field == "email":
        m = re.fullmatch(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", text)
        return text.lower() if m else None
    if field == "id_last4":
        m = re.fullmatch(r"(?:\D{0,30}?)(\d{4})", text)
        return m[1] if m and len(re.sub(r"\D", "", text)) == 4 else None
    return None


# Identity statements ("my phone is ...", "SSN last four is ...") must not look like requests such as
# "change my phone number", so values and their labels are removed before intent classification.
_ID_LABELS = re.compile(
    r"\b(?:my\s+|her\s+|his\s+)?(?:full\s+)?(?:name|phone(?:\s+number)?|cell|mobile|e-?mail(?:\s+address)?|dob|date of birth|birthday|born|"
    r"ssn|social(?:\s+security)?(?:\s+number)?|national\s+id|last\s+(?:4|four)(?:\s+digits)?|policy(?:\s+number)?)\s*(?:is|=|:|was)?", re.I)


def _request_text(scrub: str, ident: dict) -> str:
    if not any(ident.values()):
        return scrub
    text = scrub
    if ident.get("full_name"):
        text = re.sub(re.escape(ident["full_name"]), " ", text, flags=re.I)
    text = re.sub(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", " ", text)
    text = re.sub(r"\b(?:i am|i'm|this is)\s+(?:the\s+)?policyholder\b", " ", text, flags=re.I)
    return _ID_LABELS.sub(" ", text)


CLAIM_PHRASE = re.compile(
    r"\b(?:my|the|this|that|our|her|his)\s+(?:(?:denied|rejected|open|closed|pending|healthcare|health|medical|dental|auto|car)\s+)*"
    r"claims?\b(?:\s+(?:from|in|back in)\s+(?:last\s+)?[a-z]+)?", re.I)


def leading_name(raw: str):
    """'Margaret Chen born 1985-03-15 ...' -> 'Margaret Chen' (a capitalised name opening the message)."""
    m = re.match(r"\s*([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){1,2})(?=\s*(?:,|\.|;|$|\s(?:born|dob|date|with|and|my|here|ssn|phone|email|calling|i'm|im)\b))", str(raw or ""))
    if not m or any(w.lower() in NAME_STOP or w.lower() in {"hi", "hello", "hey", "yes", "no"} for w in m[1].split()):
        return None
    return m[1]


def looks_like_attempt(field: str, raw: str) -> bool:
    """The caller clearly tried to answer the requested field but the value is malformed."""
    text = str(raw or "").strip()
    digits = re.sub(r"\D", "", text)
    if field == "phone":
        return 5 <= len(digits) <= 15 and len(re.sub(r"[\d\s()+.-]", "", text)) <= 25
    if field == "id_last4":
        return 1 <= len(digits) <= 9 and len(text) <= 40
    if field == "dob":
        return bool(re.search(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", text))
    if field == "email":
        return "@" in text or bool(re.search(r"\bat\b.*\bdot\b", text, re.I))
    return False


def regex_extract(raw: str) -> dict:
    text = str(raw or "")
    lower = text.lower()
    ident = {k: None for k in ["full_name", "dob", "phone", "email", "id_last4", "policy_number"]}

    scrub = text
    dob = _extract_dob(text)
    if dob:
        ident["dob"], span = dob
        scrub = scrub.replace(span, " ", 1)
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if m:
        ident["email"] = m[0].lower()
        scrub = scrub.replace(m[0], " ", 1)
    m = re.search(r"\bPOL[-\s]?(\d{3,})\b", text, I)
    if m:
        ident["policy_number"] = f"POL-{m[1]}"
        scrub = scrub.replace(m[0], " ", 1)
    case_m = re.search(r"\bCL[-\s]?(\d{3,})\b", text, I)
    if case_m:
        scrub = scrub.replace(case_m[0], " ", 1)
    m = re.search(r"(?:ssn|social(?: security)?|last\s*(?:4|four)|national\s*id|id\s*(?:number)?\s*ending)[^0-9]{0,25}(\d{4})\b", scrub, I)
    if m:
        ident["id_last4"] = m[1]
        scrub = scrub.replace(m[0], " ", 1)
    m = re.search(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", scrub)
    if m:
        ident["phone"] = "+1" + re.sub(r"\D", "", m[0])[-10:]
    ident["full_name"] = _extract_name(text)

    hints = {"case_id": f"CL-{case_m[1]}" if case_m else None, "case_type": None, "status": None, "month": None, "year": None}
    if re.search(r"\b(health|healthcare|medical|hospital|doctor|clinic|pathology|lab)\b", scrub, I):
        hints["case_type"] = "healthcare"
    elif re.search(r"\b(dental|dentist|teeth|tooth)\b", scrub, I):
        hints["case_type"] = "dental"
    elif re.search(r"\b(auto|car|vehicle|accident|collision|repair)\b", scrub, I):
        hints["case_type"] = "auto"
    if re.search(r"\b(denied|denial|rejected|declined)\b", scrub, I):
        hints["status"] = "denied"
    elif re.search(r"\b(open|in progress|pending)\b", scrub, I) and re.search(r"claim", scrub, I):
        hints["status"] = "open"
    elif re.search(r"\b(closed|settled|completed)\b", scrub, I):
        hints["status"] = "closed"
    mm = re.search(_MONTH, scrub, I)
    modal_may = bool(mm) and mm[1].lower() == "may" and re.match(r"\s+(i|we|be|have|you)\b", scrub[mm.end():], I)
    if mm and not modal_may:
        hints["month"] = _month_index(mm[1])
    yr = re.search(r"\b(20[12]\d)\b", scrub)
    if yr:
        hints["year"] = int(yr[1])

    request_text = _request_text(scrub, ident)
    intent_result = classify(request_text)
    # "my denied healthcare claim from January" says WHICH claim, not WHAT the caller wants; if removing it
    # reveals a confident specific request ("how do I upload the documents for ..."), that is the intent.
    without_claim = CLAIM_PHRASE.sub(" ", request_text)
    if without_claim != request_text and intent_result["intent"] in (None, "general_claim_question", "denial_question"):
        alt = classify(without_claim)
        if alt["intent"] not in (None, "general_claim_question") and alt["confidence"] >= max(CONFIDENT, intent_result["confidence"]):
            intent_result = alt
    intent = intent_result["intent"]
    additional = []
    for clause in re.split(r"[?.!;,]|\band\s+(?:also\s+)?|\balso\b|\bplus\b", without_claim, flags=re.I):
        if len(clause.split()) < 3:
            continue
        r = classify(clause)
        if r["intent"] and r["confidence"] >= CONFIDENT and r["intent"] != intent and r["intent"] not in additional:
            additional.append(r["intent"])

    wants_human = bool(re.search(
        r"\b(speak|talk|transfer|connect|put me through|get me)\b.{0,25}\b(human|person|agent|representative|rep|someone|supervisor|manager|operator)\b"
        r"|\b(real person|human agent|live agent)\b", lower))
    rep = re.search(r"\b(?:i am|i'm|this is)\s+(?:her|his|their|the)?\s*(son|daughter|husband|wife|spouse|partner|mother|father|parent|caregiver|guardian|representative|attorney)\b", text, I) \
        or re.search(r"\bcalling (?:on behalf of|for) my (mother|mom|father|dad|wife|husband|parent)\b", text, I)
    caller_role = None
    rep_relationship = None
    if re.search(r"\b(i am|i'm) the policyholder\b|\bmy (own )?(claim|policy)\b", lower):
        caller_role = "policyholder"
    if rep:
        caller_role = "representative"
        rep_relationship = None if "calling" in rep[0].lower() else rep[1].lower()
        if rep_relationship == "representative":
            rep_relationship = None
    rep_name = None
    if caller_role == "representative":
        rep_name = ident["full_name"]
        ph = re.search(r"\b(?:mother|mom|father|dad|wife|husband|parent|behalf of|calling for)(?:'s)?(?:\s+(?:is|named|called))?\s+([A-Z][a-z'-]+\s+[A-Z][a-z'-]+)", text)
        ident["full_name"] = ph[1] if ph else None

    if re.match(r"\s*(yes|yeah|yep|yup|sure|ok(ay)?|please do|please|go ahead|do it|send it|that works|correct|sounds good|absolutely)\b", lower):
        affirm = True
    elif re.match(r"\s*(no|nope|nah|no thanks|don't|do not|skip|not now|not needed)\b", lower):
        affirm = False
    else:
        affirm = None
    refuses = bool(re.search(r"\b(already told you|not (going to|gonna) (give|tell)|won'?t (give|tell|share)|why do you need|none of your business|just tell me|i don'?t want to (give|share))\b", lower))
    wants_to_end = bool(re.search(r"\b(that'?s all|that is all|nothing else|i'?m done|we'?re done|bye|goodbye|no more questions|that'?s it|all set|wrap up)\b", lower))
    switch_claim = bool(re.search(
        r"\b(other|another|different|next|second|rest of (?:my|the))\s+(?:\w+\s+)?claims?\b|\b(?:list|all|show)\s+(?:of\s+)?(?:my\s+)?claims\b"
        r"|\bwhat claims\b|\bclaims? (?:do i|i) have\b|\bback to (?:the )?list\b", lower))

    emotion, intensity = "neutral", 0
    if re.search(r"\b(ridiculous|unacceptable|absurd|furious|outrageous|wtf|damn|hell|stupid|useless)\b|!{2,}", lower):
        emotion, intensity = "angry", 2
    elif re.search(r"\b(frustrat|annoy|fed up|again\?|already told|waste of time|seriously)", lower):
        emotion, intensity = "frustrated", 2
    elif re.search(r"\b(worried|anxious|scared|stress|nervous|afraid|can'?t afford|panick)", lower):
        emotion, intensity = "anxious", 2
    elif re.search(r"\b(confused|don'?t understand|makes no sense|what do you mean|lost)\b", lower):
        emotion, intensity = "confused", 2
    elif re.search(r"\b(sad|upset|devastated|hopeless)\b", lower):
        emotion, intensity = "sad", 2

    has_id = any(ident.values())
    domain = bool(re.search(
        r"\b(claim|policy|insur|denied|denial|document|coverage|deductible|premium|reimburse|payment|appeal|upload|portal|"
        r"verify|verification|status|pathology|office note|estimate|accident|ssn|dob|birth|email|phone|consent|approv|pay|"
        r"paid|money|amount|cover|healthcare|dental|auto|human|representative|agent|summary|refund|bill|deadline|submit)\w*", lower))
    greeting = bool(re.match(r"\s*(hi|hello|hey|good (morning|afternoon|evening))\b", lower)) and len(lower) < 40
    is_question = bool(re.search(r"\?|^\s*(what|who|how|why|when|where|can you|could you|tell me|explain|write|give me)\b", lower))
    in_scope = has_id or domain or intent is not None or greeting or affirm is not None or wants_to_end or refuses or not is_question

    if wants_human:
        dialog_act = "request_human"
    elif refuses:
        dialog_act = "refuse"
    elif has_id:
        dialog_act = "provide_info"
    elif wants_to_end:
        dialog_act = "goodbye"
    elif affirm is True:
        dialog_act = "affirm"
    elif affirm is False:
        dialog_act = "deny"
    elif greeting:
        dialog_act = "greeting"
    elif is_question:
        dialog_act = "ask_question"
    else:
        dialog_act = "other"

    return {
        "identity": ident,
        "caller_role": caller_role,
        "rep_name": rep_name,
        "rep_relationship": rep_relationship,
        "intent": intent,
        "additional_intents": additional[:2],
        "document_updates": [],
        "case_hints": hints,
        "selected_case_id": hints["case_id"],
        "followup_topics": [],
        "dialog_act": dialog_act,
        "in_scope": in_scope,
        "emotion": emotion,
        "emotion_intensity": intensity,
        "refuses_verification": refuses,
        "wants_human": wants_human,
        "affirm": affirm,
        "wants_to_end": wants_to_end,
        "switch_claim": switch_claim,
        "abusive": bool(re.search(r"\b(fuck\w*|shit\w*|bitch\w*|asshole|bastard|idiot\w*|moron\w*|screw you|shut up|stupid (?:bot|machine|ai|agent)|you(?:'re| are) (?:useless|worthless|an idiot))\b", lower)),
        "question": text.strip() if is_question else None,
        "intent_retrieval": {"confidence": intent_result["confidence"], "candidates": intent_result["candidates"]},
        "source": "regex",
    }
