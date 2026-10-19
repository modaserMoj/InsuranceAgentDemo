import json
import os
import re
from pathlib import Path

FIXTURE_DIR = Path(
    os.environ.get("FIXTURE_DIR")
    or Path(__file__).resolve().parent.parent / "apps" / "insurance_claims" / "fixtures"
)


def _load(name: str):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


POLICYHOLDERS = _load("policyholders.json")
CLAIMS = _load("claims.json")
REPRESENTATIVES = _load("representatives.json")
CONSENT_SCENARIOS = _load("consent_scenarios.json")
GUIDELINES = _load("required_document_guideline.json")
CLAIM_SCHEMA = _load("claim_schema.json")

PII_FIELDS = ["full_name", "dob", "phone", "email", "id_last4"]
FOLLOWUP_TOPICS = [g["topic"] for g in GUIDELINES["claim_followup_guidance"]]

# The alternatives topic has no match_any in the fixture; these phrases trigger it in offline mode.
UNAVAILABLE_PHRASES = [
    "don't have", "do not have", "can't get", "cannot get", "can't find", "unable to get",
    "lost", "not available", "no longer have", "doesn't exist", "won't give",
]


def norm_name(s) -> str:
    s = re.sub(r"[^a-z\s]", " ", str(s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def norm_phone(s) -> str:
    return re.sub(r"\D", "", str(s or ""))[-10:]


def norm_email(s) -> str:
    return str(s or "").strip().lower()


def _field_matches(holder: dict, field: str, value):
    if not value:
        return None
    if field == "full_name":
        return any(norm_name(n) == norm_name(value) for n in [holder["name"], *holder.get("name_aliases", [])])
    if field == "dob":
        return holder["dob"] == value
    if field == "phone":
        return any(norm_phone(p) == norm_phone(value) for p in [holder["phone"], *holder.get("phone_aliases", [])])
    if field == "email":
        return any(norm_email(e) == norm_email(value) for e in [holder["email"], *holder.get("email_aliases", [])])
    if field == "id_last4":
        return holder["id_last4"] == str(value)
    return None


def verify_identity(claimed: dict) -> dict:
    """Best-matching policyholder for the claimed PII. Policy number narrows the lookup
    but never counts toward the 3-field threshold."""
    pool = POLICYHOLDERS
    policy = (claimed.get("policy_number") or "").upper()
    if policy:
        by_policy = [h for h in pool if h["policy_number"].upper() == policy]
        if by_policy:
            pool = by_policy
    best = None
    for h in pool:
        matched, mismatched = [], []
        for f in PII_FIELDS:
            r = _field_matches(h, f, claimed.get(f))
            if r is True:
                matched.append(f)
            elif r is False:
                mismatched.append(f)
        policy_mismatch = bool(policy) and h["policy_number"].upper() != policy
        score = len(matched) - len(mismatched) - (1 if policy_mismatch else 0)
        if best is None or score > best["score"]:
            best = {"holder": h, "matched": matched, "mismatched": mismatched, "policy_mismatch": policy_mismatch, "score": score}
    verified = bool(best) and len(best["matched"]) >= 3 and not best["mismatched"] and not best["policy_mismatch"]
    return {
        "verified": verified,
        "party_id": best["holder"]["party_id"] if verified else None,
        "matched": best["matched"] if best else [],
        "mismatched": best["mismatched"] if best else [],
    }


def get_policyholder(party_id):
    return next((h for h in POLICYHOLDERS if h["party_id"] == party_id), None)


def list_claims(party_id):
    return [c for c in CLAIMS if c["party_id"] == party_id]


def get_claim(party_id, case_id):
    cid = str(case_id or "").upper()
    return next((c for c in CLAIMS if c["party_id"] == party_id and c["case_id"].upper() == cid), None)


def find_representative(rep_name, party_id):
    for r in REPRESENTATIVES:
        if r["buyer_party_id"] == party_id and (not rep_name or norm_name(r["rep_name"]) == norm_name(rep_name)):
            return r
    return None


def consent_sequence(scenario: str) -> list:
    return (CONSENT_SCENARIOS.get(scenario) or CONSENT_SCENARIOS["default"])["status_sequence"]


def check_consent(consent: dict) -> str:
    """Simulated consent service: each check advances through the scenario's status sequence."""
    seq = consent_sequence(consent["scenario"])
    status = seq[consent["checks"]] if consent["checks"] < len(seq) else "timeout"
    consent["checks"] += 1
    consent["status"] = status
    return status


def _lookup_doc_key(mapping: dict, doc: str):
    d = doc.lower()
    return next((k for k in mapping if k in d or d in k), None)


def document_guidance(claim: dict) -> dict:
    g = GUIDELINES
    docs = claim.get("documents_needed") or []
    out_docs = []
    for doc in docs:
        k = _lookup_doc_key(g["document_guidance"], doc)
        ak = _lookup_doc_key(g["document_alternative_guidance"], doc)
        out_docs.append({
            "document": doc,
            "requirements": g["document_guidance"][k]["en"] if k else None,
            "if_unavailable": (g["document_alternative_guidance"][ak] if ak else g["document_alternative_guidance"]["default"])["en"],
        })
    return {
        "general": g["default_guidance"]["en"],
        "case_type": (g["case_type_guidance"].get(claim["case_type"]) or {}).get("en"),
        "documents": out_docs,
        "average_processing_time": g["claim_followup_settings"]["average_processing_time_after_submission"]["en"],
        "human_review_rule": g["claim_followup_settings"]["human_review_after_document_alternatives_exhausted"]["en"],
    }


def _fill(tpl: str, vars_: dict) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: str(vars_.get(m.group(1), m.group(0))), tpl)


def followup_guidance(claim: dict, utterance: str, intent, topics_hint=()) -> list:
    """Picks follow-up guidance topics by phrase match; topics_hint lets the LLM choose bounded topics directly."""
    docs = claim.get("documents_needed") or []
    vars_ = {
        "case_id": claim["case_id"],
        "documents": " and ".join(docs),
        "average_processing_time_after_submission": GUIDELINES["claim_followup_settings"]["average_processing_time_after_submission"]["en"],
    }
    text = str(utterance or "").lower()
    out = []
    for item in GUIDELINES["claim_followup_guidance"]:
        if item.get("requires_documents") and not docs:
            continue
        if intent and intent not in item["intent_hints"] and item["topic"] not in topics_hint:
            continue
        phrases = item.get("match_any") or UNAVAILABLE_PHRASES
        if any(p in text for p in phrases) or item["topic"] in topics_hint:
            out.append({"topic": item["topic"], "guidance": _fill(item["en"], vars_)})
    return out


def followup_fallback() -> str:
    return GUIDELINES["claim_followup_fallback"]["en"]


def field_definitions() -> dict:
    return {k: v["description"] for k, v in CLAIM_SCHEMA["field_descriptions"].items()}


def mask_email(email: str) -> str:
    user, domain = email.split("@")
    return f"{user[0]}{'*' * max(len(user) - 1, 2)}@{domain}"
