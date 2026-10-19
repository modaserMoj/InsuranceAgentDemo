"""Deterministic last line of defense: any account data the caller isn't entitled to see
(everything before verification; other parties' data after) blocks the LLM reply."""
import re

from .data import CLAIMS, POLICYHOLDERS, norm_phone

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _long_date(iso: str) -> str:
    y, m, d = (int(x) for x in iso.split("-"))
    return f"{MONTHS[m - 1]} {d}"


def _money_forms(amount) -> list[str]:
    n = float(amount)
    if not n:
        return []
    return [f"{n:,.2f}", f"{n:,.0f}" if n.is_integer() else f"{n:,}", f"{n:g}"]


def find_leaks(text: str, s: dict) -> list[str]:
    lower = text.lower()
    digits = re.sub(r"\D", "", text)
    caller_said = " \n ".join(t["text"].lower() for t in s["transcript"] if t["role"] == "user")
    allowed = s["identity"]["party_id"] if s["identity"]["verified"] else None
    leaks: list[str] = []

    def probe(label, needle):
        if not needle:
            return
        n = str(needle).lower()
        if len(n) < 4 or n in caller_said:
            return
        if re.search(rf"(^|[^a-z0-9]){re.escape(n)}([^a-z0-9]|$)", lower):
            leaks.append(label)

    for c in CLAIMS:
        if c["party_id"] == allowed:
            continue
        cid = c["case_id"]
        probe(f"claim id {cid}", cid)
        probe(f"denial reason of {cid}", c.get("denial_reason"))
        for d in c.get("documents_needed") or []:
            probe(f"required document of {cid}", d)
        if c.get("appeal_deadline"):
            probe(f"appeal deadline of {cid}", _long_date(c["appeal_deadline"]))
        probe(f"filing date of {cid}", _long_date(c["created_at"]))
        for f in ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee"):
            for m in _money_forms(c[f]):
                if m.lower() not in caller_said and re.search(rf"\$\s?{re.escape(m)}\b", text):
                    leaks.append(f"{f} of {cid}")
    for h in POLICYHOLDERS:
        if h["party_id"] == allowed:
            continue
        pid = h["party_id"]
        probe(f"email of {pid}", h["email"])
        probe(f"policy number of {pid}", h["policy_number"])
        probe(f"dob of {pid}", h["dob"])
        p = norm_phone(h["phone"])
        if p in digits and p not in re.sub(r"\D", "", caller_said):
            leaks.append(f"phone of {pid}")
    return list(dict.fromkeys(leaks))
