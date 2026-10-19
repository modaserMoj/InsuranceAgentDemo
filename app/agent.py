from .engine import nlu_context, settle_unanswered, step
from .guard import find_leaks
from .llm import LLMError, phrase
from .nlu import interpret
from .providers import Provider

NO_ANSWER = "[[NO_ANSWER]]"


def run_turn(s: dict, message: str, provider: Provider | None = None) -> dict:
    """One turn: understand -> deterministic SOP step -> phrase -> disclosure guard.
    With provider=None the turn runs offline (regex understanding + SOP templates)."""
    context = nlu_context(s)
    s["transcript"].append({"role": "user", "text": message})

    nlu, nlu_error = interpret(message, context, provider=provider)
    out = step(s, message, nlu)

    reply, source, notes = out["draft"], "template", []
    llm_no_answer = False
    if nlu_error:
        notes.append(nlu_error)
    if provider is not None and s["phase"] == "VERIFY_ID":
        # Identity verification is a strict form: every reply is the exact SOP template, never LLM-phrased.
        # (Evals caught the LLM inventing a verification rule when it was allowed to rephrase this phase.)
        source = "template (strict form)"
    elif provider is not None:
        try:
            text = phrase(provider, {
                "PHASE": s["phase"],
                "DIRECTIVES": [{"act": p["act"], "instruction": p.get("instruction", p["say"])} for p in out["parts"]],
                "FACTS": out["facts"],
                "DRAFT": out["draft"],
                "EMOTION": s["emotion"] if s["emotion"]["intensity"] >= 2 else {"label": "neutral", "intensity": 0},
                "RECENT_TURNS": s["transcript"][-8:],
                **({"IF_YOU_CANNOT_ANSWER": "The previous question also could not be answered from the file. If you cannot answer "
                    "this one from FACTS either, offer to connect the caller with a claims representative instead of asking "
                    "whether there is anything else."}
                   if s["counters"]["unanswered_streak"] >= 1 and s.get("_turn_answered") else {}),
            })
            if NO_ANSWER in text:
                text = text.replace(NO_ANSWER, "").strip()
                llm_no_answer = True
                notes.append("LLM flagged the question as not answerable from the claim file")
            leaks = find_leaks(text, s)
            if leaks:
                notes.append(f"guard blocked LLM reply (would disclose: {', '.join(leaks)}); used SOP template")
            elif text:
                reply, source = text, "llm"
        except LLMError as e:
            notes.append(f"phrasing fell back to template: {e}")

    reply = settle_unanswered(s, llm_no_answer, reply)
    draft_leaks = find_leaks(reply, s)
    if draft_leaks:
        notes.append(f"template leak detected: {', '.join(draft_leaks)}")
        reply = "I'm sorry, I need to verify your identity before I can share any details."

    s["transcript"].append({"role": "assistant", "text": reply})
    s["trace"][-1].update(reply_source=source, notes=notes)
    return {"reply": reply, "replySource": source, "notes": notes}
