"""LLM-simulated callers. Used by the "Watch a demo" button and by the eval suite.

A persona is a role-play brief: who the caller is, what they know, how they behave, and what they want.
The simulator only ever sees the conversation from the caller's side; it has no access to the engine.
"""
from .providers import Provider, chat

END = "[[END]]"

PERSONAS = {
    "margaret_happy_path": {
        "label": "Margaret: denied claim, cooperative",
        "brief": (
            "You are Margaret Chen, a policyholder. Your details: full name Margaret Chen, date of birth 1985-03-15, "
            "phone 650-521-2836, email margaret@email.com, SSN last four 4472, policy POL-9921. "
            "You are calling about your denied healthcare claim from January. Mention that early, before you are verified. "
            "Answer identity questions one at a time when asked. Then ask why it was denied, ask how to submit the "
            "missing documents, mention you already have the office note but the lab can't find the pathology report, "
            "and finally say that's all and accept the email summary."
        ),
        "style": "polite, conversational, a little worried",
    },
    "margaret_frustrated": {
        "label": "Margaret: frustrated, pushes back on verification",
        "brief": (
            "You are Margaret Chen (DOB 1985-03-15, SSN last four 4472, phone 650-521-2836). You are angry that your "
            "healthcare claim from January was denied. At first refuse to verify: say you already told them who you "
            "are and demand to know why it was denied. Push back twice. After the agent explains why verification is "
            "needed a second time, grudgingly give your details one at a time. Once you have the denial reason, say "
            "you're done and decline the email."
        ),
        "style": "frustrated, short sentences, exclamation marks, never abusive",
    },
    "david_representative": {
        "label": "David: son calling for his mother",
        "brief": (
            "You are David Chen, calling on behalf of your mother Margaret Chen. You are her son. You know her full "
            "name (Margaret Chen), her date of birth (1985-03-15) and her SSN last four (4472). Introduce yourself and "
            "your relationship up front. When the agent says it is waiting for your mother's consent, say she just "
            "approved it on her phone. Then ask about the status of her open auto claim and how much is expected. "
            "Then wrap up and skip the email."
        ),
        "style": "calm, helpful, slightly impatient",
    },
    "confused_caller": {
        "label": "Confused caller asking off-topic things",
        "brief": (
            "You are Ma Tian (DOB 1964-09-10, phone 650-208-8799, email matian@example.com, national ID last four "
            "6688). You are not good with technology and get confused. First ask 'what is RL?' because you saw it on "
            "the news. Then follow the agent's lead and verify one field at a time. Ask what a deductible is. Then ask "
            "why your claim was denied and what you need to send. Then say thank you, that's all, and accept the email."
        ),
        "style": "confused, polite, sometimes asks the agent to repeat things",
    },
}

SIM_SYSTEM = """You are role-playing a CALLER contacting an insurance company's claims support chat. Stay in character.
Write ONLY the caller's next message: one to three short sentences, plain text, no quotes, no stage directions.
Never invent identity details beyond your brief. Follow the agent's instructions where your brief allows
(for example, give one detail at a time when asked). If the agent offers to transfer you to a human and your brief
does not mention that, say no and continue.
When your goal is complete, the agent has said goodbye, or you have been transferred to a human, reply with exactly
[[END]]."""


def next_caller_message(provider: Provider, persona_id: str, transcript: list[dict]) -> tuple[str, bool]:
    """Returns (message, done). transcript items are {"role": "user"|"assistant", "text": ...} from the agent's view."""
    persona = PERSONAS[persona_id]
    lines = "\n".join(("CALLER (you): " if t["role"] == "user" else "AGENT: ") + t["text"] for t in transcript[-14:])
    user = (f"YOUR BRIEF:\n{persona['brief']}\n\nYOUR STYLE: {persona['style']}\n\n"
            f"CONVERSATION SO FAR:\n{lines or '(the agent has just greeted you)'}\n\nYour next message:")
    text = chat(provider, SIM_SYSTEM, user, 400).strip().strip('"')
    if not text or END in text:
        return "", True
    return text, False
