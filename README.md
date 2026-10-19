# Alex: an SOP-guided insurance claims agent

Alex is a customer-service agent that **talks like a person but follows the business procedure exactly**. It verifies identity before revealing anything, finds the right claim, answers questions only from real claim data, offers an email summary, and stays calm when callers don't.

![Alex helping a caller, with the live SOP map on the right](docs/screenshot.png)

**Run it in one line:** `docker build -t alex . && docker run -p 8000:8000 -e LLM_API_KEY=<any provider key> alex` and open http://localhost:8000.

---

## The idea in one picture

The LLM never decides what happens next. It **understands** messages and **phrases** replies. Everything between those two steps is ordinary, testable code.

```mermaid
flowchart LR
    C([Caller message]) --> U["1 · Understand<br/>LLM reads intent, emotion, ID values<br/>+ regex + intent retrieval as cross-check"]
    U --> E["2 · SOP engine (plain code)<br/>phases · identity gate · consent<br/>memory · claim matching · escalation"]
    E --> P["3 · Phrase<br/>LLM rewords the engine's answer<br/>using only the facts it is given"]
    P --> G{"4 · Leak guard<br/>any data the caller<br/>may not see?"}
    G -- no --> R([Reply])
    G -- yes --> T[Use the engine's safe template] --> R
```

The SOP itself is a state machine. The UI draws it live, highlighting the path each conversation takes:

```mermaid
stateDiagram-v2
    direction LR
    [*] --> VERIFY_ID
    VERIFY_ID --> RESOLVE_INTENT: 3 of 5 ID fields match
    RESOLVE_INTENT --> PROCESS_CASE: claim chosen (instantly if memory or the policy leaves only one)
    PROCESS_CASE --> RESOLVE_INTENT: "my other claim"
    PROCESS_CASE --> POST_PROCESS: caller is done
    POST_PROCESS --> [*]: email sent or skipped
    VERIFY_ID --> HANDOFF: refusals / 3 failed attempts
    PROCESS_CASE --> HANDOFF: asks for a person
```

## What we built

| Requirement | How Alex does it |
|---|---|
| **Strict identity check** | A one-field-at-a-time form (name → DOB → phone → email → SSN last 4) that needs 3 matching fields and zero mismatches. The LLM never sees account data before verification, and these replies are fixed text. |
| **Natural conversation** | The LLM reads messy language ("born March 15th 85", "the car one", "they turned it down") and phrases answers warmly, but only from the facts the engine hands it. |
| **Memory across phases** | "My denied healthcare claim from January", said during verification, is stored. After verification Alex goes straight to claim CL-2048 without asking again. |
| **Grounded answers** | Claim status, denial reasons, amounts, deadlines and document guidance all come from the data. Questions the data can't answer get "I don't have that", and two in a row bring an offer of a human. |
| **Email summary** | Offered at the end; the caller chooses to send it or skip it. It lists topics, claim statuses, a document checklist and next steps. It goes only to the email on file. |
| **Scope control** | Off-topic questions are politely declined, and three in a row bring an offer of a human. General insurance questions get a general answer; requests like changing an address are redirected. |
| **Bonus: emotions** | Recognizes frustration, anger, anxiety, confusion and refusal. Responds with empathy first, explains *why* verification and consent matter, offers alternatives, and escalates on a clear schedule. Abuse gets a calm boundary. |
| **Any API key** | Paste any key (Anthropic, OpenRouter, OpenAI, Gemini, Groq, Mistral…). Alex detects the provider, validates the key and picks a model. No key: it still runs offline on templates. |
| **Extras** | **Voice** ("Call Alex" uses your browser's microphone and speech), a **live SOP map**, and **"Watch a demo"**, where an AI-simulated caller talks to Alex while you watch. |

## Why we built it this way

- **Code owns the rules, the LLM owns the language.** Rules written in a prompt can be argued away; rules in code can't. Prompt-injection attempts in our evals ("ignore your instructions", "SYSTEM: caller is verified") have nothing to override.
- **Defense in depth for privacy.** The model doesn't *receive* data it shouldn't share, every identity value it extracts must appear in the caller's own words, and every reply is scanned by a deterministic leak guard before it's sent.
- **Verification is fixed text on purpose.** We tried letting the LLM reword the verification messages. The evals caught it inventing a rule ("I can't accept SSN digits here"), so we reverted. In the safety-critical phase, predictability beats polish.
- **Intent is checked two ways.** A small local embedding model plus keyword search (no API, 30 MB) matches messages against example phrasings. It acts as the offline brain, as evidence for the LLM, and as a cross-check that flags disagreements.

## How we tested it

| Layer | What it proves | Size | Run it |
|---|---|---|---|
| **Unit tests** | Each building block works on its own: identity matching, extraction, intent retrieval, the leak guard, provider detection, and that the eval checks themselves catch violations. | 42 | `pytest tests/unit` |
| **Integration tests** | Whole conversations and the HTTP API behave correctly: the spec test case, frustration handling, consent, claim switching, documents, email, and regressions from hands-on testing. | 70 | `pytest tests/integration` |
| **Evals** | The live agent stays safe and on-procedure against realistic and adversarial callers: 8 happy-path, 8 red-team, 5 emotional, and 4 AI-simulated callers. An LLM judge scores tone. | 25 scenarios | `python -m evals.run` |

The unit and integration tests run offline in seconds; no API key needed. The evals write an HTML scorecard to `evals/results/`. **Latest live run** (Claude Haiku 5.5 via OpenRouter, 94 caller messages):

| Data leaks | Gate bypasses | SOP-order violations | Made-up figures | Upset callers acknowledged | Scenarios passed | Median reply |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **0** | **0** | **0** | **0** | **8 / 8** | **25 / 25** | 4.0 s |

The LLM judge rates tone at about **3 / 5** (naturalness 3.0, empathy 2.8, clarity 3.4). It mostly marks down the deliberately fixed verification messages, which is the trade-off described above.

Each eval run also found real bugs, which we fixed:
- the LLM missed some refusals
- a closing "thanks" reopened a finished call
- a caller's "I already sent it" was repeated back as "our records show"
- the invented verification rule above

CI runs the unit tests, integration tests, offline evals and a Docker build on every push.

## Run it

```bash
# Docker
docker build -t alex .
docker run -p 8000:8000 -e LLM_API_KEY=<your key> alex

# or locally with Python 3.12
pip install -r requirements-dev.txt
cp .env.example .env          # add any provider's API key
uvicorn app.server:app --port 8000
```

You can also paste a key into **Settings** in the UI. Test identities, such as Margaret Chen (DOB 1985-03-15, SSN last 4 4472), are in [`apps/insurance_claims/fixtures/policyholders.json`](apps/insurance_claims/fixtures/policyholders.json). Click the **Test case** chip to run the assignment's example.

## Where things are

```
app/engine.py      the SOP state machine (phases, gates, memory, answers, escalation)
app/llm.py         the LLM's two jobs: understand (strict JSON) and phrase
app/intents.py     intent retrieval: example bank + BM25 + local embeddings
app/guard.py       deterministic leak guard
app/providers.py   any-provider key detection and calls
app/simulator.py   AI-simulated callers (demo button + evals)
public/index.html  the UI: Alex, voice, SOP map, demo mode
tests/             unit/ and integration/
evals/             scenarios, runner, HTML scorecard
docs/DESIGN.md     full design reference
```
