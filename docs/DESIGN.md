# Design reference

> The short version is the [README](../README.md). This page is the detailed reference: every phase, guardrail and configuration option.

A conversational insurance claims support agent that follows a fixed business workflow but still talks naturally:

```
VERIFY_ID  ->  RESOLVE_INTENT  ->  PROCESS_CASE  ->  POST_PROCESS
   (strict)       (bounded LLM)      (grounded LLM)     (caller's choice)
                                  any phase -> HANDOFF (human)
```

The key idea: **the LLM never controls the workflow.** It does two narrow jobs on every turn:

1. **Understand.** It turns the caller's message into strict JSON (identity fields, intent, case hints, emotion, scope, yes/no). This uses Claude structured outputs.
2. **Phrase.** It rewrites the engine's chosen actions into natural language, using only the facts the engine hands it.

Everything in between is a deterministic SOP engine ([app/engine.py](../app/engine.py)). The engine decides the phase transitions, the identity gate, consent, which claim matches, what may be disclosed, when to escalate, and whether an email is sent.

## Quick start

### Option A: Docker

```bash
docker build -t sop-claims-agent .
docker run --rm -p 8000:8000 -e OPENROUTER_API_KEY=sk-or-... sop-claims-agent
# open http://localhost:8000
```

### Option B: Python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                   # put your key in OPENROUTER_API_KEY
uvicorn app.server:app --port 8000     # http://localhost:8000
pytest                                 # unit + integration tests, no API key needed
```

### API auth token

**Any major provider's API key works.** Paste it into the UI's **API key** field, or set it in `.env` or with `docker run -e`. The server works out which provider the key belongs to, validates it with that provider's free model-list endpoint, and picks a model. The header badge then shows the result, for example "✓ claude-haiku-5-5 via Anthropic", or "✗ API key rejected" with the reason.

| Key looks like | Provider | Default model |
|---|---|---|
| `sk-ant-...` | Anthropic (official SDK) | `claude-haiku-5-5` |
| `sk-or-...` | OpenRouter | `anthropic/claude-haiku-5.5` |
| `sk-proj-...` / `sk-...` | OpenAI (then DeepSeek) | newest `gpt-*-mini` |
| `AIza...` | Google Gemini | newest `gemini-*-flash` |
| `gsk_...` | Groq | `gpt-oss-120b` / Llama 70B |
| `xai-...` | xAI | newest `grok-*-mini` |
| no prefix | Mistral, Together, DeepSeek, OpenAI (tried in turn) | provider's small chat model |

A key with a recognizable prefix is only ever sent to its own provider. Environment variables are `LLM_API_KEY` (any provider), or the provider-specific ones: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `XAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY` and `TOGETHER_API_KEY`.

Structured output for the understanding step uses each provider's strict JSON schema mode. If an endpoint doesn't support that, the app drops to plain JSON mode, then to prompt-only JSON. Every result is checked and normalized against the schema before the engine uses it.

With no key, or a rejected one, the app runs in **offline mode**: regex understanding plus the SOP's template replies. The workflow is identical; only the language is less flexible.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | auto | Force a model (also settable per conversation in the UI's **Model** field) |
| `LLM_PROVIDER` | auto | Skip detection: `anthropic`, `openrouter`, `openai`, `gemini`, `groq`, `xai`, `mistral`, `deepseek` or `together` |
| `LLM_BASE_URL` | none | Any other OpenAI-compatible endpoint (vLLM, Ollama, a gateway) |
| `LLM_EFFORT` | `low` | Reasoning effort; low keeps chat latency down |
| `DEMO_TODAY` | `2026-03-05` in Docker, the real date otherwise | "Today" for appeal-deadline math. The fixtures date from early 2026. |
| `PORT` | `8000` | HTTP port |

## Using the test UI

- **Left: the chat.** Type as the caller, or click a sample chip. The chips cover the spec test case, the frustrated caller, out-of-scope questions, a representative calling, follow-up questions, a missing document, asking for a human, and wrap-up/yes/no.
- **Right: the SOP inspector.** It shows the live phase stepper and the identity gate (which PII fields are provided and which matched). It also shows cross-phase memory (with the phase each hint was captured in), the selected case, guardrail counters and the last turn's engine trace: actions taken, the parsed understanding, whether the reply came from the LLM or the template, and any guard notes. It also shows the simulated email outbox and the handoff note.
- **Header controls:**
  - The consent scenario (`default` approves, `timeout` never approves) for representative calls.
  - An offline toggle.

**Spec test case.** Paste this, or click "Test case":

> I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.

In one turn the agent:

1. Verifies her: full name, date of birth and SSN last four match. The policy number is used for lookup only.
2. Resolves `denied + healthcare + January` from memory to claim **CL-2048**.
3. Explains the denial, the two missing documents and the appeal deadline.
4. Offers to walk her through submitting the documents.

A longer demo script:

1. "Hi, this is Margaret Chen, calling about my denied healthcare claim from January." The agent stays in VERIFY_ID and stores the hint.
2. "I already told you who I am. This is ridiculous. Just tell me why my claim was denied." The agent acknowledges the frustration, explains why verification is required, and lists the accepted ID options. Nothing is disclosed.
3. "DOB 1985-03-15, SSN last four is 4472." The agent verifies her, finds CL-2048 without asking again, and explains the denial.
4. "Yes." The agent gives the document requirements and the submission method.
5. "What is RL?" The agent politely declines and steers back to the claim. Three off-topic messages in a row trigger an offer of a human.
6. "How long does it take after I submit?" and "What if I can't get the pathology report?" The agent answers from the guideline fixture.
7. "No, that's all." The agent offers an email summary, and the caller can say yes or no.

## How the phases differ in freedom

| Phase | What the LLM may do | What is hard-coded |
|---|---|---|
| **VERIFY_ID** (strict form) | Extract values from messy text ("born March 15th 85", "margaret at email dot com"), detect a representative, detect refusal or emotion, and capture intent and case hints for later. Its replies are **never** reworded by the LLM; this phase always uses the exact SOP templates. | Works like a form: it asks for one field at a time in a fixed order (full name → DOB → phone → email → SSN last 4) and won't move on until the current field is given in a valid format. A message is accepted only if it contains the field being asked for; other valid values in the same message are kept, so the one-message spec test case still works. A message that jumps ahead without the current field is rejected with "please send only the value I ask for". The caller can "skip" a field (alternate ID fields) only while 3 fields are still reachable. Verification needs **3 matching fields with zero mismatches**; the policy number is lookup-only. Failed attempts don't reveal which field was wrong; the form restarts, and after 3 failures the call goes to a human. Representatives must be listed for that policyholder and get the policyholder's consent. The phrasing model receives **no account data** in this phase. |
| **RESOLVE_INTENT** (bounded) | Interpret "the car one", "the one they rejected in January", "the second one". It can only select a `case_id` from the claims the engine offered. | Matching against the verified caller's claims only: case type, status, month and year from memory. One match is selected; several are listed as options; none falls back to the full list. |
| **PROCESS_CASE** (grounded) | Answer follow-up questions freely, in any order, choosing among the guideline topics. It can answer **several questions in one message** (up to 3), and it reports what the caller says about each document (have it / can't get it / already sent). It flags questions the file can't answer instead of guessing. | The facts are limited to the selected claim, the field definitions, the document and follow-up guidance from the fixtures, and a per-claim **document checklist**. Each document gets its own next step, and if none can be obtained, the fixture's human-review rule triggers an offer of a human. **Two unanswerable questions in a row** replace the "anything else?" question with an offer of a claims representative. Switching to another claim goes back through RESOLVE_INTENT. |
| **POST_PROCESS** | Phrase the offer and the closing. | It always offers an email summary to the **email on file** (masked; other addresses are refused). Sending happens only on an explicit yes. The summary is built deterministically from tracked topics, claim statuses and follow-up items, so it can't hallucinate. |

## Safety layers

1. **Data minimization.** Before verification, neither model receives any account or claim data. They see only the caller's own words.
2. **Grounding check on extraction.** Every identity value the LLM extracts must be traceable to the caller's message, or it is dropped. For example, a hallucinated SSN never reaches the verifier ([app/nlu.py](../app/nlu.py)).
3. **Deterministic output guard.** Every LLM reply is scanned for data the caller isn't entitled to see: claim ids, denial reasons, documents, dates, amounts, and other people's PII. That means everything before verification, and other parties' data after it. If anything is found, the engine's safe template is used instead ([app/guard.py](../app/guard.py)).
4. **Template floor.** If the model fails (refusal, timeout, bad key), the turn still completes correctly with the SOP template, and the reason is shown in the inspector.
5. **Refusal fallbacks.** Opus/Sonnet 5.x requests use `fallbacks: "default"`, so a safety-classifier false positive is retried on a fallback model instead of failing the turn.

## Intent detection

**The intent set.** Five intents come straight from the guideline fixture's `intent_hints`: `status_inquiry`, `denial_question`, `document_submission`, `next_steps` and `general_claim_question`. Two more are backed by fields in the claims data: `payment_question` and `appeal`. Two catch-alls cover everything else:
- **`general_insurance_question`** is for a concept question, for example "what's a deductible?". After verification the LLM gives a short general answer that is marked as general information and never describes the caller's own plan. Offline, it explains that a representative can help.
- **`other_request`** is for an insurance request this chat can't serve: an address change, a new claim, coverage on their own plan, billing or a complaint. The agent says so, lists what it *can* do, offers a human, and adds the request to the email's follow-up items.

Off-topic messages ("what is RL?") are separate: they're refused by the scope check, not routed as requests.

**How it's detected: hybrid retrieval plus the LLM** ([app/intents.py](../app/intents.py)).
1. Every intent has a bank of example phrasings, seeded with the fixture's own `match_any` phrases.
2. Identity values and their labels are stripped first, so "my phone is …" isn't read as "change my phone number".
3. The message is then scored against every example with **BM25** (exact terms) and a **small local embedding model** ([model2vec](https://github.com/MinishLab/model2vec) `potion-base-8M`). That's 30 MB, bundled in `models/`, CPU-only, with no API key and no network.
4. The hybrid score is a confidence value. Below 0.38 there is no intent: the message is a greeting, an answer, or noise.

With an LLM key, the top three retrieval candidates are passed to the LLM as evidence. The LLM makes the final call, and retrieval fills in only when the LLM returns nothing and retrieval is confident. Disagreements are flagged in the inspector panel. Without a key, retrieval alone decides.

**Measured on phrasings that are not in the example bank:**
- 15 of 16 correct on a fresh set.
- 21 or more of 22 on the set in [tests/test_intents.py](../tests/test_intents.py).
- Every off-topic or bare-value message scores below the threshold.

**Intent also narrows claim matching.** "They turned down my healthcare claim from January" matches both a January 2025 and a January 2026 healthcare claim, but a denial question selects the denied one.

## Cross-phase memory

Every turn's understanding is stored no matter which phase it arrives in. That includes intent, case hints (type, status, month, year, id), identity fields, representative details, and a question asked before verification. The engine stays in its current phase and uses each item only once its gate allows.

- An intent or hint given during VERIFY_ID is acknowledged in the caller's own words ("I've noted that you're calling about your denied healthcare claim from January"). After verification it resolves the case directly.
- Hints that arrive during PROCESS_CASE for a different claim send the conversation back through RESOLVE_INTENT.

## Emotional support and SOP recovery

**Recognition.** Every message gets:
- an emotion label (frustrated, angry, anxious, confused, sad) with an intensity from 0 to 3
- a separate **refusal** flag ("I already told you", "why do you need that", "just tell me")
- an **abusive** flag (insults or cursing at the agent, as opposed to strong frustration about the situation)

The LLM supplies these, and regex covers the obvious cases offline.

**Empathy first, never copy-pasted.** At intensity 2 or more, the reply opens with an acknowledgement before any SOP step. Each emotion has escalating variants, so a second acknowledgement sounds different and more conciliatory, and intensity 3 goes straight to the strongest. Mild signals don't trigger it, so plain questions get plain answers.

**Explain, offer alternatives, keep moving: identity verification.** On a refusal, the agent:
1. explains why claim details are protected (in softer wording on each repeat)
2. spells out the allowed alternatives, for example: "I have your full name already, so I need 2 more. If you don't have your date of birth handy, just say "skip" and we can use your phone number on file or email address on file instead."
3. continues the form

The gate itself is deterministic code, so no amount of pushback reveals anything.

**The same for consent.** If a representative pushes back while consent is pending, the agent:
1. acknowledges it and explains that consent is the policyholder's legal protection
2. re-checks the consent status
3. lists the options: the policyholder approves on their phone, the policyholder joins the call, or a human transfer

**Knowing when to stop.**

| Situation | Agent behaviour |
|---|---|
| 1st refusal (verification or consent) | Acknowledge, explain, offer alternatives |
| 2nd refusal | The same, plus an offer of a human |
| 4th refusal | Automatic handoff, with a note for the human agent |
| Strongly upset twice after verification, or one message at intensity 3 | One offer to connect a person, replacing the usual "anything else?" question |
| Abuse | A calm boundary ("I need us to keep the conversation respectful") while still helping; a second abusive message brings an offer of a person |
| Caller asks for a human | Transferred immediately, at any point |

**Honesty under pressure.** When an upset caller says "I already sent those documents", the agent records that as *the caller's account*. It says the file still lists them as missing and recommends re-uploading them. It never claims "our records show it was received".

## Project layout

```
app/
  engine.py      SOP state machine: phases, gates, memory, case matching, answers, checklist, escalation, email
  data.py        fixture loading + tools (verify identity, claims, consent, guidance)
  extract.py     deterministic understanding: regex identity values + signals; intent via retrieval
  intents.py     intent bank + hybrid BM25 / local-embedding retrieval with confidence
  nlu.py         merges LLM + deterministic understanding, drops ungrounded identity values
  llm.py         the LLM's two jobs: structured understanding and constrained phrasing
  providers.py   any-provider key detection, validation, model choice, chat calls
  guard.py       deterministic disclosure guard
  simulator.py   LLM-simulated caller personas (demo button + evals)
  agent.py       one turn: understand -> engine step -> phrase -> guard
  server.py      FastAPI app + JSON API
public/index.html   chat UI: Alex mascot, voice, SOP map, demo mode, inspector
models/             bundled 30 MB embedding model for intent retrieval
tests/unit/         building blocks in isolation
tests/integration/  full conversations + HTTP API
evals/              scenarios, runner, HTML scorecard (python -m evals.run [--offline])
```

### HTTP API

- `POST /api/sessions` `{ api_key?, model?, consent_scenario?, offline? }` returns `{ reply, llm, state }`.
- `POST /api/sessions/:id/messages` `{ text }` returns `{ reply, replySource, notes, state }`.
- `POST /api/sessions/:id/simulate` `{ persona }` returns `{ text, done }`. This is the next line from an AI-simulated caller.
- `GET /api/personas`, `GET /api/sessions/:id`, `GET /api/config`, `GET /healthz`.

## Notes and limitations

- Sessions live in memory, and the email is simulated (it appears in the inspector's outbox).
- The phrasing step adds one LLM call per turn on top of understanding. Set `ANTHROPIC_EFFORT`, or a faster `ANTHROPIC_MODEL`, if latency matters more than nuance.
- `CL-2048`'s appeal deadline (2026-03-18) is computed against `DEMO_TODAY`. With the real current date, the agent correctly says the deadline has passed and points the caller to human review.
