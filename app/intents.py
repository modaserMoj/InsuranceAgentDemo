"""Intent detection by hybrid retrieval over example utterances.

Each intent has a bank of example statements (seeded with the guideline fixture's own match_any phrases).
A caller message is scored against every example with BM25 (exact terms) and a small local embedding
model (meaning), and the best-scoring intent wins. The score doubles as a confidence value: below the
threshold the message does not clearly match any intent and the caller is redirected.
"""
import logging
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import GUIDELINES

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "potion-base-8M"
MODEL_NAME = "minishlab/potion-base-8M"
EMBED_WEIGHT = 0.7
BM25_CAP = 6.0
CONFIDENT = 0.50
MIN_SCORE = 0.38

INTENT_DESCRIPTIONS = {
    "status_inquiry": "where a claim stands, whether it was approved, processed or paid",
    "denial_question": "why a claim was denied and what is needed to fix it",
    "document_submission": "what documents are needed, their format, and how/where to send them",
    "next_steps": "what the caller should do next, timelines, what happens after submitting",
    "payment_question": "amounts: how much was paid, expected reimbursement, allowed maximum",
    "appeal": "appealing or disputing a claim decision, appeal deadlines",
    "general_claim_question": "some other question about a specific claim on file",
    "general_insurance_question": "a general insurance concept, not about their own account (deductible, in-network, copay...)",
    "other_request": "an insurance request this assistant cannot serve: update personal info, file a new claim, change or cancel a policy, pay a premium or bill, check their own plan's coverage, make a complaint",
}
INTENTS = list(INTENT_DESCRIPTIONS)

EXAMPLES = {
    "status_inquiry": [
        "what is the status of my claim", "where is my claim at", "has my claim been processed yet",
        "any update on my claim", "is my claim still open", "was my claim approved", "did my claim go through",
        "check on my claim", "what's happening with my claim", "is my claim finished", "has it been settled",
        "is my auto claim done", "I want to know how my claim is going", "track my claim",
        "has my claim been paid out yet", "is the claim complete",
    ],
    "denial_question": [
        "why was my claim denied", "why did you deny my claim", "my claim got rejected why",
        "they turned down my claim", "what was the reason for the denial", "why was it declined",
        "I got a denial letter", "my claim was refused", "what went wrong with my claim",
        "why didn't my claim get approved", "explain the denial", "you guys denied my claim",
        "calling about my denied claim", "what's missing from my claim",
        "my claim was not approved", "they said no to my claim", "my claim wasn't accepted",
    ],
    "document_submission": [
        "what documents do you need", "how do I submit the documents", "where do I upload the files",
        "can I send the paperwork by fax", "what format should the file be", "can I send a photo of the report",
        "do you accept scans", "how do I send you the pathology report", "where do I mail the documents",
        "what should the office note include", "is a pdf okay", "I want to upload the missing paperwork",
        "what if I don't have the document", "I can't get the report from the lab",
    ],
    "next_steps": [
        "what do I do now", "what are the next steps", "what happens next", "how long does it take after I submit",
        "when will I hear back", "how soon do I need to send it", "what should I do to fix this",
        "how will I know you got it", "will I get a confirmation", "how long is the review",
        "when should I expect an answer", "what's the timeline",
    ],
    "payment_question": [
        "how much did you pay", "how much will I get reimbursed", "what's the expected reimbursement",
        "why did I get paid less", "what is the allowed amount", "when will I get my money",
        "how much was paid on the claim", "what's the net pay", "did they send the payment",
        "how much is covered on this claim", "why is the payment zero",
    ],
    "appeal": [
        "I want to appeal", "how do I appeal the decision", "can I dispute this", "what's the appeal deadline",
        "I want to contest the denial", "how do I get it reconsidered", "can I fight this decision",
        "is it too late to appeal", "request a review of the denial", "I disagree with the decision",
    ],
    "general_claim_question": [
        "I have a question about my claim", "can you tell me about my claim", "I'm calling about a claim",
        "help me with my claim", "something about my healthcare claim", "question regarding my dental claim",
        "I need help with my car claim", "explain my claim to me",
    ],
    "general_insurance_question": [
        "what is a deductible", "what does in-network mean", "what's the difference between copay and coinsurance",
        "what is an out of pocket maximum", "what does allowed amount mean", "what is a premium",
        "how does insurance reimbursement work", "what is an explanation of benefits", "what does EOB mean",
        "what's a pre-authorization", "what does out of network mean",
    ],
    "other_request": [
        "I want to update my address", "change my phone number", "update my email on file",
        "I need to file a new claim", "I want to start a new claim", "cancel my policy",
        "I want to change my plan", "pay my premium", "pay my bill", "is my surgery covered",
        "does my plan cover physical therapy", "I want to make a complaint", "add a dependent to my policy",
        "I lost my insurance card", "send me a new id card", "how much is my monthly premium",
        "does my policy cover this procedure", "is this treatment covered by my insurance", "am I covered for this",
    ],
}

# Fold the fixture's own trigger phrases into the bank under the first intent each topic is tagged with.
for _item in GUIDELINES["claim_followup_guidance"]:
    _target = _item["intent_hints"][0]
    for _phrase in _item.get("match_any", []):
        if len(_phrase.split()) >= 2 and _phrase not in EXAMPLES[_target]:
            EXAMPLES[_target].append(_phrase)

STOPWORDS = set("""a an the i me my we our you your it its is are was were be been am do does did to of in on for at by
with about from and or but so if this that these those there here can could would should will just please hi hello hey
im i'm it's that's what's""".split())


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9']+", text.lower())
    out = []
    for t in toks:
        t = t.strip("'")
        if not t or t in STOPWORDS:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if len(t) > len(suffix) + 3 and t.endswith(suffix):
                t = t[: -len(suffix)]
                break
        out.append(t)
    return out


class _BM25:
    def __init__(self, docs: list[list[str]], k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / len(docs)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            denom_base = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            for t in query:
                if t in tf:
                    out[i] += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + denom_base)
        return out


class IntentIndex:
    def __init__(self):
        self.labels, self.texts = [], []
        for intent, phrases in EXAMPLES.items():
            for p in phrases:
                self.labels.append(intent)
                self.texts.append(p)
        self.bm25 = _BM25([_tokens(t) for t in self.texts])
        self.model = _load_model()
        if self.model is not None:
            self.matrix = self._embed(self.texts)

    def _embed(self, texts):
        vecs = np.asarray(self.model.encode(texts), dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)

    def rank(self, text: str, top: int = 3) -> list[dict]:
        q = _tokens(text)
        if not q:
            return []
        lexical = np.minimum(self.bm25.scores(q) / BM25_CAP, 1.0)
        if self.model is not None:
            semantic = np.clip(self.matrix @ self._embed([text])[0], 0.0, 1.0)
            hybrid = EMBED_WEIGHT * semantic + (1 - EMBED_WEIGHT) * lexical
        else:
            hybrid = lexical
        best: dict[str, tuple[float, int]] = {}
        for i, (label, score) in enumerate(zip(self.labels, hybrid)):
            if label not in best or score > best[label][0]:
                best[label] = (float(score), i)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top]
        return [{"intent": k, "score": round(v[0], 3), "closest_example": self.texts[v[1]]} for k, v in ranked]


def _load_model():
    try:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from model2vec import StaticModel

        return StaticModel.from_pretrained(str(MODEL_DIR) if MODEL_DIR.exists() else MODEL_NAME)
    except Exception as e:  # missing package or no network: fall back to lexical-only retrieval
        log.warning("embedding model unavailable, using BM25 only: %s", e)
        return None


@lru_cache(maxsize=1)
def index() -> IntentIndex:
    return IntentIndex()


def classify(text: str) -> dict:
    """Returns {"intent": best intent or None, "confidence": score, "candidates": top-3 with closest examples}."""
    candidates = index().rank(text)
    if not candidates or candidates[0]["score"] < MIN_SCORE:
        return {"intent": None, "confidence": candidates[0]["score"] if candidates else 0.0, "candidates": candidates}
    return {"intent": candidates[0]["intent"], "confidence": candidates[0]["score"], "candidates": candidates}


def retrieval_mode() -> str:
    return "hybrid (BM25 + embeddings)" if index().model is not None else "BM25 only"
