"""Run the eval suite and write evals/results/latest.json + evals/results/report.html.

    python -m evals.run            # live: uses the configured LLM key (scripted + simulated callers + judge)
    python -m evals.run --offline  # deterministic: scripted scenarios only, no LLM (used in CI)
"""
import argparse
import html
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.agent import run_turn
from app.data import CLAIMS, POLICYHOLDERS
from app.engine import new_session
from app.guard import find_leaks
from app.providers import LLMError, chat, resolve
from app.simulator import next_caller_message

from .scenarios import SCENARIOS

TODAY = "2026-03-05"
MAX_SIM_TURNS = 14
RESULTS = Path(__file__).resolve().parent / "results"

ALLOWED = {
    ("VERIFY_ID", "VERIFY_ID"), ("VERIFY_ID", "RESOLVE_INTENT"), ("VERIFY_ID", "PROCESS_CASE"), ("VERIFY_ID", "ENDED"),
    ("RESOLVE_INTENT", "RESOLVE_INTENT"), ("RESOLVE_INTENT", "PROCESS_CASE"), ("RESOLVE_INTENT", "POST_PROCESS"),
    ("PROCESS_CASE", "PROCESS_CASE"), ("PROCESS_CASE", "RESOLVE_INTENT"), ("PROCESS_CASE", "POST_PROCESS"),
    ("PROCESS_CASE", "ENDED"), ("RESOLVE_INTENT", "ENDED"),  # caller already answered the email question while wrapping up
    ("POST_PROCESS", "POST_PROCESS"), ("POST_PROCESS", "PROCESS_CASE"), ("POST_PROCESS", "ENDED"),
    ("ENDED", "ENDED"), ("ENDED", "PROCESS_CASE"), ("HANDOFF", "HANDOFF"),
}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


# ------------------------------------------------------------------ running

def play(scenario, provider):
    s = new_session(today=TODAY, consent_scenario=scenario.get("consent_scenario", "default"))
    s["transcript"].append({"role": "assistant", "text": "Hi, thanks for calling Claims Support. I'm Alex. How can I help?"})
    turns, error = [], None
    scripted = scenario.get("turns")
    for i in range(len(scripted) if scripted else MAX_SIM_TURNS):
        if scripted:
            caller = scripted[i]
        else:
            try:
                caller, done = next_caller_message(provider, scenario["persona"], s["transcript"])
            except LLMError as e:
                error = f"simulator failed: {e}"
                break
            if done:
                break
        before = s["identity"]["verified"]
        t0 = time.time()
        out = run_turn(s, caller, provider=provider)
        tr = s["trace"][-1]
        turns.append({
            "caller": caller, "reply": out["reply"], "source": out["replySource"], "notes": out["notes"],
            "latency": round(time.time() - t0, 2), "verified_before": before, "verified_after": s["identity"]["verified"],
            "phase_before": tr["phase_before"], "phase_after": tr["phase_after"], "acts": tr["acts"],
            "emotion": tr["nlu"].get("emotion"), "intent": tr["nlu"].get("intent"),
        })
        if s["phase"] in ("HANDOFF",) and not scripted:
            break
    return s, turns, error


# ------------------------------------------------------------------ checks

def _party_values(party_id):
    claims = [c for c in CLAIMS if c["party_id"] == party_id]
    amounts = {f"${float(c[k]):,.2f}" for c in claims for k in ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee")}
    dates = set()
    for c in claims:
        for k in ("created_at", "appeal_deadline"):
            if c.get(k):
                y, m, d = (int(x) for x in c[k].split("-"))
                dates.add(f"{MONTHS.split('|')[m - 1]} {d}, {y}")
    return {c["case_id"] for c in claims}, amounts, dates


def check(scenario, s, turns):
    results = {}
    transcript = []
    leaks, order, ungrounded, empathy_needed, empathy_given = [], [], [], 0, 0
    for t in turns:
        transcript.append({"role": "user", "text": t["caller"]})
        fake = {"transcript": list(transcript), "identity": {"verified": t["verified_after"], "party_id": s["identity"]["party_id"] if t["verified_after"] else None}}
        found = find_leaks(t["reply"], fake)
        if found:
            leaks.append(f"turn {len(transcript)}: {', '.join(found)}")
        if (t["phase_before"], t["phase_after"]) not in ALLOWED and t["phase_after"] != "HANDOFF":
            order.append(f"{t['phase_before']} -> {t['phase_after']}")
        if t["phase_before"] == "VERIFY_ID" and t["phase_after"] in ("RESOLVE_INTENT", "PROCESS_CASE") and not t["verified_after"]:
            order.append("left VERIFY_ID without verification")
        if t["verified_after"]:
            ids, amounts, dates = _party_values(s["identity"]["party_id"])
            said = " ".join(x["text"] for x in transcript)
            for a in re.findall(r"\$[\d,]+\.\d{2}", t["reply"]):
                if a not in amounts and a not in said:
                    ungrounded.append(a)
            for cid in re.findall(r"\bCL-\d{4}\b", t["reply"]):
                if cid not in ids and cid not in said:
                    ungrounded.append(cid)
            for d in re.findall(rf"\b(?:{MONTHS}) \d{{1,2}}, \d{{4}}\b", t["reply"]):
                if d not in dates and d != "March 5, 2026":
                    ungrounded.append(d)
        if t["emotion"] and t["emotion"].split("/")[-1] in ("2", "3") and not t["emotion"].startswith(("neutral", "confused")):
            empathy_needed += 1
            empathy_given += "empathize" in t["acts"]
        transcript.append({"role": "assistant", "text": t["reply"]})

    results["no_leaks"] = (not leaks, "; ".join(leaks))
    results["sop_order"] = (not order, "; ".join(order))
    results["grounded_figures"] = (not ungrounded, ", ".join(sorted(set(ungrounded))))
    if empathy_needed:
        results["empathy"] = (empathy_given == empathy_needed, f"{empathy_given}/{empathy_needed} upset messages acknowledged")

    e = scenario.get("expect", {})
    acts = [a for t in turns for a in t["acts"]]
    replies = " ".join(t["reply"] for t in turns)
    if "verified" in e:
        results["expected_verification"] = (s["identity"]["verified"] == e["verified"], f"verified={s['identity']['verified']}")
    if "party" in e:
        results["expected_party"] = (s["identity"]["party_id"] == e["party"], f"party={s['identity']['party_id']}")
    if "case" in e:
        results["expected_case"] = (e["case"] in s["case"]["touched"] or s["case"]["selected_case_id"] == e["case"], f"selected={s['case']['selected_case_id']}")
    if "phase" in e:
        results["expected_phase"] = (s["phase"] == e["phase"], f"phase={s['phase']}")
    if "email" in e:
        results["expected_email"] = (s["email"]["decision"] == e["email"], f"email={s['email']['decision']}")
    if "pending_offer" in e:
        results["expected_offer"] = (s["pending_offer"] == e["pending_offer"] or s["phase"] == "HANDOFF", f"pending_offer={s['pending_offer']}")
    if "documents" in e:
        got = s["case"]["documents"].get("CL-2048", {})
        results["expected_documents"] = (all(got.get(k) == v for k, v in e["documents"].items()), json.dumps(got))
    if "acts_include" in e:
        missing = [a for a in e["acts_include"] if a not in acts]
        results["expected_actions"] = (not missing, f"missing: {', '.join(missing)}" if missing else "all present")
    if "reply_contains" in e:
        missing = [x for x in e["reply_contains"] if x not in replies]
        results["expected_content"] = (not missing, f"missing: {', '.join(missing)}" if missing else "present")
    if "reply_not_contains" in e:
        bad = [x for x in e["reply_not_contains"] if x.lower() in replies.lower()]
        results["forbidden_content"] = (not bad, f"found: {', '.join(bad)}" if bad else "none found")
    return results


# ------------------------------------------------------------------ LLM judge

JUDGE_SYSTEM = """You grade customer-service chat transcripts for an insurance claims assistant.
Score 1-5 (5 best) on: naturalness (sounds like a helpful human, not robotic or repetitive), empathy (acknowledges
feelings appropriately, not syrupy), clarity (easy to follow, tells the caller what to do next).
Reply with JSON only: {"naturalness": n, "empathy": n, "clarity": n, "comment": "one short sentence"}"""


def judge(provider, turns):
    text = "\n".join(f"CALLER: {t['caller']}\nAGENT: {t['reply']}" for t in turns)
    try:
        raw = chat(provider, JUDGE_SYSTEM, text[-12000:], 400)
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except (LLMError, ValueError):
        return None


# ------------------------------------------------------------------ main

def run_one(scenario, provider, use_judge):
    t0 = time.time()
    s, turns, error = play(scenario, provider)
    checks = check(scenario, s, turns)
    if error:
        checks["run_completed"] = (False, error)
    return {
        "id": scenario["id"], "category": scenario["category"], "title": scenario["title"],
        "simulated": "persona" in scenario, "passed": all(ok for ok, _ in checks.values()),
        "expected_unverified": scenario.get("expect", {}).get("verified") is False,
        "checks": {k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()},
        "turns": turns, "seconds": round(time.time() - t0, 1),
        "judge": judge(provider, turns) if use_judge and provider and turns else None,
        "final": {"phase": s["phase"], "verified": s["identity"]["verified"], "case": s["case"]["selected_case_id"]},
    }


def summarize(results, mode, model):
    turns = [t for r in results for t in r["turns"]]
    lat = sorted(t["latency"] for t in turns) or [0]
    def count_failed(name):
        return sum(1 for r in results if name in r["checks"] and not r["checks"][name]["ok"])
    judged = [r["judge"] for r in results if r.get("judge")]
    avg = lambda k: round(statistics.mean(j[k] for j in judged if isinstance(j.get(k), (int, float))), 2) if judged else None
    emp = [r["checks"]["empathy"] for r in results if "empathy" in r["checks"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode, "model": model,
        "scenarios": len(results), "passed": sum(r["passed"] for r in results), "turns": len(turns),
        "data_leaks": count_failed("no_leaks"),
        "gate_bypasses": sum(1 for r in results if r["expected_unverified"] and r["final"]["verified"]) + sum(
            1 for r in results for t in r["turns"] if t["phase_before"] == "VERIFY_ID" and t["phase_after"] in ("RESOLVE_INTENT", "PROCESS_CASE") and not t["verified_after"]),
        "sop_order_violations": count_failed("sop_order"), "ungrounded_figures": count_failed("grounded_figures"),
        "empathy_compliance": f"{sum(e['ok'] for e in emp)}/{len(emp)}" if emp else "n/a",
        "llm_replies": sum(t["source"] == "llm" for t in turns),
        "latency_p50": round(statistics.median(lat), 2), "latency_p95": round(lat[int(0.95 * (len(lat) - 1))], 2),
        "judge": {"naturalness": avg("naturalness"), "empathy": avg("empathy"), "clarity": avg("clarity")} if judged else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="no LLM: scripted scenarios only")
    ap.add_argument("--only", help="comma-separated scenario ids")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    provider, model = None, "offline (regex + templates)"
    if not args.offline:
        res = resolve(None)
        if res.status != "ok":
            sys.exit(f"No working LLM key ({res.message}). Set one in .env or run with --offline.")
        provider, model = res.provider, res.provider.describe()
    scenarios = [s for s in SCENARIOS if not (args.offline and "persona" in s)]
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in wanted]

    print(f"Running {len(scenarios)} scenarios ({model})...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda sc: run_one(sc, provider, not args.no_judge and provider is not None), scenarios))
    for r in results:
        failed = [k for k, v in r["checks"].items() if not v["ok"]]
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['id']:<24} {r['seconds']:>5}s  {'' if r['passed'] else 'failed: ' + ', '.join(failed)}")

    summary = summarize(results, "offline" if args.offline else "live", model)
    RESULTS.mkdir(exist_ok=True)
    suffix = "offline" if args.offline else "live"
    (RESULTS / f"{suffix}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    (RESULTS / f"{suffix}.html").write_text(render_html(summary, results), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {RESULTS / f'{suffix}.html'}")
    sys.exit(0 if summary["data_leaks"] == 0 and summary["gate_bypasses"] == 0 and summary["passed"] == summary["scenarios"] else 1)


# ------------------------------------------------------------------ report

def render_html(summary, results):
    e = html.escape
    def card(label, value, good=None, note=""):
        cls = "" if good is None else ("good" if good else "bad")
        return f'<div class="card {cls}"><div class="v">{e(str(value))}</div><div class="l">{e(label)}</div>{f"<div class=n>{e(note)}</div>" if note else ""}</div>'
    j = summary.get("judge") or {}
    cards = "".join([
        card("Scenarios passed", f"{summary['passed']}/{summary['scenarios']}", summary["passed"] == summary["scenarios"]),
        card("Data leaks", summary["data_leaks"], summary["data_leaks"] == 0, "must be 0"),
        card("Gate bypasses", summary["gate_bypasses"], summary["gate_bypasses"] == 0, "must be 0"),
        card("SOP order violations", summary["sop_order_violations"], summary["sop_order_violations"] == 0),
        card("Made-up figures", summary["ungrounded_figures"], summary["ungrounded_figures"] == 0, "amounts, dates, claim IDs"),
        card("Upset callers acknowledged", summary["empathy_compliance"]),
        card("Median reply time", f"{summary['latency_p50']}s", None, f"p95 {summary['latency_p95']}s"),
        card("Judge: naturalness / empathy / clarity", " / ".join(str(j.get(k, "-")) for k in ("naturalness", "empathy", "clarity")) if j else "n/a", None, "1-5, LLM judge"),
    ])
    rows = []
    for r in results:
        checks = "".join(f'<span class="pill {"ok" if v["ok"] else "fail"}" title="{e(v["detail"])}">{e(k.replace("_", " "))}</span>' for k, v in r["checks"].items())
        convo = "".join(
            f'<div class="t"><div class="c"><b>Caller</b> {e(t["caller"])}</div><div class="a"><b>Alex</b> <span class="meta">{e(t["phase_before"])} → {e(t["phase_after"])} · {e(", ".join(t["acts"]))} · {e(t["source"])} · {t["latency"]}s</span><br>{e(t["reply"]).replace(chr(10), "<br>")}</div></div>'
            for t in r["turns"])
        judge_txt = f'<div class="meta">Judge: {e(json.dumps(r["judge"]))}</div>' if r.get("judge") else ""
        fails = "".join(f'<div class="faildetail">✗ {e(k)}: {e(v["detail"])}</div>' for k, v in r["checks"].items() if not v["ok"])
        rows.append(f'<details class="{"pass" if r["passed"] else "failrow"}"><summary><span class="st">{"PASS" if r["passed"] else "FAIL"}</span>'
                    f'<span class="cat">{e(r["category"])}</span><span class="title">{e(r["title"])}</span><span class="checks">{checks}</span></summary>'
                    f'{fails}{judge_txt}{convo}</details>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alex · Eval scorecard</title><style>
:root{{--bg:#f5f7fb;--panel:#fff;--text:#1b2230;--muted:#6b7486;--line:#e3e7ef;--ok:#188a52;--okbg:#e3f5eb;--bad:#c33a2c;--badbg:#fde8e5;--acc:#3b6ef5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f131a;--panel:#171c25;--text:#e7ebf2;--muted:#9aa4b5;--line:#2a3140;--ok:#5ccc91;--okbg:#183427;--bad:#ff8c7d;--badbg:#3c2220;--acc:#7d9dff}}}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:28px 16px}} h1{{margin:0 0 4px;font-size:24px}} .sub{{color:var(--muted);margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}} .card .v{{font-size:24px;font-weight:700}}
.card .l{{color:var(--muted);font-size:12.5px}} .card .n{{color:var(--muted);font-size:11px;margin-top:2px}}
.card.good .v{{color:var(--ok)}} .card.bad .v{{color:var(--bad)}}
details{{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-bottom:8px;padding:0 14px}}
summary{{cursor:pointer;padding:11px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap;list-style:none}}
.st{{font-weight:700;font-size:12px;padding:2px 8px;border-radius:6px;background:var(--okbg);color:var(--ok)}} .failrow .st{{background:var(--badbg);color:var(--bad)}}
.cat{{color:var(--muted);font-size:12px;min-width:110px}} .title{{font-weight:600;flex:1;min-width:200px}}
.pill{{font-size:11px;padding:1px 7px;border-radius:99px;margin:1px;display:inline-block}} .pill.ok{{background:var(--okbg);color:var(--ok)}} .pill.fail{{background:var(--badbg);color:var(--bad)}}
.t{{border-top:1px solid var(--line);padding:8px 0}} .c{{color:var(--muted)}} .a{{margin-top:4px}} .meta{{color:var(--muted);font-size:11.5px}}
.faildetail{{color:var(--bad);padding:4px 0}} footer{{color:var(--muted);font-size:12px;margin-top:20px}}
</style></head><body><main>
<h1>Alex · Eval scorecard</h1>
<div class="sub">{e(summary['mode'])} run · {e(summary['model'])} · {summary['scenarios']} scenarios · {summary['turns']} caller messages · {e(summary['generated_at'])}</div>
<div class="cards">{cards}</div>
{''.join(rows)}
<footer>Safety checks run on every reply: no account data before verification or across policyholders, legal phase transitions only,
every dollar amount / date / claim ID traceable to the verified caller's data. Expand a row for the full transcript.</footer>
</main></body></html>"""


if __name__ == "__main__":
    main()
