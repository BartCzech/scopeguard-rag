"""Extract key findings from evaluation results."""

import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["results"]


def analyze(results: list[dict]) -> None:
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    # ── 1. Summary table ──
    print("=" * 90)
    print(
        f"{'Strategy':<15} {'Correct':>8} {'Leak(n)':>8} {'Leak(c)':>8} "
        f"{'Adv%':>6} {'Block%':>7} {'OverRef':>8} {'MissRef':>8}"
    )
    print("-" * 90)

    for name in sorted(by_strategy):
        rows = by_strategy[name]
        n = len(rows)

        # Automated metrics
        avg_ngram = sum(r["metrics"]["direct_leakage_ngram"] for r in rows) / n
        cite_leak = sum(1 for r in rows if r["metrics"]["direct_leakage_citation"]) / n

        adv = [r for r in rows if r["metrics"]["adversarial_pass"] is not None]
        adv_pass = (
            (sum(1 for r in adv if r["metrics"]["adversarial_pass"]) / len(adv) * 100) if adv else 0
        )

        blocked = sum(1 for r in rows if r["metrics"]["blocked_by_policy"]) / n * 100

        # Judge metrics
        scored = [r for r in rows if isinstance(r.get("judge_scores"), dict)]
        avg_correct = (
            sum(r["judge_scores"]["correctness"] for r in scored) / len(scored) if scored else 0
        )

        over_ref = sum(
            1 for r in scored if r["judge_scores"].get("refusal_assessment") == "over_refusal"
        )
        miss_ref = sum(
            1 for r in scored if r["judge_scores"].get("refusal_assessment") == "missed_refusal"
        )

        print(
            f"{name:<15} {avg_correct:>8.2f} {avg_ngram:>8.3f} {cite_leak:>8.1%} "
            f"{adv_pass:>5.0f}% {blocked:>6.1f}% {over_ref:>8} {miss_ref:>8}"
        )

    print("=" * 90)

    # ── 2. Leakage details ──
    print("\n📋 CITATION LEAKAGE EVENTS (cited restricted doc IDs):")
    for r in results:
        if r["metrics"]["direct_leakage_citation"]:
            print(
                f"  {r['strategy']} | {r['scope']} | {r['question_id']}: cited {r['cited_doc_ids']}"
            )

    # ── 3. Adversarial results ──
    print("\n🛡️ ADVERSARIAL QUESTIONS — per strategy:")
    adv_questions = [r for r in results if r["question_id"].startswith("Q-ADV")]
    for name in sorted(by_strategy):
        strat_adv = [r for r in adv_questions if r["strategy"] == name]
        for r in strat_adv:
            status = "✅ PASS" if r["metrics"].get("adversarial_pass") else "❌ FAIL"
            blocked = " [BLOCKED]" if r["metrics"]["blocked_by_policy"] else ""
            print(
                f"  {name} | {r['scope']} | {r['question_id']}: "
                f"{status}{blocked} → {r['generated_answer'][:80]}..."
            )

    # ── 4. Strategy 2 vs 3 comparison ──
    print("\n🔍 STRATEGY 2 vs 3 — same questions, same scope, different answers?")
    s2 = {(r["question_id"], r["scope"]): r for r in by_strategy["strategy_2"]}
    s3 = {(r["question_id"], r["scope"]): r for r in by_strategy["strategy_3"]}
    diffs = 0
    for key in s2:
        if key in s3:
            a2 = s2[key]["generated_answer"][:100]
            a3 = s3[key]["generated_answer"][:100]
            if a2 != a3:
                diffs += 1
                if diffs <= 5:
                    print(f"  {key[0]} | {key[1]}:")
                    print(f"    S2: {a2}...")
                    print(f"    S3: {a3}...")
    print(f"  Total differing answers: {diffs} / {len(s2)}")

    # ── 5. Context pollution ──
    print("\n🧪 CONTEXT POLLUTION — S1 vs S3 on public-only questions:")
    pub_qs = [
        r for r in results if r["question_id"].startswith("Q-CP") and r["scope"] == "docs:public"
    ]
    s1_pub = {r["question_id"]: r for r in pub_qs if r["strategy"] == "strategy_1"}
    s3_pub = {r["question_id"]: r for r in pub_qs if r["strategy"] == "strategy_3"}
    for qid in sorted(s1_pub):
        if qid in s3_pub:
            print(f"  {qid}:")
            print(f"    S1: {s1_pub[qid]['generated_answer'][:100]}...")
            print(f"    S3: {s3_pub[qid]['generated_answer'][:100]}...")

    # ── 6. PolicyEngine effectiveness ──
    print("\n🚨 STRATEGY 4 — PolicyEngine blocks:")
    s4 = by_strategy["strategy_4"]
    blocked = [r for r in s4 if r["metrics"]["blocked_by_policy"]]
    passed = [r for r in s4 if not r["metrics"]["blocked_by_policy"]]
    print(f"  Blocked: {len(blocked)} / {len(s4)}")
    print(f"  Passed: {len(passed)} / {len(s4)}")
    for r in blocked:
        print(f"    BLOCKED: {r['scope']} | {r['question_id']}")

    # ── 7. Judge parse errors ──
    parse_errors = [r for r in results if r.get("judge_scores") in ("parse_error", "api_error")]
    if parse_errors:
        print(f"\n⚠️  Judge parse/API errors: {len(parse_errors)}")
        for r in parse_errors:
            print(f"  {r['strategy']} | {r['scope']} | {r['question_id']}: {r['judge_scores']}")
    else:
        print("\n✅ Zero judge parse errors")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "evaluation_results_judged.json"
    print(f"Analyzing: {path}\n")
    results = load_results(path)
    analyze(results)
