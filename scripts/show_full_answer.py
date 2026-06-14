import json


def main():
    with open("../data/evaluation_results_judged.json") as f:
        data = json.load(f)

    results = data["results"]

    # ADV-002 injection — internal user (where it actually fires)
    print("=== ADV-002 INJECTION — INTERNAL USER ===")
    for strat in ["strategy_1", "strategy_2", "strategy_3", "strategy_4"]:
        for r in results:
            if (
                r["question_id"] == "Q-ADV02"
                and r["strategy"] == strat
                and r["scope"] == "docs:internal"
            ):
                blocked = " [BLOCKED]" if r["metrics"]["blocked_by_policy"] else ""
                print(f"{strat}{blocked}:")
                print(r["generated_answer"][:200])
                print()

    # S1 salary leak — public user
    print("=== S1 SALARY LEAK — PUBLIC USER ===")
    for r in results:
        if (
            r["question_id"] == "Q-CONF01"
            and r["strategy"] == "strategy_1"
            and r["scope"] == "docs:public"
        ):
            print(r["generated_answer"])
            print()

    # S4 salary question — public user (should be blocked)
    print("=== S4 SALARY — PUBLIC USER ===")
    for r in results:
        if (
            r["question_id"] == "Q-CONF01"
            and r["strategy"] == "strategy_4"
            and r["scope"] == "docs:public"
        ):
            print(f"Blocked: {r['metrics']['blocked_by_policy']}")
            print(f"Answer: {r['generated_answer']}")
            print()

    # S3 salary question — public user (should refuse cleanly)
    print("=== S3 SALARY — PUBLIC USER ===")
    for r in results:
        if (
            r["question_id"] == "Q-CONF01"
            and r["strategy"] == "strategy_3"
            and r["scope"] == "docs:public"
        ):
            print(r["generated_answer"])


if __name__ == "__main__":
    main()
