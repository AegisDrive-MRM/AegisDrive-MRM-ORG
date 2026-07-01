"""CLI entry point for the lightweight world-model paper reproduction.

Usage from mrm_demo/:
    python -m paper_reproduction.reproduce_demo --scenario S4
    python -m paper_reproduction.reproduce_demo --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
MRM_DIR = CURRENT_DIR.parent
if str(MRM_DIR) not in sys.path:
    sys.path.insert(0, str(MRM_DIR))

from decision_engine import decide_minimum_risk, format_decision
from risk_state import find_scenario, load_scenarios
from paper_reproduction.evaluate_candidates import run_paper_world_model


def _print_result(state, as_json: bool = False) -> None:
    decision = decide_minimum_risk(state)
    wm = run_paper_world_model(state, decision)
    if as_json:
        print(json.dumps({"scenario": state.to_dict(), "decision": decision.to_dict(), "paper_world_model": wm.to_dict()}, ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 96)
    print(f"{state.scenario_id} | {state.name}")
    print("=" * 96)
    print(format_decision(decision))
    print("\n[Lightweight paper reproduction: candidate rollout]")
    print("-" * 112)
    print(f"{'status':<15} {'action':<25} {'1s risk':>8} {'3s risk':>8} {'5s risk':>8} {'score':>8} {'penalty':>8}")
    print("-" * 112)
    for pred in wm.predictions:
        risks = {p.horizon_s: p.risk for p in pred.future_risks}
        print(
            f"{pred.status:<15} {pred.label[:24]:<25} "
            f"{risks.get(1, 0):>8.3f} {risks.get(3, 0):>8.3f} {risks.get(5, 0):>8.3f} "
            f"{pred.score:>8.3f} {pred.penalty:>8.3f}"
        )
        print(f"  - {pred.reason}")
    print("-" * 112)
    print(f"Recommended: {wm.recommended_label} [{wm.recommended_action_id}]")
    print(wm.explanation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lightweight T-ITS world-model reproduction for AegisDrive MRM")
    parser.add_argument("--scenario", "-s", default="S4")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    scenarios = load_scenarios(MRM_DIR / "scenarios.json")
    if args.all:
        for scenario in scenarios:
            _print_result(scenario, as_json=args.json)
    else:
        _print_result(find_scenario(scenarios, args.scenario), as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
