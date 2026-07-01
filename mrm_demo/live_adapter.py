"""Optional live adapter from JSON runtime states to MRM decision.

This file avoids hard-wiring demo_internal.py and demo_external.py together. If
those two modules write compact JSON snapshots, this adapter can read them,
merge them into RiskState, and run the baseline plus world-model predictor.

Example:
    python live_adapter.py --internal runtime/internal_state.json --external runtime/external_state.json --once --world-model paper
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

try:
    from .decision_engine import decide_minimum_risk
    from .risk_state import DriverState, RiskState, RoadState, SystemState
    from .world_model_mock import run_world_model_mock
    from .paper_reproduction import run_paper_world_model
except ImportError:
    from decision_engine import decide_minimum_risk
    from risk_state import DriverState, RiskState, RoadState, SystemState
    from world_model_mock import run_world_model_mock
    from paper_reproduction import run_paper_world_model


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _driver_from_snapshot(raw: Dict[str, Any]) -> DriverState:
    raw = dict(raw or {})
    trs = raw.get("trs", raw.get("takeover_score"))
    if trs is not None:
        try:
            trs_val = int(trs)
            if trs_val <= 0:
                raw.setdefault("no_response_sec", 3.0)
                raw.setdefault("eye_closed_sec", 2.2)
            elif trs_val == 1:
                raw.setdefault("distracted_sec", 2.0)
                raw.setdefault("response_latency_sec", 2.6)
            elif trs_val == 2:
                raw.setdefault("response_latency_sec", 1.2)
        except Exception:
            pass
    mapping = {
        "eyes_closed_sec": "eye_closed_sec",
        "eye_closed": "eye_closed_sec",
        "yawning_frequency_min": "yawn_frequency_min",
        "head_yaw": "head_yaw_deg",
        "head_pitch": "head_pitch_deg",
        "no_response": "no_response_sec",
    }
    for src, dst in mapping.items():
        if src in raw and dst not in raw:
            if src in {"eye_closed", "no_response"} and isinstance(raw[src], bool):
                raw[dst] = 2.5 if raw[src] else 0.0
            else:
                raw[dst] = raw[src]
    return DriverState.from_dict(raw)


def _road_from_snapshot(raw: Dict[str, Any]) -> RoadState:
    raw = dict(raw or {})
    mapping = {
        "ttc": "min_ttc_sec",
        "min_ttc": "min_ttc_sec",
        "distance": "front_distance_m",
        "front_distance": "front_distance_m",
        "rrs": "road_risk_hint",
        "risk_level": "road_risk_hint",
        "confidence": "lane_confidence",
    }
    for src, dst in mapping.items():
        if src in raw and dst not in raw:
            raw[dst] = raw[src]
    return RoadState.from_dict(raw)


def _system_from_snapshot(raw: Dict[str, Any]) -> SystemState:
    raw = dict(raw or {})
    if "confidence" in raw and "perception_confidence" not in raw:
        raw["perception_confidence"] = raw["confidence"]
    return SystemState.from_dict(raw)


def build_live_risk_state(internal: Dict[str, Any], external: Dict[str, Any]) -> RiskState:
    system_raw: Dict[str, Any] = {}
    for source in (external, internal):
        if isinstance(source.get("system"), dict):
            system_raw.update(source["system"])
    # Also allow system fields at the top level of external snapshot.
    for k in ["system_failure", "odd_exit", "sensor_degraded", "perception_confidence", "automation_status"]:
        if k in external:
            system_raw[k] = external[k]

    driver_raw = internal.get("driver", internal)
    road_raw = external.get("road", external)

    return RiskState(
        scenario_id="LIVE",
        name="live_json_adapter",
        description="Live JSON adapter output from cabin and road perception snapshots.",
        expected_strategy="",
        driver=_driver_from_snapshot(driver_raw),
        road=_road_from_snapshot(road_raw),
        system=_system_from_snapshot(system_raw),
    )


def _select_model(kind: str, state: RiskState, decision):
    if kind == "paper":
        return run_paper_world_model(state, decision)
    return run_world_model_mock(state, decision)


def run_once(internal_path: Path, external_path: Path, model_kind: str) -> Dict[str, Any]:
    state = build_live_risk_state(_load_json(internal_path), _load_json(external_path))
    decision = decide_minimum_risk(state)
    wm = _select_model(model_kind, state, decision)
    return {
        "scenario": state.to_dict(),
        "decision": decision.to_dict(),
        "world_model_kind": model_kind,
        "world_model": wm.to_dict(),
    }


def print_summary(result: Dict[str, Any]) -> None:
    d = result["decision"]
    a = d["assessment"]
    wm = result["world_model"]
    print("=" * 88)
    print(
        f"TRS={a['takeover_score']}/3 | road={a['road_risk_score']:.2f} | "
        f"system={a['system_risk_score']:.2f} | fused={a['fused_risk_score']:.2f}"
    )
    print(f"Baseline: {d['baseline_strategy_label']} [{d['baseline_strategy']}] | FSM={d['fsm_state']}")
    print(f"World model({result['world_model_kind']}): {wm['recommended_label']} [{wm['recommended_action_id']}]")
    print("Reasons: " + "; ".join(d["reasons"][:4]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live JSON adapter for AegisDrive MRM")
    parser.add_argument("--internal", default="runtime/internal_state.json")
    parser.add_argument("--external", default="runtime/external_state.json")
    parser.add_argument("--world-model", choices=["mock", "paper"], default="paper")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    internal_path = Path(args.internal)
    external_path = Path(args.external)
    while True:
        result = run_once(internal_path, external_path, args.world_model)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_summary(result)
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
