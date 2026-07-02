"""Regression tests for the IDMS -> RiskState live adapter.

Run from this directory with:
    python test_live_adapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrm_demo.decision_engine import decide_minimum_risk
from mrm_demo.live_adapter import (
    build_risk_state_from_idms,
    driver_state_to_adapter_fields,
    road_state_to_adapter_fields,
    system_state_to_adapter_fields,
)


def _candidate(decision, action_id: str):
    for action in decision.candidate_actions:
        if action.action_id == action_id:
            return action
    raise AssertionError(f"missing candidate action: {action_id}")


def test_normal_low_risk() -> None:
    face_data = {
        "has_face": True,
        "driver_available": True,
        "monitoring_valid": True,
        "perclos": 0.03,
        "is_drowsy": False,
        "is_yawning": False,
        "is_distracted": False,
        "yaw": 2.0,
        "pitch": 0.0,
    }
    vehicle_data = {
        "front_distance_m": 80.0,
        "min_ttc_sec": 9.0,
        "relative_speed_mps": 0.5,
        "road_risk_hint": 0,
    }
    system_status = {
        "automation_status": "normal",
        "sensor_degraded": False,
        "perception_confidence": 0.96,
        "system_failure": False,
        "odd_exit": False,
    }

    driver_fields = driver_state_to_adapter_fields(face_data=face_data)
    state = build_risk_state_from_idms(
        face_data=face_data,
        vehicle_data=vehicle_data,
        system_status=system_status,
        scenario_id="normal_low_risk",
    )
    decision = decide_minimum_risk(state)

    assert driver_fields["takeover_readiness"] >= 2.5
    assert decision.assessment.fused_risk_score < 0.25
    assert decision.fsm_state == "NORMAL"
    assert decision.baseline_strategy == "continue_monitoring"


def test_no_face_unavailable() -> None:
    face_data = {
        "has_face": False,
        "monitoring_valid": False,
        "unavailable_reason": "no_face",
        "perclos": 0.0,
    }

    driver_fields = driver_state_to_adapter_fields(face_data=face_data)
    state = build_risk_state_from_idms(
        face_data=face_data,
        vehicle_data={"front_distance_m": 60.0, "min_ttc_sec": 8.0},
        system_status={"automation_status": "normal", "perception_confidence": 0.9},
        scenario_id="no_face_unavailable",
    )
    decision = decide_minimum_risk(state)

    assert driver_fields["driver_available"] is False
    assert driver_fields["monitoring_valid"] is False
    assert driver_fields["takeover_readiness"] == 0.0
    assert decision.assessment.takeover_score == 0
    assert decision.assessment.fused_risk_score >= 0.25
    assert decision.baseline_strategy != "request_takeover"


def test_missing_face_data() -> None:
    driver_fields = driver_state_to_adapter_fields(face_data=None, driver_state=None)
    state = build_risk_state_from_idms(
        face_data=None,
        driver_state=None,
        vehicle_data={"front_distance_m": 75.0, "min_ttc_sec": 10.0},
        system_status={"automation_status": "normal", "perception_confidence": 0.95},
        scenario_id="missing_face_data",
    )
    decision = decide_minimum_risk(state)

    assert driver_fields["monitoring_valid"] is False or driver_fields["driver_available"] is False
    assert state.driver.no_response_sec >= 2.0
    assert decision.assessment.takeover_score == 0
    assert decision.fsm_state != "NORMAL"


def test_critical_ttc_selects_most_dangerous_target() -> None:
    vehicle_data = [
        {"class_name": "truck", "distance_m": 45.0, "ttc": 4.5, "relative_speed": -3.0},
        {
            "class_name": "car",
            "distance": 9.0,
            "min_ttc_sec": 0.8,
            "rel_speed": -8.0,
            "warning_level": 3,
        },
    ]

    road_fields = road_state_to_adapter_fields(vehicle_data=vehicle_data)
    state = build_risk_state_from_idms(
        face_data={"has_face": True, "monitoring_valid": True, "perclos": 0.04},
        vehicle_data=vehicle_data,
        system_status={"automation_status": "normal", "perception_confidence": 0.92},
        scenario_id="critical_ttc",
    )
    decision = decide_minimum_risk(state)

    assert state.road.min_ttc_sec == 0.8
    assert road_fields["road_risk"] >= 0.75
    assert decision.assessment.road_risk_level >= 2
    assert decision.baseline_strategy != "continue_monitoring"
    assert decision.fsm_state in {"WARNING", "TAKEOVER_REQUEST", "MRM_PREPARE", "MRM_EXECUTE"}


def test_sensor_degraded_limits_side_strategies() -> None:
    system_status = SimpleNamespace(
        automation_status="degraded",
        sensor_degraded=True,
        perception_confidence=0.35,
        system_failure=False,
        odd_exit=False,
        camera_ok=False,
        radar_ok=True,
        lane_perception_ok=False,
    )

    system_fields = system_state_to_adapter_fields(system_status=system_status)
    state = build_risk_state_from_idms(
        face_data={"has_face": True, "monitoring_valid": True, "perclos": 0.02},
        vehicle_data={
            "front_distance_m": 55.0,
            "min_ttc_sec": 7.0,
            "lane_relevance": 0.4,
        },
        system_status=system_status,
        scenario_id="sensor_degraded",
    )
    decision = decide_minimum_risk(state)

    assert system_fields["system_risk"] >= 0.5
    assert state.system.sensor_degraded is True
    assert _candidate(decision, "lane_change_avoidance").allowed is False
    assert _candidate(decision, "shoulder_stop").allowed is False


def test_driver_unavailable_high_road_risk_enters_mrm() -> None:
    driver_state = SimpleNamespace(
        driver_available=False,
        monitoring_valid=True,
        unavailable_reason="no_response",
        driver_risk=0.95,
        fatigue_score=0.0,
        attention_score=0.0,
        takeover_readiness=0.0,
        no_response_sec=3.5,
    )

    state = build_risk_state_from_idms(
        driver_state=driver_state,
        vehicle_data={
            "front_distance_m": 13.0,
            "min_ttc_sec": 1.2,
            "relative_speed_mps": -6.0,
            "warning_level": 2,
        },
        system_status={"automation_status": "normal", "perception_confidence": 0.85},
        scenario_id="driver_unavailable_high_road_risk",
    )
    decision = decide_minimum_risk(state)

    assert state.driver.no_response_sec >= 2.0
    assert decision.assessment.takeover_score == 0
    assert decision.baseline_strategy != "request_takeover"
    assert decision.fsm_state in {"MRM_PREPARE", "MRM_EXECUTE"}
    assert decision.baseline_strategy in {
        "strong_deceleration",
        "emergency_brake",
        "keep_lane_safe_stop",
    }


if __name__ == "__main__":
    test_normal_low_risk()
    test_no_face_unavailable()
    test_missing_face_data()
    test_critical_ttc_selects_most_dangerous_target()
    test_sensor_degraded_limits_side_strategies()
    test_driver_unavailable_high_road_risk_enters_mrm()
    print("All live adapter tests passed.")
