"""IDMS output adapter for the AegisDrive MRM RiskState pipeline.

The adapter accepts cabin, road, fusion, and system outputs from the original
IDMS modules as dictionaries, dataclass instances, or plain objects. It only
normalizes those signals into ``RiskState``; the final MRM strategy remains in
``decision_engine.decide_minimum_risk``.

The legacy JSON snapshot helpers are kept so existing live snapshot demos can
continue to call ``run_once`` and ``build_live_risk_state``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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


RISK_LEVEL_TO_SCORE = {
    "safe": 0.0,
    "normal": 0.0,
    "none": 0.0,
    "low": 0.3,
    "caution": 0.35,
    "medium": 0.5,
    "moderate": 0.5,
    "high": 0.75,
    "danger": 0.75,
    "urgent": 0.75,
    "critical": 1.0,
    "emergency": 1.0,
    "failure": 1.0,
}


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict, dataclass instance, or plain object."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return asdict(obj).get(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def get_first_value(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Return the first present, non-None value among ``keys``."""
    for key in keys:
        value = get_value(obj, key, None)
        if value is not None:
            return value
    return default


def clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value into the requested range.

    Invalid inputs return ``low``. This is conservative for normalized risk
    scores, where unknown values should be handled explicitly before clamping.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(low)
    if math.isnan(number) or math.isinf(number):
        return float(low)
    return max(float(low), min(float(high), number))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(number) or math.isinf(number):
        return float(default)
    return number


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "ok", "available", "valid"}:
            return True
        if normalized in {"0", "false", "no", "n", "none", "unavailable", "invalid"}:
            return False
    return bool(value)


def _risk_score(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        return RISK_LEVEL_TO_SCORE.get(value.strip().lower(), default)
    score = _to_float(value, default)
    if score > 3.0:
        return clamp(score / 100.0)
    if score > 1.0:
        return clamp(score / 3.0)
    return clamp(score)


def _risk_level(score: Any) -> int:
    score = _risk_score(score)
    if score >= 0.75:
        return 3
    if score >= 0.50:
        return 2
    if score >= 0.25:
        return 1
    return 0


def _level_to_score(level: Any, default: float = 0.0) -> float:
    if level is None:
        return default
    if isinstance(level, str):
        return RISK_LEVEL_TO_SCORE.get(level.strip().lower(), default)
    return clamp(_to_float(level, default) / 3.0)


def _value_from_sources(sources: Iterable[Any], keys: Sequence[str], default: Any = None) -> Any:
    for source in sources:
        value = get_first_value(source, keys, None)
        if value is not None:
            return value
    return default


def _estimate_face_driver_fields(face_data: Any) -> Dict[str, float]:
    perclos = clamp(get_value(face_data, "perclos", 0.0))
    blink_freq = _to_float(get_value(face_data, "blink_freq", 0.0))
    yaw = _to_float(get_value(face_data, "yaw", 0.0))
    pitch = _to_float(get_value(face_data, "pitch", 0.0))
    drowsy_frames = _to_float(get_value(face_data, "drowsy_frames", 0.0))
    yawn_frames = _to_float(get_value(face_data, "yawn_frames", 0.0))
    distracted_frames = _to_float(get_value(face_data, "distracted_frames", 0.0))
    nod_frames = _to_float(get_value(face_data, "nod_frames", 0.0))

    is_drowsy = _to_bool(get_value(face_data, "is_drowsy", False))
    is_yawning = _to_bool(get_value(face_data, "is_yawning", False))
    is_distracted = _to_bool(get_value(face_data, "is_distracted", False))
    is_nodding = _to_bool(get_value(face_data, "is_nodding", False))

    perclos_score = clamp(perclos / 0.30)
    blink_score = clamp((blink_freq - 20.0) / 15.0)
    yaw_score = clamp((abs(yaw) - 15.0) / 30.0)
    pitch_score = clamp((abs(pitch) - 20.0) / 25.0)

    fatigue_score = max(
        0.60 * perclos_score + 0.20 * blink_score,
        0.75 if is_drowsy else 0.0,
        0.45 if is_yawning else 0.0,
        0.50 if is_nodding else 0.0,
        clamp(drowsy_frames / 20.0),
        clamp(yawn_frames / 15.0) * 0.6,
        clamp(nod_frames / 15.0) * 0.6,
    )
    attention_score = max(
        0.75 if is_distracted else 0.0,
        0.55 * yaw_score + 0.25 * pitch_score,
        clamp(distracted_frames / 20.0),
        0.35 if is_yawning else 0.0,
        0.35 if is_nodding else 0.0,
    )
    driver_risk = max(
        0.55 * fatigue_score + 0.45 * attention_score,
        fatigue_score * attention_score,
    )
    return {
        "fatigue_score": round(clamp(fatigue_score), 4),
        "attention_score": round(clamp(attention_score), 4),
        "driver_risk": round(clamp(driver_risk), 4),
    }


def driver_state_to_adapter_fields(face_data: Any = None, driver_state: Any = None) -> Dict[str, Any]:
    """Normalize cabin-side IDMS output into adapter fields."""
    no_input = face_data is None and driver_state is None

    raw_has_face = get_value(face_data, "has_face", None)
    has_face = _to_bool(raw_has_face, default=True) if raw_has_face is not None else None
    driver_available = _to_bool(
        _value_from_sources([driver_state, face_data], ["driver_available"], None),
        default=True,
    )
    monitoring_valid = _to_bool(
        _value_from_sources([driver_state, face_data], ["monitoring_valid"], None),
        default=True,
    )

    if no_input:
        driver_available = False
        monitoring_valid = False
    elif has_face is False:
        driver_available = False
        monitoring_valid = False

    unavailable_reason = _value_from_sources(
        [driver_state, face_data],
        ["unavailable_reason"],
        "",
    )
    if no_input and not unavailable_reason:
        unavailable_reason = "no_driver_data"
    elif has_face is False and not unavailable_reason:
        unavailable_reason = "no_face"
    elif not monitoring_valid and not unavailable_reason:
        unavailable_reason = "monitoring_invalid"

    face_estimate = _estimate_face_driver_fields(face_data) if face_data is not None else {
        "fatigue_score": 0.0,
        "attention_score": 0.0,
        "driver_risk": 0.0,
    }

    fatigue_score = _risk_score(
        _value_from_sources(
            [driver_state],
            ["fatigue_score", "int_fatigue_score"],
            face_estimate["fatigue_score"],
        )
    )
    attention_score = _risk_score(
        _value_from_sources(
            [driver_state],
            ["attention_score", "int_attention_score"],
            face_estimate["attention_score"],
        )
    )
    driver_risk = _risk_score(
        _value_from_sources(
            [driver_state],
            ["driver_risk", "int_score"],
            face_estimate["driver_risk"],
        )
    )

    # No face is an availability failure, not a fatigue observation.
    if not driver_available or not monitoring_valid:
        fatigue_score = 0.0 if has_face is False or no_input else fatigue_score
        attention_score = 0.0 if has_face is False or no_input else attention_score
        driver_risk = max(driver_risk, 0.75)

    takeover_raw = _value_from_sources([driver_state], ["takeover_readiness", "trs"], None)
    if takeover_raw is not None:
        takeover_readiness = _to_float(takeover_raw, 0.0)
        if 0.0 <= takeover_readiness <= 1.0:
            takeover_readiness *= 3.0
        takeover_readiness = clamp(takeover_readiness, 0.0, 3.0)
    else:
        takeover_readiness = clamp(3.0 * (1.0 - driver_risk), 0.0, 3.0)

    if not driver_available:
        takeover_readiness = 0.0
    elif not monitoring_valid:
        takeover_readiness = min(takeover_readiness, 1.0)

    no_response_sec = _to_float(
        _value_from_sources([driver_state, face_data], ["no_response_sec", "no_response"], None),
        default=-1.0,
    )
    if no_response_sec < 0.0:
        if not driver_available:
            no_response_sec = 3.0
        elif not monitoring_valid:
            no_response_sec = 2.0
        else:
            no_response_sec = 0.0

    return {
        "driver_available": bool(driver_available),
        "monitoring_valid": bool(monitoring_valid),
        "unavailable_reason": str(unavailable_reason or ""),
        "fatigue_score": round(clamp(fatigue_score), 4),
        "attention_score": round(clamp(attention_score), 4),
        "driver_risk": round(clamp(driver_risk), 4),
        "takeover_readiness": round(clamp(takeover_readiness, 0.0, 3.0), 3),
        "no_response_sec": round(max(0.0, no_response_sec), 3),
    }


def _target_ttc(target: Any) -> Any:
    return get_first_value(target, ["min_ttc_sec", "ttc", "min_ttc", "ext_min_ttc"], None)


def _target_distance(target: Any) -> Any:
    return get_first_value(
        target,
        ["front_distance_m", "distance_m", "distance", "front_distance", "ext_min_dist"],
        None,
    )


def _select_most_dangerous_target(vehicle_data: Any) -> Any:
    if isinstance(vehicle_data, Mapping):
        return vehicle_data
    if not isinstance(vehicle_data, list):
        return None
    if not vehicle_data:
        return None

    def sort_key(target: Any) -> tuple[int, float, float]:
        ttc = _target_ttc(target)
        dist = _target_distance(target)
        level = _level_to_score(get_first_value(target, ["warning_level", "road_risk_hint", "risk_hint"], 0))
        if ttc is not None:
            return (0, _to_float(ttc, 99.0), -level)
        return (1, _to_float(dist, 99.0), -level)

    return sorted(vehicle_data, key=sort_key)[0]


def road_state_to_adapter_fields(vehicle_data: Any = None, road_state: Any = None) -> Dict[str, Any]:
    """Normalize road-side perception output into adapter fields.

    ``relative_speed_mps`` follows the MRM demo convention: negative means the
    distance to the object is shrinking. If upstream naming is ambiguous, the
    value is passed through with this convention documented here.
    """
    target = _select_most_dangerous_target(vehicle_data)
    sources = [target, road_state]
    no_road_input = target is None and road_state is None

    front_distance_m = _to_float(
        _value_from_sources(
            sources,
            ["front_distance_m", "distance_m", "distance", "front_distance", "ext_min_dist"],
            45.0 if no_road_input else 80.0,
        ),
        45.0,
    )
    min_ttc_sec = _to_float(
        _value_from_sources(
            sources,
            ["min_ttc_sec", "ttc", "min_ttc", "ext_min_ttc"],
            6.0 if no_road_input else 99.0,
        ),
        6.0,
    )
    relative_speed_mps = _to_float(
        _value_from_sources(
            sources,
            ["relative_speed_mps", "relative_speed", "rel_speed"],
            0.0,
        ),
        0.0,
    )

    warning_score = _level_to_score(
        _value_from_sources(
            sources,
            ["warning_level", "road_risk_hint", "risk_hint", "ext_max_level"],
            None,
        ),
        default=0.0,
    )
    direct_road_risk = _risk_score(
        _value_from_sources(sources, ["road_risk"], None),
        default=0.0,
    )

    if min_ttc_sec <= 0.0:
        ttc_score = 0.0
    elif min_ttc_sec <= 1.0:
        ttc_score = 0.95
    elif min_ttc_sec <= 1.5:
        ttc_score = 0.85
    elif min_ttc_sec <= 2.0:
        ttc_score = 0.70
    elif min_ttc_sec <= 3.0:
        ttc_score = 0.55
    elif min_ttc_sec <= 5.0:
        ttc_score = 0.30
    else:
        ttc_score = 0.05

    distance_score = clamp((45.0 - front_distance_m) / 41.0)
    closing_score = clamp(-relative_speed_mps / 12.0) if relative_speed_mps < 0.0 else 0.0
    distance_closing_score = 0.0
    if relative_speed_mps < 0.0:
        distance_closing_score = max(distance_score, 0.55 * distance_score + 0.45 * closing_score)

    road_risk = max(
        direct_road_risk,
        warning_score,
        0.62 * ttc_score + 0.25 * distance_score + 0.13 * closing_score,
        distance_closing_score,
    )
    if no_road_input:
        road_risk = max(road_risk, 0.32)
    if min_ttc_sec <= 1.0:
        road_risk = max(road_risk, 0.85)
    elif min_ttc_sec <= 2.0:
        road_risk = max(road_risk, 0.60)
    if front_distance_m <= 12.0 and relative_speed_mps < 0.0:
        road_risk = max(road_risk, 0.70)

    lane_available = _to_bool(
        _value_from_sources(
            sources,
            ["lane_available", "adjacent_lane_clear"],
            False,
        ),
        default=False,
    )
    shoulder_available = _to_bool(
        _value_from_sources(sources, ["shoulder_available"], False),
        default=False,
    )
    side_confidence = clamp(
        _value_from_sources(
            sources,
            ["side_confidence", "lane_confidence", "confidence", "lane_relevance"],
            0.50 if no_road_input else 1.0,
        )
    )

    return {
        "front_distance_m": round(max(0.0, front_distance_m), 3),
        "min_ttc_sec": round(max(0.0, min_ttc_sec), 3),
        "relative_speed_mps": round(relative_speed_mps, 3),
        "road_risk": round(clamp(road_risk), 4),
        "lane_available": bool(lane_available),
        "shoulder_available": bool(shoulder_available),
        "side_confidence": round(side_confidence, 4),
    }


def system_state_to_adapter_fields(system_status: Any = None, fusion_result: Any = None) -> Dict[str, Any]:
    """Normalize automation and perception reliability into adapter fields."""
    no_system_input = system_status is None
    sources = [system_status]

    camera_ok = get_value(system_status, "camera_ok", None)
    radar_ok = get_value(system_status, "radar_ok", None)
    lidar_ok = get_value(system_status, "lidar_ok", None)
    lane_ok = get_value(system_status, "lane_perception_ok", None)
    sensor_flags = [camera_ok, radar_ok, lidar_ok, lane_ok]
    any_sensor_bad = any(flag is False for flag in sensor_flags)

    sensor_degraded = _to_bool(
        _value_from_sources(sources, ["sensor_degraded"], any_sensor_bad),
        default=any_sensor_bad,
    ) or any_sensor_bad
    perception_confidence = clamp(
        _value_from_sources(sources, ["perception_confidence"], 0.75 if no_system_input else 1.0)
    )
    system_failure = _to_bool(_value_from_sources(sources, ["system_failure"], False), default=False)
    odd_exit = _to_bool(_value_from_sources(sources, ["odd_exit"], False), default=False)
    automation_status = str(
        _value_from_sources(sources, ["automation_status"], "unknown" if no_system_input else "normal")
        or "unknown"
    )

    system_risk = 0.25 if no_system_input else 0.0
    if sensor_degraded:
        system_risk += 0.35
    if perception_confidence < 0.5:
        system_risk += 0.35
    else:
        system_risk += 0.20 * (1.0 - perception_confidence)
    if system_failure:
        system_risk = max(system_risk, 0.85)
    if odd_exit:
        system_risk = max(system_risk, 0.75)
    if automation_status.lower() in {"degraded", "failure", "minimal_risk"}:
        system_risk += 0.15
    if _to_bool(get_value(system_status, "takeover_request_active", False)):
        system_risk += 0.10

    if fusion_result is not None and system_status is None:
        if _to_bool(get_value(fusion_result, "int_unavailable", False)):
            system_risk = max(system_risk, 0.45)
        urgency = get_value(fusion_result, "alert_urgency", None)
        if urgency is not None:
            system_risk = max(system_risk, RISK_LEVEL_TO_SCORE.get(str(urgency).lower(), 0.0))
        fused_level = get_value(fusion_result, "fused_level", None)
        if fused_level is not None:
            system_risk = max(system_risk, 0.55 * _level_to_score(fused_level))

    return {
        "sensor_degraded": bool(sensor_degraded),
        "perception_confidence": round(perception_confidence, 4),
        "system_failure": bool(system_failure),
        "odd_exit": bool(odd_exit),
        "automation_status": automation_status,
        "system_risk": round(clamp(system_risk), 4),
    }


def fusion_result_to_adapter_fields(
    fusion_result: Any = None,
    driver_fields: Dict[str, Any] | None = None,
    road_fields: Dict[str, Any] | None = None,
    system_fields: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Normalize fused IDMS risk output or estimate it from adapter fields."""
    driver_fields = driver_fields or {}
    road_fields = road_fields or {}
    system_fields = system_fields or {}

    if fusion_result is not None:
        fused_score = _risk_score(get_first_value(fusion_result, ["fused_score"], None), default=0.0)
        if get_value(fusion_result, "fused_score", None) is None:
            fused_score = max(
                _risk_score(get_value(fusion_result, "int_score", 0.0)),
                _risk_score(get_value(fusion_result, "ext_score", 0.0)),
                _risk_score(get_value(fusion_result, "cross_score", 0.0)),
            )
        fused_level_raw = get_first_value(fusion_result, ["fused_level", "fused_text"], None)
        fused_level = _risk_level(fused_score) if fused_level_raw is None else _risk_level(_level_to_score(fused_level_raw, fused_score))
        alert_urgency = str(get_value(fusion_result, "alert_urgency", "") or "")
        if not alert_urgency:
            alert_urgency = "emergency" if fused_level >= 3 else "urgent" if fused_level == 2 else "normal" if fused_level == 1 else "none"
        should_alert = _to_bool(get_value(fusion_result, "should_alert", fused_level >= 2))
    else:
        driver_risk = _risk_score(driver_fields.get("driver_risk"), default=0.55)
        road_risk = _risk_score(road_fields.get("road_risk"), default=0.35)
        system_risk = _risk_score(system_fields.get("system_risk"), default=0.25)
        max_component = max(driver_risk, road_risk, system_risk)
        weighted = 0.50 * road_risk + 0.30 * driver_risk + 0.20 * system_risk
        fused_score = max(0.85 * max_component, weighted)

        if driver_fields.get("driver_available") is False or driver_fields.get("monitoring_valid") is False:
            fused_score = max(fused_score, 0.55)
        if _to_float(road_fields.get("min_ttc_sec"), 99.0) <= 1.0:
            fused_score = max(fused_score, 0.85)
        elif _to_float(road_fields.get("min_ttc_sec"), 99.0) <= 1.5:
            fused_score = max(fused_score, 0.75)
        if system_fields.get("system_failure") is True or system_fields.get("odd_exit") is True:
            fused_score = max(fused_score, 0.80)
        if system_fields.get("sensor_degraded") is True and _to_float(system_fields.get("perception_confidence"), 1.0) < 0.5:
            fused_score = max(fused_score, 0.55)

        fused_score = clamp(fused_score)
        fused_level = _risk_level(fused_score)
        alert_urgency = "emergency" if fused_level >= 3 else "urgent" if fused_level == 2 else "normal" if fused_level == 1 else "none"
        should_alert = fused_level >= 2

    component_scores = {
        "driver": _risk_score(driver_fields.get("driver_risk"), default=0.0),
        "road": _risk_score(road_fields.get("road_risk"), default=0.0),
        "system": _risk_score(system_fields.get("system_risk"), default=0.0),
    }
    risk_source = [name for name, score in component_scores.items() if score >= 0.5]
    if not risk_source:
        risk_source = [max(component_scores, key=component_scores.get)]

    return {
        "fused_score": round(clamp(fused_score), 4),
        "fused_level": int(fused_level),
        "risk_source": risk_source,
        "alert_urgency": alert_urgency,
        "should_alert": bool(should_alert),
    }


def _driver_fields_to_risk_state(face_data: Any, fields: Dict[str, Any]) -> DriverState:
    perclos = clamp(get_value(face_data, "perclos", fields["fatigue_score"] * 0.30))
    is_yawning = _to_bool(get_value(face_data, "is_yawning", False))
    is_distracted = _to_bool(get_value(face_data, "is_distracted", fields["attention_score"] >= 0.5))
    is_drowsy = _to_bool(get_value(face_data, "is_drowsy", fields["fatigue_score"] >= 0.75))
    yaw = _to_float(get_value(face_data, "yaw", 0.0), 0.0)
    pitch = _to_float(get_value(face_data, "pitch", 0.0), 0.0)

    return DriverState(
        eye_closed_sec=2.1 if is_drowsy else 0.0,
        yawn_frequency_min=4.0 if is_yawning else (2.0 if fields["fatigue_score"] >= 0.45 else 0.0),
        perclos=perclos,
        distracted_sec=2.0 if is_distracted else (1.0 if fields["attention_score"] >= 0.45 else 0.0),
        head_yaw_deg=yaw,
        head_pitch_deg=pitch,
        no_response_sec=fields["no_response_sec"],
        response_latency_sec=2.6 if fields["takeover_readiness"] <= 1.0 and fields["driver_available"] else 0.0,
        hands_on_wheel=bool(fields["driver_available"]),
    )


def _road_fields_to_risk_state(fields: Dict[str, Any]) -> RoadState:
    side_confidence = clamp(fields["side_confidence"])
    return RoadState(
        front_distance_m=fields["front_distance_m"],
        min_ttc_sec=fields["min_ttc_sec"],
        relative_speed_mps=fields["relative_speed_mps"],
        road_risk_hint=_risk_level(fields["road_risk"]),
        lane_confidence=side_confidence,
        adjacent_lane_clear=bool(fields["lane_available"]),
        lane_change_info_reliable=side_confidence >= 0.55,
        shoulder_available=bool(fields["shoulder_available"]),
        shoulder_confidence=side_confidence if fields["shoulder_available"] else min(side_confidence, 0.3),
    )


def _system_fields_to_risk_state(system_status: Any, fields: Dict[str, Any]) -> SystemState:
    return SystemState(
        automation_status=fields["automation_status"],
        system_failure=fields["system_failure"],
        odd_exit=fields["odd_exit"],
        sensor_degraded=fields["sensor_degraded"],
        perception_confidence=fields["perception_confidence"],
        takeover_request_active=_to_bool(get_value(system_status, "takeover_request_active", False)),
        degradation_note=str(get_value(system_status, "degradation_note", "") or ""),
    )


def build_risk_state_from_idms(
    face_data: Any = None,
    driver_state: Any = None,
    vehicle_data: Any = None,
    road_state: Any = None,
    fusion_result: Any = None,
    system_status: Any = None,
    scenario_id: str = "live_idms",
    description: str = "Live IDMS adapter input",
) -> RiskState:
    """Build a ``RiskState`` from raw IDMS outputs."""
    effective_road_state = road_state
    if effective_road_state is None and vehicle_data is None and fusion_result is not None:
        effective_road_state = fusion_result

    driver_fields = driver_state_to_adapter_fields(face_data=face_data, driver_state=driver_state)
    road_fields = road_state_to_adapter_fields(vehicle_data=vehicle_data, road_state=effective_road_state)
    system_fields = system_state_to_adapter_fields(system_status=system_status, fusion_result=fusion_result)
    fusion_fields = fusion_result_to_adapter_fields(
        fusion_result=fusion_result,
        driver_fields=driver_fields,
        road_fields=road_fields,
        system_fields=system_fields,
    )

    system_state = _system_fields_to_risk_state(system_status, system_fields)
    if fusion_fields["should_alert"] and not system_state.takeover_request_active:
        system_state.takeover_request_active = fusion_fields["fused_level"] >= 2
    if fusion_fields["fused_level"] >= 2 and not system_state.degradation_note:
        system_state.degradation_note = (
            "IDMS fusion indicates elevated risk from "
            + ",".join(fusion_fields["risk_source"])
        )

    return RiskState(
        scenario_id=str(scenario_id),
        name=str(scenario_id),
        description=str(description),
        expected_strategy="",
        driver=_driver_fields_to_risk_state(face_data, driver_fields),
        road=_road_fields_to_risk_state(road_fields),
        system=system_state,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _driver_from_snapshot(raw: Dict[str, Any]) -> DriverState:
    raw = dict(raw or {})
    return build_risk_state_from_idms(face_data=raw, driver_state=raw).driver


def _road_from_snapshot(raw: Dict[str, Any]) -> RoadState:
    return build_risk_state_from_idms(vehicle_data=dict(raw or {})).road


def _system_from_snapshot(raw: Dict[str, Any]) -> SystemState:
    return build_risk_state_from_idms(system_status=dict(raw or {})).system


def build_live_risk_state(internal: Dict[str, Any], external: Dict[str, Any]) -> RiskState:
    """Compatibility wrapper for the old internal/external JSON snapshots."""
    internal = dict(internal or {})
    external = dict(external or {})

    system_raw: Dict[str, Any] = {}
    for source in (external, internal):
        if isinstance(source.get("system"), dict):
            system_raw.update(source["system"])
    for key in [
        "system_failure",
        "odd_exit",
        "sensor_degraded",
        "perception_confidence",
        "automation_status",
        "takeover_request_active",
        "degradation_note",
    ]:
        if key in external:
            system_raw[key] = external[key]

    driver_raw = internal.get("driver", internal.get("driver_state", internal))
    face_raw = internal.get("face_data", internal.get("face", internal))
    vehicle_raw = external.get("vehicle_data", external.get("vehicles", external.get("targets", external.get("road", external))))

    state = build_risk_state_from_idms(
        face_data=face_raw,
        driver_state=driver_raw,
        vehicle_data=vehicle_raw,
        system_status=system_raw,
        scenario_id="LIVE",
        description="Live JSON adapter output from cabin and road perception snapshots.",
    )
    state.name = "live_json_adapter"
    return state


def _select_model(kind: str, state: RiskState, decision: Any) -> Any:
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


def run_adapter_demo() -> Dict[str, Any]:
    """Run one conservative sample: unavailable driver, critical TTC, degraded perception."""
    state = build_risk_state_from_idms(
        face_data={
            "has_face": False,
            "monitoring_valid": False,
            "unavailable_reason": "no_face",
        },
        vehicle_data=[
            {
                "class_name": "car",
                "distance_m": 8.0,
                "ttc": 0.8,
                "relative_speed_mps": -7.0,
                "warning_level": 3,
            }
        ],
        system_status={
            "automation_status": "degraded",
            "sensor_degraded": True,
            "perception_confidence": 0.38,
            "system_failure": False,
            "odd_exit": False,
            "lane_perception_ok": False,
            "degradation_note": "demo: driver unavailable, critical TTC, degraded perception",
        },
        scenario_id="live_idms_demo",
        description="Driver unavailable with critical front TTC and degraded perception.",
    )
    decision = decide_minimum_risk(state)
    wm = run_world_model_mock(state, decision)
    result = {
        "scenario": state.to_dict(),
        "decision": decision.to_dict(),
        "world_model_kind": "mock",
        "world_model": wm.to_dict(),
    }
    print_summary(result)
    return result


def main(argv: List[str] | None = None) -> int:
    if argv is None and len(sys.argv) == 1:
        run_adapter_demo()
        return 0

    parser = argparse.ArgumentParser(description="Live JSON adapter for AegisDrive MRM")
    parser.add_argument("--internal", default="runtime/internal_state.json")
    parser.add_argument("--external", default="runtime/external_state.json")
    parser.add_argument("--world-model", choices=["mock", "paper"], default="paper")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--demo", action="store_true", help="run a built-in adapter demo once")
    args = parser.parse_args(argv)

    if args.demo:
        run_adapter_demo()
        return 0

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
