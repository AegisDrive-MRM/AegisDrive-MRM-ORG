"""
RiskState and scoring utilities for the AegisDrive minimum-risk-decision demo.

This file is intentionally independent from the existing IDMS runtime.  The next
engineering step is to replace the mock scenario fields with real outputs from:
- src.internal.*  -> driver takeover readiness signals
- src.external.*  -> road risk, distance and TTC signals
- src.core.*      -> multimodal risk fusion signals
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import json


RISK_LEVEL_LABELS = {
    0: "SAFE / 低风险",
    1: "CAUTION / 关注",
    2: "DANGER / 高风险",
    3: "CRITICAL / 极高风险",
}

TAKEOVER_LABELS = {
    0: "不可接管",
    1: "接管能力不足",
    2: "有条件可接管",
    3: "可接管",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a number into [low, high]."""
    return max(low, min(high, float(value)))


def _pick_fields(cls: type, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields that belong to a dataclass."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in dict(raw or {}).items() if k in names}


def _risk_level_from_score(score: float) -> int:
    score = clamp(score)
    if score < 0.25:
        return 0
    if score < 0.50:
        return 1
    if score < 0.75:
        return 2
    return 3


def _inverse_scale(value: float, safe_value: float, danger_value: float) -> float:
    """
    Map a metric to risk. Lower values are more dangerous.

    Example: distance=5m is dangerous, distance=45m is safe.
    """
    if safe_value == danger_value:
        return 0.0
    return clamp((safe_value - float(value)) / (safe_value - danger_value))


@dataclass
class DriverState:
    """Mocked cabin-side driver monitoring output."""

    eye_closed_sec: float = 0.0
    yawn_frequency_min: float = 0.0
    perclos: float = 0.0
    distracted_sec: float = 0.0
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    no_response_sec: float = 0.0
    response_latency_sec: float = 0.0
    hands_on_wheel: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DriverState":
        return cls(**_pick_fields(cls, raw))


@dataclass
class RoadState:
    """Mocked road-side perception and risk output."""

    front_distance_m: float = 80.0
    min_ttc_sec: float = 99.0
    relative_speed_mps: float = 0.0  # negative means ego vehicle is closing in.
    ego_speed_kmh: float = 60.0
    object_class: str = "none"
    road_risk_hint: int = 0
    lane_confidence: float = 1.0
    adjacent_lane_clear: bool = False
    lane_change_info_reliable: bool = True
    shoulder_available: bool = False
    shoulder_confidence: float = 0.0
    rear_risk: float = 0.0
    weather: str = "clear"

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RoadState":
        return cls(**_pick_fields(cls, raw))


@dataclass
class SystemState:
    """Mocked system status and perception reliability output."""

    automation_status: str = "normal"
    system_failure: bool = False
    odd_exit: bool = False
    sensor_degraded: bool = False
    perception_confidence: float = 1.0
    takeover_request_active: bool = False
    degradation_note: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SystemState":
        return cls(**_pick_fields(cls, raw))


@dataclass
class RiskState:
    """Structured input consumed by the minimum-risk decision module."""

    scenario_id: str
    name: str
    description: str
    expected_strategy: str = ""
    driver: DriverState = field(default_factory=DriverState)
    road: RoadState = field(default_factory=RoadState)
    system: SystemState = field(default_factory=SystemState)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RiskState":
        return cls(
            scenario_id=str(raw.get("scenario_id", "S?")),
            name=str(raw.get("name", "Unnamed scenario")),
            description=str(raw.get("description", "")),
            expected_strategy=str(raw.get("expected_strategy", "")),
            driver=DriverState.from_dict(raw.get("driver", {})),
            road=RoadState.from_dict(raw.get("road", {})),
            system=SystemState.from_dict(raw.get("system", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskAssessment:
    takeover_score: int
    takeover_numeric: float
    takeover_label: str
    takeover_reasons: List[str]
    driver_risk_score: float
    road_risk_score: float
    road_risk_level: int
    road_risk_label: str
    system_risk_score: float
    fused_risk_score: float
    fused_risk_level: int
    fused_risk_label: str
    risk_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_takeover_readiness(state: RiskState) -> tuple[int, float, str, List[str]]:
    """
    Convert driver monitoring signals into a 0-3 takeover readiness score.

    3 = driver can take over; 0 = driver should not be relied on.
    """
    d = state.driver
    s = state.system
    score = 3.0
    reasons: List[str] = []

    if d.no_response_sec >= 2.0:
        score = min(score, 0.25)
        reasons.append(f"驾驶员无响应 {d.no_response_sec:.1f}s，不能依赖其完成接管")
    elif d.no_response_sec >= 1.0:
        score -= 1.1
        reasons.append(f"驾驶员响应迟缓 {d.no_response_sec:.1f}s")

    if d.eye_closed_sec >= 2.0:
        score -= 1.8
        reasons.append(f"闭眼持续 {d.eye_closed_sec:.1f}s，疲劳风险高")
    elif d.eye_closed_sec >= 1.0:
        score -= 1.0
        reasons.append(f"闭眼持续 {d.eye_closed_sec:.1f}s，需要强提醒")

    if d.perclos >= 0.25:
        score -= 1.3
        reasons.append(f"PERCLOS={d.perclos:.2f}，眼睛闭合占比偏高")
    elif d.perclos >= 0.15:
        score -= 0.7
        reasons.append(f"PERCLOS={d.perclos:.2f}，存在疲劳倾向")

    if d.yawn_frequency_min >= 4.0:
        score -= 0.7
        reasons.append(f"哈欠频率 {d.yawn_frequency_min:.1f}/min，疲劳信号明显")
    elif d.yawn_frequency_min >= 2.0:
        score -= 0.35
        reasons.append(f"哈欠频率 {d.yawn_frequency_min:.1f}/min，轻度疲劳")

    abs_yaw = abs(d.head_yaw_deg)
    if d.distracted_sec >= 2.0 or abs_yaw >= 40.0:
        score -= 1.2
        reasons.append(f"分心/偏航明显：{d.distracted_sec:.1f}s, yaw={d.head_yaw_deg:.0f}°")
    elif d.distracted_sec >= 0.8 or abs_yaw >= 30.0:
        score -= 0.7
        reasons.append(f"短时分心：{d.distracted_sec:.1f}s, yaw={d.head_yaw_deg:.0f}°")

    if d.head_pitch_deg <= -30.0:
        score -= 0.8
        reasons.append(f"低头/点头姿态 pitch={d.head_pitch_deg:.0f}°")

    if d.response_latency_sec >= 2.5:
        score -= 0.7
        reasons.append(f"接管反应时间估计 {d.response_latency_sec:.1f}s 偏长")

    if not d.hands_on_wheel:
        score -= 0.4
        reasons.append("双手未稳定在方向盘上")

    if s.takeover_request_active and d.no_response_sec >= 1.5:
        score = 0.0
        reasons.append("接管请求已触发但驾驶员未响应，接管准备度强制降为 0")

    score = clamp(score, 0.0, 3.0)
    if score >= 2.5:
        level = 3
    elif score >= 1.6:
        level = 2
    elif score >= 0.8:
        level = 1
    else:
        level = 0

    if not reasons:
        reasons.append("驾驶员状态稳定，未发现明显疲劳或分心信号")

    return level, score, TAKEOVER_LABELS[level], reasons


def compute_road_risk(state: RiskState) -> tuple[float, int, str, List[str]]:
    """Score front-road collision risk and perception uncertainty."""
    r = state.road
    reasons: List[str] = []

    if r.min_ttc_sec <= 0.8:
        ttc_score = 1.0
    elif r.min_ttc_sec <= 1.5:
        ttc_score = 0.85
    elif r.min_ttc_sec <= 3.0:
        ttc_score = 0.60
    elif r.min_ttc_sec <= 5.0:
        ttc_score = 0.30
    else:
        ttc_score = 0.05

    if r.min_ttc_sec < 6.0:
        reasons.append(f"最小 TTC={r.min_ttc_sec:.1f}s")

    distance_score = _inverse_scale(r.front_distance_m, safe_value=45.0, danger_value=4.0)
    if r.front_distance_m < 25.0:
        reasons.append(f"前车距离 {r.front_distance_m:.1f}m")

    closing_score = clamp(-r.relative_speed_mps / 12.0) if r.relative_speed_mps < 0 else 0.0
    if closing_score > 0.05:
        reasons.append(f"相对速度 {r.relative_speed_mps:.1f}m/s，距离正在缩短")

    hint_score = clamp(r.road_risk_hint / 3.0)
    lane_uncertainty = 1.0 - clamp(r.lane_confidence)

    score = 0.55 * ttc_score + 0.22 * distance_score + 0.15 * closing_score + 0.08 * hint_score
    score = max(score, 0.78 * hint_score)
    score += 0.08 * lane_uncertainty + 0.05 * clamp(r.rear_risk)
    score = clamp(score)

    if lane_uncertainty > 0.35:
        reasons.append(f"车道线/车道相关性置信度较低：{r.lane_confidence:.2f}")
    if r.rear_risk > 0.4:
        reasons.append(f"后向风险偏高：{r.rear_risk:.2f}")
    if not reasons:
        reasons.append("道路风险较低，前向距离和 TTC 均处于安全区间")

    level = _risk_level_from_score(score)
    return score, level, RISK_LEVEL_LABELS[level], reasons


def compute_system_risk(state: RiskState) -> tuple[float, List[str]]:
    """Score automation failure, ODD exit and perception reliability."""
    s = state.system
    reasons: List[str] = []
    score = 0.0

    if s.system_failure:
        score += 0.45
        reasons.append("自动驾驶系统发生失效/降级")
    if s.odd_exit:
        score += 0.35
        reasons.append("当前场景触发 ODD 退出")
    if s.sensor_degraded:
        score += 0.25
        reasons.append("传感器退化或关键感知不可用")

    conf_penalty = 1.0 - clamp(s.perception_confidence)
    score += 0.35 * conf_penalty
    if conf_penalty > 0.30:
        reasons.append(f"感知可信度较低：{s.perception_confidence:.2f}")

    if s.automation_status.lower() in {"degraded", "failure", "minimal_risk"}:
        score += 0.10
        reasons.append(f"系统状态为 {s.automation_status}")

    if s.degradation_note:
        reasons.append(s.degradation_note)

    if not reasons:
        reasons.append("系统状态正常，感知可信度较高")

    return clamp(score), reasons


def evaluate_risk_state(state: RiskState) -> RiskAssessment:
    """Compute all risk scores consumed by the decision engine."""
    takeover_score, takeover_numeric, takeover_label, takeover_reasons = compute_takeover_readiness(state)
    road_score, road_level, road_label, road_reasons = compute_road_risk(state)
    system_score, system_reasons = compute_system_risk(state)

    driver_risk = clamp((3.0 - takeover_numeric) / 3.0)
    fused = 0.40 * road_score + 0.35 * driver_risk + 0.25 * system_score

    if takeover_score <= 1 and road_score >= 0.50:
        fused += 0.15
    if takeover_score == 0 and system_score >= 0.40:
        fused += 0.10
    if state.system.sensor_degraded and road_score >= 0.50:
        fused += 0.08
    fused = clamp(fused)

    fused_level = _risk_level_from_score(fused)
    risk_reasons = road_reasons + system_reasons

    return RiskAssessment(
        takeover_score=takeover_score,
        takeover_numeric=round(takeover_numeric, 2),
        takeover_label=takeover_label,
        takeover_reasons=takeover_reasons,
        driver_risk_score=round(driver_risk, 3),
        road_risk_score=round(road_score, 3),
        road_risk_level=road_level,
        road_risk_label=road_label,
        system_risk_score=round(system_score, 3),
        fused_risk_score=round(fused, 3),
        fused_risk_level=fused_level,
        fused_risk_label=RISK_LEVEL_LABELS[fused_level],
        risk_reasons=risk_reasons,
    )


def load_scenarios(path: str | Path | None = None) -> List[RiskState]:
    """Load demo scenarios from scenarios.json."""
    if path is None:
        path = Path(__file__).with_name("scenarios.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "scenarios" in raw:
        raw = raw["scenarios"]
    return [RiskState.from_dict(item) for item in raw]


def find_scenario(scenarios: Sequence[RiskState], scenario_id: str) -> RiskState:
    wanted = scenario_id.strip().lower()
    for scenario in scenarios:
        if scenario.scenario_id.lower() == wanted or scenario.name.lower() == wanted:
            return scenario
    available = ", ".join(s.scenario_id for s in scenarios)
    raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {available}")


def percent(value: float) -> str:
    return f"{100.0 * clamp(value):.0f}%"
