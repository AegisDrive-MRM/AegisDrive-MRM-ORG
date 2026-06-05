"""
Rule-based minimum-risk decision engine.

This is the non-world-model baseline required by the three-person plan.  It is
implemented as a transparent state machine so that the team can explain every
transition during a stage report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

try:  # Package mode: python -m mrm_demo.app
    from .risk_state import RiskAssessment, RiskState, clamp, evaluate_risk_state
except ImportError:  # Script mode: python app.py
    from risk_state import RiskAssessment, RiskState, clamp, evaluate_risk_state


FSM_STATE_LABELS = {
    "NORMAL": "正常监测",
    "WARNING": "预警增强",
    "TAKEOVER_REQUEST": "请求驾驶员接管",
    "MRM_PREPARE": "最小风险处置准备",
    "MRM_EXECUTE": "执行最小风险处置",
    "MRC_REACHED": "已达到最小风险状态",
}

STRATEGY_LABELS = {
    "continue_monitoring": "继续监测，不触发接管",
    "attention_warning": "增强视觉/声音提醒",
    "request_takeover": "请求驾驶员接管",
    "mild_deceleration": "轻度减速并保持车道",
    "strong_deceleration": "强制减速并保持车道",
    "emergency_brake": "紧急制动，保持车道",
    "keep_lane_safe_stop": "车道内减速至安全停车",
    "shoulder_stop": "靠边/路肩安全停车",
    "lane_change_avoidance": "变道避让",
    "hold_position": "保持驻车/危险报警",
}


@dataclass
class CandidateAction:
    action_id: str
    label: str
    allowed: bool
    reason: str
    effect_hint: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DecisionResult:
    fsm_state: str
    fsm_state_label: str
    baseline_strategy: str
    baseline_strategy_label: str
    urgency: str
    should_request_takeover: bool
    assessment: RiskAssessment
    reasons: List[str] = field(default_factory=list)
    candidate_actions: List[CandidateAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["assessment"] = self.assessment.to_dict()
        data["candidate_actions"] = [a.to_dict() for a in self.candidate_actions]
        return data


def build_candidate_actions(state: RiskState, assessment: RiskAssessment) -> List[CandidateAction]:
    """Build the shared candidate strategy set for baseline and world-model mock."""
    r = state.road
    s = state.system
    actions: List[CandidateAction] = []

    low_risk = assessment.fused_risk_level <= 1 and not (s.system_failure or s.odd_exit)
    actions.append(
        CandidateAction(
            "continue_monitoring",
            STRATEGY_LABELS["continue_monitoring"],
            allowed=low_risk,
            reason="仅在综合风险较低且系统未失效时允许" if not low_risk else "风险处于低位",
            effect_hint="不主动干预，维持监测",
        )
    )

    takeover_allowed = assessment.takeover_score >= 2
    actions.append(
        CandidateAction(
            "request_takeover",
            STRATEGY_LABELS["request_takeover"],
            allowed=takeover_allowed,
            reason="驾驶员接管准备度不足" if not takeover_allowed else "驾驶员具备接管基础",
            effect_hint="通过视觉/声音提醒将控制权交给驾驶员",
        )
    )

    actions.append(
        CandidateAction(
            "attention_warning",
            STRATEGY_LABELS["attention_warning"],
            allowed=assessment.takeover_score >= 1,
            reason="驾驶员完全无响应时不应只报警" if assessment.takeover_score < 1 else "可用于低到中等风险提醒",
            effect_hint="增强提示，但车辆动作保持保守",
        )
    )

    actions.append(
        CandidateAction(
            "mild_deceleration",
            STRATEGY_LABELS["mild_deceleration"],
            allowed=True,
            reason="低侵入式 MRM 准备动作",
            effect_hint="降低闭合速度，给系统和驾驶员争取时间",
        )
    )

    strong_allowed = (
        assessment.road_risk_level >= 1
        or assessment.fused_risk_level >= 1
        or assessment.takeover_score <= 1
        or s.system_failure
        or s.odd_exit
    )
    actions.append(
        CandidateAction(
            "strong_deceleration",
            STRATEGY_LABELS["strong_deceleration"],
            allowed=strong_allowed,
            reason="高风险但仍有制动距离时使用" if strong_allowed else "低风险场景不需要强制减速",
            effect_hint="快速降低 TTC 风险，同时保持车道稳定",
        )
    )

    emergency_allowed = assessment.road_risk_level >= 2 or r.min_ttc_sec <= 1.2
    actions.append(
        CandidateAction(
            "emergency_brake",
            STRATEGY_LABELS["emergency_brake"],
            allowed=emergency_allowed,
            reason="只有在高/极高碰撞风险下才优先考虑" if not emergency_allowed else "TTC 或道路风险已达到紧急阈值",
            effect_hint="最大限度降低前向碰撞风险，但会增加舒适性和后向风险代价",
        )
    )

    keep_lane_stop_allowed = (
        assessment.takeover_score <= 1
        or s.sensor_degraded
        or s.system_failure
        or s.odd_exit
        or assessment.road_risk_level >= 2
    )
    actions.append(
        CandidateAction(
            "keep_lane_safe_stop",
            STRATEGY_LABELS["keep_lane_safe_stop"],
            allowed=keep_lane_stop_allowed,
            reason="传感器退化、驾驶员不可接管或侧向信息不足时的保守兜底动作" if keep_lane_stop_allowed else "低风险且驾驶员可接管时不需要停车",
            effect_hint="不变道，逐步减速至最小风险状态",
        )
    )

    shoulder_allowed = (
        r.shoulder_available
        and r.shoulder_confidence >= 0.55
        and r.lane_confidence >= 0.55
        and s.perception_confidence >= 0.55
        and not s.sensor_degraded
    )
    shoulder_reason = (
        "路肩可用且感知可信度满足要求"
        if shoulder_allowed
        else "路肩不可用、置信度不足或传感器退化"
    )
    actions.append(
        CandidateAction(
            "shoulder_stop",
            STRATEGY_LABELS["shoulder_stop"],
            allowed=shoulder_allowed,
            reason=shoulder_reason,
            effect_hint="在右侧路肩或安全区域完成最小风险停车",
        )
    )

    lane_change_allowed = (
        assessment.road_risk_level >= 2
        and r.adjacent_lane_clear
        and r.lane_change_info_reliable
        and r.lane_confidence >= 0.65
        and s.perception_confidence >= 0.70
        and not s.sensor_degraded
    )
    actions.append(
        CandidateAction(
            "lane_change_avoidance",
            STRATEGY_LABELS["lane_change_avoidance"],
            allowed=lane_change_allowed,
            reason="相邻车道安全且侧向感知可靠" if lane_change_allowed else "侧向信息不足，Demo 中默认禁止冒险变道",
            effect_hint="侧向避让前方目标，但对感知质量要求最高",
        )
    )

    return actions


def _action_allowed(candidate_actions: List[CandidateAction], action_id: str) -> bool:
    for action in candidate_actions:
        if action.action_id == action_id:
            return action.allowed
    return False


def decide_minimum_risk(state: RiskState) -> DecisionResult:
    """Run the rule-based baseline state machine."""
    assessment = evaluate_risk_state(state)
    candidates = build_candidate_actions(state, assessment)
    r = state.road
    s = state.system

    reasons: List[str] = []
    reasons.extend(assessment.takeover_reasons[:2])
    reasons.extend(assessment.risk_reasons[:3])

    if r.ego_speed_kmh <= 3.0 and assessment.fused_risk_level <= 1:
        fsm = "MRC_REACHED"
        strategy = "hold_position"
        urgency = "low"
        reasons.append("车辆已接近静止，维持驻车并打开危险报警")
    elif (
        assessment.fused_risk_level == 0
        and assessment.takeover_score == 3
        and assessment.road_risk_level == 0
        and not (s.system_failure or s.odd_exit or s.sensor_degraded)
    ):
        fsm = "NORMAL"
        strategy = "continue_monitoring"
        urgency = "low"
        reasons.append("综合风险低，继续监测即可")
    elif s.sensor_degraded or s.perception_confidence < 0.55:
        fsm = "MRM_EXECUTE" if (s.odd_exit or assessment.road_risk_level >= 1) else "MRM_PREPARE"
        strategy = "keep_lane_safe_stop"
        urgency = "high" if assessment.road_risk_level >= 2 else "medium"
        reasons.append("感知不可靠，禁止冒险变道，采用车道内安全停车")
    elif assessment.takeover_score >= 2 and (assessment.road_risk_level >= 2 or s.system_failure or s.odd_exit):
        fsm = "TAKEOVER_REQUEST"
        strategy = "request_takeover"
        urgency = "high" if assessment.road_risk_level >= 2 else "medium"
        reasons.append("驾驶员仍具备接管基础，优先发起接管请求")
    elif assessment.takeover_score >= 2 and (assessment.fused_risk_level <= 1 or assessment.road_risk_level <= 1):
        fsm = "WARNING"
        strategy = "attention_warning"
        urgency = "medium"
        reasons.append("当前主要是注意力或中等道路风险，先增强提醒")
    else:
        should_execute = assessment.fused_risk_level >= 2 or assessment.takeover_score <= 1 or s.system_failure or s.odd_exit
        fsm = "MRM_EXECUTE" if should_execute else "MRM_PREPARE"
        urgency = "high" if assessment.fused_risk_level >= 2 else "medium"

        if r.min_ttc_sec <= 1.0 or (assessment.road_risk_level == 3 and assessment.takeover_score == 0):
            strategy = "emergency_brake"
            urgency = "critical"
            reasons.append("TTC 已接近极限，触发紧急制动兜底")
        elif _action_allowed(candidates, "shoulder_stop") and r.min_ttc_sec > 1.2 and assessment.road_risk_level <= 2:
            strategy = "shoulder_stop"
            reasons.append("驾驶员不可接管且路肩可用，建议靠边/路肩停车")
        elif assessment.road_risk_level >= 2:
            strategy = "strong_deceleration"
            reasons.append("高道路风险但侧向/路肩条件不足，先强制减速保持车道")
        else:
            strategy = "mild_deceleration"
            reasons.append("进入 MRM 准备：轻度减速并持续观察")

    return DecisionResult(
        fsm_state=fsm,
        fsm_state_label=FSM_STATE_LABELS[fsm],
        baseline_strategy=strategy,
        baseline_strategy_label=STRATEGY_LABELS[strategy],
        urgency=urgency,
        should_request_takeover=strategy in {"request_takeover", "attention_warning"},
        assessment=assessment,
        reasons=reasons,
        candidate_actions=candidates,
    )


def format_decision(result: DecisionResult) -> str:
    """Human-readable decision summary for command-line demo."""
    a = result.assessment
    lines = [
        f"FSM 状态: {result.fsm_state_label} ({result.fsm_state})",
        f"Baseline 策略: {result.baseline_strategy_label} [{result.baseline_strategy}]",
        f"紧急程度: {result.urgency}",
        f"接管准备度: {a.takeover_score}/3 - {a.takeover_label}",
        f"道路风险: {a.road_risk_score:.2f} - {a.road_risk_label}",
        f"系统风险: {a.system_risk_score:.2f}",
        f"综合风险: {a.fused_risk_score:.2f} - {a.fused_risk_label}",
        "理由: " + "；".join(result.reasons),
    ]
    return "\n".join(lines)
