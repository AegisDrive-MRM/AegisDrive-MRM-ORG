"""
Mock world-model assisted prediction.

The real research direction can later replace this file with a learned temporal
model or a model-based simulator.  The current mock version is deliberately
rule-based so the demo is runnable without camera, GPU or training data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List

try:
    from .decision_engine import CandidateAction, DecisionResult
    from .risk_state import RiskState, clamp
except ImportError:
    from decision_engine import CandidateAction, DecisionResult
    from risk_state import RiskState, clamp


HORIZONS = (1, 3, 5)


@dataclass
class FutureRiskPoint:
    horizon_s: int
    risk: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ActionPrediction:
    action_id: str
    label: str
    allowed: bool
    status: str
    score: float
    future_risks: List[FutureRiskPoint]
    reason: str

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["future_risks"] = [p.to_dict() for p in self.future_risks]
        return data


@dataclass
class WorldModelResult:
    recommended_action_id: str
    recommended_label: str
    explanation: str
    predictions: List[ActionPrediction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "recommended_action_id": self.recommended_action_id,
            "recommended_label": self.recommended_label,
            "explanation": self.explanation,
            "predictions": [p.to_dict() for p in self.predictions],
        }


def _constraint_penalty(state: RiskState, action: CandidateAction, decision: DecisionResult | None = None) -> float:
    r = state.road
    s = state.system
    penalty = 0.0
    if not action.allowed:
        return 1.0

    road_level = decision.assessment.road_risk_level if decision is not None else 0
    fused = decision.assessment.fused_risk_score if decision is not None else 0.0
    baseline = decision.baseline_strategy if decision is not None else ""

    # Avoid recommending unnecessary interventions in low-risk scenes.
    if fused < 0.25 and action.action_id != "continue_monitoring":
        penalty += 0.18

    if action.action_id == "request_takeover":
        if not (s.system_failure or s.odd_exit or road_level >= 2):
            penalty += 0.14
        if s.takeover_request_active:
            penalty += 0.02
    if action.action_id == "mild_deceleration" and road_level == 0:
        penalty += 0.08
    if action.action_id == "strong_deceleration":
        penalty += 0.03 if road_level >= 2 else 0.16
        if s.sensor_degraded:
            penalty += 0.10
    if action.action_id == "emergency_brake":
        penalty += 0.06 + 0.10 * clamp(r.rear_risk)
        if r.min_ttc_sec > 1.0 and baseline != "emergency_brake":
            penalty += 0.18
        elif road_level < 3 and r.min_ttc_sec > 1.0:
            penalty += 0.18
    if action.action_id == "keep_lane_safe_stop":
        if road_level <= 1 and not (s.sensor_degraded or s.system_failure or s.odd_exit):
            penalty += 0.18
        elif road_level == 0:
            penalty += 0.10
        if s.sensor_degraded:
            penalty -= 0.07
    if action.action_id == "lane_change_avoidance":
        penalty += 0.10 + 0.08 * (1.0 - clamp(r.lane_confidence))
    if action.action_id == "shoulder_stop":
        penalty += 0.06 * (1.0 - clamp(r.shoulder_confidence))
        if r.min_ttc_sec < 1.2:
            penalty += 0.12
        if baseline == "shoulder_stop":
            penalty -= 0.06
    if action.action_id == baseline:
        penalty -= 0.08
    return max(0.0, penalty)


def _risk_after_action(state: RiskState, decision: DecisionResult, action: CandidateAction, horizon_s: int) -> float:
    """Rule-based surrogate for future 1/3/5s risk."""
    base = decision.assessment.fused_risk_score
    road = state.road
    system = state.system
    takeover = decision.assessment.takeover_score
    h = float(horizon_s)

    closing_pressure = clamp(-road.relative_speed_mps / 12.0) if road.relative_speed_mps < 0 else 0.0
    uncertainty = 1.0 - clamp(system.perception_confidence)
    lane_uncertainty = 1.0 - clamp(road.lane_confidence)
    rear_risk = clamp(road.rear_risk)
    critical_ttc = road.min_ttc_sec <= 1.0

    action_id = action.action_id
    delta = 0.0

    if not action.allowed:
        delta = 0.28 + 0.04 * h
    elif action_id == "continue_monitoring":
        if base < 0.25 and decision.assessment.road_risk_level == 0:
            delta = 0.008 * h
        else:
            delta = 0.05 * h + 0.07 * closing_pressure * h + 0.04 * uncertainty * h
    elif action_id == "attention_warning":
        if takeover >= 2:
            delta = -0.02 * min(h, 3.0) + 0.02 * closing_pressure * h
        else:
            delta = 0.04 * h + 0.10
    elif action_id == "request_takeover":
        if takeover >= 2:
            delta = -0.06 * min(h, 3.0) - 0.035 * max(0.0, h - 3.0)
        else:
            delta = 0.06 * h + 0.15
    elif action_id == "mild_deceleration":
        delta = -0.045 * h + 0.035 * closing_pressure * max(0.0, h - 1.0)
        if critical_ttc:
            delta += 0.12
    elif action_id == "strong_deceleration":
        delta = -0.085 * h + 0.04 * rear_risk
        if road.min_ttc_sec > 3.0:
            delta += 0.04
    elif action_id == "emergency_brake":
        delta = -0.16 * h + 0.12 * rear_risk
        if decision.assessment.road_risk_level < 3:
            delta += 0.05
    elif action_id == "keep_lane_safe_stop":
        delta = -0.060 * h + 0.050 * lane_uncertainty + 0.025 * rear_risk
        if system.sensor_degraded:
            delta -= 0.015 * h  # conservative action benefits under uncertainty.
    elif action_id == "shoulder_stop":
        # Transition is not immediately risk-free, but 3-5s benefit is strong.
        delta = 0.03 if h <= 1.0 else -0.11 * h
        delta += 0.05 * (1.0 - clamp(road.shoulder_confidence)) + 0.03 * lane_uncertainty
    elif action_id == "lane_change_avoidance":
        # Side movement has a one-second transition risk even when allowed.
        delta = 0.06 if h <= 1.0 else -0.075 * h
        delta += 0.08 * lane_uncertainty + 0.05 * uncertainty
    else:
        delta = 0.0

    return round(clamp(base + delta), 3)


def _weighted_score(points: List[FutureRiskPoint], penalty: float) -> float:
    # Put more weight on near-term risk; emergency scenarios care about 1s and 3s first.
    weights = {1: 0.45, 3: 0.35, 5: 0.20}
    score = sum(weights[p.horizon_s] * p.risk for p in points) + penalty
    return round(score, 3)


def _explain_prediction(state: RiskState, decision: DecisionResult, action: CandidateAction, points: List[FutureRiskPoint]) -> str:
    first, middle, last = points[0].risk, points[1].risk, points[-1].risk
    if not action.allowed:
        return action.reason
    if action.action_id == "request_takeover" and decision.assessment.takeover_score >= 2:
        return "驾驶员具备接管基础，预计提醒后风险逐步下降"
    if action.action_id == "request_takeover":
        return "驾驶员不可接管，仅请求接管会造成风险继续上升"
    if action.action_id == "emergency_brake":
        return "最快降低前向碰撞风险，但需关注后向风险和舒适性代价"
    if action.action_id == "shoulder_stop":
        return "路肩可用时，1s 内有横向过渡风险，3-5s 后风险明显下降"
    if action.action_id == "lane_change_avoidance":
        return "只有侧向感知可靠时才允许；否则不作为最小风险策略"
    if last < first - 0.10:
        return "风险趋势下降，适合作为保守备选"
    if middle > first + 0.05:
        return "短时风险上升，当前场景不优先采用"
    return "风险变化有限，更多用于过渡或低风险场景"


def run_world_model_mock(state: RiskState, decision: DecisionResult) -> WorldModelResult:
    """Evaluate candidate actions at 1/3/5 seconds and choose the lowest-risk action."""
    predictions: List[ActionPrediction] = []
    for action in decision.candidate_actions:
        points = [
            FutureRiskPoint(h, _risk_after_action(state, decision, action, h))
            for h in HORIZONS
        ]
        score = _weighted_score(points, _constraint_penalty(state, action, decision))
        predictions.append(
            ActionPrediction(
                action_id=action.action_id,
                label=action.label,
                allowed=action.allowed,
                status="待定",
                score=score,
                future_risks=points,
                reason=_explain_prediction(state, decision, action, points),
            )
        )

    allowed_predictions = [p for p in predictions if p.allowed]
    if allowed_predictions:
        recommended = min(allowed_predictions, key=lambda p: p.score)
    else:
        recommended = min(predictions, key=lambda p: p.score)

    for pred in predictions:
        if not pred.allowed:
            pred.status = "禁止"
        elif pred.action_id == recommended.action_id:
            pred.status = "推荐"
        elif pred.score <= recommended.score + 0.08:
            pred.status = "备选"
        else:
            pred.status = "不优先"

    predictions.sort(key=lambda p: (0 if p.allowed else 1, p.score))
    explanation = (
        f"世界模型 mock 对候选策略进行 1/3/5 秒风险推演后，推荐 "
        f"“{recommended.label}”。该结果用于辅助 baseline，不直接替代安全兜底规则。"
    )
    return WorldModelResult(
        recommended_action_id=recommended.action_id,
        recommended_label=recommended.label,
        explanation=explanation,
        predictions=predictions,
    )
