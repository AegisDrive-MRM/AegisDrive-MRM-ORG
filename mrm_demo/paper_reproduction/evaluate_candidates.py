"""Candidate evaluation for the lightweight T-ITS paper reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List

try:
    from ..decision_engine import CandidateAction, DecisionResult
    from ..risk_state import RiskState
    from .risk_cost import action_constraint_penalty
    from .world_model_predictor import DEFAULT_HORIZONS, ActionRollout, RolloutPoint, rollout_action
except ImportError:
    from decision_engine import CandidateAction, DecisionResult
    from risk_state import RiskState
    from paper_reproduction.risk_cost import action_constraint_penalty
    from paper_reproduction.world_model_predictor import DEFAULT_HORIZONS, ActionRollout, RolloutPoint, rollout_action


@dataclass
class PaperActionPrediction:
    action_id: str
    label: str
    allowed: bool
    status: str
    score: float
    future_risks: List[RolloutPoint]
    reason: str
    rollout: ActionRollout
    penalty: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "allowed": self.allowed,
            "status": self.status,
            "score": self.score,
            "future_risks": [p.to_dict() for p in self.future_risks],
            "reason": self.reason,
            "penalty": self.penalty,
            "rollout": self.rollout.to_dict(),
        }


@dataclass
class PaperWorldModelResult:
    recommended_action_id: str
    recommended_label: str
    explanation: str
    predictions: List[PaperActionPrediction] = field(default_factory=list)
    method_note: str = "lightweight_dawm_mpc_reproduction"

    def to_dict(self) -> Dict[str, object]:
        return {
            "recommended_action_id": self.recommended_action_id,
            "recommended_label": self.recommended_label,
            "explanation": self.explanation,
            "method_note": self.method_note,
            "predictions": [p.to_dict() for p in self.predictions],
        }


def _weighted_score(points: List[RolloutPoint], penalty: float) -> float:
    weights = {1: 0.46, 3: 0.34, 5: 0.20}
    score = sum(weights.get(p.horizon_s, 0.0) * p.risk for p in points) + penalty
    return round(score, 4)


def _trend(points: List[RolloutPoint]) -> str:
    if not points:
        return "no rollout points"
    first = points[0].risk
    last = points[-1].risk
    if last < first - 0.12:
        return "predicted risk decreases clearly"
    if last > first + 0.08:
        return "predicted risk increases"
    return "predicted risk is almost flat"


def _explain(action: CandidateAction, prediction: PaperActionPrediction | None, decision: DecisionResult) -> str:
    if not action.allowed:
        return action.reason
    if prediction is None:
        return "rollout finished"
    risks = prediction.future_risks
    trend = _trend(risks)
    if action.action_id == "request_takeover" and decision.assessment.takeover_score >= 2:
        return "driver takeover is feasible; " + trend
    if action.action_id == "request_takeover":
        return "driver takeover is unsafe because TRS is low"
    if action.action_id == "emergency_brake":
        return "front collision risk drops fastest, but comfort and rear-risk cost are high; " + trend
    if action.action_id == "shoulder_stop":
        return "lateral transition cost is paid first, then terminal risk can fall if shoulder confidence is high; " + trend
    if action.action_id == "lane_change_avoidance":
        return "side maneuver depends strongly on adjacent-lane reliability; " + trend
    if action.action_id == "keep_lane_safe_stop":
        return "conservative MRM action, suitable when perception or driver takeover is unreliable; " + trend
    if action.action_id in {"mild_deceleration", "strong_deceleration"}:
        return "longitudinal braking increases TTC and reduces closing pressure; " + trend
    return trend


def evaluate_action(state: RiskState, decision: DecisionResult, action: CandidateAction, horizons: Iterable[int] = DEFAULT_HORIZONS) -> PaperActionPrediction:
    rollout = rollout_action(state, decision, action, horizons=horizons, dt=0.2, keep_dense_trace=False)
    penalty = action_constraint_penalty(state, decision, action, rollout.profile)
    score = _weighted_score(rollout.future_risks, penalty)
    pred = PaperActionPrediction(
        action_id=action.action_id,
        label=action.label,
        allowed=action.allowed,
        status="pending",
        score=score,
        future_risks=rollout.future_risks,
        reason="",
        rollout=rollout,
        penalty=penalty,
    )
    pred.reason = _explain(action, pred, decision)
    return pred


def run_paper_world_model(state: RiskState, decision: DecisionResult, horizons: Iterable[int] = DEFAULT_HORIZONS) -> PaperWorldModelResult:
    """Run the lightweight world-model reproduction on all candidate MRM actions."""
    predictions = [evaluate_action(state, decision, action, horizons=horizons) for action in decision.candidate_actions]
    allowed = [p for p in predictions if p.allowed]
    recommended = min(allowed if allowed else predictions, key=lambda p: p.score)

    for p in predictions:
        if not p.allowed:
            p.status = "forbidden"
        elif p.action_id == recommended.action_id:
            p.status = "recommended"
        elif p.score <= recommended.score + 0.08:
            p.status = "backup"
        else:
            p.status = "not_preferred"

    predictions.sort(key=lambda p: (0 if p.allowed else 1, p.score))
    explanation = (
        "Lightweight paper reproduction: each MRM candidate is rolled out for 1/3/5 s "
        "with a compact driving-aware state model; the action with the lowest risk-aware "
        "weighted cost is selected as the world-model-assisted recommendation."
    )
    return PaperWorldModelResult(
        recommended_action_id=recommended.action_id,
        recommended_label=recommended.label,
        explanation=explanation,
        predictions=predictions,
    )
