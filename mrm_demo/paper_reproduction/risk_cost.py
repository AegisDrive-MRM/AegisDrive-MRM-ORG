"""Risk-cost utilities for the lightweight world-model reproduction.

This file defines a compact state/action abstraction and a risk cost function.
It is intended to be interpretable enough for a stage report while keeping the
same high-level mechanism used by world-model planning papers:

    current state + candidate action -> future state rollout -> risk cost
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

try:
    from ..decision_engine import CandidateAction, DecisionResult
    from ..risk_state import RiskState, clamp
except ImportError:
    from decision_engine import CandidateAction, DecisionResult
    from risk_state import RiskState, clamp


@dataclass(frozen=True)
class ActionProfile:
    """Numeric action profile used by the rollout model."""

    action_id: str
    label: str
    ego_accel_mps2: float
    front_accel_mps2: float = 0.0
    comfort_cost: float = 0.0
    transition_cost: float = 0.0
    requires_driver: bool = False
    requires_shoulder: bool = False
    requires_adjacent_lane: bool = False
    lateral_move: bool = False
    target_stop: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RolloutState:
    """Minimal traffic state for short-horizon risk rollout."""

    t: float
    ego_speed_mps: float
    front_distance_m: float
    relative_speed_mps: float
    min_ttc_sec: float
    lane_confidence: float
    perception_confidence: float
    shoulder_confidence: float
    rear_risk: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _ttc(distance_m: float, relative_speed_mps: float) -> float:
    """Return TTC. Negative relative speed means the ego vehicle is closing in."""
    if distance_m <= 0.1:
        return 0.0
    if relative_speed_mps < -0.05:
        return distance_m / max(-relative_speed_mps, 1e-3)
    return 99.0


def action_profile(action: CandidateAction) -> ActionProfile:
    """Map a semantic MRM action to a numeric rollout profile."""
    profiles = {
        "continue_monitoring": ActionProfile(action.action_id, action.label, ego_accel_mps2=0.0, comfort_cost=0.00),
        "attention_warning": ActionProfile(action.action_id, action.label, ego_accel_mps2=-0.2, comfort_cost=0.01),
        "request_takeover": ActionProfile(action.action_id, action.label, ego_accel_mps2=-0.4, comfort_cost=0.02, requires_driver=True),
        "mild_deceleration": ActionProfile(action.action_id, action.label, ego_accel_mps2=-1.4, comfort_cost=0.04),
        "strong_deceleration": ActionProfile(action.action_id, action.label, ego_accel_mps2=-3.2, comfort_cost=0.10),
        "emergency_brake": ActionProfile(action.action_id, action.label, ego_accel_mps2=-6.0, comfort_cost=0.24),
        "keep_lane_safe_stop": ActionProfile(action.action_id, action.label, ego_accel_mps2=-2.6, comfort_cost=0.08, target_stop=True),
        "shoulder_stop": ActionProfile(action.action_id, action.label, ego_accel_mps2=-2.2, comfort_cost=0.12, transition_cost=0.16, requires_shoulder=True, lateral_move=True, target_stop=True),
        "lane_change_avoidance": ActionProfile(action.action_id, action.label, ego_accel_mps2=-1.2, comfort_cost=0.16, transition_cost=0.22, requires_adjacent_lane=True, lateral_move=True),
        "hold_position": ActionProfile(action.action_id, action.label, ego_accel_mps2=0.0, comfort_cost=0.00, target_stop=True),
    }
    return profiles.get(action.action_id, ActionProfile(action.action_id, action.label, ego_accel_mps2=0.0, comfort_cost=0.05))


def initial_rollout_state(state: RiskState) -> RolloutState:
    """Extract the compact rollout state from the shared RiskState."""
    road = state.road
    system = state.system
    return RolloutState(
        t=0.0,
        ego_speed_mps=max(0.0, road.ego_speed_kmh / 3.6),
        front_distance_m=max(0.0, road.front_distance_m),
        relative_speed_mps=road.relative_speed_mps,
        min_ttc_sec=max(0.0, road.min_ttc_sec),
        lane_confidence=clamp(road.lane_confidence),
        perception_confidence=clamp(system.perception_confidence),
        shoulder_confidence=clamp(road.shoulder_confidence),
        rear_risk=clamp(road.rear_risk),
    )


def step_rollout(x: RolloutState, profile: ActionProfile, dt: float) -> RolloutState:
    """Advance the compact traffic state with a constant-acceleration model."""
    ego_acc = profile.ego_accel_mps2
    front_acc = profile.front_accel_mps2

    ego_speed_next = max(0.0, x.ego_speed_mps + ego_acc * dt)
    rel_acc = front_acc - ego_acc
    rel_speed_next = x.relative_speed_mps + rel_acc * dt

    # Distance evolves using current relative speed plus relative acceleration.
    distance_next = x.front_distance_m + x.relative_speed_mps * dt + 0.5 * rel_acc * dt * dt
    distance_next = max(0.0, distance_next)
    ttc_next = _ttc(distance_next, rel_speed_next)

    return RolloutState(
        t=round(x.t + dt, 3),
        ego_speed_mps=ego_speed_next,
        front_distance_m=distance_next,
        relative_speed_mps=rel_speed_next,
        min_ttc_sec=ttc_next,
        lane_confidence=x.lane_confidence,
        perception_confidence=x.perception_confidence,
        shoulder_confidence=x.shoulder_confidence,
        rear_risk=x.rear_risk,
    )


def _piecewise_ttc_risk(ttc: float) -> float:
    if ttc <= 0.8:
        return 1.0
    if ttc <= 1.5:
        return 0.82
    if ttc <= 3.0:
        return 0.55
    if ttc <= 5.0:
        return 0.26
    return 0.04


def _distance_risk(distance_m: float) -> float:
    # 45 m is treated as safe, 3 m as dangerous.
    return clamp((45.0 - distance_m) / (45.0 - 3.0))


def _driver_takeover_risk(decision: DecisionResult, profile: ActionProfile, horizon_s: float) -> float:
    driver_risk = decision.assessment.driver_risk_score
    trs = decision.assessment.takeover_score
    if profile.requires_driver:
        if trs >= 2:
            # A takeover request helps only after a short response delay.
            return clamp(driver_risk - 0.08 * max(0.0, horizon_s - 1.0))
        return clamp(driver_risk + 0.30)
    if profile.target_stop or profile.ego_accel_mps2 <= -1.0:
        # MRM actions do not need the driver, so driver risk is down-weighted.
        return clamp(0.55 * driver_risk)
    return driver_risk


def risk_cost(state: RiskState, decision: DecisionResult, x: RolloutState, profile: ActionProfile) -> float:
    """Compute the world-model risk cost for one predicted future state."""
    road = state.road
    system = state.system
    assessment = decision.assessment

    collision = 0.62 * _piecewise_ttc_risk(x.min_ttc_sec) + 0.23 * _distance_risk(x.front_distance_m)
    collision += 0.15 * max(0.0, -x.relative_speed_mps) / 12.0
    collision = clamp(collision)

    driver_term = _driver_takeover_risk(decision, profile, x.t)
    uncertainty = 1.0 - clamp(x.perception_confidence)
    lane_uncertainty = 1.0 - clamp(x.lane_confidence)

    lateral_penalty = 0.0
    if profile.lateral_move:
        # Lateral actions carry higher short-term risk, especially if side info is weak.
        lateral_penalty += profile.transition_cost * max(0.0, 1.0 - x.t / 3.0)
        lateral_penalty += 0.12 * lane_uncertainty
        if profile.requires_shoulder:
            lateral_penalty += 0.10 * (1.0 - clamp(road.shoulder_confidence))
        if profile.requires_adjacent_lane:
            lateral_penalty += 0.10 * (0.0 if road.adjacent_lane_clear else 1.0)

    rear_penalty = 0.0
    if profile.ego_accel_mps2 <= -3.0:
        rear_penalty = 0.13 * clamp(x.rear_risk)
    if profile.ego_accel_mps2 <= -5.0:
        rear_penalty += 0.05 * clamp(x.rear_risk)

    system_term = 0.45 * assessment.system_risk_score + 0.20 * uncertainty
    if system.sensor_degraded and profile.lateral_move:
        system_term += 0.20

    # Prior keeps the prediction anchored to current fused assessment.
    prior = 0.20 * assessment.fused_risk_score
    risk = prior + 0.46 * collision + 0.18 * driver_term + 0.16 * system_term
    risk += lateral_penalty + rear_penalty

    # Stopping actions are rewarded after the transition if speed is lower.
    if profile.target_stop and x.t >= 3.0:
        stop_reward = 0.06 * clamp((state.road.ego_speed_kmh / 3.6 - x.ego_speed_mps) / 12.0)
        risk -= stop_reward

    # If collision is imminent, the world model should value immediate braking
    # more than comfort. This keeps critical-TTC scenes aligned with safety.
    if profile.action_id == "emergency_brake" and state.road.min_ttc_sec <= 1.0 and x.t >= 1.0:
        risk -= 0.12

    # A feasible shoulder stop pays a lateral transition cost first, but should
    # become attractive once it approaches the minimal-risk condition.
    if profile.requires_shoulder and road.shoulder_available and x.t >= 3.0:
        risk -= 0.08 * clamp(road.shoulder_confidence)
        if x.t >= 5.0:
            risk -= 0.04 * clamp(road.shoulder_confidence)

    return round(clamp(risk), 4)


def action_constraint_penalty(state: RiskState, decision: DecisionResult, action: CandidateAction, profile: ActionProfile) -> float:
    """Action-level penalty used after the rollout."""
    if not action.allowed:
        return 1.0

    road = state.road
    system = state.system
    penalty = profile.comfort_cost

    if profile.requires_driver and decision.assessment.takeover_score < 2:
        penalty += 0.45
    if profile.requires_shoulder and not road.shoulder_available:
        penalty += 0.50
    if profile.requires_shoulder and road.shoulder_confidence < 0.55:
        penalty += 0.20
    if profile.requires_adjacent_lane and not road.adjacent_lane_clear:
        penalty += 0.40
    if profile.lateral_move and (system.sensor_degraded or system.perception_confidence < 0.65):
        penalty += 0.35
    if action.action_id == decision.baseline_strategy:
        penalty -= 0.08

    # In MRM scenes where the driver is unreliable, actions that do not make
    # progress toward a minimal-risk condition should pay a terminal penalty.
    if decision.assessment.takeover_score <= 1 and not profile.target_stop and action.action_id in {"mild_deceleration", "strong_deceleration"}:
        penalty += 0.12

    if action.action_id == "emergency_brake" and road.min_ttc_sec > 1.5:
        penalty += 0.08
    if action.action_id == "emergency_brake" and road.min_ttc_sec <= 1.0:
        penalty = max(0.0, penalty - 0.18)

    return max(0.0, round(penalty, 4))
