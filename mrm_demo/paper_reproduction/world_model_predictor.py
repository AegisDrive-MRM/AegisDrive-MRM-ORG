"""Short-horizon world-model rollout for MRM candidate actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List

try:
    from ..decision_engine import CandidateAction, DecisionResult
    from ..risk_state import RiskState
    from .risk_cost import (
        ActionProfile,
        RolloutState,
        action_profile,
        initial_rollout_state,
        risk_cost,
        step_rollout,
    )
except ImportError:
    from decision_engine import CandidateAction, DecisionResult
    from risk_state import RiskState
    from paper_reproduction.risk_cost import (
        ActionProfile,
        RolloutState,
        action_profile,
        initial_rollout_state,
        risk_cost,
        step_rollout,
    )


DEFAULT_HORIZONS = (1, 3, 5)


@dataclass
class RolloutPoint:
    horizon_s: int
    risk: float
    front_distance_m: float
    min_ttc_sec: float
    ego_speed_kmh: float
    relative_speed_mps: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ActionRollout:
    action_id: str
    label: str
    profile: ActionProfile
    future_risks: List[RolloutPoint] = field(default_factory=list)
    dense_trace: List[RolloutPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "profile": self.profile.to_dict(),
            "future_risks": [p.to_dict() for p in self.future_risks],
            "dense_trace": [p.to_dict() for p in self.dense_trace],
        }


def _point_from_state(state: RiskState, decision: DecisionResult, x: RolloutState, profile: ActionProfile, horizon_s: int) -> RolloutPoint:
    return RolloutPoint(
        horizon_s=horizon_s,
        risk=risk_cost(state, decision, x, profile),
        front_distance_m=round(x.front_distance_m, 2),
        min_ttc_sec=round(x.min_ttc_sec, 2),
        ego_speed_kmh=round(x.ego_speed_mps * 3.6, 2),
        relative_speed_mps=round(x.relative_speed_mps, 2),
    )


def rollout_action(
    state: RiskState,
    decision: DecisionResult,
    action: CandidateAction,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    dt: float = 0.2,
    keep_dense_trace: bool = False,
) -> ActionRollout:
    """Roll out one semantic action and sample risk at selected horizons."""
    profile = action_profile(action)
    horizon_set = {int(h) for h in horizons}
    max_horizon = max(horizon_set)
    x = initial_rollout_state(state)
    future: List[RolloutPoint] = []
    dense: List[RolloutPoint] = []

    step_count = int(round(max_horizon / dt))
    for step in range(1, step_count + 1):
        x = step_rollout(x, profile, dt)
        t_int = int(round(x.t))
        if keep_dense_trace and step % max(1, int(round(1.0 / dt))) == 0:
            dense.append(_point_from_state(state, decision, x, profile, t_int))
        if abs(x.t - t_int) < 1e-6 and t_int in horizon_set:
            future.append(_point_from_state(state, decision, x, profile, t_int))

    # Defensive fallback for unusual dt values.
    got = {p.horizon_s for p in future}
    for h in sorted(horizon_set - got):
        xx = initial_rollout_state(state)
        steps = int(round(h / dt))
        for _ in range(steps):
            xx = step_rollout(xx, profile, dt)
        future.append(_point_from_state(state, decision, xx, profile, h))

    future.sort(key=lambda p: p.horizon_s)
    return ActionRollout(action.action_id, action.label, profile, future, dense)
