"""Adaptive Event-Triggered controller for AET-RAG.

Implements equations (urgency, spatial, slack, disruption score, adaptive
threshold) and the trigger decision rule from the paper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AETConfig:
    theta_0: float = 0.7
    theta_min: float = 0.25
    gamma: float = 0.015
    w_urgency: float = 0.5
    w_spatial: float = 0.25
    w_slack: float = 0.25
    d_max: float = 60.0
    epsilon: float = 1.0
    # Safety-net guards: trigger re-optimization whenever a single component
    # alone signals a near-certain feasibility loss, even if the convex
    # combination D_t still sits below the adaptive threshold. Set to a value
    # > 1.0 to disable a guard.
    safety_urgency: float = 0.85
    safety_slack: float = 0.95


@dataclass
class RouteState:
    """Lightweight snapshot of the current solver solution used by the gate."""
    coords: np.ndarray
    arrival_times: Dict[int, float]
    deadlines: Dict[int, float]
    customers_remaining: List[int]
    edge_routes: Dict[Tuple[int, int], int]   # (i,j) -> vehicle_id
    network_diameter: float


@dataclass
class TriggerLog:
    event_id: str
    timestamp: float
    U: float
    S: float
    R: float
    D: float
    theta: float
    trigger: bool
    reason: str
    elapsed_since_last: float


class AETController:
    def __init__(self, cfg: AETConfig):
        self.cfg = cfg
        self.t_last: float = 0.0
        self.logs: List[TriggerLog] = []

    # ----- components -------------------------------------------------------
    def urgency(self, p: float, d_hat: float) -> float:
        return float(p * min(1.0, d_hat / max(self.cfg.d_max, 1e-9)))

    def spatial(self, event_pos: np.ndarray, state: RouteState) -> float:
        if not state.customers_remaining:
            return 0.0
        cluster = state.coords[state.customers_remaining]
        centroid = cluster.mean(axis=0)
        d = float(np.linalg.norm(event_pos - centroid))
        return float(max(0.0, 1.0 - d / max(state.network_diameter, 1e-9)))

    def slack_risk(
        self, affected_id: Tuple[int, int], d_hat: float, state: RouteState
    ) -> float:
        # Slack = min over downstream customers (deadline - ETA) along the same vehicle
        i, j = affected_id
        veh = state.edge_routes.get((i, j))
        if veh is None:
            # event not on any active edge: low downstream risk
            return 0.0
        slacks = [
            state.deadlines[c] - state.arrival_times[c]
            for c in state.customers_remaining
            if c in state.arrival_times and c in state.deadlines
        ]
        if not slacks:
            return 1.0
        slack = max(0.0, min(slacks))
        return float(1.0 - min(1.0, slack / (d_hat + self.cfg.epsilon)))

    def threshold(self, t: float) -> float:
        dt = max(0.0, t - self.t_last)
        return float(
            self.cfg.theta_min
            + (self.cfg.theta_0 - self.cfg.theta_min) * math.exp(-self.cfg.gamma * dt)
        )

    # ----- core decision ---------------------------------------------------
    def evaluate(
        self,
        z,                                        # ExtractedTuple
        state: RouteState,
        event_position: Optional[np.ndarray] = None,
    ) -> TriggerLog:
        if event_position is None:
            i, j = z.affected_id
            event_position = (state.coords[i] + state.coords[j]) / 2.0
        U = self.urgency(z.delay_probability, z.severity_minutes)
        S = self.spatial(event_position, state)
        R = self.slack_risk(tuple(z.affected_id), z.severity_minutes, state)
        D = (
            self.cfg.w_urgency * U
            + self.cfg.w_spatial * S
            + self.cfg.w_slack * R
        )
        theta = self.threshold(z.timestamp)
        trig = D >= theta
        reason = "D>=theta" if trig else "D<theta"
        if not trig and U >= self.cfg.safety_urgency:
            trig = True
            reason = "safety_U"
        if not trig and R >= self.cfg.safety_slack:
            trig = True
            reason = "safety_R"
        log = TriggerLog(
            event_id=z.event_id,
            timestamp=z.timestamp,
            U=U, S=S, R=R, D=D, theta=theta,
            trigger=trig,
            reason=reason,
            elapsed_since_last=z.timestamp - self.t_last,
        )
        self.logs.append(log)
        return log

    def mark_solver_run(self, t: float) -> None:
        self.t_last = t

    def to_records(self) -> List[dict]:
        return [vars(l) for l in self.logs]
