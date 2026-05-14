"""Orchestrate the three synchronization scenarios on a shared event stream."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .aet_controller import AETConfig, AETController, RouteState
from .ortools_solver import SolverResult, solve

logger = logging.getLogger(__name__)


def _build_route_state(
    instance, result: SolverResult, served_only: bool = True
) -> RouteState:
    diameter = float(
        np.linalg.norm(instance.coords[:, None, :] - instance.coords[None, :, :], axis=-1).max()
    )
    remaining = [n for n in result.arrival_times.keys() if n != instance.depot]
    deadlines = {
        n: float(instance.time_windows[n, 1]) for n in remaining
    }
    return RouteState(
        coords=instance.coords,
        arrival_times={k: float(v) for k, v in result.arrival_times.items()},
        deadlines=deadlines,
        customers_remaining=remaining,
        edge_routes=result.edge_to_vehicle,
        network_diameter=diameter,
    )


def _apply_event_to_travel(
    travel: np.ndarray, affected_id: Tuple[int, int], severity_minutes: float
) -> np.ndarray:
    i, j = affected_id
    updated = travel.copy()
    if 0 <= i < updated.shape[0] and 0 <= j < updated.shape[1] and i != j:
        updated[i, j] += float(severity_minutes)
    return updated


def _local_eta_propagation(state: RouteState, severity: float) -> None:
    """Add the predicted delay to all downstream ETAs of the affected route."""
    for n in list(state.arrival_times.keys()):
        state.arrival_times[n] = state.arrival_times[n] + severity * 0.5


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
@dataclass
class ScenarioOutput:
    scenario: str
    instance_id: str
    seed: int
    solver_calls: int
    final_result: SolverResult
    per_event_log: pd.DataFrame
    runtime_total: float = 0.0


def run_static(instance, events, cfg) -> ScenarioOutput:
    travel = instance.travel_time.copy()
    res = solve(
        instance, travel,
        time_limit_seconds=cfg["solver"]["time_limit_seconds"],
        first_solution=cfg["solver"]["first_solution_strategy"],
        metaheuristic=cfg["solver"]["local_search_metaheuristic"],
    )
    log = pd.DataFrame({"event_id": [e.event_id for e in events],
                        "trigger": [False] * len(events),
                        "solver_called": [False] * len(events)})
    return ScenarioOutput("static", instance.instance_id, 0, 1, res, log, res.runtime_seconds)


def run_continuous(instance, events, extractor, cfg) -> ScenarioOutput:
    travel = instance.travel_time.copy()
    res = solve(
        instance, travel,
        time_limit_seconds=cfg["solver"]["time_limit_seconds"],
        first_solution=cfg["solver"]["first_solution_strategy"],
        metaheuristic=cfg["solver"]["local_search_metaheuristic"],
    )
    runtime_total = res.runtime_seconds
    rows = []
    calls = 1
    for ev in events:
        z = extractor.extract(ev)
        travel = _apply_event_to_travel(travel, z.affected_id, z.severity_minutes)
        res = solve(
            instance, travel,
            time_limit_seconds=cfg["solver"]["time_limit_seconds"],
            first_solution=cfg["solver"]["first_solution_strategy"],
            metaheuristic=cfg["solver"]["local_search_metaheuristic"],
        )
        calls += 1
        runtime_total += res.runtime_seconds
        rows.append({"event_id": ev.event_id, "trigger": True, "solver_called": True,
                     "runtime": res.runtime_seconds})
    log = pd.DataFrame(rows)
    return ScenarioOutput("continuous", instance.instance_id, 0, calls, res, log, runtime_total)


def run_aet_rag(instance, events, extractor, cfg, seed: int) -> ScenarioOutput:
    aet_cfg = AETConfig(
        theta_0=cfg["aet"]["theta_0"],
        theta_min=cfg["aet"]["theta_min"],
        gamma=cfg["aet"]["gamma"],
        w_urgency=cfg["aet"]["weights"]["urgency"],
        w_spatial=cfg["aet"]["weights"]["spatial"],
        w_slack=cfg["aet"]["weights"]["slack"],
        d_max=cfg["aet"]["d_max_minutes"],
        epsilon=cfg["aet"]["epsilon"],
        safety_urgency=float(cfg["aet"].get("safety_urgency", 0.85)),
        safety_slack=float(cfg["aet"].get("safety_slack", 0.95)),
        confidence_min_safety=float(cfg["aet"].get("confidence_min_safety", 0.60)),
        confidence_scaling=bool(cfg["aet"].get("confidence_scaling", True)),
    )
    ctrl = AETController(aet_cfg)
    travel = instance.travel_time.copy()
    res = solve(
        instance, travel,
        time_limit_seconds=cfg["solver"]["time_limit_seconds"],
        first_solution=cfg["solver"]["first_solution_strategy"],
        metaheuristic=cfg["solver"]["local_search_metaheuristic"],
    )
    state = _build_route_state(instance, res)
    runtime_total = res.runtime_seconds
    calls = 1
    rows = []
    for ev in events:
        z = extractor.extract(ev)
        log = ctrl.evaluate(z, state)
        solver_called = False
        if log.trigger:
            travel = _apply_event_to_travel(travel, z.affected_id, z.severity_minutes)
            res = solve(
                instance, travel,
                time_limit_seconds=cfg["solver"]["time_limit_seconds"],
                first_solution=cfg["solver"]["first_solution_strategy"],
                metaheuristic=cfg["solver"]["local_search_metaheuristic"],
            )
            calls += 1
            runtime_total += res.runtime_seconds
            state = _build_route_state(instance, res)
            ctrl.mark_solver_run(z.timestamp)
            solver_called = True
        else:
            _local_eta_propagation(state, z.severity_minutes)
        rows.append({
            "event_id": ev.event_id, "timestamp": z.timestamp,
            "U": log.U, "S": log.S, "R": log.R,
            "D": log.D, "theta": log.theta,
            "confidence": log.confidence,
            "trigger": log.trigger, "solver_called": solver_called,
            "reason": log.reason,
            "severity_minutes": z.severity_minutes,
            "delay_probability": z.delay_probability,
        })
    df = pd.DataFrame(rows)
    return ScenarioOutput("aet_rag", instance.instance_id, seed, calls, res, df, runtime_total)


# ---------------------------------------------------------------------------
# Lightweight baselines (no LLM extraction needed)
# ---------------------------------------------------------------------------
import re  # noqa: E402

# Heuristic vocabulary for keyword-triggered re-optimization. Words are
# intentionally broad (English + a few common synonyms) and match the
# severe-incident lexicon used by SVRPBench-style event generators.
_KEYWORD_PATTERN = re.compile(
    r"\b(accident|crash|collision|closed|closure|"
    r"blocked|blockade|police|impassable|severe|major|"
    r"emergency|breakdown|flood|fire|explosion)\b",
    re.IGNORECASE,
)


def run_keyword(instance, events, cfg) -> ScenarioOutput:
    """Keyword-trigger baseline. Re-optimizes whenever the raw textual event
    matches a small severe-incident lexicon. No LLM, no semantic scoring."""
    travel = instance.travel_time.copy()
    res = solve(
        instance, travel,
        time_limit_seconds=cfg["solver"]["time_limit_seconds"],
        first_solution=cfg["solver"]["first_solution_strategy"],
        metaheuristic=cfg["solver"]["local_search_metaheuristic"],
    )
    runtime_total = res.runtime_seconds
    rows = []
    calls = 1
    for ev in events:
        trig = bool(_KEYWORD_PATTERN.search(getattr(ev, "message", "") or ""))
        solver_called = False
        if trig:
            # Use the ground-truth severity to perturb travel, mirroring how
            # the LLM-based scenarios consume the parsed delay.
            sev = float(getattr(ev, "true_delay", 0.0))
            aff = tuple(getattr(ev, "affected_id", (0, 0)))
            travel = _apply_event_to_travel(travel, aff, sev)
            res = solve(
                instance, travel,
                time_limit_seconds=cfg["solver"]["time_limit_seconds"],
                first_solution=cfg["solver"]["first_solution_strategy"],
                metaheuristic=cfg["solver"]["local_search_metaheuristic"],
            )
            calls += 1
            runtime_total += res.runtime_seconds
            solver_called = True
        rows.append({"event_id": ev.event_id, "trigger": trig,
                     "solver_called": solver_called})
    log = pd.DataFrame(rows)
    return ScenarioOutput("keyword", instance.instance_id, 0, calls, res, log, runtime_total)


def run_periodic(instance, events, cfg) -> ScenarioOutput:
    """Fixed-period baseline. Re-optimizes every `periodic_every_n` events,
    independent of textual content. Represents a naive time-triggered policy."""
    every = int(cfg.get("baselines", {}).get("periodic_every_n", 10))
    every = max(1, every)
    travel = instance.travel_time.copy()
    res = solve(
        instance, travel,
        time_limit_seconds=cfg["solver"]["time_limit_seconds"],
        first_solution=cfg["solver"]["first_solution_strategy"],
        metaheuristic=cfg["solver"]["local_search_metaheuristic"],
    )
    runtime_total = res.runtime_seconds
    rows = []
    calls = 1
    for idx, ev in enumerate(events):
        trig = ((idx + 1) % every == 0)
        solver_called = False
        if trig:
            sev = float(getattr(ev, "true_delay", 0.0))
            aff = tuple(getattr(ev, "affected_id", (0, 0)))
            travel = _apply_event_to_travel(travel, aff, sev)
            res = solve(
                instance, travel,
                time_limit_seconds=cfg["solver"]["time_limit_seconds"],
                first_solution=cfg["solver"]["first_solution_strategy"],
                metaheuristic=cfg["solver"]["local_search_metaheuristic"],
            )
            calls += 1
            runtime_total += res.runtime_seconds
            solver_called = True
        rows.append({"event_id": ev.event_id, "trigger": trig,
                     "solver_called": solver_called})
    log = pd.DataFrame(rows)
    return ScenarioOutput("periodic", instance.instance_id, 0, calls, res, log, runtime_total)
