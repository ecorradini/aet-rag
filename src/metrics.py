"""Aggregate metrics across scenarios."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def route_nervousness(per_event_log: pd.DataFrame) -> float:
    """Proxy for nervousness: fraction of consecutive solver invocations."""
    if "solver_called" not in per_event_log or len(per_event_log) < 2:
        return 0.0
    return float(per_event_log["solver_called"].astype(int).diff().abs().mean())


def summarize(outputs: List["ScenarioOutput"], dataset_source: str = "unknown") -> pd.DataFrame:
    rows = []
    for o in outputs:
        r = o.final_result
        rows.append({
            "dataset_source": dataset_source,
            "scenario": o.scenario,
            "instance_id": o.instance_id,
            "seed": o.seed,
            "solver_calls": o.solver_calls,
            "total_travel": r.total_travel,
            "total_lateness": r.total_lateness,
            "unserved": r.unserved,
            "served_ratio": 1 - r.unserved / max(1, r.unserved + len(r.arrival_times) - 1),
            "objective": r.objective,
            "runtime_total_seconds": o.runtime_total,
            "route_nervousness": route_nervousness(o.per_event_log),
            "feasible": r.feasible,
        })
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean + std grouped by (dataset_source, scenario) when available."""
    numeric = summary.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    group_cols = [c for c in ("dataset_source", "scenario") if c in summary.columns]
    if not group_cols:
        group_cols = ["scenario"]
    numeric = [c for c in numeric if c not in group_cols]
    return summary.groupby(group_cols)[numeric].agg(["mean", "std", "min", "max"])


def solver_call_reduction(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset reduction in solver_calls of each scenario vs `continuous`.

    Returns a DataFrame indexed by (dataset_source, scenario) with one column
    `reduction_vs_continuous` in [-inf, 1]. Higher is better.
    """
    if "dataset_source" not in summary.columns:
        summary = summary.assign(dataset_source="all")
    rows = []
    for src, df in summary.groupby("dataset_source"):
        base = df.loc[df["scenario"] == "continuous", "solver_calls"].mean()
        if not base or base <= 0:
            continue
        means = df.groupby("scenario")["solver_calls"].mean()
        for scen, val in means.items():
            rows.append({
                "dataset_source": src,
                "scenario": scen,
                "solver_calls_mean": float(val),
                "continuous_baseline": float(base),
                "reduction_vs_continuous": float((base - val) / base),
            })
    return pd.DataFrame(rows)
