"""Hyperparameter sensitivity analysis through controller replay.

We re-run the AET trigger logic on the per-event (U, S, R, confidence)
streams already persisted in the AET-RAG logs, sweeping the convex weights
(w_u, w_s, w_l) and the initial threshold theta_0. For every configuration
we report the mean number of triggers per instance, the high-severity
recall (fraction of severe events that fire a trigger), and the
false-positive rate on mild events. The cost is negligible because no LLM
nor OR-Tools call is performed.

Usage:
    python -m src.sensitivity --logs outputs/raw --events outputs/raw \
                              --out outputs/metrics/sensitivity.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from .aet_controller import AETConfig


def _replay(events: List[dict], log: List[dict], cfg: AETConfig) -> Tuple[int, int, int, int, int]:
    """Replay the controller decision on a stream with fixed U,S,R,c."""
    t_last = 0.0
    triggers = 0
    high_seen = high_trig = mild_seen = mild_fp = 0
    by_eid = {r["event_id"]: r for r in log}
    for ev in events:
        eid = ev["event_id"]
        r = by_eid.get(eid)
        if r is None:
            continue
        U = float(r["U"]); S = float(r["S"]); R = float(r["R"])
        c = float(r.get("confidence", 1.0) or 1.0)
        t = float(r["timestamp"])
        D_raw = cfg.w_urgency * U + cfg.w_spatial * S + cfg.w_slack * R
        D = c * D_raw if cfg.confidence_scaling else D_raw
        theta = cfg.theta_min + (cfg.theta_0 - cfg.theta_min) * math.exp(-cfg.gamma * max(0.0, t - t_last))
        trig = D >= theta
        if not trig and c >= cfg.confidence_min_safety:
            if U >= cfg.safety_urgency or R >= cfg.safety_slack:
                trig = True
        if trig:
            triggers += 1
            t_last = t
        is_high = (ev.get("severity_class") == "high")
        if is_high:
            high_seen += 1
            high_trig += int(trig)
        else:
            mild_seen += 1
            mild_fp += int(trig)
    return triggers, high_seen, high_trig, mild_seen, mild_fp


def _dataset_of(instance_id: str) -> str:
    if instance_id.startswith("enrips"):
        return "euro"
    if instance_id.startswith("sol"):
        return "solomon"
    if instance_id.startswith("svrp"):
        return "svrp"
    if instance_id.startswith("syn"):
        return "synthetic"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="outputs/raw")
    ap.add_argument("--events", default="outputs/raw")
    ap.add_argument("--out", default="outputs/metrics/sensitivity.csv")
    args = ap.parse_args()

    pairs: List[Tuple[str, str, str]] = []
    for ev_path in sorted(glob.glob(os.path.join(args.events, "*_events.csv"))):
        base = os.path.basename(ev_path)[: -len("_events.csv")]
        log_path = os.path.join(args.logs, f"{base}_aet_rag_log.csv")
        if os.path.exists(log_path):
            pairs.append((base, ev_path, log_path))

    # Sweep grid. Always renormalize the weights to sum to 1.
    grid: List[Dict[str, float]] = []
    for wu in (0.30, 0.40, 0.50, 0.60, 0.70):
        # split the remaining mass equally between spatial and slack as a
        # simple, interpretable two-parameter family
        ws = (1.0 - wu) / 2.0
        wl = (1.0 - wu) / 2.0
        grid.append({"w_urgency": wu, "w_spatial": ws, "w_slack": wl,
                     "theta_0": 0.70})
    for th0 in (0.55, 0.65, 0.75, 0.85):
        grid.append({"w_urgency": 0.50, "w_spatial": 0.25, "w_slack": 0.25,
                     "theta_0": th0})

    rows = []
    for params in grid:
        cfg = AETConfig(
            theta_0=params["theta_0"],
            w_urgency=params["w_urgency"],
            w_spatial=params["w_spatial"],
            w_slack=params["w_slack"],
        )
        by_dataset: Dict[str, List[Tuple[int, int, int, int, int]]] = {}
        all_stats: List[Tuple[int, int, int, int, int]] = []
        for base, ev_path, log_path in pairs:
            events = list(csv.DictReader(open(ev_path)))
            log = list(csv.DictReader(open(log_path)))
            s = _replay(events, log, cfg)
            all_stats.append(s)
            by_dataset.setdefault(_dataset_of(base), []).append(s)
        # Aggregate
        def _agg(stats):
            n = len(stats)
            if n == 0:
                return (0.0, 0.0, 0.0)
            calls = mean(x[0] for x in stats)
            rec = mean((x[2] / x[1]) if x[1] else 1.0 for x in stats)
            fpr = mean((x[4] / x[3]) if x[3] else 0.0 for x in stats)
            return (calls, rec, fpr)
        row = {**params}
        for d, stats in by_dataset.items():
            c, r, f = _agg(stats)
            row[f"calls_{d}"] = round(c, 2)
            row[f"recall_{d}"] = round(r, 3)
            row[f"fpr_{d}"] = round(f, 3)
        c, r, f = _agg(all_stats)
        row["calls_all"] = round(c, 2)
        row["recall_all"] = round(r, 3)
        row["fpr_all"] = round(f, 3)
        rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"\nWrote {len(rows)} configurations to {args.out}")


if __name__ == "__main__":
    main()
