"""End-to-end experiment runner.

Usage:
    python -m src.run_experiments --config config.yaml --scenario all
    python -m src.run_experiments --config config.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from tqdm import tqdm

from . import event_generator, metrics
from .baselines import (
    run_aet_rag,
    run_continuous,
    run_keyword,
    run_periodic,
    run_static,
)
from .data_loader import load_instances
from .rag_extractor import build_extractor


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy_logger in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def run(config_path: str, scenarios: List[str], smoke: bool,
        dataset_filter: List[str] | None = None) -> None:
    _configure_logging()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if smoke:
        # Legacy single-source smoke shortcut. With multi-source configs use
        # config_smoke.yaml instead, which respects per-source `n_instances`.
        if not cfg.get("dataset", {}).get("sources"):
            cfg["experiment"]["n_instances"] = 1
            cfg["experiment"]["customer_sizes"] = [20]
        cfg["experiment"]["events_per_instance"] = min(
            5, int(cfg["experiment"].get("events_per_instance", 5))
        )
        cfg["solver"]["time_limit_seconds"] = 5

    out_root = Path(cfg["experiment"]["output_dir"])
    raw_dir = out_root / "raw"
    metrics_dir = out_root / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(cfg)
    if dataset_filter:
        keep = set(dataset_filter)
        instances = [i for i in instances if getattr(i, "source", "unknown") in keep]
        if not instances:
            raise SystemExit(f"No instances matched --dataset filter {dataset_filter}")
    seed = cfg["experiment"]["seed"]
    summaries = []

    for inst in tqdm(instances, desc="instances"):
        events = event_generator.generate_for_instance(
            inst.instance_id,
            inst.stochastic_events,
            seed=seed,
            events_per_instance=cfg["experiment"]["events_per_instance"],
        )
        event_generator.to_dataframe(events).to_csv(
            raw_dir / f"{inst.instance_id}_events.csv", index=False
        )
        extractor = build_extractor(cfg, seed=seed)

        outs = []
        if "static" in scenarios:
            outs.append(run_static(inst, events, cfg))
        if "continuous" in scenarios:
            # Bound solver calls under smoke
            ev_for_continuous = events[:10] if smoke else events
            outs.append(run_continuous(inst, ev_for_continuous, extractor, cfg))
        if "periodic" in scenarios:
            outs.append(run_periodic(inst, events, cfg))
        if "keyword" in scenarios:
            outs.append(run_keyword(inst, events, cfg))
        if "aet_rag" in scenarios:
            outs.append(run_aet_rag(inst, events, extractor, cfg, seed=seed))

        for o in outs:
            o.per_event_log.to_csv(
                raw_dir / f"{inst.instance_id}_{o.scenario}_log.csv", index=False
            )
        summaries.append(metrics.summarize(outs, dataset_source=getattr(inst, "source", "unknown")))

    new_summary = pd.concat(summaries, ignore_index=True)

    summary_path = metrics_dir / "summary.csv"
    if (dataset_filter or scenarios != ["static", "continuous", "aet_rag"]) and summary_path.exists():
        prev = pd.read_csv(summary_path)
        # Drop the (dataset, scenario) rows we just recomputed, keep the rest.
        touched_ds = set(new_summary["dataset_source"].unique())
        touched_sc = set(new_summary["scenario"].unique())
        mask_drop = prev["dataset_source"].isin(touched_ds) & prev["scenario"].isin(touched_sc)
        summary = pd.concat([prev[~mask_drop], new_summary], ignore_index=True)
    else:
        summary = new_summary
    summary.to_csv(summary_path, index=False)
    agg = metrics.aggregate(summary)
    agg.to_csv(metrics_dir / "aggregate.csv")
    reduction = metrics.solver_call_reduction(summary)
    reduction.to_csv(metrics_dir / "solver_call_reduction.csv", index=False)

    print("\n=== AGGREGATE (mean by dataset x scenario) ===")
    group_cols = [c for c in ("dataset_source", "scenario") if c in summary.columns]
    print(summary.groupby(group_cols).mean(numeric_only=True))
    print("\nSolver-call reduction vs continuous (per dataset):")
    print(reduction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scenario", default="all",
                        help="comma-separated: static,continuous,aet_rag or 'all'")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run for sanity checking the pipeline")
    parser.add_argument("--dataset", default="",
                        help="comma-separated list of dataset sources to run; "
                             "empty = all. Use to rerun a subset and merge into "
                             "the existing metrics CSVs.")
    args = parser.parse_args()
    if args.scenario == "all":
        scenarios = ["static", "continuous", "periodic", "keyword", "aet_rag"]
    else:
        scenarios = [s.strip() for s in args.scenario.split(",") if s.strip()]
    ds_filter = [s.strip() for s in args.dataset.split(",") if s.strip()] or None
    run(args.config, scenarios, args.smoke, dataset_filter=ds_filter)


if __name__ == "__main__":
    main()
