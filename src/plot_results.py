"""Generate the figures referenced in paper/root.tex.

All figures are saved as PDF under outputs/figures/ and a copy under
../paper/imgs/ for easy inclusion via \\includegraphics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, name: str, out_dirs):
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, bbox_inches="tight")
    plt.close(fig)


def fig_solver_calls(summary: pd.DataFrame, dirs):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.barplot(data=summary, x="scenario", y="solver_calls",
                order=["static", "continuous", "aet_rag"], ax=ax, errorbar="sd")
    ax.set_ylabel("OR-Tools invocations per instance")
    ax.set_xlabel("")
    _save(fig, "solver_calls_by_scenario.pdf", dirs)


def fig_tw_violations(summary: pd.DataFrame, dirs):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.boxplot(data=summary, x="scenario", y="total_lateness",
                order=["static", "continuous", "aet_rag"], ax=ax)
    ax.set_ylabel("Total time-window lateness (min)")
    ax.set_xlabel("")
    _save(fig, "time_window_violations.pdf", dirs)


def fig_threshold_trace(raw_dir: Path, dirs):
    candidates = sorted(raw_dir.glob("*_aet_rag_log.csv"))
    if not candidates:
        return
    df = pd.read_csv(candidates[0])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(df["timestamp"], df["D"], label="$D_t$", color="C0")
    ax.plot(df["timestamp"], df["theta"], label=r"$\theta_t$", color="C3", linestyle="--")
    triggers = df[df["trigger"]]
    for t in triggers["timestamp"]:
        ax.axvline(t, color="gray", alpha=0.3, linewidth=0.7)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("score")
    ax.legend()
    _save(fig, "threshold_trace.pdf", dirs)


def fig_tradeoff(summary: pd.DataFrame, dirs):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for sc, grp in summary.groupby("scenario"):
        ax.scatter(grp["solver_calls"], grp["served_ratio"], label=sc, alpha=0.7)
    ax.set_xlabel("Solver calls per instance")
    ax.set_ylabel("Served-customer ratio")
    ax.legend()
    _save(fig, "tradeoff_feasibility_cost.pdf", dirs)


def fig_nervousness(summary: pd.DataFrame, dirs):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.barplot(data=summary, x="scenario", y="route_nervousness",
                order=["static", "continuous", "aet_rag"], ax=ax, errorbar="sd")
    ax.set_ylabel("Route nervousness (proxy)")
    ax.set_xlabel("")
    _save(fig, "route_nervousness.pdf", dirs)


def fig_latency(summary: pd.DataFrame, dirs):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.boxplot(data=summary, x="scenario", y="runtime_total_seconds",
                order=["static", "continuous", "aet_rag"], ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Total solver runtime per instance (s, log)")
    ax.set_xlabel("")
    _save(fig, "latency_distribution.pdf", dirs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/metrics/summary.csv")
    parser.add_argument("--raw", default="outputs/raw")
    parser.add_argument("--figures", default="outputs/figures")
    parser.add_argument("--paper-imgs", default="../paper/imgs")
    args = parser.parse_args()

    summary = pd.read_csv(args.metrics)
    out_dirs = [Path(args.figures), Path(args.paper_imgs)]
    fig_solver_calls(summary, out_dirs)
    fig_tw_violations(summary, out_dirs)
    fig_threshold_trace(Path(args.raw), out_dirs)
    fig_tradeoff(summary, out_dirs)
    fig_nervousness(summary, out_dirs)
    fig_latency(summary, out_dirs)
    print("Figures written to:", [str(d) for d in out_dirs])


if __name__ == "__main__":
    main()
