# Experiments for AET-RAG

Reproducible code for the experimental section of *Adaptive Event-Triggered
Synchronization in Logistics Digital Twins: A RAG-Driven Framework for
Unstructured Data*.

## Layout

```
experiments/
  requirements.txt
  config.yaml
  src/
    download_data.py     # fetch SVRPBench (HF), Solomon (SINTEF), EURO-NeurIPS 2022 (git), synthetic no-op
    data_loader.py       # Instance dataclass + multi-source loader (synthetic/solomon/svrpbench/euro_neurips_2022)
    event_generator.py   # numerical disruption -> driver-style textual events
    rag_extractor.py     # Oracle + optional OpenAI LLM extractor (pydantic-validated)
    aet_controller.py    # U/S/R, D_t, theta_t, trigger decision
    ortools_solver.py    # CVRPTW wrapper around Google OR-Tools
    baselines.py         # static / continuous / AET-RAG scenarios
    metrics.py           # per-scenario + aggregate metrics
    run_experiments.py   # CLI orchestrator
    plot_results.py      # figures referenced in paper/root.tex
  data/                  # SVRPBench cache (populated by download_data)
  outputs/
    raw/                 # per-instance event logs, per-scenario per-event logs
    metrics/             # summary.csv, aggregate.csv, solver_call_reduction.csv
    figures/             # PDFs mirrored into ../paper/imgs/
```

## Quickstart

```bash
cd experiments
pip install -r requirements.txt

# Download all four dataset families (SVRPBench HF, Solomon SINTEF zip,
# EURO-NeurIPS 2022 git clone, synthetic no-op). Use `--only NAME` to fetch
# just one source, e.g. `--only svrpbench,solomon`.
python -m src.download_data --config config.yaml

# Smoke run: 2 instances per source, single seed, 10 events, 5s solver budget.
python -m src.run_experiments --config config_smoke.yaml --scenario all

# Full run (config.yaml drives per-source `n_instances` and seeds).
python -m src.run_experiments --config config.yaml --scenario all

# Build figures referenced in paper/root.tex.
python -m src.plot_results
```

## Datasets

The pipeline groups results by `dataset_source`. The four families are:

| Source              | Provider                                | Notes                                                      |
| ------------------- | --------------------------------------- | ---------------------------------------------------------- |
| `svrpbench`         | HuggingFace `MBZUAI/svrp-bench`         | TWCVRP subsets, log-normal + Poisson stochastics            |
| `solomon`           | SINTEF Solomon-100 zip                  | R1/R2/RC1/RC2 families + log-normal contextual overlay      |
| `euro_neurips_2022` | `ortec/euro-neurips-vrp-2022-quickstart`| Asymmetric duration matrices, 200--1000 customers (capped)  |
| `synthetic`         | in-process generator                    | Controlled ground-truth disruptions for ablations           |

Each adapter normalizes coords, demands, time windows and travel-time
matrix into minutes and tags `Instance.source`. The Solomon and EURO-NeurIPS
families do not ship per-instance event streams: a mean-preserving log-normal
overlay (Serrano et al., 2024) plus a Poisson severe-incident process are
synthesized at load time to produce the ground-truth `stochastic_events`
DataFrame consumed by the RAG event generator.

## Reproducibility notes

- All randomness flows from `experiment.seed` in `config.yaml`.
- The same event stream is fed to **all three** scenarios (static, continuous,
  AET-RAG) so differences can be attributed to synchronization policy only.
- `rag.mode: oracle` runs without any external API: ground-truth severity is
  perturbed with controlled Gaussian noise to isolate the controller from
  LLM-side variability. Switch to `llm` (and export `OPENAI_API_KEY`) to obtain
  the RAG extraction-quality row of Table III.
- Hyperparameters (`theta_0`, `theta_min`, `gamma`, weights) are exposed in
  `config.yaml`; the sensitivity sweep is performed by editing this file or
  passing alternative configs.

## Outputs that feed the paper

| Paper artifact (root.tex)          | Produced by                                |
| ---------------------------------- | ------------------------------------------ |
| Table I `tab:main_results`         | `outputs/metrics/aggregate.csv`            |
| Table II `tab:ablation`            | re-runs with edited `aet.weights`/`theta_t`|
| Table III `tab:rag_quality`        | re-runs with `rag.mode` in {oracle, llm}   |
| Fig. `fig:solver_calls`            | `plot_results.fig_solver_calls`            |
| Fig. `fig:threshold_trace`         | `plot_results.fig_threshold_trace`         |
| Fig. `fig:tw_violations`           | `plot_results.fig_tw_violations`           |
| Fig. `fig:tradeoff`                | `plot_results.fig_tradeoff`                |
| Fig. `fig:sensitivity`             | sweep + custom plotting                    |
