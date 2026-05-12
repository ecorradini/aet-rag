"""Load CVRPTW instances from multiple stochastic-routing benchmarks.

Four dataset families are supported, each producing the same `Instance`
representation (coords, demands, time_windows, service_times, travel_time,
stochastic_events) tagged with `instance.source` for downstream grouping:

  * synthetic         -- in-process generator (`_make_synthetic`)
  * solomon           -- Solomon 1987 100-customer .txt files (via vrplib)
  * svrpbench         -- HuggingFace `MBZUAI/svrp-bench` JSONL cache
  * euro_neurips_2022 -- ORTEC 2022 ASYM-VRPTW instances (uses repo `tools.py`)

The driver is `load_instances(config)`, which iterates over
`config['dataset']['sources']`. Legacy single-source configs are honoured.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instance dataclass
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    instance_id: str
    depot: int
    n_customers: int
    coords: np.ndarray                 # shape (n+1, 2)
    demands: np.ndarray                # shape (n+1,)
    time_windows: np.ndarray           # shape (n+1, 2)
    service_times: np.ndarray          # shape (n+1,)
    travel_time: np.ndarray            # shape (n+1, n+1) baseline minutes
    vehicle_capacity: int
    num_vehicles: int
    stochastic_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = "unknown"            # dataset family for groupby in metrics


# ---------------------------------------------------------------------------
# Common: synthesize a per-instance ground-truth event stream
# ---------------------------------------------------------------------------
def _synth_events_for(
    instance_id: str,
    coords: np.ndarray,
    time_windows: np.ndarray,
    n_events: int,
    seed: int,
    sigma: float = 0.20,
    severe_fraction: float = 0.10,
    severe_range: tuple = (30.0, 120.0),
) -> pd.DataFrame:
    """Build a stochastic-event stream compatible with event_generator.

    Mean-preserving log-normal travel-time noise (Serrano et al. 2024) plus a
    Poisson-like severe-incident overlay aligned with SVRPBench semantics.
    """
    rng = np.random.default_rng(seed)
    n = coords.shape[0] - 1
    horizon = float(time_windows[:, 1].max()) if len(time_windows) else 480.0
    ev_times = np.sort(rng.uniform(0, horizon, size=n_events))

    edges_i = rng.integers(0, n + 1, size=n_events)
    edges_j = rng.integers(0, n + 1, size=n_events)
    same = edges_i == edges_j
    edges_j[same] = (edges_j[same] + 1) % (n + 1)

    # Mean-preserving log-normal: T~ = d * exp(eps), eps~N(-sigma^2/2, sigma^2).
    # Baseline edge "minutes" magnitude: derive from coord scale to keep delays
    # in a sensible range without needing the actual matrix here.
    span = float(np.linalg.norm(coords.max(0) - coords.min(0))) or 1.0
    base = max(1.0, span / 50.0)  # crude per-edge baseline in minutes
    eps = rng.normal(-(sigma ** 2) / 2, sigma, size=n_events)
    severities = base * (np.exp(eps) - 1.0)
    severities = np.abs(severities) + rng.uniform(0.5, 3.0, size=n_events)

    incidents = rng.random(size=n_events) < severe_fraction
    if incidents.any():
        severities[incidents] = rng.uniform(*severe_range, size=incidents.sum())

    classes = np.where(
        severities < 5, "mild",
        np.where(severities < 20, "medium", "severe"),
    )
    probs = np.clip(
        rng.beta(2, 2, size=n_events) + (severities / 60.0).clip(0, 0.5),
        0, 1,
    )
    return pd.DataFrame({
        "event_id": [f"{instance_id}_evt{i:04d}" for i in range(n_events)],
        "timestamp": ev_times,
        "affected_type": "edge",
        "affected_i": edges_i,
        "affected_j": edges_j,
        "true_delay": severities,
        "true_probability": probs,
        "severity_class": classes,
    })


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------
def _make_synthetic(instance_id: str, n: int, seed: int) -> Instance:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n + 1, 2))
    coords[0] = [50.0, 50.0]
    demands = np.concatenate([[0], rng.integers(1, 10, size=n)])

    horizon = 480
    a = rng.uniform(0, horizon - 120, size=n)
    width = rng.choice([60, 90, 120, 180], size=n)
    tw = np.stack([a, a + width], axis=1)
    tw = np.concatenate([[[0.0, horizon]], tw])

    service = np.concatenate([[0.0], rng.uniform(3, 10, size=n)])

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    speed_kmh = 30.0
    travel = dist / speed_kmh * 60.0
    travel = travel + rng.normal(0, 0.5, size=travel.shape).clip(-1, 1)
    np.fill_diagonal(travel, 0.0)
    travel = np.maximum(travel, 0.0)

    n_events = max(10, n // 2)
    ev_times = np.sort(rng.uniform(0, horizon, size=n_events))
    edges_i = rng.integers(0, n + 1, size=n_events)
    edges_j = rng.integers(0, n + 1, size=n_events)
    same = edges_i == edges_j
    edges_j[same] = (edges_j[same] + 1) % (n + 1)
    severities = rng.lognormal(mean=1.5, sigma=1.0, size=n_events)
    incidents = rng.random(size=n_events) < 0.10
    severities[incidents] += rng.exponential(scale=30.0, size=incidents.sum())
    classes = np.where(severities < 5, "mild",
                       np.where(severities < 20, "medium", "severe"))
    probs = np.clip(rng.beta(2, 2, size=n_events) + (severities / 60).clip(0, 0.5), 0, 1)
    events = pd.DataFrame({
        "event_id": [f"{instance_id}_evt{i:04d}" for i in range(n_events)],
        "timestamp": ev_times,
        "affected_type": "edge",
        "affected_i": edges_i,
        "affected_j": edges_j,
        "true_delay": severities,
        "true_probability": probs,
        "severity_class": classes,
    })

    num_vehicles = max(2, int(math.ceil(demands.sum() / 30)))
    return Instance(
        instance_id=instance_id, depot=0, n_customers=n,
        coords=coords, demands=demands, time_windows=tw, service_times=service,
        travel_time=travel, vehicle_capacity=30, num_vehicles=num_vehicles,
        stochastic_events=events, source="synthetic",
    )


# ---------------------------------------------------------------------------
# Solomon 1987 (.txt via vrplib)
# ---------------------------------------------------------------------------
def _from_solomon_file(path: Path, seed: int, overlay: dict, events_per_instance: int) -> Optional[Instance]:
    try:
        import vrplib                                              # type: ignore
    except ImportError:
        logger.warning("solomon: `vrplib` package not installed; skipping %s", path)
        return None
    try:
        data = vrplib.read_instance(str(path), instance_format="solomon")
    except Exception as exc:
        logger.debug("solomon: parse failed for %s: %s", path, exc)
        return None

    coords = np.asarray(data["node_coord"], dtype=float)
    demands = np.asarray(data["demand"], dtype=float)
    tw = np.asarray(data["time_window"], dtype=float)
    service = np.asarray(data["service_time"], dtype=float)
    travel = np.asarray(data["edge_weight"], dtype=float)  # symmetric, == distance
    np.fill_diagonal(travel, 0.0)

    capacity = int(data["capacity"])
    # Solomon doesn't fix m; heuristic upper bound.
    num_vehicles = int(data.get("vehicles") or max(5, int(math.ceil(demands.sum() / capacity))))

    instance_id = f"sol_{path.stem}"
    events = _synth_events_for(
        instance_id, coords, tw,
        n_events=events_per_instance,
        seed=seed,
        sigma=float(overlay.get("lognormal_sigma", 0.20)),
        severe_fraction=float(overlay.get("poisson_severe_fraction", 0.10)),
        severe_range=tuple(overlay.get("severe_minutes", [30, 120])),
    )
    return Instance(
        instance_id=instance_id, depot=0, n_customers=coords.shape[0] - 1,
        coords=coords, demands=demands, time_windows=tw, service_times=service,
        travel_time=travel, vehicle_capacity=capacity, num_vehicles=num_vehicles,
        stochastic_events=events, source="solomon",
    )


def _load_solomon(spec: dict, events_per_instance: int, seed: int) -> List[Instance]:
    cache_dir = Path(spec["cache_dir"])
    files = sorted(cache_dir.rglob("*.txt"))
    families = spec.get("families")
    if families:
        families = tuple(f.upper() for f in families)
        files = [p for p in files if p.stem.upper().startswith(families)]
    n = int(spec.get("n_instances", 10))
    overlay = spec.get("stochastic_overlay", {})
    rng = np.random.default_rng(seed)
    out: List[Instance] = []
    for p in files:
        if len(out) >= n:
            break
        inst = _from_solomon_file(p, int(rng.integers(0, 10**9)), overlay, events_per_instance)
        if inst is not None:
            out.append(inst)
    logger.info("solomon: loaded %d instances (filter=%s)", len(out), families)
    return out


# ---------------------------------------------------------------------------
# SVRPBench HuggingFace JSONL (canonical MBZUAI/svrp-bench schema)
# ---------------------------------------------------------------------------
def _from_svrpbench_hf(instance_id: str, row: dict, seed: int, events_per_instance: int) -> Optional[Instance]:
    """Canonical SVRPBench schema.

    Required keys: `locations` (N+1,2), `demands` (N+1,), `num_vehicles`,
    `vehicle_capacities` (m,). Optional: `time_windows`, `time_matrix`,
    `service_times`, `appear_times` (minutes from midnight, 0..1440).
    When `time_windows` is missing (canonical SVRPBench CVRP subsets), they
    are synthesized from `appear_times` with a fixed 180-minute width, which
    matches the SVRPBench paper's residential TW convention.
    """
    locs = row.get("locations") or row.get("coordinates") or row.get("coords")
    if locs is None:
        return None
    coords = np.asarray(locs, dtype=float)
    n_plus = coords.shape[0]
    rng = np.random.default_rng(seed)

    demands = np.asarray(row.get("demands"), dtype=float)
    service = np.asarray(row.get("service_times") or np.zeros(n_plus), dtype=float)

    tw = row.get("time_windows")
    if tw is None:
        # Synthesize TW from `appear_times`; default to a daily horizon otherwise.
        appear = row.get("appear_times")
        if appear is not None and len(appear) == n_plus:
            a = np.asarray(appear, dtype=float)
            widths = rng.choice([90.0, 120.0, 180.0, 240.0], size=n_plus)
            tw = np.stack([a, np.minimum(a + widths, 1440.0)], axis=1)
            tw[0] = [0.0, 1440.0]
        else:
            horizon = 480.0
            a = rng.uniform(0, horizon - 120, size=n_plus)
            widths = rng.choice([60.0, 90.0, 120.0, 180.0], size=n_plus)
            tw = np.stack([a, a + widths], axis=1)
            tw[0] = [0.0, horizon]
    tw = np.asarray(tw, dtype=float)

    travel = row.get("time_matrix") or row.get("travel_time") or row.get("distance_matrix")
    if travel is None:
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        # Treat coords as km scale; 30 km/h -> minutes.
        scale = float(dist.max()) or 1.0
        if scale > 100:  # raw metres
            travel = dist / 1000.0 / 30.0 * 60.0
        else:
            travel = dist / 30.0 * 60.0
    travel = np.asarray(travel, dtype=float)
    np.fill_diagonal(travel, 0.0)

    caps = row.get("vehicle_capacities")
    if isinstance(caps, (list, tuple, np.ndarray)) and len(caps):
        capacity = int(np.asarray(caps).max())
        num_vehicles = int(row.get("num_vehicles") or len(caps))
    else:
        capacity = int(row.get("capacity") or row.get("vehicle_capacity") or 30)
        num_vehicles = int(row.get("num_vehicles") or 10)

    events = _synth_events_for(
        instance_id, coords, tw,
        n_events=events_per_instance, seed=seed,
        sigma=0.20, severe_fraction=0.10,
    )
    return Instance(
        instance_id=instance_id, depot=0, n_customers=n_plus - 1,
        coords=coords, demands=demands, time_windows=tw, service_times=service,
        travel_time=travel, vehicle_capacity=capacity, num_vehicles=num_vehicles,
        stochastic_events=events, source="svrpbench",
    )


def _load_svrpbench(spec: dict, events_per_instance: int, seed: int) -> List[Instance]:
    cache_dir = Path(spec["cache_dir"])
    jsonl = cache_dir / "instances.jsonl"
    if not jsonl.exists():
        logger.warning("svrpbench: %s missing — run `download_data --only svrpbench`.", jsonl)
        return []
    rows: List[dict] = []
    with jsonl.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n = int(spec.get("n_instances", 15))
    rng = np.random.default_rng(seed)
    out: List[Instance] = []
    for k, row in enumerate(rows):
        if len(out) >= n:
            break
        inst = _from_svrpbench_hf(f"svrp_{k:03d}", row,
                                  int(rng.integers(0, 10**9)),
                                  events_per_instance)
        if inst is not None:
            out.append(inst)
    logger.info("svrpbench: loaded %d/%d instances", len(out), len(rows))
    return out


# ---------------------------------------------------------------------------
# EURO-NeurIPS 2022 (ORTEC ASYM-VRPTW)
# ---------------------------------------------------------------------------
def _ortec_tools(cache_dir: Path):
    """Import the ORTEC quickstart `tools.py` lazily."""
    candidates = [cache_dir / "tools.py", cache_dir / "repo" / "tools.py"]
    for tools_py in candidates:
        if tools_py.exists():
            spec = importlib.util.spec_from_file_location("ortec_tools", tools_py)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["ortec_tools"] = mod
            spec.loader.exec_module(mod)
            return mod
    return None


def _from_euro_neurips_file(path: Path, tools_mod, seed: int,
                            events_per_instance: int,
                            max_customers: Optional[int] = None) -> Optional[Instance]:
    try:
        data = tools_mod.read_vrplib(str(path))
    except Exception as exc:
        logger.debug("euro_neurips_2022: parse failed for %s: %s", path, exc)
        return None
    coords  = np.asarray(data["coords"], dtype=float)
    demands = np.asarray(data["demands"], dtype=float)
    tw_sec  = np.asarray(data["time_windows"], dtype=float)
    svc_sec = np.asarray(data["service_times"], dtype=float)
    dur_sec = np.asarray(data["duration_matrix"], dtype=float)  # asymmetric
    capacity = int(data["capacity"])

    if max_customers is not None and coords.shape[0] - 1 > max_customers:
        # Take the first max_customers+1 nodes (depot + N).
        keep = max_customers + 1
        coords  = coords[:keep]
        demands = demands[:keep]
        tw_sec  = tw_sec[:keep]
        svc_sec = svc_sec[:keep]
        dur_sec = dur_sec[:keep, :keep]

    # Convert seconds -> minutes (paper's canonical unit).
    tw = tw_sec / 60.0
    service = svc_sec / 60.0
    travel = dur_sec / 60.0
    np.fill_diagonal(travel, 0.0)

    # ORTEC instances don't include a fleet size; use a feasible upper bound.
    num_vehicles = max(5, int(math.ceil(demands.sum() / capacity)))
    instance_id = f"enrips_{path.stem}"
    events = _synth_events_for(
        instance_id, coords, tw,
        n_events=events_per_instance, seed=seed,
        sigma=0.20, severe_fraction=0.10,
    )
    return Instance(
        instance_id=instance_id, depot=0, n_customers=coords.shape[0] - 1,
        coords=coords, demands=demands, time_windows=tw, service_times=service,
        travel_time=travel, vehicle_capacity=capacity, num_vehicles=num_vehicles,
        stochastic_events=events, source="euro_neurips_2022",
    )


def _load_euro_neurips(spec: dict, events_per_instance: int, seed: int) -> List[Instance]:
    cache_dir = Path(spec["cache_dir"])
    instances_dir = cache_dir / spec.get("instances_dir", "instances")
    if not instances_dir.exists():
        logger.warning("euro_neurips_2022: %s missing — run download_data first.",
                       instances_dir)
        return []
    tools_mod = _ortec_tools(cache_dir)
    if tools_mod is None:
        logger.warning("euro_neurips_2022: tools.py not found in %s.", cache_dir)
        return []
    files = sorted(instances_dir.glob("*.txt"))
    n = int(spec.get("n_instances", 5))
    max_n = spec.get("max_customers")
    rng = np.random.default_rng(seed)
    out: List[Instance] = []
    for p in files:
        if len(out) >= n:
            break
        inst = _from_euro_neurips_file(
            p, tools_mod, int(rng.integers(0, 10**9)),
            events_per_instance, max_n,
        )
        if inst is not None:
            out.append(inst)
    logger.info("euro_neurips_2022: loaded %d instances", len(out))
    return out


def _load_synthetic(spec: dict, exp_cfg: dict, seed: int) -> List[Instance]:
    sizes = spec.get("customer_sizes") or exp_cfg.get("customer_sizes", [100])
    n = int(spec.get("n_instances") or exp_cfg.get("n_instances", 10))
    rng = np.random.default_rng(seed)
    out: List[Instance] = []
    per_size = max(1, n // len(sizes))
    for size in sizes:
        for k in range(per_size):
            s = int(rng.integers(0, 10**9))
            out.append(_make_synthetic(f"syn_n{size}_{k:03d}", size, s))
    return out[:n]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
_LOADERS = {
    "synthetic":         _load_synthetic,
    "solomon":           _load_solomon,
    "svrpbench":         _load_svrpbench,
    "euro_neurips_2022": _load_euro_neurips,
}


def _legacy_sources(cfg: dict) -> List[dict]:
    ds = cfg.get("dataset", {})
    src = ds.get("source", "synthetic")
    mapping = {"synthetic": "synthetic",
               "svrpbench_hf": "svrpbench",
               "svrpbench_github": "euro_neurips_2022"}
    name = mapping.get(src, "synthetic")
    return [{
        "name": name, "enabled": True,
        "cache_dir": ds.get("cache_dir", f"data/{name}"),
        "n_instances": cfg.get("experiment", {}).get("n_instances", 10),
        "hf_repo": ds.get("hf_repo"),
    }]


def load_instances(config: dict, max_instances: Optional[int] = None) -> List[Instance]:
    exp = config["experiment"]
    seed = int(exp["seed"])
    events_per_instance = int(exp.get("events_per_instance", 100))
    sources = config.get("dataset", {}).get("sources") or _legacy_sources(config)
    out: List[Instance] = []
    for spec in sources:
        name = spec["name"]
        if not spec.get("enabled", True):
            logger.info("%s: disabled, skipping.", name)
            continue
        loader = _LOADERS.get(name)
        if loader is None:
            logger.warning("Unknown source `%s` in config, skipping.", name)
            continue
        try:
            if name == "synthetic":
                batch = loader(spec, exp, seed)
            elif name in ("solomon", "svrpbench", "euro_neurips_2022"):
                batch = loader(spec, events_per_instance, seed)
            else:
                batch = []
        except Exception as exc:
            logger.exception("Loader for %s failed: %s", name, exc)
            batch = []
        logger.info("source=%s -> %d instance(s)", name, len(batch))
        out.extend(batch)
    if not out:
        logger.warning("No instances loaded from any source; falling back to synthetic.")
        out = _load_synthetic({}, exp, seed)
    return out[: max_instances or len(out)]
