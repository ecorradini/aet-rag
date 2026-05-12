"""Download all dataset families declared in `dataset.sources` of the config.

Supports four sources, each with its own fetcher:
  * svrpbench         -> HuggingFace `MBZUAI/svrp-bench` (canonical, public)
  * solomon           -> SINTEF ZIP (Solomon 1987 100-customer set)
  * euro_neurips_2022 -> git clone ortec/euro-neurips-vrp-2022-quickstart
  * synthetic         -> no-op (instances generated at load time)

Legacy single-source mode (`dataset.source == "synthetic"|"svrpbench_hf"|...`)
is preserved as a fallback for backward compatibility.

Usage:
    python -m src.download_data --config config.yaml
    python -m src.download_data --config config.yaml --only solomon,euro_neurips_2022
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-source fetchers
# ---------------------------------------------------------------------------
def fetch_svrpbench(spec: dict) -> bool:
    """Download SVRPBench (NeurIPS 2025) from the canonical HuggingFace repo."""
    cache_dir = Path(spec["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset                          # type: ignore
    except ImportError:
        logger.warning("svrpbench: `datasets` package not installed; skipping.")
        return False
    repo = spec.get("hf_repo", "MBZUAI/svrp-bench")
    split = spec.get("hf_split", "train")
    # MBZUAI/svrp-bench only ships a "test" split; auto-fallback.
    try:
        ds = load_dataset(repo, split=split, cache_dir=str(cache_dir))
    except Exception as exc:
        msg = str(exc)
        if "Unknown split" in msg or "split" in msg.lower():
            for alt in ("test", "train", "validation"):
                if alt == split:
                    continue
                try:
                    ds = load_dataset(repo, split=alt, cache_dir=str(cache_dir))
                    logger.info("svrpbench: split=%r unavailable, using %r.", split, alt)
                    split = alt
                    break
                except Exception:
                    ds = None
            if ds is None:
                logger.warning("svrpbench: no usable split found: %s", exc)
                return False
        else:
            logger.warning("svrpbench: load_dataset(%s) failed: %s", repo, exc)
            return False

    subsets = spec.get("subsets")
    if subsets:
        ds = ds.filter(lambda r: r.get("subset_name") in set(subsets))
        logger.info("svrpbench: filtered to %d rows in %s", len(ds), subsets)

    out = cache_dir / "instances.jsonl"
    with out.open("w") as f:
        for row in ds:
            f.write(json.dumps(_jsonable(row)) + "\n")
    logger.info("svrpbench: %d rows written to %s", len(ds), out)
    return True


def fetch_solomon(spec: dict) -> bool:
    """Download Solomon 1987 100-customer instance suite from SINTEF."""
    cache_dir = Path(spec["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = list(cache_dir.rglob("*.txt"))
    if existing:
        logger.info("solomon: cache already populated (%d files)", len(existing))
        return True
    url = spec.get("url",
        "https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip")
    logger.info("solomon: downloading %s", url)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (AET-RAG experiments)",
            "Accept": "*/*",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            blob = resp.read()
    except Exception as exc:                                       # pragma: no cover
        logger.warning("solomon: download failed: %s", exc)
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            z.extractall(cache_dir)
    except zipfile.BadZipFile as exc:
        logger.warning("solomon: ZIP parse failed: %s", exc)
        return False
    n = len(list(cache_dir.rglob("*.txt")))
    logger.info("solomon: extracted %d .txt instances to %s", n, cache_dir)
    return n > 0


def fetch_euro_neurips(spec: dict) -> bool:
    """Shallow-clone the ORTEC EURO-NeurIPS 2022 Dynamic VRPTW quickstart."""
    cache_dir = Path(spec["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = cache_dir / spec.get("instances_dir", "instances")
    if instances_dir.exists() and any(instances_dir.glob("*.txt")):
        logger.info("euro_neurips_2022: already cloned (%s)", cache_dir)
        return True
    repo = spec.get("git_repo",
        "https://github.com/ortec/euro-neurips-vrp-2022-quickstart.git")
    # Use a sub-directory to keep ORTEC's tools.py importable.
    target = cache_dir / "repo"
    if target.exists():
        # Repo exists but instances missing -> wipe.
        import shutil
        shutil.rmtree(target)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo, str(target)],
            check=True, timeout=300,
        )
    except Exception as exc:                                       # pragma: no cover
        logger.warning("euro_neurips_2022: git clone failed: %s", exc)
        return False
    # Symlink the canonical paths for the loader (use absolute paths so the
    # links resolve regardless of CWD).
    try:
        if not instances_dir.exists():
            os.symlink((target / "instances").resolve(), instances_dir)
        tools_link = cache_dir / "tools.py"
        if not tools_link.exists():
            os.symlink((target / "tools.py").resolve(), tools_link)
    except OSError:
        import shutil
        shutil.copytree(target / "instances", instances_dir, dirs_exist_ok=True)
        shutil.copy(target / "tools.py", cache_dir / "tools.py")
    # Verify resolution (broken symlinks happen if cache_dir was relative).
    if not any(instances_dir.glob("*.txt")):
        # Recreate as absolute symlink, or fall back to copy.
        try:
            if instances_dir.is_symlink():
                instances_dir.unlink()
            os.symlink((target / "instances").resolve(), instances_dir)
        except OSError:
            pass
    n = len(list(instances_dir.glob("*.txt")))
    logger.info("euro_neurips_2022: %d instance files available", n)
    return n > 0


def fetch_synthetic(spec: dict) -> bool:
    cache_dir = Path(spec.get("cache_dir", "data/synthetic"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("synthetic: nothing to download (instances generated at load time).")
    return True


FETCHERS = {
    "svrpbench":         fetch_svrpbench,
    "solomon":           fetch_solomon,
    "euro_neurips_2022": fetch_euro_neurips,
    "synthetic":         fetch_synthetic,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _jsonable(row):
    """Convert HF dataset row (which may contain numpy arrays) to plain JSON."""
    try:
        import numpy as np
        out = {}
        for k, v in row.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                out[k] = v.item()
            else:
                out[k] = v
        return out
    except Exception:
        return dict(row)


def _legacy_specs(cfg: dict) -> List[dict]:
    """Build a one-element source list from the legacy `dataset.source` key."""
    ds = cfg.get("dataset", {})
    src = ds.get("source", "synthetic")
    mapping = {
        "synthetic":          "synthetic",
        "svrpbench_hf":       "svrpbench",
        "svrpbench_github":   "euro_neurips_2022",  # historical alias; rarely used
    }
    name = mapping.get(src, "synthetic")
    return [{
        "name": name, "enabled": True,
        "hf_repo": ds.get("hf_repo", "MBZUAI/svrp-bench"),
        "hf_split": ds.get("hf_split", "train"),
        "cache_dir": ds.get("cache_dir", f"data/{name}"),
    }]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def download(cfg: dict, only: Iterable[str] | None = None) -> dict:
    sources = cfg.get("dataset", {}).get("sources") or _legacy_specs(cfg)
    results = {}
    only_set = set(only) if only else None
    for spec in sources:
        name = spec["name"]
        if not spec.get("enabled", True):
            logger.info("%s: disabled in config, skipping.", name)
            continue
        if only_set and name not in only_set:
            continue
        fetcher = FETCHERS.get(name)
        if fetcher is None:
            logger.warning("Unknown dataset source `%s`, skipping.", name)
            continue
        logger.info("=== Fetching %s ===", name)
        results[name] = fetcher(spec)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--only", default=None,
                        help="comma-separated subset of source names to fetch")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    results = download(cfg, only=only)
    print("\n=== Download summary ===")
    for name, ok in results.items():
        print(f"  {name:22s} {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
