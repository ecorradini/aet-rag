"""Convert numerical stochastic delays into synthetic driver-style textual events.

In oracle mode this is the *ground truth* against which the RAG extractor is
later evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


MILD_TEMPLATES = [
    "slow traffic at the junction, about {d:.0f} minutes late",
    "stuck behind a garbage truck for a minute or two",
    "minor congestion, moving slowly but flowing",
    "slight delay at the crosswalk, nothing serious",
]
MEDIUM_TEMPLATES = [
    "heavier than usual traffic on this stretch, expect ~{d:.0f} min delay",
    "construction work narrowing the lanes, slow but moving",
    "double-parked van blocking half the lane",
    "school zone congestion, queues forming",
]
SEVERE_TEMPLATES = [
    "major accident, police closed all lanes",
    "road completely blocked by a landslide, rerouting required",
    "huge protest on Main Street, traffic at a standstill for {d:.0f} minutes",
    "fire trucks blocking the whole intersection, no way through",
]


@dataclass
class TextEvent:
    event_id: str
    timestamp: float
    affected_type: str
    affected_id: tuple
    true_delay: float
    true_probability: float
    severity_class: str
    message: str


def generate_for_instance(
    instance_id: str,
    events: pd.DataFrame,
    seed: int,
    events_per_instance: int,
) -> List[TextEvent]:
    rng = np.random.default_rng(seed)
    if events.empty:
        return []
    sample = events.sample(
        n=min(events_per_instance, len(events)),
        random_state=int(rng.integers(0, 10 ** 9)),
    ).sort_values("timestamp").reset_index(drop=True)

    out: List[TextEvent] = []
    for k, row in sample.iterrows():
        cls = row["severity_class"]
        if cls == "mild":
            tmpl = MILD_TEMPLATES
        elif cls == "medium":
            tmpl = MEDIUM_TEMPLATES
        else:
            tmpl = SEVERE_TEMPLATES
        msg = rng.choice(tmpl).format(d=row["true_delay"])
        out.append(
            TextEvent(
                event_id=f"{instance_id}_e{k:04d}",
                timestamp=float(row["timestamp"]),
                affected_type=row["affected_type"],
                affected_id=(int(row["affected_i"]), int(row["affected_j"])),
                true_delay=float(row["true_delay"]),
                true_probability=float(row["true_probability"]),
                severity_class=cls,
                message=msg,
            )
        )
    return out


def to_dataframe(events: List[TextEvent]) -> pd.DataFrame:
    return pd.DataFrame([e.__dict__ for e in events])
