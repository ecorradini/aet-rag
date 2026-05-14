"""RAG-based semantic extraction with an oracle fallback.

The oracle mode injects controlled Gaussian noise around the ground-truth
severity/probability and returns the affected entity unchanged. The LLM mode
calls a configured OpenAI-compatible chat model with a strict JSON schema.

The module also supports a persistent cache on disk: whenever a real LLM
call is performed, the resulting (event_id -> extracted tuple) mapping is
appended to a JSONL file so that subsequent runs can replay the extractions
without re-invoking the API. This is essential to keep wall-clock cost of
ablation runs (e.g. modifying the controller, replaying with different
weights, adding new baselines) bounded.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ExtractedTuple(BaseModel):
    event_id: str
    timestamp: float
    affected_type: str
    affected_id: Tuple[int, int]
    delay_probability: float = Field(ge=0.0, le=1.0)
    severity_minutes: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


SYSTEM_PROMPT = """You are a logistics dispatcher assistant. Given a driver
note, you must output STRICT JSON with the schema:
{"affected_type": "edge"|"node", "affected_id": [i, j],
 "delay_probability": float in [0,1],
 "severity_minutes": float >= 0,
 "confidence": float in [0,1],
 "rationale": short string}
Do not output anything outside the JSON object."""


@dataclass
class OracleExtractor:
    severity_std: float = 2.0
    probability_std: float = 0.05
    confidence_mean: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def extract(self, event) -> ExtractedTuple:
        sev = max(0.0, event.true_delay + self._rng.normal(0, self.severity_std))
        prob = float(np.clip(event.true_probability + self._rng.normal(0, self.probability_std), 0, 1))
        conf = float(np.clip(self._rng.normal(self.confidence_mean, 0.05), 0, 1))
        return ExtractedTuple(
            event_id=event.event_id,
            timestamp=event.timestamp,
            affected_type=event.affected_type,
            affected_id=event.affected_id,
            delay_probability=prob,
            severity_minutes=sev,
            confidence=conf,
            rationale="oracle",
        )


@dataclass
class LLMExtractor:
    """OpenAI-compatible RAG extractor. Falls back to oracle on failure.

    Optionally persists every successful extraction to a JSONL cache so that
    repeated runs (different controllers, different weights, ablations)
    can be replayed without re-invoking the API.
    """
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: Optional[str] = None
    oracle_fallback: Optional[OracleExtractor] = None
    cache_path: Optional[str] = None
    _cache: Dict[str, dict] = field(default_factory=dict, init=False)
    tokens_in: int = field(default=0, init=False)
    tokens_out: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    api_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._client = None
        if self.cache_path:
            self._load_cache()
        key = self.api_key or os.environ.get(self.api_key_env)
        # User-friendly fallback: if `api_key_env` was set to the key itself
        # (starts with "sk-"), treat it as the literal key.
        if not key and self.api_key_env.startswith("sk-"):
            key = self.api_key_env
        if not key:
            logger.warning(
                "OpenAI key not found (env var `%s` is empty); "
                "LLM extractor will use cache+oracle fallback.",
                self.api_key_env,
            )
            return
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=key)
            logger.info("OpenAI client ready (model=%s).", self.model)
        except Exception as exc:
            logger.warning("openai client unavailable (%s); fallback to oracle.", exc)

    def _load_cache(self) -> None:
        p = Path(self.cache_path)
        if not p.exists():
            return
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._cache[rec["event_id"]] = rec
                except Exception:
                    continue
        logger.info("Loaded %d cached LLM extractions from %s",
                    len(self._cache), self.cache_path)

    def _save_cache_entry(self, payload: dict) -> None:
        if not self.cache_path:
            return
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    def extract(self, event) -> ExtractedTuple:
        # 1. cache hit
        if event.event_id in self._cache:
            self.cache_hits += 1
            rec = self._cache[event.event_id]
            rec = dict(rec)
            rec["event_id"] = event.event_id
            rec["timestamp"] = event.timestamp
            if "affected_id" in rec and isinstance(rec["affected_id"], list):
                rec["affected_id"] = tuple(rec["affected_id"])
            try:
                return ExtractedTuple(**rec)
            except ValidationError:
                pass  # fall through to LLM/oracle
        # 2. no client -> oracle
        if self._client is None:
            assert self.oracle_fallback is not None
            return self.oracle_fallback.extract(event)
        # 3. fresh LLM call
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Driver note: {event.message}\n"
                            f"Suggested affected entity: {event.affected_type} "
                            f"{event.affected_id}"
                        ),
                    },
                ],
            )
            self.api_calls += 1
            try:
                usage = resp.usage
                self.tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
            except Exception:
                pass
            payload = json.loads(resp.choices[0].message.content or "{}")
            payload["event_id"] = event.event_id
            payload["timestamp"] = event.timestamp
            if "affected_id" not in payload:
                payload["affected_id"] = list(event.affected_id)
            extracted = ExtractedTuple(**payload)
            # persist
            self._save_cache_entry({
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "affected_type": extracted.affected_type,
                "affected_id": list(extracted.affected_id),
                "delay_probability": extracted.delay_probability,
                "severity_minutes": extracted.severity_minutes,
                "confidence": extracted.confidence,
                "rationale": extracted.rationale,
            })
            return extracted
        except (ValidationError, Exception) as exc:
            logger.debug("LLM extraction failed (%s); using oracle.", exc)
            assert self.oracle_fallback is not None
            return self.oracle_fallback.extract(event)


def build_extractor(config: dict, seed: int):
    rag_cfg = config["rag"]
    oracle = OracleExtractor(
        severity_std=rag_cfg.get("noise_severity_std", 2.0),
        probability_std=rag_cfg.get("noise_probability_std", 0.05),
        seed=seed,
    )
    if rag_cfg.get("mode", "oracle") == "oracle":
        return oracle
    cache_path = rag_cfg.get("cache_path")
    if cache_path is None:
        # default cache lives under the experiment output dir
        out_dir = config.get("experiment", {}).get("output_dir", "outputs")
        cache_path = str(Path(out_dir) / "rag_cache.jsonl")
    return LLMExtractor(
        model=rag_cfg.get("llm_model", "gpt-4o-mini"),
        api_key_env=rag_cfg.get("openai_api_key_env", "OPENAI_API_KEY"),
        api_key=rag_cfg.get("openai_api_key"),
        oracle_fallback=oracle,
        cache_path=cache_path,
    )
