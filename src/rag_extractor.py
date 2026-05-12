"""RAG-based semantic extraction with an oracle fallback.

The oracle mode injects controlled Gaussian noise around the ground-truth
severity/probability and returns the affected entity unchanged. The LLM mode
calls a configured OpenAI-compatible chat model with a strict JSON schema.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

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
    """OpenAI-compatible RAG extractor. Falls back to oracle on failure."""
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: Optional[str] = None
    oracle_fallback: Optional[OracleExtractor] = None

    def __post_init__(self) -> None:
        self._client = None
        key = self.api_key or os.environ.get(self.api_key_env)
        # User-friendly fallback: if `api_key_env` was set to the key itself
        # (starts with "sk-"), treat it as the literal key.
        if not key and self.api_key_env.startswith("sk-"):
            key = self.api_key_env
        if not key:
            logger.warning(
                "OpenAI key not found (env var `%s` is empty); "
                "LLM extractor will use oracle fallback.",
                self.api_key_env,
            )
            return
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=key)
            logger.info("OpenAI client ready (model=%s).", self.model)
        except Exception as exc:
            logger.warning("openai client unavailable (%s); fallback to oracle.", exc)

    def extract(self, event) -> ExtractedTuple:
        if self._client is None:
            assert self.oracle_fallback is not None
            return self.oracle_fallback.extract(event)
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
            payload = json.loads(resp.choices[0].message.content or "{}")
            payload["event_id"] = event.event_id
            payload["timestamp"] = event.timestamp
            if "affected_id" not in payload:
                payload["affected_id"] = list(event.affected_id)
            return ExtractedTuple(**payload)
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
    return LLMExtractor(
        model=rag_cfg.get("llm_model", "gpt-4o-mini"),
        api_key_env=rag_cfg.get("openai_api_key_env", "OPENAI_API_KEY"),
        api_key=rag_cfg.get("openai_api_key"),
        oracle_fallback=oracle,
    )
