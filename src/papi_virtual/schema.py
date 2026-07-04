from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import json


@dataclass(slots=True)
class KernelRecord:
    step_idx: int
    region: str
    latency_ms: float
    flops: float
    bytes_moved: float
    batch_size: int
    context_len: int
    decode_parallelism: int
    metadata: dict[str, Any]

    @property
    def arithmetic_intensity(self) -> float:
        if self.bytes_moved <= 0:
            return 0.0
        return self.flops / self.bytes_moved

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "KernelRecord":
        payload = json.loads(line)
        return cls(**payload)


@dataclass(slots=True)
class DispatchDecision:
    policy_name: str
    step_idx: int
    region: str
    target: str
    estimated_latency_ms: float
    baseline_latency_ms: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class SimulationSummary:
    policy_name: str
    total_latency_ms: float
    baseline_latency_ms: float
    speedup_vs_gpu_only: float
    oracle_gap_closed: float
    dispatches_to_pim: int
    total_records: int

