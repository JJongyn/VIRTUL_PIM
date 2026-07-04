from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Iterable
import json

from .schema import KernelRecord


class DecodeProfiler:
    """
    Lightweight trace writer.

    This is intentionally runtime-agnostic. A caller can feed records from
    PyTorch hooks, custom instrumentation, or offline parsers.
    """

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[KernelRecord] = []

    def add_record(self, record: KernelRecord) -> None:
        self._records.append(record)

    def dump(self) -> None:
        with self.output_path.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(record.to_json())
                handle.write("\n")

    def summary(self) -> dict[str, float]:
        if not self._records:
            return {"records": 0, "total_latency_ms": 0.0}
        return {
            "records": float(len(self._records)),
            "total_latency_ms": sum(record.latency_ms for record in self._records),
        }


def load_trace(path: str | Path) -> list[KernelRecord]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [KernelRecord.from_json(line) for line in handle if line.strip()]


def write_simulation_report(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def synthetic_trace(
    num_steps: int = 24,
    base_batch: int = 1,
    base_context: int = 1024,
) -> list[KernelRecord]:
    """
    Deterministic fallback trace for testing the pipeline without a GPU run.
    """
    records: list[KernelRecord] = []
    for step_idx in range(num_steps):
        context_len = base_context + step_idx * 256
        batch_size = min(base_batch + step_idx // 6, 8)
        decode_parallelism = batch_size

        records.append(
            KernelRecord(
                step_idx=step_idx,
                region="attention",
                latency_ms=0.25 + 0.02 * step_idx,
                flops=2.6e9 + step_idx * 1.5e8,
                bytes_moved=1.5e9 + step_idx * 1.6e8,
                batch_size=batch_size,
                context_len=context_len,
                decode_parallelism=decode_parallelism,
                metadata={"source": "synthetic"},
            )
        )
        records.append(
            KernelRecord(
                step_idx=step_idx,
                region="mlp",
                latency_ms=0.20 + 0.01 * step_idx,
                flops=1.6e10 + step_idx * 5e8,
                bytes_moved=8e7 + step_idx * 3e6,
                batch_size=batch_size,
                context_len=context_len,
                decode_parallelism=decode_parallelism,
                metadata={"source": "synthetic"},
            )
        )
    return records


def profile_callable(
    fn,
    region: str,
    *,
    step_idx: int,
    flops: float,
    bytes_moved: float,
    batch_size: int,
    context_len: int,
    decode_parallelism: int,
    metadata: dict | None = None,
) -> KernelRecord:
    start = perf_counter()
    fn()
    elapsed_ms = (perf_counter() - start) * 1e3
    return KernelRecord(
        step_idx=step_idx,
        region=region,
        latency_ms=elapsed_ms,
        flops=flops,
        bytes_moved=bytes_moved,
        batch_size=batch_size,
        context_len=context_len,
        decode_parallelism=decode_parallelism,
        metadata=metadata or {},
    )
