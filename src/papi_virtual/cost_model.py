from __future__ import annotations

from dataclasses import dataclass

from .schema import KernelRecord


@dataclass(slots=True)
class VirtualPIMConfig:
    # PAPI reports roofline analysis on NVIDIA A100 with 312 TFLOPS and
    # 1935 GB/s peak bandwidth. We adopt these as the default GPU reference.
    gpu_peak_bw_gbps: float = 1935.0
    pim_peak_bw_gbps: float = 2400.0
    gpu_peak_tflops: float = 312.0
    pim_peak_tflops: float = 18.0
    transfer_overhead_ms: float = 0.020
    sync_overhead_ms: float = 0.010
    min_offload_bytes: float = 256 * 1024
    # PAPI shows attention remains memory-bound while FC can transition with
    # decoding parallelism. LoL-PIM further emphasizes that long context makes
    # attention/KV memory demand grow with context length. We reflect these
    # prior observations using region-aware modifiers rather than pretending to
    # have a cycle-accurate PIM simulator.
    attention_long_context_bw_bonus: float = 0.20
    fc_parallelism_penalty: float = 0.25
    non_attention_penalty: float = 0.10


class VirtualPIMCostModel:
    """
    Prior-work-grounded what-if latency model.

    Grounding:
    - GPU reference defaults follow the PAPI roofline setup on A100
      (312 TFLOPS, 1935 GB/s).
    - Region behavior follows qualitative observations from PAPI and LoL-PIM:
      attention/KV paths become increasingly memory-sensitive with longer
      contexts, while FC/MLP paths are more compute-sensitive and can become
      less attractive for memory-centric execution as parallelism rises.

    This is still a virtual backend model, not a hardware-faithful simulator.
    """

    def __init__(self, config: VirtualPIMConfig | None = None) -> None:
        self.config = config or VirtualPIMConfig()

    def estimate_gpu_latency_ms(self, record: KernelRecord) -> float:
        if record.latency_ms > 0:
            return record.latency_ms
        compute_ms = self._compute_ms(record.flops, self.config.gpu_peak_tflops)
        memory_ms = self._memory_ms(record.bytes_moved, self.config.gpu_peak_bw_gbps)
        return max(compute_ms, memory_ms)

    def estimate_pim_latency_ms(self, record: KernelRecord) -> float:
        baseline_ms = self.estimate_gpu_latency_ms(record)
        gpu_compute_ms = self._compute_ms(record.flops, self.config.gpu_peak_tflops)
        gpu_memory_ms = self._memory_ms(record.bytes_moved, self.config.gpu_peak_bw_gbps)
        total_gpu_ms = max(gpu_compute_ms, gpu_memory_ms)
        if total_gpu_ms <= 0:
            total_gpu_ms = 1e-9

        compute_share = gpu_compute_ms / total_gpu_ms
        memory_share = gpu_memory_ms / total_gpu_ms

        effective_pim_bw = self.config.pim_peak_bw_gbps
        effective_pim_tflops = self.config.pim_peak_tflops

        if record.region == "attention":
            context_factor = min(record.context_len / 1024.0, 1.0)
            effective_pim_bw *= 1.0 + self.config.attention_long_context_bw_bonus * context_factor
        else:
            effective_pim_tflops *= max(1.0 - self.config.non_attention_penalty, 1e-6)

        if record.region == "mlp":
            parallelism_factor = min(record.decode_parallelism / 32.0, 1.0)
            effective_pim_tflops *= max(1.0 - self.config.fc_parallelism_penalty * parallelism_factor, 1e-6)

        scaled_compute_ms = baseline_ms * compute_share * (self.config.gpu_peak_tflops / effective_pim_tflops)
        scaled_memory_ms = baseline_ms * memory_share * (self.config.gpu_peak_bw_gbps / effective_pim_bw)
        orchestration_ms = self.config.transfer_overhead_ms + self.config.sync_overhead_ms
        return max(scaled_compute_ms, scaled_memory_ms) + orchestration_ms

    def is_pim_candidate(self, record: KernelRecord) -> bool:
        if record.bytes_moved < self.config.min_offload_bytes:
            return False
        return record.region in {"attention", "kv_cache", "memory_op"}

    @staticmethod
    def _compute_ms(flops: float, peak_tflops: float) -> float:
        if peak_tflops <= 0:
            return float("inf")
        return flops / (peak_tflops * 1e12) * 1e3

    @staticmethod
    def _memory_ms(bytes_moved: float, peak_bw_gbps: float) -> float:
        if peak_bw_gbps <= 0:
            return float("inf")
        return bytes_moved / (peak_bw_gbps * 1e9) * 1e3
