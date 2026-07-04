from __future__ import annotations

from dataclasses import dataclass

from .cost_model import VirtualPIMCostModel
from .schema import KernelRecord


class BasePolicy:
    name = "base"

    def reset(self) -> None:
        return None

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        raise NotImplementedError


class GPUOnlyPolicy(BasePolicy):
    name = "gpu_only"

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        return "gpu"


class StaticAttentionPolicy(BasePolicy):
    name = "static_attention"

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        if cost_model.is_pim_candidate(record):
            return "pim"
        return "gpu"


@dataclass(slots=True)
class ThresholdPolicy(BasePolicy):
    name: str = "threshold"
    intensity_threshold: float = 6.0
    min_context_len: int = 2048

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        if not cost_model.is_pim_candidate(record):
            return "gpu"
        if record.context_len < self.min_context_len:
            return "gpu"
        if record.arithmetic_intensity <= self.intensity_threshold:
            return "pim"
        return "gpu"


@dataclass(slots=True)
class OnlinePredictorPolicy(BasePolicy):
    name: str = "online_predictor"
    memory_pressure_threshold: float = 0.12
    latency_ratio_threshold: float = 0.98
    min_context_len: int = 256
    min_bytes_moved: float = 32768.0

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        if not cost_model.is_pim_candidate(record):
            return "gpu"

        memory_pressure = self._memory_pressure(record)
        gpu_est = cost_model.estimate_gpu_latency_ms(record)
        pim_est = cost_model.estimate_pim_latency_ms(record)
        context_ready = record.context_len >= self.min_context_len
        bytes_ready = record.bytes_moved >= self.min_bytes_moved

        if context_ready and bytes_ready and memory_pressure >= self.memory_pressure_threshold and gpu_est / max(pim_est, 1e-9) >= self.latency_ratio_threshold:
            return "pim"
        return "gpu"

    @staticmethod
    def _memory_pressure(record: KernelRecord) -> float:
        bytes_per_token = record.bytes_moved / max(record.decode_parallelism, 1)
        context_factor = min(record.context_len / 8192.0, 1.0)
        normalized = min(bytes_per_token / (8 * 1024 * 1024), 1.0)
        return 0.5 * context_factor + 0.5 * normalized


class OraclePolicy(BasePolicy):
    name = "oracle"

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        gpu_est = cost_model.estimate_gpu_latency_ms(record)
        pim_est = cost_model.estimate_pim_latency_ms(record)
        return "pim" if pim_est < gpu_est else "gpu"


@dataclass(slots=True)
class AdaptiveFeaturePolicy(BasePolicy):
    name: str = "adaptive_feature"
    min_context_len: int = 384
    min_bytes_moved: float = 32768.0
    score_threshold: float = 0.58
    use_memory_pressure: bool = True
    use_context: bool = True
    use_batch: bool = True
    use_bytes: bool = True
    use_intensity: bool = True
    use_latency: bool = True
    use_region_bias: bool = True

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        if not cost_model.is_pim_candidate(record):
            return "gpu"
        if record.context_len < self.min_context_len:
            return "gpu"
        if record.bytes_moved < self.min_bytes_moved:
            return "gpu"

        memory_pressure = OnlinePredictorPolicy._memory_pressure(record)
        context_score = min(record.context_len / 1024.0, 1.0)
        batch_score = min(record.batch_size / 4.0, 1.0)
        byte_score = min(record.bytes_moved / (2 * 1024 * 1024), 1.0)
        intensity_score = max(0.0, 1.0 - min(record.arithmetic_intensity / 2.0, 1.0))
        latency_score = min(record.latency_ms / 50.0, 1.0)
        region_bias = 0.20 if record.region == "attention" else 0.05

        score = (
            (0.22 * memory_pressure if self.use_memory_pressure else 0.0)
            + (0.22 * context_score if self.use_context else 0.0)
            + (0.14 * batch_score if self.use_batch else 0.0)
            + (0.14 * byte_score if self.use_bytes else 0.0)
            + (0.14 * intensity_score if self.use_intensity else 0.0)
            + (0.08 * latency_score if self.use_latency else 0.0)
            + (region_bias if self.use_region_bias else 0.0)
        )
        return "pim" if score >= self.score_threshold else "gpu"


@dataclass(slots=True)
class KVRegimePolicy(BasePolicy):
    name: str = "kv_regime"
    min_bytes_moved: float = 32768.0
    min_context_len: int = 256
    early_kv_bytes_threshold: float = 2_000_000.0
    sustained_kv_bytes_threshold: float = 1_500_000.0
    kv_per_token_threshold: float = 8192.0
    score_threshold: float = 0.66

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        if not cost_model.is_pim_candidate(record):
            return "gpu"
        if record.bytes_moved < self.min_bytes_moved:
            return "gpu"
        if record.context_len < self.min_context_len:
            return "gpu"

        metadata = record.metadata or {}
        kv_bytes = float(metadata.get("kv_bytes_est", 0.0) or 0.0)
        kv_bytes_per_token = float(metadata.get("kv_bytes_per_token_est", 0.0) or 0.0)
        memory_pressure = float(metadata.get("memory_pressure_proxy", 0.0) or 0.0)

        context_score = min(record.context_len / 768.0, 1.0)
        kv_score = min(kv_bytes / (8 * 1024 * 1024), 1.0)
        kv_token_score = min(kv_bytes_per_token / 16384.0, 1.0)
        intensity_score = max(0.0, 1.0 - min(record.arithmetic_intensity / 2.0, 1.0))
        latency_score = min(record.latency_ms / 50.0, 1.0)
        region_bias = 0.18 if record.region == "attention" else 0.05

        # Early regime entry is allowed when KV pressure is already large enough
        # even at moderate context lengths, which matches the newer-model behavior.
        if (
            record.region == "attention"
            and kv_bytes >= self.early_kv_bytes_threshold
            and kv_bytes_per_token >= self.kv_per_token_threshold
            and memory_pressure >= 0.995
        ):
            return "pim"

        score = (
            0.22 * context_score
            + 0.26 * kv_score
            + 0.20 * kv_token_score
            + 0.08 * memory_pressure
            + 0.12 * intensity_score
            + 0.04 * latency_score
            + region_bias
        )

        if (
            record.region == "attention"
            and kv_bytes >= self.sustained_kv_bytes_threshold
            and score >= self.score_threshold
        ):
            return "pim"
        return "gpu"


@dataclass(slots=True)
class ContextKVHysteresisPolicy(BasePolicy):
    name: str = "context_kv_hysteresis"
    min_bytes_moved: float = 32768.0
    context_gate_len: int = 256
    context_strong_len: int = 320
    kv_bytes_on: float = 1_500_000.0
    kv_bytes_off: float = 1_200_000.0
    kv_per_token_on: float = 6000.0
    kv_per_token_off: float = 4500.0
    memory_pressure_on: float = 0.990
    memory_pressure_off: float = 0.982
    score_on: float = 0.58
    score_off: float = 0.50
    _active: bool = False

    def _kv_inputs(self, record: KernelRecord) -> tuple[float, float, float, float]:
        metadata = record.metadata or {}
        kv_bytes = float(metadata.get("kv_bytes_est", 0.0) or 0.0)
        kv_bytes_per_token = float(metadata.get("kv_bytes_per_token_est", 0.0) or 0.0)
        memory_pressure = float(metadata.get("memory_pressure_proxy", 0.0) or 0.0)

        context_score = min(max(record.context_len - self.context_gate_len, 0) / max(self.context_strong_len - self.context_gate_len, 1), 1.0)
        kv_score = min(kv_bytes / (8 * 1024 * 1024), 1.0)
        kv_token_score = min(kv_bytes_per_token / 16384.0, 1.0)
        pressure_score = min(max((memory_pressure - 0.96) / 0.04, 0.0), 1.0)
        score = 0.30 * context_score + 0.30 * kv_score + 0.22 * kv_token_score + 0.18 * pressure_score
        return kv_bytes, kv_bytes_per_token, memory_pressure, score

    def reset(self) -> None:
        self._active = False

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        if not cost_model.is_pim_candidate(record):
            return "gpu"
        if record.region != "attention":
            return "gpu"
        if record.bytes_moved < self.min_bytes_moved:
            return "gpu"
        if record.context_len < self.context_gate_len:
            self._active = False
            return "gpu"

        kv_bytes, kv_bytes_per_token, memory_pressure, score = self._kv_inputs(record)

        if self._active:
            stay_on = (
                kv_bytes >= self.kv_bytes_off
                and kv_bytes_per_token >= self.kv_per_token_off
                and memory_pressure >= self.memory_pressure_off
                and score >= self.score_off
            )
            self._active = stay_on
            return "pim" if stay_on else "gpu"

        turn_on = (
            kv_bytes >= self.kv_bytes_on
            and kv_bytes_per_token >= self.kv_per_token_on
            and memory_pressure >= self.memory_pressure_on
            and score >= self.score_on
        )
        self._active = turn_on
        return "pim" if turn_on else "gpu"


@dataclass(slots=True)
class ContextKVRefinedPolicy(ContextKVHysteresisPolicy):
    name: str = "context_kv_refined"

    def reset(self) -> None:
        self._active = False

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        if not cost_model.is_pim_candidate(record):
            return "gpu"
        if record.region != "attention":
            return "gpu"
        if record.bytes_moved < self.min_bytes_moved:
            return "gpu"
        if record.context_len < self.context_gate_len:
            return "gpu"

        kv_bytes, kv_bytes_per_token, memory_pressure, score = self._kv_inputs(record)
        turn_on = (
            kv_bytes >= self.kv_bytes_on
            and kv_bytes_per_token >= self.kv_per_token_on
            and memory_pressure >= self.memory_pressure_on
            and score >= self.score_on
        )
        return "pim" if turn_on else "gpu"


def _model_name(record: KernelRecord) -> str:
    return str(record.metadata.get("model_name", "")).lower()


def _family_context_prior(record: KernelRecord) -> tuple[int, float]:
    model_name = _model_name(record)
    if "smollm" in model_name:
        return 256, 0.07
    if "opt" in model_name:
        return 320, 0.05
    if "qwen" in model_name:
        return 384, 0.03
    if "pythia" in model_name:
        return 384, 0.02
    if "gpt2" in model_name:
        return 384, 0.00
    return 384, 0.02


@dataclass(slots=True)
class AdaptiveFamilyPolicy(BasePolicy):
    name: str = "adaptive_family"
    min_context_len: int = 384
    min_bytes_moved: float = 32768.0
    score_threshold: float = 0.60

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        if not cost_model.is_pim_candidate(record):
            return "gpu"
        if record.bytes_moved < self.min_bytes_moved:
            return "gpu"

        family_min_context, family_bias = _family_context_prior(record)
        effective_min_context = min(self.min_context_len, family_min_context)
        if record.context_len < effective_min_context:
            return "gpu"

        memory_pressure = OnlinePredictorPolicy._memory_pressure(record)
        context_norm = max(record.context_len - effective_min_context, 0) / max(1024 - effective_min_context, 1)
        context_score = min(max(context_norm, 0.0), 1.0)
        batch_score = min(record.batch_size / 4.0, 1.0)
        byte_score = min(record.bytes_moved / (2 * 1024 * 1024), 1.0)
        intensity_score = max(0.0, 1.0 - min(record.arithmetic_intensity / 2.0, 1.0))
        latency_score = min(record.latency_ms / 50.0, 1.0)
        region_bias = 0.20 if record.region == "attention" else 0.05

        score = (
            0.18 * memory_pressure
            + 0.18 * context_score
            + 0.10 * batch_score
            + 0.14 * byte_score
            + 0.14 * intensity_score
            + 0.08 * latency_score
            + region_bias
            + family_bias
        )
        return "pim" if score >= self.score_threshold else "gpu"


@dataclass(slots=True)
class AdaptiveScorePolicy(BasePolicy):
    name: str = "adaptive_score"
    min_context_len: int = 384
    min_relative_gain: float = 0.05
    dispatch_score_threshold: float = 0.20

    def choose_target(self, record: KernelRecord, cost_model: VirtualPIMCostModel) -> str:
        if record.context_len < self.min_context_len:
            return "gpu"

        gpu_est = cost_model.estimate_gpu_latency_ms(record)
        pim_est = cost_model.estimate_pim_latency_ms(record)
        relative_gain = (gpu_est - pim_est) / max(gpu_est, 1e-9)
        if relative_gain < self.min_relative_gain:
            return "gpu"

        memory_pressure = OnlinePredictorPolicy._memory_pressure(record)
        intensity_score = max(0.0, 1.0 - min(record.arithmetic_intensity / 2.0, 1.0))
        context_score = min(record.context_len / 1024.0, 1.0)
        batch_score = min(record.batch_size / 4.0, 1.0)
        region_bias = 0.15 if record.region == "attention" else 0.05

        dispatch_score = (
            0.35 * relative_gain
            + 0.25 * memory_pressure
            + 0.15 * intensity_score
            + 0.15 * context_score
            + 0.10 * batch_score
            + region_bias
        )
        return "pim" if dispatch_score >= self.dispatch_score_threshold else "gpu"
