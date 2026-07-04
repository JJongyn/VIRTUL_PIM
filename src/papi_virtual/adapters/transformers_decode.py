from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import contextlib

from ..profiler import DecodeProfiler
from ..schema import KernelRecord


@dataclass(slots=True)
class TraceConfig:
    model_name: str
    prompt: str
    max_new_tokens: int = 32
    batch_size: int = 1
    device: str = "cuda"
    torch_dtype: str = "auto"
    quantization: str = "none"
    output_path: str = "artifacts/traces/transformers_decode.jsonl"
    attention_regions: bool = True
    mlp_regions: bool = True


class _RegionTracer:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    def timed(self, region: str):
        @contextlib.contextmanager
        def manager():
            start = perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (perf_counter() - start) * 1e3
                self.records.append((region, elapsed_ms))

        return manager()


def _maybe_sync_torch(device: str) -> None:
    try:
        import torch
    except ImportError:
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class _ModuleTimingCollector:
    def __init__(self, device: str) -> None:
        self.device = device
        self._starts: dict[int, float] = {}
        self.attention_latency_ms = 0.0
        self.mlp_latency_ms = 0.0
        self.attention_modules = 0
        self.mlp_modules = 0

    def reset(self) -> None:
        self._starts.clear()
        self.attention_latency_ms = 0.0
        self.mlp_latency_ms = 0.0

    def pre_hook(self, kind: str):
        def hook(module, inputs):
            _maybe_sync_torch(self.device)
            self._starts[id(module)] = perf_counter()

        return hook

    def post_hook(self, kind: str):
        def hook(module, inputs, output):
            _maybe_sync_torch(self.device)
            start = self._starts.pop(id(module), None)
            if start is None:
                return
            elapsed_ms = (perf_counter() - start) * 1e3
            if kind == "attention":
                self.attention_latency_ms += elapsed_ms
            elif kind == "mlp":
                self.mlp_latency_ms += elapsed_ms

        return hook


def _classify_module_name(name: str) -> str | None:
    lowered = name.lower()
    if any(token in lowered for token in ("attn", "attention", "self_attn")):
        return "attention"
    if any(token in lowered for token in ("mlp", "feed_forward", "feedforward", "ffn", "fc1", "fc2")):
        return "mlp"
    return None


def _register_timing_hooks(model, device: str):
    collector = _ModuleTimingCollector(device)
    handles = []
    seen_attention = 0
    seen_mlp = 0
    for name, module in model.named_modules():
        kind = _classify_module_name(name)
        if kind is None:
            continue
        handles.append(module.register_forward_pre_hook(collector.pre_hook(kind)))
        handles.append(module.register_forward_hook(collector.post_hook(kind)))
        if kind == "attention":
            seen_attention += 1
        elif kind == "mlp":
            seen_mlp += 1
    collector.attention_modules = seen_attention
    collector.mlp_modules = seen_mlp
    return collector, handles


def estimate_attention_flops(hidden_size: int, num_heads: int, context_len: int, batch_size: int) -> float:
    head_dim = hidden_size // max(num_heads, 1)
    qk = 2.0 * batch_size * num_heads * context_len * head_dim
    av = 2.0 * batch_size * num_heads * context_len * head_dim
    return qk + av


def estimate_attention_bytes(hidden_size: int, context_len: int, batch_size: int, dtype_bytes: int = 2) -> float:
    q_bytes = batch_size * hidden_size * dtype_bytes
    kv_bytes = 2 * batch_size * context_len * hidden_size * dtype_bytes
    out_bytes = batch_size * hidden_size * dtype_bytes
    return float(q_bytes + kv_bytes + out_bytes)


def estimate_attention_byte_breakdown(
    hidden_size: int,
    context_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    q_bytes = float(batch_size * hidden_size * dtype_bytes)
    kv_bytes = float(2 * batch_size * context_len * hidden_size * dtype_bytes)
    out_bytes = float(batch_size * hidden_size * dtype_bytes)
    total = q_bytes + kv_bytes + out_bytes
    kv_per_token = 0.0 if context_len <= 0 else kv_bytes / context_len
    memory_pressure_proxy = 0.0 if total <= 0 else kv_bytes / total
    return {
        "q_bytes_est": q_bytes,
        "kv_bytes_est": kv_bytes,
        "attn_out_bytes_est": out_bytes,
        "attention_bytes_total_est": total,
        "kv_bytes_per_token_est": kv_per_token,
        "memory_pressure_proxy": memory_pressure_proxy,
    }


def estimate_mlp_flops(hidden_size: int, intermediate_size: int, batch_size: int) -> float:
    return float(4.0 * batch_size * hidden_size * intermediate_size)


def estimate_mlp_bytes(hidden_size: int, intermediate_size: int, batch_size: int, dtype_bytes: int = 2) -> float:
    activations = batch_size * (hidden_size + intermediate_size + hidden_size) * dtype_bytes
    weights = (hidden_size * intermediate_size * 2) * dtype_bytes
    return float(activations + weights)


def build_records_from_layer_timings(
    *,
    step_idx: int,
    batch_size: int,
    context_len: int,
    decode_parallelism: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    attention_latency_ms: float,
    mlp_latency_ms: float,
    metadata: dict[str, Any] | None = None,
) -> list[KernelRecord]:
    payload = dict(metadata or {})
    payload.update(
        {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "dtype_bytes_est": 2,
        }
    )
    attn_breakdown = estimate_attention_byte_breakdown(hidden_size, context_len, batch_size)
    attention_payload = dict(payload)
    attention_payload.update(attn_breakdown)
    attention_payload["memory_pressure_proxy"] = attn_breakdown["memory_pressure_proxy"]
    mlp_payload = dict(payload)
    mlp_payload["memory_pressure_proxy"] = 0.0
    return [
        KernelRecord(
            step_idx=step_idx,
            region="attention",
            latency_ms=attention_latency_ms,
            flops=estimate_attention_flops(hidden_size, num_heads, context_len, batch_size),
            bytes_moved=estimate_attention_bytes(hidden_size, context_len, batch_size),
            batch_size=batch_size,
            context_len=context_len,
            decode_parallelism=decode_parallelism,
            metadata=attention_payload,
        ),
        KernelRecord(
            step_idx=step_idx,
            region="mlp",
            latency_ms=mlp_latency_ms,
            flops=estimate_mlp_flops(hidden_size, intermediate_size, batch_size),
            bytes_moved=estimate_mlp_bytes(hidden_size, intermediate_size, batch_size),
            batch_size=batch_size,
            context_len=context_len,
            decode_parallelism=decode_parallelism,
            metadata=mlp_payload,
        ),
    ]


def trace_transformers_decode(config: TraceConfig) -> str:
    """
    Collect a coarse decode trace from Hugging Face Transformers.

    This adapter is intentionally conservative. It measures per-token end-to-end
    decode step latency and derives coarse region records using model metadata.
    For the first implementation cut, this is more robust than invasive kernel
    hooks. If needed, finer-grained hooks can replace this later.
    """

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Transformers adapter requires `torch` and `transformers` to be installed."
        ) from exc

    quant_mode = config.quantization.lower()
    quantization_config = None
    model_kwargs: dict[str, Any] = {}
    if quant_mode != "none":
        if not config.device.startswith("cuda"):
            raise RuntimeError("Quantized loading currently requires a CUDA device.")
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Quantized loading requires `bitsandbytes`. Install it in the experiment environment first."
            ) from exc
        if quant_mode == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quant_mode == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported quantization mode: {config.quantization}")
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": config.device}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pretrained_kwargs = dict(model_kwargs)
    if config.torch_dtype != "auto":
        pretrained_kwargs["torch_dtype"] = getattr(torch, config.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **pretrained_kwargs,
    )
    if quant_mode == "none":
        model.to(config.device)
    model.eval()
    collector, handles = _register_timing_hooks(model, config.device)

    prompts = [config.prompt for _ in range(max(config.batch_size, 1))]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(config.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(config.device)

    profiler = DecodeProfiler(config.output_path)
    batch_size = int(input_ids.shape[0])
    hidden_size = int(getattr(model.config, "hidden_size"))
    num_heads = int(getattr(model.config, "num_attention_heads"))
    intermediate_size = int(getattr(model.config, "intermediate_size", hidden_size * 4))

    past_key_values = None
    current_ids = input_ids
    current_mask = attention_mask

    try:
        with torch.inference_mode():
            for step_idx in range(config.max_new_tokens):
                collector.reset()
                _maybe_sync_torch(config.device)
                step_start = perf_counter()
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                _maybe_sync_torch(config.device)
                elapsed_ms = (perf_counter() - step_start) * 1e3

                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values

                context_len = int(input_ids.shape[1] + step_idx)
                decode_parallelism = batch_size

                attention_latency_ms = collector.attention_latency_ms
                mlp_latency_ms = collector.mlp_latency_ms
                if attention_latency_ms <= 0.0 and mlp_latency_ms <= 0.0:
                    attn_flops = estimate_attention_flops(hidden_size, num_heads, context_len, batch_size)
                    mlp_flops = estimate_mlp_flops(hidden_size, intermediate_size, batch_size)
                    total_flops = max(attn_flops + mlp_flops, 1.0)
                    attention_latency_ms = elapsed_ms * attn_flops / total_flops
                    mlp_latency_ms = elapsed_ms * mlp_flops / total_flops

                for record in build_records_from_layer_timings(
                    step_idx=step_idx,
                    batch_size=batch_size,
                    context_len=context_len,
                    decode_parallelism=decode_parallelism,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    attention_latency_ms=attention_latency_ms,
                    mlp_latency_ms=mlp_latency_ms,
                    metadata={
                        "source": "transformers_adapter",
                        "model_name": config.model_name,
                        "device": config.device,
                        "quantization": quant_mode,
                        "step_latency_ms": elapsed_ms,
                        "attention_modules": collector.attention_modules,
                        "mlp_modules": collector.mlp_modules,
                        "hooked_attention_latency_ms": collector.attention_latency_ms,
                        "hooked_mlp_latency_ms": collector.mlp_latency_ms,
                    },
                ):
                    profiler.add_record(record)

                current_ids = next_token
                if current_mask is not None:
                    current_mask = torch.cat(
                        [current_mask, torch.ones((batch_size, 1), dtype=current_mask.dtype, device=current_mask.device)],
                        dim=1,
                    )
    finally:
        for handle in handles:
            handle.remove()

    profiler.dump()
    return str(Path(config.output_path))
