from __future__ import annotations

from typing import Iterable

from .cost_model import VirtualPIMCostModel
from .policies import BasePolicy, GPUOnlyPolicy, OraclePolicy
from .schema import DispatchDecision, KernelRecord, SimulationSummary


def simulate_policy(
    records: Iterable[KernelRecord],
    policy: BasePolicy,
    cost_model: VirtualPIMCostModel,
) -> tuple[list[DispatchDecision], SimulationSummary]:
    decisions: list[DispatchDecision] = []
    total_latency_ms = 0.0
    baseline_latency_ms = 0.0
    oracle_latency_ms = 0.0

    policy.reset()

    for record in records:
        gpu_est = cost_model.estimate_gpu_latency_ms(record)
        pim_est = cost_model.estimate_pim_latency_ms(record)
        baseline_latency_ms += gpu_est
        oracle_latency_ms += min(gpu_est, pim_est)

        target = policy.choose_target(record, cost_model)
        estimated = pim_est if target == "pim" else gpu_est
        total_latency_ms += estimated

        decisions.append(
            DispatchDecision(
                policy_name=policy.name,
                step_idx=record.step_idx,
                region=record.region,
                target=target,
                estimated_latency_ms=estimated,
                baseline_latency_ms=gpu_est,
                metadata={"gpu_est_ms": gpu_est, "pim_est_ms": pim_est},
            )
        )

    summary = summarize(policy.name, decisions, total_latency_ms, baseline_latency_ms, oracle_latency_ms)
    return decisions, summary


def summarize(
    policy_name: str,
    decisions: list[DispatchDecision],
    total_latency_ms: float,
    baseline_latency_ms: float,
    oracle_latency_ms: float,
) -> SimulationSummary:
    gpu_only_gain = max(baseline_latency_ms - total_latency_ms, 0.0)
    oracle_gain = max(baseline_latency_ms - oracle_latency_ms, 0.0)
    gap_closed = 0.0 if oracle_gain == 0 else gpu_only_gain / oracle_gain
    speedup = 1.0 if total_latency_ms == 0 else baseline_latency_ms / total_latency_ms
    return SimulationSummary(
        policy_name=policy_name,
        total_latency_ms=total_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        speedup_vs_gpu_only=speedup,
        oracle_gap_closed=gap_closed,
        dispatches_to_pim=sum(1 for d in decisions if d.target == "pim"),
        total_records=len(decisions),
    )


def default_policies() -> list[BasePolicy]:
    from .policies import AdaptiveFamilyPolicy, AdaptiveFeaturePolicy, AdaptiveScorePolicy, KVRegimePolicy, OnlinePredictorPolicy, StaticAttentionPolicy, ThresholdPolicy

    return [
        GPUOnlyPolicy(),
        StaticAttentionPolicy(),
        ThresholdPolicy(),
        OnlinePredictorPolicy(),
        AdaptiveFeaturePolicy(),
        KVRegimePolicy(),
        AdaptiveFamilyPolicy(),
        AdaptiveScorePolicy(),
        OraclePolicy(),
    ]
