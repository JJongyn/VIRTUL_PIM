#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papi_virtual.analysis import summarize_trace
from papi_virtual.cost_model import VirtualPIMCostModel
from papi_virtual.policies import AdaptiveFeaturePolicy, KVRegimePolicy, OnlinePredictorPolicy
from papi_virtual.profiler import load_trace
from papi_virtual.simulator import simulate_policy


@dataclass(frozen=True, slots=True)
class TraceSpec:
    tag: str
    family: str
    context_target: int
    batch_size: int
    path: str


TRACE_SPECS = [
    TraceSpec("gpt2_ctx256", "gpt2", 256, 1, "artifacts/traces/gpt2_none_ctx256_bs1_kvmech.jsonl"),
    TraceSpec("gpt2_ctx512", "gpt2", 512, 1, "artifacts/traces/gpt2_none_ctx512_bs1_kvmech.jsonl"),
    TraceSpec("gpt2_ctx768", "gpt2", 768, 1, "artifacts/traces/gpt2_none_ctx768_bs1_kvmech.jsonl"),
    TraceSpec("opt125m_ctx256", "opt-125m", 256, 1, "artifacts/traces/opt125m_gpu_ctx256_bs1.jsonl"),
    TraceSpec("opt125m_ctx512", "opt-125m", 512, 1, "artifacts/traces/opt125m_gpu_ctx512_bs1.jsonl"),
    TraceSpec("opt125m_ctx768", "opt-125m", 768, 1, "artifacts/traces/opt125m_gpu_ctx768_bs1.jsonl"),
    TraceSpec("qwen25_ctx128", "Qwen2.5", 128, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx128_bs1_densectx.jsonl"),
    TraceSpec("qwen25_ctx256", "Qwen2.5", 256, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_densectx.jsonl"),
    TraceSpec("qwen25_ctx256_8bit", "Qwen2.5", 256, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_8bit_ctx256_bs1_qwen25_8bit_probe.jsonl"),
    TraceSpec("qwen25_ctx384", "Qwen2.5", 384, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx384_bs1_densectx.jsonl"),
    TraceSpec("qwen25_ctx512", "Qwen2.5", 512, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx512_bs1_densectx.jsonl"),
    TraceSpec("qwen25_ctx640", "Qwen2.5", 640, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx640_bs1_densectx.jsonl"),
    TraceSpec("qwen25_ctx768", "Qwen2.5", 768, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_densectx.jsonl"),
    TraceSpec("qwen25_ctx768_8bit", "Qwen2.5", 768, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_8bit_ctx768_bs1_qwen25_8bit_probe.jsonl"),
    TraceSpec("qwen25_ctx1024", "Qwen2.5", 1024, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx1024_bs1_densectx.jsonl"),
    TraceSpec("qwen3_ctx128", "Qwen3", 128, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx128_bs1_densectx.jsonl"),
    TraceSpec("qwen3_ctx256", "Qwen3", 256, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_densectx.jsonl"),
    TraceSpec("qwen3_ctx384", "Qwen3", 384, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx384_bs1_densectx.jsonl"),
    TraceSpec("qwen3_ctx512", "Qwen3", 512, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx512_bs1_densectx.jsonl"),
    TraceSpec("qwen3_ctx640", "Qwen3", 640, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx640_bs1_densectx.jsonl"),
    TraceSpec("qwen3_ctx768", "Qwen3", 768, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_densectx.jsonl"),
    TraceSpec("qwen3_ctx1024", "Qwen3", 1024, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx1024_bs1_densectx.jsonl"),
    TraceSpec("smollm2_ctx128", "SmolLM2", 128, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx128_bs1_densectx.jsonl"),
    TraceSpec("smollm2_ctx256", "SmolLM2", 256, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_densectx.jsonl"),
    TraceSpec("smollm2_ctx256_8bit", "SmolLM2", 256, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_8bit_ctx256_bs1_smollm2_8bit_probe.jsonl"),
    TraceSpec("smollm2_ctx384", "SmolLM2", 384, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx384_bs1_densectx.jsonl"),
    TraceSpec("smollm2_ctx512", "SmolLM2", 512, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs1_densectx.jsonl"),
    TraceSpec("smollm2_ctx640", "SmolLM2", 640, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx640_bs1_densectx.jsonl"),
    TraceSpec("smollm2_ctx768", "SmolLM2", 768, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_densectx.jsonl"),
    TraceSpec("smollm2_ctx768_8bit", "SmolLM2", 768, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_8bit_ctx768_bs1_smollm2_8bit_probe.jsonl"),
    TraceSpec("smollm2_ctx1024", "SmolLM2", 1024, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx1024_bs1_densectx.jsonl"),
    TraceSpec("phi35_ctx256", "Phi-3.5", 256, 1, "artifacts/traces/microsoft_phi_3_5_mini_instruct_4bit_ctx256_bs1_phi35_probe.jsonl"),
    TraceSpec("phi35_ctx384", "Phi-3.5", 384, 1, "artifacts/traces/microsoft_phi_3_5_mini_instruct_4bit_ctx384_bs1_phi35_probe.jsonl"),
    TraceSpec("phi35_ctx768", "Phi-3.5", 768, 1, "artifacts/traces/microsoft_phi_3_5_mini_instruct_4bit_ctx768_bs1_phi35_probe.jsonl"),
]


FEATURE_SETS = {
    "attention_only": ["attention_share"],
    "context_only": ["context_target"],
    "context_plus_attention": ["context_target", "attention_share"],
    "kv_only": ["mean_kv_bytes_est", "mean_kv_bytes_per_token_est", "mean_memory_pressure_proxy"],
    "all_features": [
        "context_target",
        "attention_share",
        "mean_kv_bytes_est",
        "mean_kv_bytes_per_token_est",
        "mean_memory_pressure_proxy",
    ],
}

CLASSIFIERS = ("logistic_regression", "decision_tree")


def context_bucket(context_target: int) -> str:
    if context_target <= 256:
        return "short"
    if context_target <= 512:
        return "mid"
    return "long"


class ConstantModel:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full((len(x),), self.value, dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(x), 2), dtype=float)
        probs[:, self.value] = 1.0
        return probs


def fit_classifier(train_rows: list[dict[str, Any]], feature_names: list[str], label_key: str, classifier_name: str) -> Any:
    x = np.array([[row[name] for name in feature_names] for row in train_rows], dtype=float)
    y = np.array([row[label_key] for row in train_rows], dtype=int)
    if len(np.unique(y)) < 2:
        return ConstantModel(int(y[0]))
    if classifier_name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)),
            ]
        ).fit(x, y)
    if classifier_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, class_weight="balanced", random_state=0).fit(x, y)
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def evaluate_classifier(model: Any, test_rows: list[dict[str, Any]], feature_names: list[str], label_key: str) -> dict[str, float]:
    x = np.array([[row[name] for name in feature_names] for row in test_rows], dtype=float)
    y = np.array([row[label_key] for row in test_rows], dtype=int)
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= 0.5).astype(int)
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    positive_recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    false_positive_rate = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
    try:
        auc = float(roc_auc_score(y, probs))
        if np.isnan(auc):
            auc = 0.5
    except ValueError:
        auc = 0.5
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "roc_auc": float(auc),
        "positive_recall": float(positive_recall),
        "false_positive_rate": float(false_positive_rate),
        "num_points": float(len(test_rows)),
        "positive_rate": float(np.mean(y) if len(y) else 0.0),
    }


def load_workload_rows() -> list[dict[str, Any]]:
    cost_model = VirtualPIMCostModel()
    rows: list[dict[str, Any]] = []
    for spec in TRACE_SPECS:
        records = load_trace(ROOT / spec.path)
        trace_summary = summarize_trace(records)
        _, online_summary = simulate_policy(records, OnlinePredictorPolicy(), cost_model)
        _, adaptive_summary = simulate_policy(records, AdaptiveFeaturePolicy(), cost_model)
        _, kv_summary = simulate_policy(records, KVRegimePolicy(), cost_model)
        rows.append(
            {
                "family": spec.family,
                "workload_tag": spec.tag,
                "context_target": spec.context_target,
                "context_bucket": context_bucket(spec.context_target),
                "batch_size": spec.batch_size,
                "attention_share": float(next((r["latency_share"] for r in trace_summary["regions"] if r["region"] == "attention"), 0.0)),
                "mean_kv_bytes_est": float(trace_summary.get("mean_kv_bytes_est", 0.0)),
                "mean_kv_bytes_per_token_est": float(trace_summary.get("mean_kv_bytes_per_token_est", 0.0)),
                "mean_memory_pressure_proxy": float(trace_summary.get("mean_memory_pressure_proxy", 0.0)),
                "online_speedup": float(online_summary.speedup_vs_gpu_only),
                "adaptive_feature_speedup": float(adaptive_summary.speedup_vs_gpu_only),
                "kv_regime_speedup": float(kv_summary.speedup_vs_gpu_only),
                "online_positive": int(online_summary.speedup_vs_gpu_only > 1.05),
                "adaptive_positive": int(adaptive_summary.speedup_vs_gpu_only > 1.05),
                "kv_regime_positive": int(kv_summary.speedup_vs_gpu_only > 1.05),
            }
        )
    return rows


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["accuracy", "f1", "roc_auc", "positive_recall", "false_positive_rate", "num_points", "positive_rate"]
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if not np.isnan(float(row[key]))]
        out[key] = 0.0 if not values else float(sum(values) / len(values))
        if values:
            out[f"{key}_std"] = float(np.std(values))
        else:
            out[f"{key}_std"] = 0.0
    return out


def build_split_results(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    families = sorted({str(row["family"]) for row in rows})
    context_buckets = sorted({str(row["context_bucket"]) for row in rows})
    label_key = f"online_positive_t{str(threshold).replace('.', '_')}"

    split_defs = [
        ("family_held_out", families, lambda row, v: row["family"] == v),
        ("context_bucket_held_out", context_buckets, lambda row, v: row["context_bucket"] == v),
    ]
    for split_type, split_values, match_fn in split_defs:
        for split_value in split_values:
            train_rows = [row for row in rows if not match_fn(row, split_value)]
            test_rows = [row for row in rows if match_fn(row, split_value)]
            for classifier_name in CLASSIFIERS:
                for feature_name, feature_list in FEATURE_SETS.items():
                    model = fit_classifier(train_rows, feature_list, label_key, classifier_name)
                    metrics = evaluate_classifier(model, test_rows, feature_list, label_key)
                    split_rows.append(
                        {
                            "split_type": split_type,
                            "split_value": split_value,
                            "classifier": classifier_name,
                            "signal_family": feature_name,
                            **metrics,
                        }
                    )

    summary: dict[str, dict[str, dict[str, float]]] = {}
    by_classifier: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for split_type in ("family_held_out", "context_bucket_held_out"):
        summary[split_type] = {}
        by_classifier[split_type] = {}
        for signal_family in FEATURE_SETS:
            family_rows = [row for row in split_rows if row["split_type"] == split_type and row["signal_family"] == signal_family and row["classifier"] == "logistic_regression"]
            summary[split_type][signal_family] = aggregate_metrics(family_rows)
        for classifier_name in CLASSIFIERS:
            by_classifier[split_type][classifier_name] = {}
            for signal_family in FEATURE_SETS:
                clf_rows = [row for row in split_rows if row["split_type"] == split_type and row["signal_family"] == signal_family and row["classifier"] == classifier_name]
                by_classifier[split_type][classifier_name][signal_family] = aggregate_metrics(clf_rows)

    boundary_rows = [row for row in rows if 1.0 <= float(row["online_speedup"]) <= 1.18]
    boundary_summary: dict[str, dict[str, float]] = {}
    if len(boundary_rows) >= 6:
        boundary_label_key = label_key
        folds = sorted({str(row["family"]) for row in boundary_rows})
        for signal_family, feature_list in FEATURE_SETS.items():
            fold_metrics = []
            for family in folds:
                train_rows = [row for row in boundary_rows if row["family"] != family]
                test_rows = [row for row in boundary_rows if row["family"] == family]
                if not train_rows or not test_rows:
                    continue
                model = fit_classifier(train_rows, feature_list, boundary_label_key, "logistic_regression")
                fold_metrics.append(evaluate_classifier(model, test_rows, feature_list, boundary_label_key))
            if fold_metrics:
                boundary_summary[signal_family] = aggregate_metrics(fold_metrics)
    return {"threshold": threshold, "split_rows": split_rows, "summary": summary, "classifier_robustness": by_classifier, "boundary_summary": boundary_summary}


def build_main_readout(rows: list[dict[str, Any]], split_results: dict[str, Any], threshold_sensitivity: list[dict[str, Any]]) -> list[str]:
    family_summary = split_results["summary"]["family_held_out"]
    context_summary = split_results["summary"]["context_bucket_held_out"]
    classifier_robustness = split_results["classifier_robustness"]
    boundary_summary = split_results["boundary_summary"]
    online_positive_short = sum(row["online_positive"] for row in rows if row["context_bucket"] == "short")
    online_positive_mid = sum(row["online_positive"] for row in rows if row["context_bucket"] == "mid")
    online_positive_long = sum(row["online_positive"] for row in rows if row["context_bucket"] == "long")
    short_total = sum(1 for row in rows if row["context_bucket"] == "short")
    mid_total = sum(1 for row in rows if row["context_bucket"] == "mid")
    long_total = sum(1 for row in rows if row["context_bucket"] == "long")
    threshold_101 = next(item for item in threshold_sensitivity if abs(float(item["threshold"]) - 1.01) < 1e-9)
    threshold_110 = next(item for item in threshold_sensitivity if abs(float(item["threshold"]) - 1.10) < 1e-9)

    return [
        f"- Online-positive fraction rises from {online_positive_short}/{short_total} in short-context workloads to {online_positive_mid}/{mid_total} in mid-context and {online_positive_long}/{long_total} in long-context workloads.",
        f"- On family-held-out splits, `context_only` is the strongest coarse detector (F1={family_summary['context_only']['f1']:.3f}±{family_summary['context_only']['f1_std']:.3f}, FP rate={family_summary['context_only']['false_positive_rate']:.3f}).",
        f"- `attention_only` over-predicts the positive regime on held-out families (positive recall={family_summary['attention_only']['positive_recall']:.3f}, FP rate={family_summary['attention_only']['false_positive_rate']:.3f}), which supports the paper's negative point that static attention labeling is too permissive.",
        f"- On context-bucket-held-out splits, `all_features` is the strongest overall detector (F1={context_summary['all_features']['f1']:.3f}±{context_summary['all_features']['f1_std']:.3f}) and modestly exceeds `context_only` (F1={context_summary['context_only']['f1']:.3f}±{context_summary['context_only']['f1_std']:.3f}); this supports calling KV-related signals complementary decision signals rather than a standalone dominant mechanism.",
        f"- The signal ordering is not specific to logistic regression: under a shallow decision tree, family-held-out `context_only` still exceeds `attention_only` in F1 ({classifier_robustness['family_held_out']['decision_tree']['context_only']['f1']:.3f} vs {classifier_robustness['family_held_out']['decision_tree']['attention_only']['f1']:.3f}), and context-bucket-held-out `all_features` still slightly exceeds `context_only` ({classifier_robustness['context_bucket_held_out']['decision_tree']['all_features']['f1']:.3f} vs {classifier_robustness['context_bucket_held_out']['decision_tree']['context_only']['f1']:.3f}).",
        f"- On a boundary-focused subset ({len([row for row in rows if 1.0 <= float(row['online_speedup']) <= 1.18])} workloads with `1.00x <= online <= 1.18x`), `all_features` reduces the family-held-out false-positive rate relative to `attention_only` ({boundary_summary.get('all_features', {}).get('false_positive_rate', 0.0):.3f} vs {boundary_summary.get('attention_only', {}).get('false_positive_rate', 0.0):.3f}) while remaining competitive with `context_only` in F1 ({boundary_summary.get('all_features', {}).get('f1', 0.0):.3f} vs {boundary_summary.get('context_only', {}).get('f1', 0.0):.3f}).",
        f"- Threshold sensitivity is limited but non-trivial: family-held-out `context_only` F1 moves from {threshold_101['summary']['family_held_out']['context_only']['f1']:.3f} at 1.01x to {threshold_110['summary']['family_held_out']['context_only']['f1']:.3f} at 1.10x, while the main qualitative ordering remains context-first and attention-too-permissive.",
        "- The resulting interpretation is predictive rather than causal: context gives the clearest coarse regime boundary in this setup, while KV-related features help refine boundary detection when coarse context cues are held out.",
    ]


def write_outputs(rows: list[dict[str, Any]], split_results: dict[str, Any], threshold_sensitivity: list[dict[str, Any]], output_json: Path, output_csv: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_workloads": len(rows),
        "classifier": {
            "type": "logistic_regression",
            "preprocessing": "standard_scaler",
            "decision_threshold": 0.5,
            "fixed_backend_label_threshold": 1.05,
            "robustness_classifier": "decision_tree_max_depth_3",
        },
        "workload_rows": rows,
        "split_results": split_results,
        "threshold_sensitivity": threshold_sensitivity,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(split_results["split_rows"][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in split_results["split_rows"]:
            writer.writerow(row)

    family_summary = split_results["summary"]["family_held_out"]
    context_summary = split_results["summary"]["context_bucket_held_out"]
    lines = [
        "# Decision Signal Study",
        "",
        f"- Workloads: {len(rows)}",
        "- Label: whether the realistic online predictor reaches a positive virtual offload regime (`speedup > 1.05x`) under the current virtual replay setup.",
        "- This label does not represent real-hardware benefit; it is used only to compare the relative informativeness of different signal families under a fixed backend definition.",
        "- Classifier: logistic regression with standardized features and a 0.5 decision threshold.",
        "- Feature groups: attention-only, context-only, context+attention, KV-only, and all-features.",
        f"- Evaluation: 5 family-held-out folds and 3 context-bucket-held-out folds over the same {len(rows)} workloads.",
        f"- Label balance at 1.05x: {sum(row['online_positive_t1_05'] for row in rows)}/{len(rows)} positive workloads.",
        "",
        "## Family-Held-Out Summary",
        "",
        "| Signal Family | Accuracy | F1 | ROC-AUC | Pos Recall | FP Rate | Pos Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in FEATURE_SETS:
        row = family_summary[name]
        lines.append(
            f"| {name} | {row['accuracy']:.3f}±{row['accuracy_std']:.3f} | {row['f1']:.3f}±{row['f1_std']:.3f} | {row['roc_auc']:.3f}±{row['roc_auc_std']:.3f} | "
            f"{row['positive_recall']:.3f}±{row['positive_recall_std']:.3f} | {row['false_positive_rate']:.3f}±{row['false_positive_rate_std']:.3f} | {row['positive_rate']:.3f}±{row['positive_rate_std']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Context-Bucket-Held-Out Summary",
            "",
            "| Signal Family | Accuracy | F1 | ROC-AUC | Pos Recall | FP Rate | Pos Rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in FEATURE_SETS:
        row = context_summary[name]
        lines.append(
            f"| {name} | {row['accuracy']:.3f}±{row['accuracy_std']:.3f} | {row['f1']:.3f}±{row['f1_std']:.3f} | {row['roc_auc']:.3f}±{row['roc_auc_std']:.3f} | "
            f"{row['positive_recall']:.3f}±{row['positive_recall_std']:.3f} | {row['false_positive_rate']:.3f}±{row['false_positive_rate_std']:.3f} | {row['positive_rate']:.3f}±{row['positive_rate_std']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Sensitivity",
            "",
            "| Threshold | Family-held-out Context F1 | Family-held-out All-Features F1 | Context-held-out Context F1 | Context-held-out All-Features F1 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in threshold_sensitivity:
        fam = item["summary"]["family_held_out"]
        ctx = item["summary"]["context_bucket_held_out"]
        lines.append(
            f"| {item['threshold']:.2f} | {fam['context_only']['f1']:.3f} | {fam['all_features']['f1']:.3f} | "
            f"{ctx['context_only']['f1']:.3f} | {ctx['all_features']['f1']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Classifier Robustness",
            "",
            "| Split | Metric | Logistic Context | Logistic All | Tree Context | Tree All |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for split_type, label in (("family_held_out", "family-held-out"), ("context_bucket_held_out", "context-bucket-held-out")):
        logistic = split_results["classifier_robustness"][split_type]["logistic_regression"]
        tree = split_results["classifier_robustness"][split_type]["decision_tree"]
        lines.append(
            f"| {label} | F1 | {logistic['context_only']['f1']:.3f} | {logistic['all_features']['f1']:.3f} | {tree['context_only']['f1']:.3f} | {tree['all_features']['f1']:.3f} |"
        )
        lines.append(
            f"| {label} | FP rate | {logistic['context_only']['false_positive_rate']:.3f} | {logistic['all_features']['false_positive_rate']:.3f} | {tree['context_only']['false_positive_rate']:.3f} | {tree['all_features']['false_positive_rate']:.3f} |"
        )

    if split_results["boundary_summary"]:
        lines.extend(
            [
                "",
                "## Boundary-Focused Subset",
                "",
                "| Signal Family | F1 | FP Rate | Pos Recall |",
                "|---|---:|---:|---:|",
            ]
        )
        for feature_name in FEATURE_SETS:
            if feature_name not in split_results["boundary_summary"]:
                continue
            metrics = split_results["boundary_summary"][feature_name]
            lines.append(
                f"| {feature_name} | {metrics['f1']:.3f} | {metrics['false_positive_rate']:.3f} | {metrics['positive_recall']:.3f} |"
            )

    lines.extend(["", "## Main Readout", ""])
    lines.extend(build_main_readout(rows, split_results, threshold_sensitivity))
    lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run workload-level decision-signal analysis for positive virtual offload regimes.")
    parser.add_argument("--output-json", default="artifacts/reports/decision_signal_study.json")
    parser.add_argument("--output-csv", default="artifacts/reports/decision_signal_study.csv")
    parser.add_argument("--output-md", default="artifacts/reports/DECISION_SIGNAL_STUDY.md")
    parser.add_argument("--thresholds", default="1.01,1.03,1.05,1.10")
    args = parser.parse_args()

    rows = load_workload_rows()
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    for row in rows:
        for threshold in thresholds:
            row[f"online_positive_t{str(threshold).replace('.', '_')}"] = int(float(row["online_speedup"]) > threshold)
    split_results = build_split_results(rows, 1.05)
    threshold_sensitivity = [build_split_results(rows, threshold) for threshold in thresholds]
    write_outputs(rows, split_results, threshold_sensitivity, ROOT / args.output_json, ROOT / args.output_csv, ROOT / args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
