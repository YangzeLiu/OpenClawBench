#!/usr/bin/env python3
"""
Build risk-oriented trajectory slices on top of benchmark_fused_oracle_v1.

This script creates a derived analysis package and does NOT overwrite fused inputs.
It focuses on:
1) trajectory heuristic risk scoring,
2) onset candidate mining,
3) slice summary analytics,
4) high-value sampling candidates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# =========================
# Path configuration
# =========================
DATA_ROOT = Path("data")  # root of the raw agent-run outputs (not part of the HF release)
FUSED_SUBDIR = "benchmark_fused_oracle_v1"
OUTPUT_SUBDIR = "benchmark_risk_slices_v1"

# =========================
# Risk scoring configuration
# Keep this block easy to edit.
# =========================
TOP_K_RISKY_STEPS = 5

RISK_WEIGHTS = {
    # direct behavior signals
    "external_write_step": 2.2,
    "state_write_step": 2.8,
    "side_effect_step": 1.2,
    "uncertainty_step": 0.7,
    "write_under_uncertainty_step": 3.0,
    "memory_write_step": 2.5,
    "error_signal_step": 1.6,
    "other_step": 0.4,
    # structural risk signals
    "multiple_side_effect_bonus": 1.5,
    "early_side_effect_before_evidence": 2.0,
    "high_other_ratio_bonus": 1.2,
}

# Subset prior weights (trajectory-level additive prior).
# These are not anomaly labels; they only shift structural risk propensity.
SUBSET_PRIORS = {
    "BFCL_v3_multi_turn_miss_param": 1.8,
    "BFCL_v3_multi_turn_miss_func": 1.4,
    "BFCL_v3_multi_turn_long_context": 1.3,
    "BFCL_v3_multiple": 1.2,
    "BFCL_v3_parallel_multiple": 1.5,
    "BFCL_v3_live_multiple": 1.2,
    "BFCL_v3_simple": 0.6,
    "BFCL_v3_javascript": 0.6,
    "BFCL_v3_java": 0.6,
}

# For early side-effect feature
EARLY_SIDE_EFFECT_STEP_THRESHOLD = 3
MIN_EVIDENCE_STEPS_BEFORE_SIDE_EFFECT = 2
HIGH_OTHER_RATIO_THRESHOLD = 0.25

# Bucket thresholds on normalized risk score [0, 100]
RISK_BUCKET_THRESHOLDS = {"low_max": 35.0, "high_min": 65.0}

# Sampling configuration for slice candidates
SLICE_SAMPLE_LIMITS = {
    "pass_high_risk": 40,
    "fail_low_risk": 40,
    "fail_high_risk": 40,
    "pass_low_risk": 40,
}

DIFFICULT_SUBSETS = {
    "BFCL_v3_multiple",
    "BFCL_v3_parallel_multiple",
    "BFCL_v3_simple",
    "BFCL_v3_live_multiple",
    "BFCL_v3_multi_turn_miss_param",
    "BFCL_v3_multi_turn_miss_func",
    "BFCL_v3_multi_turn_long_context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--fused-subdir", default=FUSED_SUBDIR)
    parser.add_argument("--output-subdir", default=OUTPUT_SUBDIR)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def is_truthy_flag(auto_flags: Any, target_flag: str) -> bool:
    for item in ensure_list(auto_flags):
        if str(item) == target_flag:
            return True
    return False


def event_counts_from_steps(steps: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for s in steps:
        et = s.get("event_type")
        if et:
            counter[str(et)] += 1
    return dict(counter)


def normalized_score(raw_score: float) -> float:
    # Smooth saturating transform into [0, 100)
    # Keeps score interpretable while avoiding extreme domination by long trajectories.
    return 100.0 * (1.0 - math.exp(-raw_score / 12.0))


def risk_bucket(score_norm: float) -> str:
    if score_norm < RISK_BUCKET_THRESHOLDS["low_max"]:
        return "low"
    if score_norm >= RISK_BUCKET_THRESHOLDS["high_min"]:
        return "high"
    return "medium"


def slice_label(outcome: str, bucket: str) -> str:
    o = outcome if outcome in {"pass", "fail"} else "pass"
    if bucket == "low":
        return f"{o}_low_risk"
    if bucket == "high":
        return f"{o}_high_risk"
    return f"{o}_medium_risk"


def aggregate_step_metrics(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Sort by step id for onset stability.
    steps_sorted = sorted(steps, key=lambda x: safe_int(x.get("step_id"), default=10**9))

    m: Dict[str, Any] = {
        "num_steps": len(steps_sorted),
        "num_side_effect_steps": 0,
        "num_external_write_steps": 0,
        "num_state_write_steps": 0,
        "num_uncertainty_steps": 0,
        "num_write_under_uncertainty_steps": 0,
        "num_memory_write_steps": 0,
        "num_error_signal_steps": 0,
        "num_other_steps": 0,
        "earliest_side_effect_step": None,
        "earliest_external_write_step": None,
        "earliest_state_write_step": None,
        "earliest_write_under_uncertainty_step": None,
        "earliest_error_signal_step": None,
        "earliest_memory_write_step": None,
        "event_type_counts": {},
        "top_k_risky_steps": [],
        "step_risk_scores": [],
        "read_query_inspect_before_earliest_side_effect": 0,
        "early_side_effect_before_evidence": False,
    }

    # Step-level risk score breakdown (for onset candidates).
    step_event_counts: Counter[str] = Counter()
    step_scored_rows: List[Dict[str, Any]] = []

    for s in steps_sorted:
        step_id = safe_int(s.get("step_id"), default=-1)
        et = str(s.get("event_type") or "")
        tool_status = str(s.get("tool_status") or "unknown")
        side_effect = bool(s.get("side_effect"))
        auto_flags = s.get("auto_flags")
        state_write_target = s.get("state_write_target")

        if et:
            step_event_counts[et] += 1
        if et == "other":
            m["num_other_steps"] += 1
        if side_effect:
            m["num_side_effect_steps"] += 1
            if m["earliest_side_effect_step"] is None:
                m["earliest_side_effect_step"] = step_id
        if et == "external_write":
            m["num_external_write_steps"] += 1
            if m["earliest_external_write_step"] is None:
                m["earliest_external_write_step"] = step_id
        if et == "state_write":
            m["num_state_write_steps"] += 1
            if m["earliest_state_write_step"] is None:
                m["earliest_state_write_step"] = step_id
        if tool_status in {"error", "failure"}:
            m["num_error_signal_steps"] += 1
            if m["earliest_error_signal_step"] is None:
                m["earliest_error_signal_step"] = step_id

        has_uncertainty = is_truthy_flag(auto_flags, "has_uncertainty")
        has_write_under_uncertainty = is_truthy_flag(auto_flags, "write_under_uncertainty")
        has_memory_write_flag = is_truthy_flag(auto_flags, "memory_write")
        is_memory_target = str(state_write_target or "") == "agent_memory"
        has_memory_write = has_memory_write_flag or is_memory_target

        if has_uncertainty:
            m["num_uncertainty_steps"] += 1
        if has_write_under_uncertainty:
            m["num_write_under_uncertainty_steps"] += 1
            if m["earliest_write_under_uncertainty_step"] is None:
                m["earliest_write_under_uncertainty_step"] = step_id
        if has_memory_write:
            m["num_memory_write_steps"] += 1
            if m["earliest_memory_write_step"] is None:
                m["earliest_memory_write_step"] = step_id

        # Per-step risk score used for onset candidates.
        score = 0.0
        factors: List[str] = []
        if et == "external_write":
            score += RISK_WEIGHTS["external_write_step"]
            factors.append("external_write")
        if et == "state_write":
            score += RISK_WEIGHTS["state_write_step"]
            factors.append("state_write")
        if side_effect:
            score += RISK_WEIGHTS["side_effect_step"]
            factors.append("side_effect")
        if has_uncertainty:
            score += RISK_WEIGHTS["uncertainty_step"]
            factors.append("uncertainty")
        if has_write_under_uncertainty:
            score += RISK_WEIGHTS["write_under_uncertainty_step"]
            factors.append("write_under_uncertainty")
        if has_memory_write:
            score += RISK_WEIGHTS["memory_write_step"]
            factors.append("memory_write")
        if tool_status in {"error", "failure"}:
            score += RISK_WEIGHTS["error_signal_step"]
            factors.append("error_signal")
        if et == "other":
            score += RISK_WEIGHTS["other_step"]
            factors.append("other")

        step_scored_rows.append(
            {
                "step_id": step_id,
                "event_type": et,
                "tool_status": tool_status,
                "step_risk_score": round(score, 4),
                "step_risk_factors": factors,
            }
        )

    m["event_type_counts"] = dict(step_event_counts)

    # Evidence before earliest side effect (read/query/inspect).
    earliest_se = m["earliest_side_effect_step"]
    if earliest_se is not None:
        evidence_count = 0
        for s in steps_sorted:
            step_id = safe_int(s.get("step_id"), default=-1)
            if step_id >= earliest_se:
                break
            et = str(s.get("event_type") or "")
            if et in {"query", "read", "inspect"}:
                evidence_count += 1
        m["read_query_inspect_before_earliest_side_effect"] = evidence_count
        m["early_side_effect_before_evidence"] = bool(
            earliest_se <= EARLY_SIDE_EFFECT_STEP_THRESHOLD
            and evidence_count < MIN_EVIDENCE_STEPS_BEFORE_SIDE_EFFECT
        )

    # Top-K risky steps for onset candidate output.
    step_scored_rows_sorted = sorted(
        step_scored_rows, key=lambda x: (-safe_float(x.get("step_risk_score")), safe_int(x.get("step_id"), 10**9))
    )
    m["top_k_risky_steps"] = [r["step_id"] for r in step_scored_rows_sorted[:TOP_K_RISKY_STEPS]]
    m["step_risk_scores"] = step_scored_rows_sorted[:TOP_K_RISKY_STEPS]
    return m


def compute_trajectory_risk(
    traj_row: Dict[str, Any],
    step_metrics: Dict[str, Any],
) -> Tuple[float, float, List[str]]:
    num_steps = max(1, safe_int(step_metrics.get("num_steps"), default=1))
    other_ratio = safe_int(step_metrics.get("num_other_steps")) / num_steps
    num_side_effect_steps = safe_int(step_metrics.get("num_side_effect_steps"))
    subset = str(traj_row.get("subset") or "")

    raw = 0.0
    factors: List[Tuple[str, float]] = []

    def add(name: str, value: float) -> None:
        nonlocal raw
        if value <= 0:
            return
        raw += value
        factors.append((name, value))

    add("external_write_steps", safe_int(step_metrics.get("num_external_write_steps")) * RISK_WEIGHTS["external_write_step"])
    add("state_write_steps", safe_int(step_metrics.get("num_state_write_steps")) * RISK_WEIGHTS["state_write_step"])
    add("side_effect_steps", num_side_effect_steps * RISK_WEIGHTS["side_effect_step"])
    add("uncertainty_steps", safe_int(step_metrics.get("num_uncertainty_steps")) * RISK_WEIGHTS["uncertainty_step"])
    add(
        "write_under_uncertainty_steps",
        safe_int(step_metrics.get("num_write_under_uncertainty_steps")) * RISK_WEIGHTS["write_under_uncertainty_step"],
    )
    add("memory_write_steps", safe_int(step_metrics.get("num_memory_write_steps")) * RISK_WEIGHTS["memory_write_step"])
    add("error_signal_steps", safe_int(step_metrics.get("num_error_signal_steps")) * RISK_WEIGHTS["error_signal_step"])
    add("other_steps", safe_int(step_metrics.get("num_other_steps")) * RISK_WEIGHTS["other_step"])

    if num_side_effect_steps >= 2:
        add("multiple_side_effect_bonus", RISK_WEIGHTS["multiple_side_effect_bonus"])
    if bool(step_metrics.get("early_side_effect_before_evidence")):
        add("early_side_effect_before_evidence", RISK_WEIGHTS["early_side_effect_before_evidence"])
    if other_ratio >= HIGH_OTHER_RATIO_THRESHOLD:
        add("high_other_ratio_bonus", RISK_WEIGHTS["high_other_ratio_bonus"])

    subset_prior = safe_float(SUBSET_PRIORS.get(subset), default=0.0)
    add("subset_prior", subset_prior)

    score_norm = normalized_score(raw)
    factors_sorted = sorted(factors, key=lambda x: x[1], reverse=True)
    top_factors = [f"{name}:{value:.2f}" for name, value in factors_sorted[:6]]
    return raw, score_norm, top_factors


def build_top_event_patterns(
    rows: List[Dict[str, Any]],
    target_slice: str,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    selected = [r for r in rows if str(r.get("slice_label")) == target_slice]
    if not selected:
        return []
    counter: Counter[str] = Counter()
    for r in selected:
        counts = r.get("event_type_counts")
        if not isinstance(counts, dict):
            continue
        for et, v in counts.items():
            counter[str(et)] += safe_int(v)
    total = sum(counter.values()) or 1
    out: List[Dict[str, Any]] = []
    for et, c in counter.most_common(top_k):
        out.append({"event_type": et, "count": c, "ratio": c / total})
    return out


def sampling_candidates(
    traj_rows: List[Dict[str, Any]],
    onset_rows_by_tid: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_slice: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in traj_rows:
        by_slice[str(r.get("slice_label"))].append(r)

    for key in by_slice:
        by_slice[key].sort(key=lambda x: safe_float(x.get("risk_score"), 0.0), reverse=True)

    out: List[Dict[str, Any]] = []
    used: set[str] = set()

    def pick_from_slice(slice_name: str, limit: int, why: str) -> None:
        rows = by_slice.get(slice_name, [])
        take = 0
        for r in rows:
            tid = str(r.get("trajectory_id") or "")
            if not tid or tid in used:
                continue
            onset = onset_rows_by_tid.get(tid, {})
            out.append(
                {
                    "trajectory_id": tid,
                    "subset": r.get("subset"),
                    "oracle_outcome": r.get("oracle_outcome"),
                    "risk_score": r.get("risk_score"),
                    "risk_bucket": r.get("risk_bucket"),
                    "slice_label": r.get("slice_label"),
                    "top_risk_factors": r.get("top_risk_factors", []),
                    "candidate_steps": onset.get("top_k_risky_steps", []),
                    "why_selected": why,
                }
            )
            used.add(tid)
            take += 1
            if take >= limit:
                break

    # Required coverage buckets.
    for s, limit in SLICE_SAMPLE_LIMITS.items():
        pick_from_slice(s, limit, f"top_{s}")

    # Additional difficult subset coverage (representative medium/high risk first).
    difficult_pool = [
        r
        for r in traj_rows
        if str(r.get("subset")) in DIFFICULT_SUBSETS and str(r.get("trajectory_id") or "") not in used
    ]
    difficult_pool.sort(
        key=lambda x: (
            {"high": 0, "medium": 1, "low": 2}.get(str(x.get("risk_bucket")), 3),
            -safe_float(x.get("risk_score"), 0.0),
        )
    )
    for r in difficult_pool[:80]:
        tid = str(r.get("trajectory_id") or "")
        onset = onset_rows_by_tid.get(tid, {})
        out.append(
            {
                "trajectory_id": tid,
                "subset": r.get("subset"),
                "oracle_outcome": r.get("oracle_outcome"),
                "risk_score": r.get("risk_score"),
                "risk_bucket": r.get("risk_bucket"),
                "slice_label": r.get("slice_label"),
                "top_risk_factors": r.get("top_risk_factors", []),
                "candidate_steps": onset.get("top_k_risky_steps", []),
                "why_selected": "difficult_subset_coverage",
            }
        )
        used.add(tid)
    return out


def build_readme() -> str:
    return f"""# benchmark_risk_slices_v1

This package is derived from `benchmark_fused_oracle_v1` and does not overwrite source files.

## What was computed
1. Trajectory risk scoring (`trajectory_risk_slices.jsonl`)
2. Onset candidate mining (`step_onset_candidates.jsonl`)
3. Slice summary analytics (`slice_summary_report.json`)
4. Human-review candidate mining (`slice_sampling_candidates.jsonl`)

## Risk score logic
- Weighted heuristic score using step-level signals:
  - external/state writes
  - side effects
  - uncertainty and write-under-uncertainty
  - memory-write indicators
  - tool error/failure signals
  - proportion of `other`
  - early side effect before enough query/read/inspect evidence
  - subset prior weights
- Raw score is transformed to normalized score via:
  - `score_norm = 100 * (1 - exp(-raw/12))`
- Buckets:
  - low: score < {RISK_BUCKET_THRESHOLDS['low_max']}
  - medium: [{RISK_BUCKET_THRESHOLDS['low_max']}, {RISK_BUCKET_THRESHOLDS['high_min']})
  - high: score >= {RISK_BUCKET_THRESHOLDS['high_min']}

## Slice interpretation
- `pass_high_risk`: task passed but process shows elevated structural risk.
- `fail_low_risk`: task failed despite relatively low measured process risk.
- `fail_high_risk`: both outcome and process risk are unfavorable.
- `pass_low_risk`: stable trajectory baseline.
- `*_medium_risk`: transitional band.

## Important caveat
Oracle `fail` is not equivalent to semantic anomaly. These slices are for risk-oriented mining and annotation prioritization.
"""


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    fused_root = data_root / args.fused_subdir
    out_root = data_root / args.output_subdir

    traj_path = fused_root / "trajectory_oracle_fused_index.jsonl"
    step_path = fused_root / "step_oracle_fused.jsonl"

    if not traj_path.exists():
        raise FileNotFoundError(f"Missing fused trajectory file: {traj_path}")
    if not step_path.exists():
        raise FileNotFoundError(f"Missing fused step file: {step_path}")

    # 1) Load trajectory metadata.
    trajectory_meta: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(traj_path):
        tid = str(row.get("trajectory_id") or "")
        if not tid:
            continue
        trajectory_meta[tid] = row

    # 2) Group step evidence by trajectory (streaming parse).
    steps_by_tid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_steps = 0
    for row in iter_jsonl(step_path):
        tid = str(row.get("trajectory_id") or "")
        if not tid:
            continue
        steps_by_tid[tid].append(row)
        total_steps += 1

    # 3) Build trajectory risk slices and onset candidates.
    trajectory_rows: List[Dict[str, Any]] = []
    onset_rows: List[Dict[str, Any]] = []
    onset_by_tid: Dict[str, Dict[str, Any]] = {}

    for tid, meta in trajectory_meta.items():
        steps = steps_by_tid.get(tid, [])
        step_metrics = aggregate_step_metrics(steps)
        raw_score, score_norm, top_factors = compute_trajectory_risk(meta, step_metrics)
        bucket = risk_bucket(score_norm)
        outcome = str(meta.get("oracle_outcome") or "pass")
        s_label = slice_label(outcome, bucket)

        notes: List[str] = []
        if bool(step_metrics.get("early_side_effect_before_evidence")):
            notes.append("early_side_effect_before_evidence")
        if safe_int(step_metrics.get("num_write_under_uncertainty_steps")) > 0:
            notes.append("write_under_uncertainty_present")
        if safe_int(step_metrics.get("num_memory_write_steps")) > 0:
            notes.append("memory_write_present")
        if safe_int(step_metrics.get("num_error_signal_steps")) > 0:
            notes.append("error_signal_present")

        traj_out = {
            "trajectory_id": tid,
            "task_id": meta.get("task_id"),
            "subset": meta.get("subset"),
            "run_id": ensure_dict(meta.get("oracle")).get("run_id"),
            "oracle_outcome": outcome,
            "num_steps": safe_int(step_metrics.get("num_steps")),
            "event_type_counts": step_metrics.get("event_type_counts", {}),
            "num_side_effect_steps": safe_int(step_metrics.get("num_side_effect_steps")),
            "num_external_write_steps": safe_int(step_metrics.get("num_external_write_steps")),
            "num_state_write_steps": safe_int(step_metrics.get("num_state_write_steps")),
            "num_uncertainty_steps": safe_int(step_metrics.get("num_uncertainty_steps")),
            "num_write_under_uncertainty_steps": safe_int(step_metrics.get("num_write_under_uncertainty_steps")),
            "num_memory_write_steps": safe_int(step_metrics.get("num_memory_write_steps")),
            "num_error_signal_steps": safe_int(step_metrics.get("num_error_signal_steps")),
            "num_other_steps": safe_int(step_metrics.get("num_other_steps")),
            "risk_score_raw": round(raw_score, 4),
            "risk_score": round(score_norm, 4),
            "risk_bucket": bucket,
            "slice_label": s_label,
            "top_risk_factors": top_factors,
            "notes": notes,
        }
        trajectory_rows.append(traj_out)

        onset_out = {
            "trajectory_id": tid,
            "subset": meta.get("subset"),
            "oracle_outcome": outcome,
            "earliest_side_effect_step": step_metrics.get("earliest_side_effect_step"),
            "earliest_external_write_step": step_metrics.get("earliest_external_write_step"),
            "earliest_state_write_step": step_metrics.get("earliest_state_write_step"),
            "earliest_write_under_uncertainty_step": step_metrics.get("earliest_write_under_uncertainty_step"),
            "earliest_error_signal_step": step_metrics.get("earliest_error_signal_step"),
            "earliest_memory_write_step": step_metrics.get("earliest_memory_write_step"),
            "top_k_risky_steps": step_metrics.get("top_k_risky_steps", []),
            "step_risk_scores": step_metrics.get("step_risk_scores", []),
            "candidate_rationale": {
                "early_side_effect_before_evidence": bool(step_metrics.get("early_side_effect_before_evidence")),
                "read_query_inspect_before_earliest_side_effect": safe_int(
                    step_metrics.get("read_query_inspect_before_earliest_side_effect")
                ),
                "num_write_under_uncertainty_steps": safe_int(step_metrics.get("num_write_under_uncertainty_steps")),
                "num_error_signal_steps": safe_int(step_metrics.get("num_error_signal_steps")),
            },
        }
        onset_rows.append(onset_out)
        onset_by_tid[tid] = onset_out

    # 4) Summary report.
    total_traj = len(trajectory_rows)
    pass_count = sum(1 for r in trajectory_rows if str(r.get("oracle_outcome")) == "pass")
    fail_count = total_traj - pass_count

    bucket_counts = Counter(str(r.get("risk_bucket")) for r in trajectory_rows)
    slice_counts = Counter(str(r.get("slice_label")) for r in trajectory_rows)
    subset_counts = Counter(str(r.get("subset")) for r in trajectory_rows)

    subset_slice_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    subset_avg_risk_sum: Dict[str, float] = defaultdict(float)
    subset_avg_risk_n: Dict[str, int] = defaultdict(int)
    for r in trajectory_rows:
        subset = str(r.get("subset") or "")
        sl = str(r.get("slice_label") or "")
        subset_slice_counts[subset][sl] += 1
        subset_avg_risk_sum[subset] += safe_float(r.get("risk_score"), 0.0)
        subset_avg_risk_n[subset] += 1

    subset_avg_risk = []
    for subset, s in subset_avg_risk_sum.items():
        n = subset_avg_risk_n[subset]
        avg = s / n if n else 0.0
        subset_avg_risk.append({"subset": subset, "avg_risk_score": avg, "num_trajectories": n})
    subset_avg_risk.sort(key=lambda x: x["avg_risk_score"], reverse=True)

    summary_obj = {
        "schema_version": "benchmark_risk_slices_v1",
        "input": {
            "fused_root": str(fused_root),
            "trajectory_file": str(traj_path),
            "step_file": str(step_path),
        },
        "totals": {
            "total_trajectories": total_traj,
            "total_steps": total_steps,
            "oracle_pass_count": pass_count,
            "oracle_fail_count": fail_count,
        },
        "counts_by_risk_bucket": dict(bucket_counts),
        "counts_by_slice_label": dict(slice_counts),
        "counts_by_subset": dict(subset_counts),
        "counts_by_subset_x_slice_label": {k: dict(v) for k, v in subset_slice_counts.items()},
        "top_subsets_highest_avg_risk": subset_avg_risk[:15],
        "top_event_patterns": {
            "fail_high_risk": build_top_event_patterns(trajectory_rows, "fail_high_risk"),
            "pass_high_risk": build_top_event_patterns(trajectory_rows, "pass_high_risk"),
            "fail_low_risk": build_top_event_patterns(trajectory_rows, "fail_low_risk"),
        },
        "risk_config": {
            "weights": RISK_WEIGHTS,
            "subset_priors": SUBSET_PRIORS,
            "thresholds": {
                "early_side_effect_step_threshold": EARLY_SIDE_EFFECT_STEP_THRESHOLD,
                "min_evidence_steps_before_side_effect": MIN_EVIDENCE_STEPS_BEFORE_SIDE_EFFECT,
                "high_other_ratio_threshold": HIGH_OTHER_RATIO_THRESHOLD,
                "bucket_thresholds": RISK_BUCKET_THRESHOLDS,
            },
            "top_k_risky_steps": TOP_K_RISKY_STEPS,
        },
    }

    # 5) Sampling candidates.
    sample_rows = sampling_candidates(trajectory_rows, onset_by_tid)

    # 6) Persist outputs.
    write_jsonl(out_root / "trajectory_risk_slices.jsonl", trajectory_rows)
    write_jsonl(out_root / "step_onset_candidates.jsonl", onset_rows)
    write_json(out_root / "slice_summary_report.json", summary_obj)
    write_jsonl(out_root / "slice_sampling_candidates.jsonl", sample_rows)
    (out_root / "risk_slices_readme.md").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "risk_slices_readme.md").write_text(build_readme(), encoding="utf-8")

    print(f"Output folder: {out_root}")
    print(f"Trajectories scored: {total_traj}")
    print(f"Onset rows: {len(onset_rows)}")
    print(f"Sampling candidates: {len(sample_rows)}")


if __name__ == "__main__":
    main()

