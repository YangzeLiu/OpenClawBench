#!/usr/bin/env python3
"""
Build LLM-ready trajectory annotation input packets from fused + risk outputs.

Inputs:
1) trajectory_oracle_fused_index.jsonl
2) step_oracle_fused.jsonl
3) trajectory_risk_slices.jsonl
4) step_onset_candidates.jsonl

Outputs:
1) annotation_input_packets.jsonl
2) input_build_report.json
3) sampling_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =========================
# Path configuration
# =========================
DATA_ROOT = Path("data")  # root of the raw agent-run outputs (not part of the HF release)
FUSED_SUBDIR = "benchmark_fused_oracle_v1"
RISK_SUBDIR = "benchmark_risk_slices_v1"
OUTPUT_SUBDIR = "5_llm_annotation_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fused-subdir", default=FUSED_SUBDIR)
    parser.add_argument("--risk-subdir", default=RISK_SUBDIR)
    parser.add_argument("--output-subdir", default=OUTPUT_SUBDIR)
    parser.add_argument("--sampling-mode", choices=["all", "stratified"], default="all")
    parser.add_argument("--target-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no truncation.")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        return "\n".join(filter(None, (to_text(i) for i in x)))
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            vv = to_text(v)
            if vv:
                parts.append(f"{k}: {vv}")
        return "\n".join(parts)
    return str(x)


def get_trajectory_id(row: Dict[str, Any]) -> Optional[str]:
    for k in ("trajectory_id", "source_trajectory_id", "id"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def get_step_id(step: Dict[str, Any], fallback_idx: int) -> str:
    for k in ("step_id", "id"):
        v = step.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, int):
            return str(v)
    return str(fallback_idx)


def extract_step_fields(step_row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    event = step_row.get("event") if isinstance(step_row.get("event"), dict) else {}
    triplet = step_row.get("triplet") if isinstance(step_row.get("triplet"), dict) else {}
    action = triplet.get("action", step_row.get("action"))
    reflection = triplet.get("reflection", step_row.get("reflection"))
    thinking = triplet.get("thinking", step_row.get("thinking"))
    auto_flags = event.get("auto_flags", step_row.get("auto_flags", []))
    if not isinstance(auto_flags, list):
        auto_flags = [to_text(auto_flags)] if auto_flags else []

    return {
        "step_id": get_step_id(step_row, idx),
        "thinking": to_text(thinking),
        "action_text": to_text(action),
        "reflection_text": to_text(reflection),
        "event_type": to_text(event.get("event_type", step_row.get("event_type"))),
        "side_effect": to_text(event.get("side_effect", step_row.get("side_effect"))),
        "artifact_target": to_text(event.get("artifact_target", step_row.get("artifact_target"))),
        "state_write_target": to_text(
            event.get("state_write_target", step_row.get("state_write_target"))
        ),
        "auto_flags": auto_flags,
    }


def step_sort_key(step: Dict[str, Any]) -> Tuple[int, str]:
    sid = step.get("step_id", "")
    if isinstance(sid, str) and sid.isdigit():
        return (0, f"{int(sid):08d}")
    return (1, str(sid))


def stratified_sample(
    rows: List[Dict[str, Any]], target_size: int, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    by_slice: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = r.get("slice_label") or "unknown"
        by_slice[key].append(r)

    for v in by_slice.values():
        rng.shuffle(v)

    labels = sorted(by_slice.keys())
    total = len(rows)
    picked: List[Dict[str, Any]] = []
    per_label: Dict[str, int] = {}

    if total <= target_size:
        return list(rows), {k: len(v) for k, v in by_slice.items()}

    # proportional allocation with ceil, then clip
    alloc: Dict[str, int] = {}
    used = 0
    for lb in labels:
        n = len(by_slice[lb])
        want = math.ceil(target_size * (n / total))
        alloc[lb] = min(want, n)
        used += alloc[lb]

    # adjust down if overshoot
    while used > target_size:
        for lb in labels:
            if used <= target_size:
                break
            if alloc[lb] > 0:
                alloc[lb] -= 1
                used -= 1

    # fill up if undershoot
    while used < target_size:
        changed = False
        for lb in labels:
            cap = len(by_slice[lb])
            if alloc[lb] < cap and used < target_size:
                alloc[lb] += 1
                used += 1
                changed = True
        if not changed:
            break

    for lb in labels:
        chunk = by_slice[lb][: alloc[lb]]
        picked.extend(chunk)
        per_label[lb] = len(chunk)

    rng.shuffle(picked)
    return picked, per_label


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    fused_root = (data_root / args.fused_subdir).resolve()
    risk_root = (data_root / args.risk_subdir).resolve()
    out_root = (data_root / args.output_subdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    traj_file = fused_root / "trajectory_oracle_fused_index.jsonl"
    step_file = fused_root / "step_oracle_fused.jsonl"
    traj_risk_file = risk_root / "trajectory_risk_slices.jsonl"
    onset_file = risk_root / "step_onset_candidates.jsonl"

    for p in (traj_file, step_file, traj_risk_file, onset_file):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input file: {p}")

    traj_meta: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(traj_file):
        tid = get_trajectory_id(row)
        if not tid:
            continue
        traj_meta[tid] = row

    steps_by_tid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(iter_jsonl(step_file), start=1):
        tid = get_trajectory_id(row)
        if not tid:
            continue
        steps_by_tid[tid].append(extract_step_fields(row, i))

    traj_risk: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(traj_risk_file):
        tid = get_trajectory_id(row)
        if tid:
            traj_risk[tid] = row

    onset_by_tid: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(onset_file):
        tid = get_trajectory_id(row)
        if tid:
            onset_by_tid[tid] = row

    packets: List[Dict[str, Any]] = []
    missing_risk = 0
    missing_onset = 0

    all_tids = sorted(set(traj_meta.keys()) | set(steps_by_tid.keys()))
    for tid in all_tids:
        tmeta = traj_meta.get(tid, {})
        rmeta = traj_risk.get(tid, {})
        ometa = onset_by_tid.get(tid, {})
        if not rmeta:
            missing_risk += 1
        if not ometa:
            missing_onset += 1

        steps = list(steps_by_tid.get(tid, []))
        steps.sort(key=step_sort_key)
        if args.max_steps and args.max_steps > 0:
            steps = steps[: args.max_steps]

        event_counts = (
            rmeta.get("event_type_counts")
            if isinstance(rmeta.get("event_type_counts"), dict)
            else {}
        )
        if not event_counts:
            c = Counter(s.get("event_type") or "unknown" for s in steps)
            event_counts = dict(c)

        packet = {
            "trajectory_id": tid,
            "subset": tmeta.get("subset", rmeta.get("subset", "")),
            "oracle_outcome": tmeta.get("oracle_outcome", rmeta.get("oracle_outcome", "")),
            "risk_bucket": rmeta.get("risk_bucket", ""),
            "risk_score": rmeta.get("risk_score", 0.0),
            "slice_label": rmeta.get("slice_label", ""),
            "onset_candidates": {
                "earliest_side_effect_step": ometa.get("earliest_side_effect_step"),
                "earliest_external_write_step": ometa.get("earliest_external_write_step"),
                "earliest_state_write_step": ometa.get("earliest_state_write_step"),
                "earliest_write_under_uncertainty_step": ometa.get(
                    "earliest_write_under_uncertainty_step"
                ),
                "earliest_error_signal_step": ometa.get("earliest_error_signal_step"),
                "top_k_risky_steps": ometa.get("top_k_risky_steps", []),
            },
            "num_steps": len(steps),
            "event_type_counts": event_counts,
            "steps": steps,
            "annotation_schema_target": {
                "trajectory_is_anomalous": "boolean",
                "dominant_episode_exists": "boolean",
                "segment_spans": [
                    {
                        "label": "normal|evolution|anomalous",
                        "start_step_id": "string",
                        "end_step_id": "string",
                    }
                ],
                "earliest_recoverable_onset": "string|null",
                "earliest_harmful_onset": "string|null",
                "earliest_state_contamination_onset": "string|null",
                "anomaly_type_l1": "string|null",
                "anomaly_type_l2": "string|null",
                "recoverability": "high|medium|low|none|null",
                "severity": "low|medium|high|critical|null",
                "evidence_steps": ["string"],
                "annotator_rationale": "string",
            },
        }
        packets.append(packet)

    selected = packets
    sampling_detail: Dict[str, int] = {}
    if args.sampling_mode == "stratified":
        selected, sampling_detail = stratified_sample(
            packets, target_size=args.target_size, seed=args.seed
        )
    elif args.sampling_mode == "all":
        sampling_detail = dict(
            Counter((r.get("slice_label") or "unknown") for r in packets)
        )

    manifest_rows = []
    for i, p in enumerate(selected, start=1):
        manifest_rows.append(
            {
                "row_id": i,
                "trajectory_id": p.get("trajectory_id"),
                "subset": p.get("subset"),
                "oracle_outcome": p.get("oracle_outcome"),
                "risk_bucket": p.get("risk_bucket"),
                "slice_label": p.get("slice_label"),
                "num_steps": p.get("num_steps"),
            }
        )

    report = {
        "schema_version": "llm_annotation_inputs_v1",
        "paths": {
            "data_root": str(data_root),
            "fused_root": str(fused_root),
            "risk_root": str(risk_root),
            "output_root": str(out_root),
        },
        "inputs": {
            "trajectory_rows": len(traj_meta),
            "step_rows_total": sum(len(v) for v in steps_by_tid.values()),
            "risk_rows": len(traj_risk),
            "onset_rows": len(onset_by_tid),
        },
        "build": {
            "packets_total": len(packets),
            "packets_selected": len(selected),
            "sampling_mode": args.sampling_mode,
            "target_size": args.target_size,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "sampling_detail_by_slice_label": sampling_detail,
            "missing_risk_rows": missing_risk,
            "missing_onset_rows": missing_onset,
        },
        "selected_distribution": {
            "slice_label_counts": dict(Counter(p.get("slice_label", "") for p in selected)),
            "risk_bucket_counts": dict(Counter(p.get("risk_bucket", "") for p in selected)),
            "oracle_outcome_counts": dict(Counter(p.get("oracle_outcome", "") for p in selected)),
            "subset_counts_top20": dict(
                Counter(p.get("subset", "") for p in selected).most_common(20)
            ),
        },
    }

    write_jsonl(out_root / "annotation_input_packets.jsonl", selected)
    write_jsonl(out_root / "sampling_manifest.jsonl", manifest_rows)
    write_json(out_root / "input_build_report.json", report)

    print(f"Output dir: {out_root}")
    print(f"Packets selected: {len(selected)} / total {len(packets)}")


if __name__ == "__main__":
    main()

