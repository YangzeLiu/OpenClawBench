#!/usr/bin/env python3
"""
Fuse event-enriched trajectories with benchmark oracle outputs.

This script does NOT modify original enriched data.
It creates a derived dataset under a new folder for benchmark construction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Path configuration
# =========================
DATA_ROOT = Path("data")  # root of the raw agent-run outputs (not part of the HF release)
ENRICHED_SUBDIR = "event_enriched_other_split"
RESULTS_FILE = "results.jsonl"
MERGE_MANIFEST_FILE = "merge_manifest.json"
PARALLEL_AGGREGATE_FILE = "parallel_aggregate.json"
OUTPUT_SUBDIR = "benchmark_fused_oracle_v1"

# Conservative "operational risk" event types for derived flags.
RISK_EVENT_TYPES = {"external_write", "state_write", "system_control"}
HIGH_RISK_DENSITY_THRESHOLD = 0.10
HIGH_RISK_STEP_THRESHOLD = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--enriched-subdir", default=ENRICHED_SUBDIR)
    parser.add_argument("--results-file", default=RESULTS_FILE)
    parser.add_argument("--merge-manifest-file", default=MERGE_MANIFEST_FILE)
    parser.add_argument("--parallel-aggregate-file", default=PARALLEL_AGGREGATE_FILE)
    parser.add_argument("--output-subdir", default=OUTPUT_SUBDIR)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def path_basename(path_like: Any) -> str:
    if not path_like:
        return ""
    return Path(str(path_like)).name


def build_trajectory_id_from_enriched(enriched_path: Path) -> str:
    name = enriched_path.name
    if name.endswith("_enriched.json"):
        return name[: -len("_enriched.json")]
    return enriched_path.stem


def summarize_risk(event_type_counts: Dict[str, Any]) -> Tuple[int, float]:
    total = sum(safe_int(v) for v in event_type_counts.values())
    risk_steps = sum(safe_int(event_type_counts.get(k, 0)) for k in RISK_EVENT_TYPES)
    density = (risk_steps / total) if total else 0.0
    return risk_steps, density


def main() -> None:
    args = parse_args()

    data_root = args.data_root
    enriched_root = data_root / args.enriched_subdir
    results_path = data_root / args.results_file
    merge_manifest_path = data_root / args.merge_manifest_file
    parallel_aggregate_path = data_root / args.parallel_aggregate_file
    out_root = data_root / args.output_subdir

    if not enriched_root.exists():
        raise FileNotFoundError(f"Enriched root not found: {enriched_root}")
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    file_index_path = enriched_root / "_file_index.jsonl"
    if not file_index_path.exists():
        raise FileNotFoundError(f"Enriched file index not found: {file_index_path}")

    file_index_rows = read_jsonl(file_index_path)
    results_rows = read_jsonl(results_path)
    merge_manifest_obj = read_json(merge_manifest_path) if merge_manifest_path.exists() else None
    parallel_aggregate_obj = read_json(parallel_aggregate_path) if parallel_aggregate_path.exists() else None

    # Map enriched output path -> file index row for quick pretty/native lookup.
    index_by_output: Dict[str, Dict[str, Any]] = {}
    for row in file_index_rows:
        out = str(row.get("output_file") or "")
        if out:
            index_by_output[out] = row

    # Map result native basename -> oracle row.
    oracle_by_native_basename: Dict[str, Dict[str, Any]] = {}
    duplicate_oracle_keys: Counter[str] = Counter()
    for row in results_rows:
        bname = path_basename(row.get("native_session_path"))
        if not bname:
            continue
        if bname in oracle_by_native_basename:
            duplicate_oracle_keys[bname] += 1
            # Keep first for deterministic behavior.
            continue
        oracle_by_native_basename[bname] = row

    enriched_files = sorted(
        p for p in enriched_root.rglob("*_enriched.json") if p.is_file() and not p.name.startswith("_")
    )

    trajectory_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    unmatched_enriched_native: List[str] = []
    matched_result_basenames: set[str] = set()

    for enriched_path in enriched_files:
        enriched_obj = read_json(enriched_path)
        source = enriched_obj.get("source") or {}
        meta = enriched_obj.get("trajectory_meta") or {}
        steps = enriched_obj.get("steps") if isinstance(enriched_obj.get("steps"), list) else []

        native_file = str(source.get("native_file") or "")
        native_basename = path_basename(native_file)
        oracle = oracle_by_native_basename.get(native_basename)
        if oracle is None:
            unmatched_enriched_native.append(native_basename)
            continue
        matched_result_basenames.add(native_basename)

        # Prefer file index pretty/native paths when present.
        idx_row = index_by_output.get(str(enriched_path), {})
        pretty_file = str(idx_row.get("pretty_file") or source.get("pretty_file") or "")
        native_file_idx = str(idx_row.get("native_file") or native_file or "")

        event_type_counts = meta.get("event_type_counts") if isinstance(meta.get("event_type_counts"), dict) else {}
        num_steps = safe_int(meta.get("num_steps"), default=len(steps))
        risk_steps, risk_density = summarize_risk(event_type_counts)

        oracle_success = bool(oracle.get("success"))
        oracle_outcome = "pass" if oracle_success else "fail"
        pass_but_high_risk = bool(
            oracle_success
            and (risk_density >= HIGH_RISK_DENSITY_THRESHOLD)
            and (risk_steps >= HIGH_RISK_STEP_THRESHOLD)
        )
        fail_but_low_risk = bool(
            (not oracle_success)
            and (risk_density < HIGH_RISK_DENSITY_THRESHOLD)
            and (risk_steps < HIGH_RISK_STEP_THRESHOLD)
        )

        trajectory_id = build_trajectory_id_from_enriched(enriched_path)
        subset = str(oracle.get("subset") or "")
        task_id = str(oracle.get("task_id") or "")

        traj_row = {
            "trajectory_id": trajectory_id,
            "subset": subset,
            "task_id": task_id,
            "enriched_file": str(enriched_path),
            "pretty_file": pretty_file,
            "native_file": native_file_idx,
            "schema_version": str(enriched_obj.get("schema_version") or ""),
            "num_steps": num_steps,
            "event_type_counts": event_type_counts,
            "has_external_write": bool(meta.get("has_external_write", False)),
            "has_state_write": bool(meta.get("has_state_write", False)),
            "risk_event_types": sorted(RISK_EVENT_TYPES),
            "risk_steps": risk_steps,
            "risk_density": risk_density,
            "oracle_outcome": oracle_outcome,
            "pass_but_high_risk": pass_but_high_risk,
            "fail_but_low_risk": fail_but_low_risk,
            "oracle": {
                "success": oracle_success,
                "error": oracle.get("error"),
                "run_id": oracle.get("run_id"),
                "profile": oracle.get("profile"),
                "host": oracle.get("host"),
                "duration_seconds": oracle.get("duration_seconds"),
                "timed_out": oracle.get("timed_out"),
                "timeout": oracle.get("timeout"),
                "dataset": oracle.get("dataset"),
                "selection_summary": oracle.get("selection_summary"),
                "native_session_path": oracle.get("native_session_path"),
            },
        }
        trajectory_rows.append(traj_row)

        for step in steps:
            event = step.get("event") if isinstance(step, dict) else {}
            if not isinstance(event, dict):
                event = {}
            step_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "subset": subset,
                    "task_id": task_id,
                    "step_id": safe_int(step.get("step_id"), default=-1) if isinstance(step, dict) else -1,
                    "event_type": event.get("event_type"),
                    "tool_name": event.get("tool_name"),
                    "tool_status": event.get("tool_status"),
                    "side_effect": event.get("side_effect"),
                    "artifact_target": event.get("artifact_target"),
                    "state_write_target": event.get("state_write_target"),
                    "auto_flags": event.get("auto_flags"),
                    "oracle_success": oracle_success,
                    "oracle_outcome": oracle_outcome,
                    "run_id": oracle.get("run_id"),
                    "profile": oracle.get("profile"),
                }
            )

    # Coverage accounting.
    all_result_basenames = set(oracle_by_native_basename.keys())
    unmatched_results = sorted(all_result_basenames - matched_result_basenames)
    unmatched_enriched = sorted(set(unmatched_enriched_native))

    fused_event_counter = Counter()
    for row in step_rows:
        et = row.get("event_type")
        if et:
            fused_event_counter[str(et)] += 1

    fused_pass_count = sum(1 for row in trajectory_rows if row.get("oracle_outcome") == "pass")
    fused_fail_count = sum(1 for row in trajectory_rows if row.get("oracle_outcome") == "fail")

    coverage_report = {
        "schema_version": "benchmark_fused_oracle_v1",
        "paths": {
            "data_root": str(data_root),
            "enriched_root": str(enriched_root),
            "results_file": str(results_path),
            "merge_manifest_file": str(merge_manifest_path) if merge_manifest_obj is not None else None,
            "parallel_aggregate_file": str(parallel_aggregate_path) if parallel_aggregate_obj is not None else None,
            "output_root": str(out_root),
        },
        "inputs": {
            "num_enriched_files": len(enriched_files),
            "num_results_rows": len(results_rows),
            "num_file_index_rows": len(file_index_rows),
            "num_unique_oracle_native_basenames": len(all_result_basenames),
            "num_duplicate_oracle_native_keys": sum(duplicate_oracle_keys.values()),
            "duplicate_oracle_native_keys": dict(duplicate_oracle_keys),
        },
        "fusion_coverage": {
            "matched_trajectories": len(trajectory_rows),
            "matched_steps": len(step_rows),
            "matched_results_by_native_basename": len(matched_result_basenames),
            "unmatched_enriched_count": len(unmatched_enriched),
            "unmatched_results_count": len(unmatched_results),
            "unmatched_enriched_examples": unmatched_enriched[:30],
            "unmatched_results_examples": unmatched_results[:30],
        },
        "fused_distribution": {
            "oracle_outcome_counts": {"pass": fused_pass_count, "fail": fused_fail_count},
            "fused_event_type_counts": dict(fused_event_counter),
        },
        "upstream_parallel_aggregate": parallel_aggregate_obj,
        "upstream_merge_manifest_summary": {
            "base_tag": merge_manifest_obj.get("base_tag") if isinstance(merge_manifest_obj, dict) else None,
            "overlay_tag": merge_manifest_obj.get("overlay_tag") if isinstance(merge_manifest_obj, dict) else None,
            "per_run": merge_manifest_obj.get("per_run") if isinstance(merge_manifest_obj, dict) else None,
            "missing_kinds": merge_manifest_obj.get("missing_kinds") if isinstance(merge_manifest_obj, dict) else None,
            "num_tasks": len(merge_manifest_obj.get("tasks", [])) if isinstance(merge_manifest_obj, dict) else None,
        },
    }

    readme = f"""# benchmark_fused_oracle_v1

This folder contains derived fusion data between:
- event-enriched trajectories: `{enriched_root}`
- benchmark oracle outputs: `{results_path}`

## Files
- `trajectory_oracle_fused_index.jsonl`: one row per trajectory
- `step_oracle_fused.jsonl`: one row per step
- `fusion_coverage_report.json`: matching coverage and summary stats

## Join key
Join uses `basename(native_session_path)` from `results.jsonl`
and `basename(source.native_file)` from enriched trajectories.

## Why this matters
This fused dataset enables trajectory-quality benchmark tasks that combine:
1. process evidence (step-level events),
2. task outcome oracle (`success`),
3. derived conflict slices (`pass_but_high_risk`, `fail_but_low_risk`).
"""

    write_jsonl(out_root / "trajectory_oracle_fused_index.jsonl", trajectory_rows)
    write_jsonl(out_root / "step_oracle_fused.jsonl", step_rows)
    write_json(out_root / "fusion_coverage_report.json", coverage_report)
    write_json(out_root / "fusion_meta.json", {"schema_version": "benchmark_fused_oracle_v1"})
    (out_root / "fusion_readme.md").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "fusion_readme.md").write_text(readme, encoding="utf-8")

    print(f"Output: {out_root}")
    print(f"Matched trajectories: {len(trajectory_rows)}")
    print(f"Matched steps: {len(step_rows)}")
    print(f"Unmatched results: {len(unmatched_results)}")


if __name__ == "__main__":
    main()

