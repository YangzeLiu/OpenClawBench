#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

DATA_ROOT_DEFAULT = Path("data")
OUTPUT_ROOT_DEFAULT = Path("data/annotation_results")
INPUT_SUBDIR_DEFAULT = "5_llm_annotation_inputs_all"
PROD_SUBDIR_DEFAULT = "silver_labels"
JUDGE_MODEL_DEFAULT = "deepseek-v4-pro"

BASE_RUNNER = Path(__file__).resolve().parent / "run_fulltax_pipeline_core.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["run", "merge"], default="run")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--dataset", choices=["BFCL", "GAIA"], required=True)
    p.add_argument("--source-model", default="")
    p.add_argument("--input-subdir", default=INPUT_SUBDIR_DEFAULT)
    p.add_argument("--prod-subdir", default=PROD_SUBDIR_DEFAULT)
    p.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--compact-mode", action="store_true")
    p.add_argument("--dynamic-max-tokens", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--include-accepted-repaired-in-main", action="store_true")
    p.add_argument("--annotation-version", default="fulltax_v1")

    # shard control
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)

    # pass-through api env name
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    return p.parse_args()


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_models(dataset_root: Path, source_model: str) -> List[Path]:
    if source_model:
        p = dataset_root / source_model
        return [p] if p.is_dir() else []
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def shard_match(i: int, shard_id: int, num_shards: int) -> bool:
    n = max(1, int(num_shards))
    sid = max(0, int(shard_id))
    if sid >= n:
        sid = sid % n
    return (i % n) == sid


def build_shard_manifest(input_packets: Path, out_manifest: Path, shard_id: int, num_shards: int) -> int:
    rows = []
    for i, r in enumerate(iter_jsonl(input_packets)):
        if not shard_match(i, shard_id, num_shards):
            continue
        tid = r.get("trajectory_id")
        if not tid:
            continue
        rows.append({"trajectory_id": tid})
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def call_base_runner(args: argparse.Namespace, model_name: str, shard_manifest: Path) -> None:
    cmd = [
        sys.executable,
        str(BASE_RUNNER),
        "--dataset", args.dataset,
        "--source-model", model_name,
        "--input-subdir", args.input_subdir,
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--id-manifest", str(shard_manifest),
        "--workers", str(args.workers),
        "--temperature", str(args.temperature),
        "--judge-model", args.judge_model,
    ]
    if args.compact_mode:
        cmd.append("--compact-mode")
    if args.dynamic_max_tokens:
        cmd.append("--dynamic-max-tokens")
    if args.resume:
        cmd.append("--resume")

    env = os.environ.copy()
    if not env.get(args.api_key_env, "").strip():
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def assign_quality_tier(rec: Dict[str, Any]) -> str:
    if rec.get("needs_review") or rec.get("truncated"):
        return "review_or_unresolved"
    was_norm = bool(rec.get("b1_normalized", False))
    was_fix = bool(rec.get("revise_inline_applied", False))
    if was_norm or was_fix:
        return "accepted_repaired"
    return "high_confidence_accepted"


def detect_review_cause(rec: Dict[str, Any]) -> str:
    if rec.get("truncated"):
        return "truncation"
    val = rec.get("validation", {}) if isinstance(rec.get("validation"), dict) else {}
    issues = [str(x) for x in val.get("issues", [])]
    if any("parse_failed" in x for x in issues):
        return "parse_failure"
    if any("missing_b2_field" in x or "missing_b1_field" in x for x in issues):
        return "schema_missing_field"
    if any("segment span" in x or "segment_spans" in x or "invalid span label" in x for x in issues):
        return "invalid_span_structure"
    if any("unsupported_onset:" in x for x in issues):
        only = all(x.startswith("unsupported_onset:") for x in issues)
        return "unsupported_onset_only" if only else "unsupported_onset_plus_other"
    if any("contradiction" in x for x in issues):
        return "subtype_field_contradiction"
    return "other"


def build_prod_tables(
    *,
    args: argparse.Namespace,
    model_name: str,
    shard_tag: str,
    shard_ids: Set[str],
    input_packets: Path,
) -> Dict[str, Any]:
    src_safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-").lower()
    judge_safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.judge_model).strip("-").lower()

    base_out = (args.output_root / args.dataset)
    final_f = base_out / f"final_annotations_fulltax_v1__src-{src_safe}__judge-{judge_safe}.jsonl"
    if not final_f.exists():
        raise RuntimeError(f"Missing base runner output: {final_f}")

    packet_meta: Dict[str, Dict[str, Any]] = {}
    for r in iter_jsonl(input_packets):
        tid = str(r.get("trajectory_id", ""))
        if tid:
            packet_meta[tid] = {
                "subset": r.get("subset"),
                "oracle_outcome": r.get("oracle_outcome"),
                "risk_bucket": r.get("risk_bucket"),
            }

    model_out = args.output_root / args.dataset / args.prod_subdir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    master_f = model_out / f"master_table__src-{src_safe}__{shard_tag}.jsonl"
    pos_f = model_out / f"positives_structured__src-{src_safe}__{shard_tag}.jsonl"
    acc_main_f = model_out / f"accepted_main__src-{src_safe}__{shard_tag}.jsonl"
    acc_rep_f = model_out / f"accepted_repaired__src-{src_safe}__{shard_tag}.jsonl"
    review_split_f = model_out / f"review_or_unresolved__src-{src_safe}__{shard_tag}.jsonl"
    review_q_f = model_out / f"review_queue__src-{src_safe}__{shard_tag}.jsonl"
    report_f = model_out / f"report__src-{src_safe}__{shard_tag}.json"

    for p in [master_f, pos_f, acc_main_f, acc_rep_f, review_split_f, review_q_f]:
        if p.exists():
            p.unlink()

    stage_counts = Counter()
    tier_counts = Counter()
    subtype_counts = Counter()
    review_causes = Counter()

    scheduled = 0
    for rec in iter_jsonl(final_f):
        tid = str(rec.get("trajectory_id", ""))
        if not tid or tid not in shard_ids:
            continue
        scheduled += 1

        ann = rec.get("annotation", {}) if isinstance(rec.get("annotation"), dict) else {}
        meta = packet_meta.get(tid, {})
        tier = assign_quality_tier(rec)
        cause = detect_review_cause(rec) if tier == "review_or_unresolved" else None

        master_row = {
            "trajectory_id": tid,
            "subset": meta.get("subset"),
            "source_model": model_name,
            "oracle_outcome": meta.get("oracle_outcome"),
            "risk_bucket": meta.get("risk_bucket"),
            "route_stage_a": rec.get("route_stage_a"),
            "candidate_subtypes_stage_a": rec.get("candidate_subtypes_stage_a", []),
            "trajectory_is_anomalous": ann.get("trajectory_is_anomalous"),
            "anomaly_type_l1": ann.get("anomaly_type_l1"),
            "anomaly_type_l2": ann.get("anomaly_type_l2"),
            "quality_tier": tier,
            "was_normalized": bool(rec.get("b1_normalized", False)),
            "revise_inline_applied": bool(rec.get("revise_inline_applied", False)),
            "needs_review": bool(rec.get("needs_review", False)),
            "final_status": rec.get("final_status"),
            "evidence_steps": ann.get("evidence_steps", []),
            "structure_fields_present": bool(ann.get("segment_spans")) and ("recoverability" in ann) and ("severity" in ann),
            "audit_priority": "highest" if tier == "review_or_unresolved" else ("high" if tier == "accepted_repaired" else "medium"),
            "review_cause": cause,
            "annotation_version": args.annotation_version,
        }
        append_jsonl(master_f, master_row)
        tier_counts[tier] += 1

        l2 = ann.get("anomaly_type_l2")
        if isinstance(l2, str):
            subtype_counts[l2] += 1

        if tier == "high_confidence_accepted":
            append_jsonl(acc_main_f, master_row)
        elif tier == "accepted_repaired":
            append_jsonl(acc_rep_f, master_row)
            if args.include_accepted_repaired_in_main:
                append_jsonl(acc_main_f, master_row)
        else:
            append_jsonl(review_split_f, master_row)
            append_jsonl(review_q_f, {
                "trajectory_id": tid,
                "review_cause": cause,
                "validation": rec.get("validation", {}),
                "annotation": ann,
            })
            review_causes[cause or "other"] += 1

        if tier in {"high_confidence_accepted", "accepted_repaired"} and ann.get("trajectory_is_anomalous") is True and isinstance(l2, str):
            append_jsonl(pos_f, {
                "trajectory_id": tid,
                "anomaly_type_l1": ann.get("anomaly_type_l1"),
                "anomaly_type_l2": l2,
                "segment_spans": ann.get("segment_spans", []),
                "earliest_recoverable_onset": ann.get("earliest_recoverable_onset"),
                "earliest_harmful_onset": ann.get("earliest_harmful_onset"),
                "earliest_state_contamination_onset": ann.get("earliest_state_contamination_onset"),
                "recoverability": ann.get("recoverability"),
                "severity": ann.get("severity"),
                "annotator_rationale_refined": ann.get("annotator_rationale_refined"),
                "quality_tier": tier,
            })

        stage_counts["rows"] += 1
        stage_counts[f"route_{rec.get('route_stage_a')}"] += 1

    report = {
        "schema_version": "annotation_runner_fulltax_v1_report",
        "dataset": args.dataset,
        "source_model": model_name,
        "judge_model": args.judge_model,
        "shard": {"shard_id": args.shard_id, "num_shards": args.num_shards},
        "stats": {
            "scheduled": scheduled,
            "processed": int(stage_counts["rows"]),
            "quality_tier_counts": dict(tier_counts),
            "subtype_counts": dict(subtype_counts),
            "review_cause_counts": dict(review_causes),
        },
        "outputs": {
            "master_table_shard": str(master_f),
            "positives_structured_shard": str(pos_f),
            "accepted_main_shard": str(acc_main_f),
            "accepted_repaired_shard": str(acc_rep_f),
            "review_or_unresolved_shard": str(review_split_f),
            "review_queue_shard": str(review_q_f),
        },
    }
    report_f.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_mode(args: argparse.Namespace) -> None:
    dataset_root = (args.data_root / args.dataset).resolve()
    models = collect_models(dataset_root, args.source_model)
    if not models:
        raise RuntimeError(f"No model folders found under {dataset_root}")

    for m in models:
        model = m.name
        input_packets = m / args.input_subdir / "annotation_input_packets.jsonl"
        if not input_packets.exists():
            print(f"[SKIP] missing input packets: {input_packets}")
            continue

        model_out = args.output_root / args.dataset / args.prod_subdir / model
        model_out.mkdir(parents=True, exist_ok=True)
        shard_tag = f"shard-{args.shard_id:03d}-of-{max(1,args.num_shards):03d}"
        shard_manifest = model_out / f"id_manifest__{shard_tag}.jsonl"

        n_manifest = build_shard_manifest(input_packets, shard_manifest, args.shard_id, args.num_shards)
        print(f"[INFO] {args.dataset}/{model} shard manifest rows={n_manifest}: {shard_manifest}")
        if n_manifest == 0:
            print(f"[SKIP] empty shard for {model}")
            continue

        call_base_runner(args, model, shard_manifest)

        shard_ids = {str(r.get("trajectory_id")) for r in iter_jsonl(shard_manifest) if r.get("trajectory_id")}
        report = build_prod_tables(
            args=args,
            model_name=model,
            shard_tag=shard_tag,
            shard_ids=shard_ids,
            input_packets=input_packets,
        )
        print(f"[DONE] {args.dataset}/{model}/{shard_tag} => processed={report['stats']['processed']}")


def merge_one_model(out_dir: Path, model_name: str, include_repaired_in_main: bool) -> None:
    src_safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-").lower()
    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    master_all = model_out / "master_table_all.jsonl"
    pos_all = model_out / "positives_structured_all.jsonl"
    acc_main_all = model_out / "accepted_main_all.jsonl"
    acc_rep_all = model_out / "accepted_repaired_all.jsonl"
    review_all = model_out / "review_or_unresolved_all.jsonl"
    summary_f = model_out / "dataset_summary_all.json"

    for p in [master_all, pos_all, acc_main_all, acc_rep_all, review_all]:
        if p.exists():
            p.unlink()

    qcnt = Counter()
    scnt = Counter()
    rcnt = Counter()

    master_shards = sorted(model_out.glob(f"master_table__src-{src_safe}__shard-*.jsonl"))
    pos_shards = sorted(model_out.glob(f"positives_structured__src-{src_safe}__shard-*.jsonl"))

    for f in master_shards:
        for r in iter_jsonl(f):
            append_jsonl(master_all, r)
            tier = str(r.get("quality_tier", ""))
            qcnt[tier] += 1
            l2 = r.get("anomaly_type_l2")
            if isinstance(l2, str):
                scnt[l2] += 1
            if tier == "high_confidence_accepted":
                append_jsonl(acc_main_all, r)
            elif tier == "accepted_repaired":
                append_jsonl(acc_rep_all, r)
                if include_repaired_in_main:
                    append_jsonl(acc_main_all, r)
            else:
                append_jsonl(review_all, r)
                rc = str(r.get("review_cause") or "other")
                rcnt[rc] += 1

    for f in pos_shards:
        for r in iter_jsonl(f):
            append_jsonl(pos_all, r)

    summary = {
        "schema_version": "fulltax_v1_merge_summary",
        "source_model": model_name,
        "totals": {
            "master": sum(1 for _ in iter_jsonl(master_all)) if master_all.exists() else 0,
            "positives": sum(1 for _ in iter_jsonl(pos_all)) if pos_all.exists() else 0,
            "accepted_main": sum(1 for _ in iter_jsonl(acc_main_all)) if acc_main_all.exists() else 0,
            "accepted_repaired": sum(1 for _ in iter_jsonl(acc_rep_all)) if acc_rep_all.exists() else 0,
            "review_or_unresolved": sum(1 for _ in iter_jsonl(review_all)) if review_all.exists() else 0,
        },
        "quality_tier_counts": dict(qcnt),
        "subtype_counts": dict(scnt),
        "review_cause_counts": dict(rcnt),
    }
    summary_f.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[MERGE DONE] {model_name} -> {model_out}")


def merge_mode(args: argparse.Namespace) -> None:
    dataset_root = (args.data_root / args.dataset).resolve()
    out_dir = (args.output_root / args.dataset / args.prod_subdir).resolve()
    models = collect_models(dataset_root, args.source_model)
    if not models:
        raise RuntimeError(f"No model folders found under {dataset_root}")
    for m in models:
        merge_one_model(out_dir, m.name, args.include_accepted_repaired_in_main)


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        run_mode(args)
    else:
        merge_mode(args)


if __name__ == "__main__":
    main()
