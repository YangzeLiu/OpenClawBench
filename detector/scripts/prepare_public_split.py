#!/usr/bin/env python3
"""Convert a public OpenClawBench split into the two-file-per-model layout
consumed by the detector code.

The Hugging Face release stores each split as one JSONL file
(`data/train.jsonl`, `data/test.jsonl`) in which every row carries the
trajectory (`steps`) and its silver label (`v2_*` fields). The training entry
point (`train/train_lora.py`, via `data.loader.load_packets` / `load_labels`)
and the zero-shot runner instead read, per source model:

    <out>/<source_model>/annotation_input_packets.jsonl   # trajectory packets
    <out>/<source_model>/final_annotations.jsonl          # labels

This script performs that split. Fields not present in the public rows
(subset, oracle_outcome, risk_*, event_type_counts, onset_candidates) fall back
to the defaults in `TrajectoryPacket.from_jsonl_row`; `onset_candidates` only
affects truncation of trajectories whose serialized body exceeds the
character cap in `data/loader.py` (the count is printed at the end).

Labels are copied unchanged. The few anomalous rows whose `v2_l2` is not one
of the 5 subtypes (retired subtypes or malformed values) are kept here and
skipped by the sixfold / sixfold_rare training targets in `data/loader.py`.

Usage (from the repo root, after downloading the dataset to hf_data/):
    python detector/scripts/prepare_public_split.py \
        --split hf_data/data/train.jsonl --out work/train
    python detector/scripts/prepare_public_split.py \
        --split hf_data/data/test.jsonl --out work/test
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True,
                    help="public split JSONL, e.g. hf_data/data/train.jsonl")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory; one sub-directory per source_model")
    args = ap.parse_args()

    writers: dict[str, tuple] = {}
    counts: Counter = Counter()
    anom: Counter = Counter()
    overlong = 0

    with args.split.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src = r["source_model"]
            if src not in writers:
                d = args.out / src
                d.mkdir(parents=True, exist_ok=True)
                writers[src] = (
                    (d / "annotation_input_packets.jsonl").open("w", encoding="utf-8"),
                    (d / "final_annotations.jsonl").open("w", encoding="utf-8"),
                )
            pf, af = writers[src]
            packet = {
                "trajectory_id": r["trajectory_id"],
                "num_steps": r["num_steps"],
                "steps": r["steps"],
            }
            # rough serialized-length estimate mirroring serialize_step caps
            est = sum(
                min(len(s.get("thinking") or ""), 1500)
                + min(len(s.get("action_text") or ""), 2000)
                + min(len(s.get("reflection_text") or ""), 1500)
                + 40
                for s in r["steps"]
            )
            if est > 100_000:
                overlong += 1
            pf.write(json.dumps(packet, ensure_ascii=False) + "\n")
            ann = {
                "trajectory_id": r["trajectory_id"],
                "annotation": {
                    "trajectory_id": r["trajectory_id"],
                    "trajectory_is_anomalous": bool(r["v2_is_anom"]),
                    "anomaly_type_l1": None,
                    "anomaly_type_l2": r["v2_l2"],
                    "earliest_harmful_onset": r["v2_onset"],
                    "evidence_steps": r.get("v2_evidence_steps") or [],
                    "annotator_rationale": r.get("v2_rationale") or "",
                },
            }
            af.write(json.dumps(ann, ensure_ascii=False) + "\n")
            counts[src] += 1
            if r["v2_is_anom"]:
                anom[src] += 1

    for pf, af in writers.values():
        pf.close()
        af.close()

    total = sum(counts.values())
    print(f"total={total} anom={sum(anom.values())} overlong(>100k chars est)={overlong}")
    for src in sorted(counts):
        print(f"  {src}: n={counts[src]} anom={anom[src]}")


if __name__ == "__main__":
    main()
