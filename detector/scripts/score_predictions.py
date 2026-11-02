"""Take predictions.jsonl + a label file -> emit eval_metrics.json
(binary F1, per-subtype F1, onset MAE, per-base breakdown).

Labels are keyed by trajectory_id only, so score one source model at a time,
using the per-model label file written by scripts/prepare_public_split.py.
For human-verified labels use scripts/score_human_verified.py.

Usage:
    python detector/scripts/score_predictions.py \
        --predictions results/predictions/deepseek-r1-bfcl.jsonl \
        --labels work/test/deepseek-r1-bfcl/final_annotations.jsonl \
        --out results/metrics/deepseek-r1-bfcl.metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import iter_jsonl, load_labels
from data.schema import DetectorPrediction
from eval.metrics import full_report, write_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True,
                   help="final_annotations.jsonl for one source model (see prepare_public_split.py)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    print(f"[info] {len(labels)} labels")

    preds: dict[str, DetectorPrediction] = {}
    base_lookup: dict[str, str] = {}
    subset_lookup: dict[str, str] = {}
    for row in iter_jsonl(args.predictions):
        tid = row.get("trajectory_id")
        if not tid:
            continue
        preds[tid] = DetectorPrediction(
            trajectory_id=tid,
            is_anomaly=row.get("is_anomaly"),
            anomaly_type_l2=row.get("anomaly_type_l2"),
            earliest_harmful_onset=row.get("earliest_harmful_onset"),
            raw_text=row.get("raw_text", ""),
        )
        if row.get("source_model"):
            base_lookup[tid] = row["source_model"]
        if row.get("subset"):
            subset_lookup[tid] = row["subset"]
    print(f"[info] {len(preds)} predictions")

    rep = full_report(preds, labels,
                      base_lookup=base_lookup or None,
                      subset_lookup=subset_lookup or None)
    write_report(args.out, rep)
    print(f"[done] -> {args.out}")
    print(f"  binary: {json.dumps(rep['binary'])}")
    print(f"  onset:  {json.dumps(rep['onset'])}")


if __name__ == "__main__":
    main()
