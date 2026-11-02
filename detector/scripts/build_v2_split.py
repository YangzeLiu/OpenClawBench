#!/usr/bin/env python3
"""
After v2 re-label finishes, build cleaner train/test split for fine-tune.

Steps:
1. Read v2 raw labels (output of annotation_pipeline/relabel_single_judge.py)
2. Filter: keep only successful + parsed records (drop errors)
3. Optional confidence filter: --conf {high,high-medium,all}
4. Drop records where v2 disagreed wildly with silver? --no, keep all v2 (silver is the noisy one)
5. Stratified 9:1 split by v2 is_anomaly + source_model
6. Output:
   - silver_v2_train_<conf>.jsonl  (~90% records, used for LoRA)
   - silver_v2_test_<conf>.jsonl   (~10% records, held-out test)
   - silver_v2_split_stats_<conf>.json (distribution summary)

The released data/train.jsonl and data/test.jsonl were built this way; this
script is only needed when re-labeling your own trajectories.
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

# Defaults are relative to the current working directory (run from the repo root).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default="results/relabel_v2.jsonl",
                    help="v2 re-label output JSONL (from annotation_pipeline/relabel_single_judge.py)")
    ap.add_argument("--src", dest="src_packets",
                    default="hf_data/data/train.jsonl",
                    help="trajectory file with full step content (source_model, trajectory_id, num_steps, steps)")
    ap.add_argument("--out-dir", default="work/v2_split")
    ap.add_argument("--conf", default="high", choices=["high", "high-medium", "all"])
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load v2 labels
    v2_records = []
    n_parse_err = 0; n_api_err = 0
    with open(args.in_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error"):
                n_api_err += 1; continue
            p = r.get("parsed")
            if not isinstance(p, dict) or "_parse_error" in p or "is_anomaly" not in p:
                n_parse_err += 1; continue
            v2_records.append(r)
    print(f"Loaded v2: {len(v2_records)} usable, {n_api_err} API err, {n_parse_err} parse err")

    # 2. Confidence filter
    keep_conf = {"high"} if args.conf == "high" else ({"high", "medium"} if args.conf == "high-medium" else None)
    if keep_conf:
        before = len(v2_records)
        v2_records = [r for r in v2_records if r["parsed"].get("confidence", "low") in keep_conf]
        print(f"After confidence filter ({args.conf}): {len(v2_records)}/{before}")

    # 3. Join with original packets so we have steps for fine-tune training
    pkt_index = {}
    with open(args.src_packets) as f:
        for line in f:
            r = json.loads(line)
            pkt_index[(r["source_model"], r["trajectory_id"])] = r

    joined = []
    n_missing = 0
    for v2 in v2_records:
        key = (v2["source_model"], v2["trajectory_id"])
        pkt = pkt_index.get(key)
        if not pkt:
            n_missing += 1; continue
        p = v2["parsed"]
        joined.append({
            "source_model": v2["source_model"],
            "trajectory_id": v2["trajectory_id"],
            "num_steps": pkt["num_steps"],
            "v2_is_anom": bool(p.get("is_anomaly")),
            "v2_l2": p.get("anomaly_type_l2"),
            "v2_onset": p.get("earliest_harmful_onset"),
            "v2_evidence_steps": p.get("evidence_steps"),
            "v2_confidence": p.get("confidence"),
            "v2_rationale": p.get("rationale"),
            "silver_is_anom": v2.get("label_is_anom"),
            "silver_l2": v2.get("label_l2"),
            "silver_route": v2.get("label_route"),
            "steps": pkt["steps"],
        })
    print(f"Joined with packets: {len(joined)} (missing {n_missing})")

    # 4. Stratified 9:1 split by (v2_is_anom, source_model)
    by_strata = defaultdict(list)
    for r in joined:
        by_strata[(r["v2_is_anom"], r["source_model"])].append(r)

    train, test = [], []
    for strata_key, recs in by_strata.items():
        random.shuffle(recs)
        n_test = max(1, round(len(recs) * args.test_frac))
        test.extend(recs[:n_test])
        train.extend(recs[n_test:])

    random.shuffle(train); random.shuffle(test)

    # 5. Stats
    def summarize(recs, name):
        anom = sum(1 for r in recs if r["v2_is_anom"])
        l2c = Counter(r["v2_l2"] for r in recs if r["v2_is_anom"])
        bm = Counter(r["source_model"] for r in recs)
        return {
            "name": name,
            "n": len(recs),
            "v2_anom": anom,
            "v2_anom_rate": round(anom / len(recs), 4) if recs else 0,
            "l2_dist": dict(l2c),
            "base_dist": dict(bm),
        }

    stats = {
        "config": {"conf_filter": args.conf, "test_frac": args.test_frac, "seed": args.seed},
        "train": summarize(train, "train"),
        "test":  summarize(test,  "test"),
    }

    # 6. Write
    train_path = out_dir / f"silver_v2_train_{args.conf}.jsonl"
    test_path  = out_dir / f"silver_v2_test_{args.conf}.jsonl"
    stats_path = out_dir / f"silver_v2_split_stats_{args.conf}.json"

    with open(train_path, "w") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(test_path,  "w") as f:
        for r in test:  f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nWrote:\n  {train_path}\n  {test_path}\n  {stats_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
