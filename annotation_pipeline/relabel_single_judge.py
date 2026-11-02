#!/usr/bin/env python3
"""
Single-judge re-labeler (DeepSeek-V4-Pro) with the v3 5-class prompt.

Resume-friendly:
- Reads existing OUT file, builds set of (source_model, trajectory_id) already
  *successfully* labeled (parsed != null AND no error).
- Skips them; re-runs everything else.
- Each result is appended (line-buffered flush) so a kill is safe.

Usage:
  DEEPSEEK_API_KEY=... python annotation_pipeline/relabel_single_judge.py \
    --in  <trajectories.jsonl> \
    --out results/relabel_v2.jsonl \
    --workers 64

Input rows need: source_model, trajectory_id, num_steps, steps[...] (e.g. data/train.jsonl).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Resolve relative to this script so the repo can be placed anywhere
REPO = Path(__file__).resolve().parent
PROMPT = REPO / "prompts/relabel_prompt_v3_5class.md"  # v3 5-class prompt used in the paper

DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE = "https://api.deepseek.com/v1"


def serialize_trajectory(rec):
    lines = [f"--- TRAJECTORY (source_model={rec['source_model']}, trajectory_id={rec['trajectory_id']}, num_steps={rec['num_steps']}) ---"]
    for s in rec["steps"]:
        sid = s.get("step_id", "?")
        ev = s.get("event_type", "")
        thinking = (s.get("thinking") or "").strip()
        action = (s.get("action_text") or "").strip()
        reflection = (s.get("reflection_text") or "").strip()
        lines.append(f"\n[step {sid}] event={ev}")
        if thinking: lines.append(f"thinking: {thinking}")
        if action:   lines.append(f"action: {action}")
        if reflection: lines.append(f"reflection: {reflection}")
    return "\n".join(lines)


def build_user_msg(prompt_text, rec):
    return prompt_text + "\n\n" + serialize_trajectory(rec) + "\n\nNow output ONLY the JSON object specified in the schema."


def call_deepseek(client, user_msg, max_tokens):
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert anomaly annotator. Output strict JSON only, matching the schema in the user message. No prose outside JSON."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
        return {"text": text, "usage": usage, "error": None}
    except Exception as e:
        return {"text": None, "usage": None, "error": f"{type(e).__name__}: {e}"}


def parse_response(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        t = text.strip()
        if t.startswith("```"):
            parts = t.split("\n")
            t = "\n".join(parts[1:-1] if parts[-1].startswith("```") else parts[1:])
        try:
            return json.loads(t)
        except Exception:
            return {"_parse_error": text[:500]}


def is_success(rec):
    if rec.get("error"):
        return False
    p = rec.get("parsed")
    if not isinstance(p, dict):
        return False
    if "_parse_error" in p:
        return False
    if "is_anomaly" not in p:
        return False
    return True


def load_done(out_path):
    """Return set of keys already successfully labeled."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if is_success(r):
                done.add((r["source_model"], r["trajectory_id"]))
    return done


def run_one(client, rec, prompt_text, max_tokens):
    user_msg = build_user_msg(prompt_text, rec)
    t0 = time.time()
    out = call_deepseek(client, user_msg, max_tokens)
    dt = time.time() - t0
    parsed = parse_response(out["text"])
    return {
        "source_model": rec["source_model"],
        "trajectory_id": rec["trajectory_id"],
        "label_is_anom": rec.get("label_is_anom"),
        "label_l2": rec.get("label_l2"),
        "label_route": rec.get("label_route"),
        "num_steps": rec.get("num_steps"),
        "raw_text": out["text"],
        "parsed": parsed,
        "usage": out["usage"],
        "error": out["error"],
        "elapsed_s": round(dt, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--workers", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    in_path  = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: missing DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)

    prompt_text = PROMPT.read_text()

    # Load all input records
    all_records = []
    with open(in_path) as f:
        for line in f:
            all_records.append(json.loads(line))
    if args.limit > 0:
        all_records = all_records[: args.limit]

    done = load_done(out_path)
    pending = [r for r in all_records if (r["source_model"], r["trajectory_id"]) not in done]
    print(f"Total input: {len(all_records)} | already done: {len(done)} | pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE)

    n_ok = 0
    n_err = 0
    t_global_start = time.time()
    fails_to_resume = []

    # Open OUT in append mode for incremental flush
    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_one, client, r, prompt_text, args.max_tokens): r
                for r in pending
            }
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                if is_success(r):
                    n_ok += 1
                else:
                    n_err += 1
                    fails_to_resume.append((r["source_model"], r["trajectory_id"], r.get("error") or "parse_failure"))
                if i % 50 == 0 or i == len(pending):
                    elapsed = time.time() - t_global_start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - i) / rate if rate > 0 else 0
                    print(f"  [{i}/{len(pending)}] ok={n_ok} err={n_err} | {elapsed:.0f}s elapsed | {rate:.2f}/s | ETA {eta:.0f}s")

    print(f"\nDone. ok={n_ok}, err={n_err}, total processed={n_ok+n_err}")
    if fails_to_resume:
        print(f"\nFirst 5 errors:")
        for sm, tid, err in fails_to_resume[:5]:
            print(f"  {sm} {tid}: {err}")
        print(f"\nRe-run the same command to resume failed-only.")


if __name__ == "__main__":
    main()
