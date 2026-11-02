"""Inference for the released detector (target_mode: sixfold_rare).

Renders the training prompt (data.loader.build_prompt_rare) with the model's
chat template, sends it to an OpenAI-compatible /v1/completions endpoint
(e.g. vLLM serving the base model with the LoRA adapter), and maps the first
output symbol (one of the six rare-symbol tokens) to (is_anomaly, anomaly_type_l2).
For Qwen-family tokenizers the template is rendered with enable_thinking=False.

Usage (from the repo root):
    python detector/eval/infer_sixfold.py \
        --packet-files work/test/deepseek-r1-bfcl/annotation_input_packets.jsonl \
        --source-models deepseek-r1-bfcl \
        --tokenizer google/gemma-3-12b-it --model detector \
        --out results/predictions/deepseek-r1-bfcl.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.loader import build_prompt_rare, load_packets, TOKEN_TO_L2_RARE
from data.schema import PROMPT_VERSION

from transformers import AutoTokenizer

TOK_PATH = "google/gemma-3-12b-it"  # overridden by --tokenizer
_tok = None
def get_tok():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    return _tok


def render_prompt(user_content):
    msgs = [
        {"role": "system", "content": "You are an expert anomaly detector. Output one symbol only."},
        {"role": "user", "content": user_content},
    ]
    return get_tok().apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False if "qwen" in TOK_PATH.lower() else None)


def call_completions(api_base, api_key, model, prompt, max_tokens=8, timeout=120, retries=3):
    body = {"model": model, "prompt": prompt, "temperature": 0.0, "max_tokens": max_tokens}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        api_base.rstrip("/") + "/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                obj = json.loads(r.read())
                return obj["choices"][0]["text"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
            last = e
    raise last or RuntimeError("inference failed")


def parse_rare(text):
    """Find the first class symbol in the output. Returns (is_anomaly, l2_or_None)."""
    if not text:
        return None, None
    # scan for any of our 6 symbols
    for ch in text:
        if ch == "◯":
            return False, None
        if ch in TOKEN_TO_L2_RARE:
            return True, TOKEN_TO_L2_RARE[ch]
    return None, None


def main():
    global TOK_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-files", nargs="+", required=True)
    ap.add_argument("--source-models", nargs="+", required=True)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("DETECTOR_API_KEY", "EMPTY"))
    ap.add_argument("--model", required=True, help="model name served by the API (e.g. the LoRA module name)")
    ap.add_argument("--tokenizer", default=TOK_PATH,
                    help="HF id or local path whose chat template is used to render prompts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()
    assert len(args.packet_files) == len(args.source_models)
    TOK_PATH = args.tokenizer

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for pf, sm in zip(args.packet_files, args.source_models):
        for tid, packet in load_packets(Path(pf), source_model=sm).items():
            user = build_prompt_rare(packet)
            full_prompt = render_prompt(user)
            tasks.append((sm, tid, packet, full_prompt))
    print(f"[info] tasks={len(tasks)} model={args.model}")

    def runone(item):
        sm, tid, packet, prompt = item
        try:
            raw = call_completions(args.api_base, args.api_key, args.model, prompt, args.max_tokens)
        except Exception as e:
            return {"source_model": sm, "trajectory_id": tid, "subset": packet.subset,
                    "is_anomaly": None, "anomaly_type_l2": None,
                    "raw_text": f"ERROR: {e}", "error": str(e),
                    "detector_model": args.model, "prompt_version": PROMPT_VERSION}
        is_anom, l2 = parse_rare(raw)
        return {"source_model": sm, "trajectory_id": tid, "subset": packet.subset,
                "is_anomaly": is_anom, "anomaly_type_l2": l2, "raw_text": raw,
                "detector_model": args.model, "prompt_version": PROMPT_VERSION}

    n = 0
    with open(args.out, "w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(runone, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"[info] {n}/{len(tasks)}", flush=True)
    print(f"[done] wrote {n} → {args.out}")


if __name__ == "__main__":
    main()
