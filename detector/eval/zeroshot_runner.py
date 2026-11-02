"""Zero-shot baseline runner.

Two backends:
- "vllm-server": OpenAI-compatible HTTP (vLLM serve / sglang / llama.cpp server)
- hosted APIs that expose an OpenAI-compatible /chat/completions endpoint
  (set --api-base accordingly)

Reads packet jsonl(s) -> calls model -> writes predictions.jsonl.
Packet files are produced by scripts/prepare_public_split.py.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import (
    PROMPT_VERSION,
    build_prompt,
    discover_gold_dirs,
    iter_jsonl,
    load_packets,
    parse_prediction,
)
from data.schema import TrajectoryPacket


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--packet-files", nargs="+", required=True,
                   help="annotation_input_packets.jsonl files (one per source model), e.g. work/test/<model>/annotation_input_packets.jsonl")
    p.add_argument("--source-models", nargs="+", default=None,
                   help="One source_model name per --packet-files (parallel order)")
    p.add_argument("--api-base", default="http://localhost:8000/v1")
    p.add_argument("--api-key-env", default="DETECTOR_API_KEY")
    p.add_argument("--model", required=True, help="model id served by the API")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def call_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int = 3,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert anomaly detector. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url=api_base.rstrip("/") + "/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(str(x.get("text", x.get("content", ""))) for x in content if isinstance(x, dict))
            return str(content)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"chat completion failed after {retries} retries: {last_err}")


def load_completed(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return {row.get("trajectory_id") for row in iter_jsonl(out_path)
            if row.get("trajectory_id")}


def collect_packets(
    packet_files: list[Path],
    source_models: list[str] | None,
) -> list[TrajectoryPacket]:
    if source_models is None:
        source_models = [""] * len(packet_files)
    if len(source_models) != len(packet_files):
        raise ValueError("--source-models must match --packet-files length")
    items: list[TrajectoryPacket] = []
    for f, src in zip(packet_files, source_models):
        for tid, p in load_packets(f, source_model=src).items():
            items.append(p)
    return items


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "EMPTY")  # vLLM accepts EMPTY token
    args.out.parent.mkdir(parents=True, exist_ok=True)

    packet_files = [Path(x) for x in args.packet_files]
    src_models = args.source_models or [None] * len(packet_files)
    packets = collect_packets(packet_files, src_models)
    if args.limit > 0:
        packets = packets[: args.limit]

    completed = load_completed(args.out) if args.resume else set()
    todo = [p for p in packets if p.trajectory_id not in completed]
    print(f"[info] total={len(packets)} completed={len(completed)} todo={len(todo)} model={args.model}")

    def work(packet: TrajectoryPacket) -> dict:
        prompt = build_prompt(packet)
        try:
            raw = call_chat_completion(
                api_base=args.api_base, api_key=api_key, model=args.model,
                prompt=prompt, max_tokens=args.max_tokens,
                temperature=args.temperature, timeout=args.timeout,
            )
        except Exception as e:
            return {
                "trajectory_id": packet.trajectory_id,
                "source_model": packet.source_model,
                "subset": packet.subset,
                "error": str(e),
                "raw_text": "",
                "is_anomaly": None,
                "anomaly_type_l2": None,
                "earliest_harmful_onset": None,
                "prompt_version": PROMPT_VERSION,
                "detector_model": args.model,
            }
        pred = parse_prediction(packet.trajectory_id, raw)
        d = asdict(pred)
        d["source_model"] = packet.source_model
        d["subset"] = packet.subset
        d["prompt_version"] = PROMPT_VERSION
        d["detector_model"] = args.model
        return d

    with args.out.open("a", encoding="utf-8") as f, \
         concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, p): p for p in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  [{i}/{len(todo)}] last_id={row.get('trajectory_id')}")
    print(f"[done] wrote -> {args.out}")


if __name__ == "__main__":
    main()
