"""LoRA fine-tune entry (multi-GPU DDP via torchrun).

Loads silver samples (per-model packets + labels, see
scripts/prepare_public_split.py), formats prompts/targets
with the SAME serializer used at inference time (data.loader.build_prompt /
build_target_json), then SFT via TRL SFTTrainer with PEFT LoRA.

Usage (from the repo root):
    PYTHONPATH=detector torchrun --nproc_per_node <N> -m train.train_lora \
        --config detector/configs/gemma3_12b.yaml

Set RESUME_CKPT=<checkpoint dir> (or RESUME_CKPT=auto for the latest one in
output_dir) to resume training from a saved checkpoint.

This file is intentionally a thin orchestration shim; the heavy deps
(transformers / peft / trl / accelerate) are imported lazily so that the rest
of the repo can be unit-tested on a CPU box without them.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml  # PyYAML

from data.loader import (
    build_prompt,
    build_target_json,
    build_silver_samples,
    load_labels,
    load_packets,
    load_quality_lookup,
    build_prompt_bin,
    build_target_binary,
    build_prompt_sixfold,
    build_target_sixfold,
    build_prompt_rare,
    build_target_rare,
)
from data.schema import PROMPT_VERSION


@dataclass
class TrainConfig:
    # Model
    base_model: str = "google/gemma-3-12b-it"
    chat_template: Optional[str] = None  # "gemma" / "qwen" / None=auto

    # Data
    silver_packets: list[str] = field(default_factory=list)
    silver_annotations: list[str] = field(default_factory=list)
    silver_master_tables: list[str] = field(default_factory=list)
    quality_filter: list[str] = field(default_factory=lambda: [
        "high_confidence_accepted", "accepted_repaired",
    ])
    dev_ratio: float = 0.05
    seed: int = 42

    # Hold-out: leave these base models out of train (for cross-base experiment)
    holdout_base_models: list[str] = field(default_factory=list)
    # Hold-out: leave these subsets out of train
    holdout_subsets: list[str] = field(default_factory=list)

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    output_dir: str = "runs/default"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 32768
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    save_steps: int = 200
    logging_steps: int = 10
    eval_steps: int = 200
    deepspeed: Optional[str] = None  # path to ds_config.json if needed
    group_by_length: bool = False
    target_mode: str = "json"
    max_normal_samples: Optional[int] = None  # cap normal samples to break majority shortcut


def load_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**(raw or {}))


def load_silver_pool(cfg: TrainConfig) -> list:
    """Load + join + filter silver samples across all configured src models."""
    if len(cfg.silver_packets) != len(cfg.silver_annotations):
        raise ValueError("silver_packets and silver_annotations must align 1:1")
    if cfg.silver_master_tables and len(cfg.silver_master_tables) != len(cfg.silver_packets):
        raise ValueError("silver_master_tables must align 1:1 with silver_packets if provided")
    out = []
    qf = set(cfg.quality_filter) if cfg.quality_filter else None
    for i, (pf, af) in enumerate(zip(cfg.silver_packets, cfg.silver_annotations)):
        packets = load_packets(Path(pf), source_model=Path(pf).parent.name)
        labels = load_labels(Path(af))
        ql = (load_quality_lookup(Path(cfg.silver_master_tables[i]))
              if cfg.silver_master_tables else None)
        samples = build_silver_samples(packets, labels, quality_filter=qf, quality_lookup=ql)
        out.extend(samples)
    if cfg.holdout_base_models:
        bad = set(cfg.holdout_base_models)
        out = [s for s in out if s.packet.source_model not in bad]
    if cfg.holdout_subsets:
        bad = set(cfg.holdout_subsets)
        out = [s for s in out if s.packet.subset not in bad]
    return out


def to_chat_examples(samples: list, system_prompt: str = "You are an expert anomaly detector. Output strict JSON only.") -> list[dict]:
    """Convert samples to chat-format dicts {messages: [...]}, ready for SFTTrainer."""
    out = []
    for s in samples:
        out.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": s.prompt},
                {"role": "assistant", "content": s.target_json},
            ],
            "trajectory_id": s.packet.trajectory_id,
            "source_model": s.packet.source_model,
            "subset": s.packet.subset,
            "length": len(s.prompt) + len(s.target_json),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build dataset and report stats; do not load model.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"[info] prompt_version={PROMPT_VERSION} base_model={cfg.base_model}")
    samples = load_silver_pool(cfg)
    if cfg.target_mode == "binary":
        for s in samples:
            s.prompt = build_prompt_bin(s.packet)
            s.target_json = build_target_binary(s.label)
        print(f"[info] target_mode=binary, sample target: {samples[0].target_json}")
    elif cfg.target_mode == "sixfold":
        kept = []
        skipped = 0
        for s in samples:
            tok = build_target_sixfold(s.label)
            if tok is None:
                skipped += 1
                continue
            s.prompt = build_prompt_sixfold(s.packet)
            s.target_json = tok
            kept.append(s)
        samples = kept
        print(f"[info] target_mode=sixfold, kept={len(kept)} skipped_rare={skipped}")
    elif cfg.target_mode == "sixfold_rare":
        kept = []
        skipped = 0
        for s in samples:
            tok = build_target_rare(s.label)
            if tok is None:
                skipped += 1
                continue
            s.prompt = build_prompt_rare(s.packet)
            s.target_json = tok
            kept.append(s)
        samples = kept
        print(f"[info] target_mode=sixfold_rare, kept={len(kept)} skipped_rare={skipped}")
    if cfg.max_normal_samples is not None:
        import random as _random
        normals = [s for s in samples if not s.label.trajectory_is_anomalous]
        anoms = [s for s in samples if s.label.trajectory_is_anomalous]
        cap = min(cfg.max_normal_samples, len(normals))
        rng = _random.Random(cfg.seed)
        kept_normals = rng.sample(normals, cap)
        samples = kept_normals + anoms
        rng.shuffle(samples)
        print(f"[info] downsample: kept_normal={cap}/{len(normals)}, all_anom={len(anoms)}, total={len(samples)}")
    print(f"[info] silver samples: {len(samples)}")

    examples = to_chat_examples(samples)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "dataset_stats.json").write_text(json.dumps({
        "n_examples": len(examples),
        "n_anomalous": sum(1 for s in samples if s.label.trajectory_is_anomalous),
        "n_normal": sum(1 for s in samples if not s.label.trajectory_is_anomalous),
        "by_source_model": _count_by(samples, lambda s: s.packet.source_model),
        "by_subset_top20": dict(sorted(
            _count_by(samples, lambda s: s.packet.subset).items(),
            key=lambda kv: -kv[1])[:20]),
        "by_l2": _count_by(samples, lambda s: s.label.anomaly_type_l2 or "__normal__"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] dataset stats written to {out_dir}/dataset_stats.json")
        return

    _run_sft(cfg, examples, out_dir)


def _count_by(samples, key_fn) -> dict:
    out: dict = {}
    for s in samples:
        k = key_fn(s) or "__none__"
        out[k] = out.get(k, 0) + 1
    return out


def _run_sft(cfg: TrainConfig, examples: list[dict], out_dir: Path) -> None:
    """Lazy import heavy deps so dry-run works without them."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    ds = Dataset.from_list(examples)
    ds = ds.train_test_split(test_size=cfg.dev_ratio, seed=cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )

    targs = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps", eval_steps=cfg.eval_steps,
        save_total_limit=12,
        report_to="none",
        length_column_name="length" if cfg.group_by_length else None,
        train_sampling_strategy="group_by_length" if cfg.group_by_length else "random",
        deepspeed=cfg.deepspeed,
        max_length=cfg.max_seq_length,
        completion_only_loss=True,
        use_liger_kernel=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model, processing_class=tok, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        peft_config=peft_cfg,
    )
    # Gemma3ForConditionalGeneration requires token_type_ids during training
    # (it's the multi-modal class). We are doing text-only SFT, so inject zeros
    # for every batch via a collator wrapper.
    _orig_collator = trainer.data_collator
    def _collator_with_token_types(features):
        batch = _orig_collator(features)
        if isinstance(batch, dict) and "input_ids" in batch and "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch
    trainer.data_collator = _collator_with_token_types
    import os
    _ckpt = os.environ.get("RESUME_CKPT", "")
    if _ckpt == "auto":
        trainer.train(resume_from_checkpoint=True)
    elif _ckpt:
        trainer.train(resume_from_checkpoint=_ckpt)
    else:
        trainer.train()
    trainer.save_model(str(out_dir / "lora_adapter"))
    tok.save_pretrained(str(out_dir / "lora_adapter"))


if __name__ == "__main__":
    main()
