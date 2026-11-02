# Detector

Code for fine-tuning and evaluating a trajectory-level anomaly detector. The main detector is Gemma 3 12B (`google/gemma-3-12b-it`) with LoRA; any model behind an OpenAI-compatible endpoint can be evaluated zero-shot with a JSON-output prompt.

## Directory layout

```
detector/
├── data/
│   ├── schema.py          # TrajectoryPacket, AnnotationLabel, DetectorPrediction, label sets
│   └── loader.py          # Prompt serialization, packet/label loading, output parsing
├── train/
│   └── train_lora.py      # LoRA fine-tune entry (multi-GPU DDP via torchrun)
├── eval/
│   ├── metrics.py         # Binary F1, per-subtype F1, onset MAE, per-base breakdown
│   ├── infer_sixfold.py   # Inference for the released detector (sixfold_rare head, /v1/completions)
│   ├── zeroshot_runner.py # JSON-prompt inference via an OpenAI-compatible chat API (zero-shot baselines)
│   └── prompts/
│       └── relabel_prompt_v3_5class.md  # 5-class annotation guide (copy of annotation_pipeline/prompts/)
├── configs/
│   └── gemma3_12b.yaml    # Configuration of the released detector (Gemma 3 12B)
└── scripts/
    ├── prepare_public_split.py  # HF split (train/test.jsonl) -> per-model packets + labels
    ├── build_v2_split.py        # Stratified 9:1 train/test split from re-labeler output
    ├── score_predictions.py     # Predictions + per-model labels -> full metrics JSON
    └── score_human_verified.py  # Predictions vs human_verified.jsonl or test.jsonl (binary P/R/F1, 6-class acc/F1)
```

## Data

The training and inference code reads, per source model, `annotation_input_packets.jsonl` (trajectories) and `final_annotations.jsonl` (labels). `scripts/prepare_public_split.py` builds this layout from the Hugging Face release:

```bash
python detector/scripts/prepare_public_split.py --split hf_data/data/train.jsonl --out work/train
python detector/scripts/prepare_public_split.py --split hf_data/data/test.jsonl  --out work/test
```

## Label space and output formats

The released labels use normal plus five anomaly subtypes:

| `anomaly_type_l2` | Meaning |
|---|---|
| (null) | Normal, no anomaly |
| `write_under_unresolved_ambiguity` | Agent commits under unresolved user-spec ambiguity |
| `premature_external_write` | External write before sufficient grounding |
| `weak_evidence_commitment` | Commitment under partial or noisy evidence |
| `capability_gap_overcommitment` | Commitment despite an explicit tool denial or empty result |
| `error_ignored_escalation` | Escalation to a higher-risk action after an unresolved error |

`target_mode` in the training config selects the prompt and target (`data/loader.py`). The released detector uses `sixfold_rare`: the model answers with one symbol per class (`◯` normal, `★` capability_gap_overcommitment, `☆` write_under_unresolved_ambiguity, `◆` weak_evidence_commitment, `◇` premature_external_write, `●` error_ignored_escalation), chosen because the base model has little prior preference among them. Training rows whose subtype is not one of the five are skipped. The default `json` mode (used by `zeroshot_runner.py`) asks for `{"is_anomaly", "anomaly_type_l2", "earliest_harmful_onset"}` and lists the full 8-subtype candidate set of the annotation taxonomy. All inference scripts write `is_anomaly` and `anomaly_type_l2` per `(source_model, trajectory_id)`.

## LoRA fine-tune

```bash
PYTHONPATH=detector torchrun --nproc_per_node 8 -m train.train_lora \
    --config detector/configs/gemma3_12b.yaml
```

All hyperparameters are in `configs/gemma3_12b.yaml`. The effective batch size is per-device batch (1) x gradient accumulation (4) x number of GPUs; the released detector used 32. The paper's schedule is 3 base epochs followed by a 2-epoch extension resumed from the end of the base run (`RESUME_CKPT=<checkpoint>` with `num_train_epochs` raised to 10, stopped after `checkpoint-1520`); see `REPRODUCE.md` section 3. Add `--dry-run` to write dataset statistics without loading the model.

## Inference

```bash
vllm serve google/gemma-3-12b-it --enable-lora \
    --lora-modules detector=runs/gemma3_12b/checkpoint-1520 --port 8000

export DETECTOR_API_KEY=EMPTY   # vLLM accepts any token
python detector/eval/infer_sixfold.py \
    --packet-files work/test/deepseek-r1-bfcl/annotation_input_packets.jsonl \
    --source-models deepseek-r1-bfcl \
    --api-base http://localhost:8000/v1 --model detector \
    --tokenizer google/gemma-3-12b-it \
    --out results/predictions/deepseek-r1-bfcl.jsonl
```

For zero-shot baselines, `eval/zeroshot_runner.py` takes the same `--packet-files` / `--source-models` / `--api-base` / `--model` arguments and uses the chat endpoint. Run one source model per output file; its `--resume` keys on `trajectory_id`, which repeats across source models.

## Scoring

```bash
# Binary P/R/F1, 6-class accuracy, per-class and macro F1, joined on (source_model, trajectory_id)
python detector/scripts/score_human_verified.py \
    --predictions results/predictions/*.jsonl \
    --labels hf_data/data/human_verified.jsonl

# Onset MAE and per-subtype F1 for JSON-format predictions (one source model)
python detector/scripts/score_predictions.py \
    --predictions results/zeroshot/deepseek-r1-bfcl.jsonl \
    --labels work/test/deepseek-r1-bfcl/final_annotations.jsonl \
    --out results/metrics/deepseek-r1-bfcl.metrics.json
```
