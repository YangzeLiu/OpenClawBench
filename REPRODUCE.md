# Reproduction Guide

This guide goes from the released data to a trained detector and its scores, and then describes how to regenerate silver labels on your own trajectories. All commands run from the repository root.

## 1. Download the dataset

```bash
pip install -r requirements.txt
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="YangzeLiu/OpenClawBench", repo_type="dataset", local_dir="hf_data")'
```

Layout after download:

```
hf_data/
└── data/
    ├── train.jsonl            # 23,857 silver-labeled trajectories
    ├── test.jsonl             # 2,646 silver-labeled trajectories
    └── human_verified.jsonl   # 440 human labels (subset: gold300 / supp140)
```

Trajectory rows: `source_model`, `trajectory_id`, `task`, `num_steps`, `steps[...]`, `v2_is_anom`, `v2_l2`, `v2_onset`, `v2_evidence_steps`, `v2_rationale`, `v2_confidence`. Human rows: `source_model`, `trajectory_id`, `split`, `subset`, `is_anomaly`, `l2`. The join key is `(source_model, trajectory_id)`; a `trajectory_id` alone is not unique across source models.

## 2. Prepare the per-model layout

The training and inference code reads one packet file and one label file per source model. Convert the public splits once:

```bash
python detector/scripts/prepare_public_split.py --split hf_data/data/train.jsonl --out work/train
python detector/scripts/prepare_public_split.py --split hf_data/data/test.jsonl  --out work/test
```

This writes `work/<split>/<source_model>/annotation_input_packets.jsonl` and `work/<split>/<source_model>/final_annotations.jsonl`.

## 3. Train the detector (LoRA)

`detector/configs/gemma3_12b.yaml` is the configuration of the released detector and already points to `work/train/`. It trains a single-symbol classification head over normal plus the five subtypes (`target_mode: sixfold_rare`) and caps the normal class with `max_normal_samples` for class-balanced sampling. Anomalous rows whose subtype is not one of the five are skipped.

```bash
# Dataset statistics only, no model weights loaded
PYTHONPATH=detector python -m train.train_lora --config detector/configs/gemma3_12b.yaml --dry-run

# Base run: 3 epochs
PYTHONPATH=detector torchrun --nproc_per_node 8 -m train.train_lora \
    --config detector/configs/gemma3_12b.yaml
```

The effective batch size is `per_device_train_batch_size` (1) x `gradient_accumulation_steps` (4) x the number of GPUs. The step numbers below come from the original run on 8 GPUs (effective batch 32, 307 optimizer steps per epoch); with a different GPU count, scale them accordingly or adjust `gradient_accumulation_steps` to keep the effective batch at 32.

The paper trains for 3 base epochs and then extends training by 2 epochs from the end of the base run. In the original run the base run ended at `checkpoint-921` (epoch 3). The extension resumed from that checkpoint with `num_train_epochs` raised to 10, which recomputes the cosine schedule over 10 epochs, and was stopped after `checkpoint-1520` (epoch 4.95). `checkpoint-1520` is the released detector. To reproduce it:

```bash
cp detector/configs/gemma3_12b.yaml detector/configs/gemma3_12b_ext.yaml
sed -i 's/^num_train_epochs: 3.0$/num_train_epochs: 10.0/' detector/configs/gemma3_12b_ext.yaml

RESUME_CKPT=runs/gemma3_12b/checkpoint-921 PYTHONPATH=detector \
    torchrun --nproc_per_node 8 -m train.train_lora \
    --config detector/configs/gemma3_12b_ext.yaml
# stop the run after runs/gemma3_12b/checkpoint-1520 has been saved
```

## 4. Inference

Serve the base model with the adapter (vLLM 0.19.1 or later is required for Gemma 3):

```bash
vllm serve google/gemma-3-12b-it \
    --enable-lora --lora-modules detector=runs/gemma3_12b/checkpoint-1520 \
    --tensor-parallel-size 8 --port 8000
```

Run the detector on each source model of the test split. `infer_sixfold.py` renders the training prompt with the Gemma chat template and parses the output symbol:

```bash
export DETECTOR_API_KEY=EMPTY
for d in work/test/*/; do
  m=$(basename "$d")
  python detector/eval/infer_sixfold.py \
      --packet-files "$d/annotation_input_packets.jsonl" --source-models "$m" \
      --api-base http://localhost:8000/v1 --model detector \
      --tokenizer google/gemma-3-12b-it \
      --out "results/predictions/$m.jsonl"
done
```

Zero-shot baselines use the JSON prompt through any OpenAI-compatible chat endpoint:

```bash
python detector/eval/zeroshot_runner.py \
    --packet-files work/test/deepseek-r1-bfcl/annotation_input_packets.jsonl \
    --source-models deepseek-r1-bfcl \
    --api-base http://localhost:8000/v1 --model google/gemma-3-12b-it \
    --out results/zeroshot/deepseek-r1-bfcl.jsonl --workers 8 --resume
```

## 5. Score predictions

Binary precision/recall/F1, 6-class accuracy, per-class F1 and macro F1 (normal plus the five subtypes), joined on `(source_model, trajectory_id)`:

```bash
# Human-verified labels (all 440, or one subset)
python detector/scripts/score_human_verified.py \
    --predictions results/predictions/*.jsonl \
    --labels hf_data/data/human_verified.jsonl \
    --out results/metrics/human_verified.json
python detector/scripts/score_human_verified.py \
    --predictions results/predictions/*.jsonl \
    --labels hf_data/data/human_verified.jsonl --subset gold300

# Silver test labels
python detector/scripts/score_human_verified.py \
    --predictions results/predictions/*.jsonl \
    --labels hf_data/data/test.jsonl \
    --out results/metrics/silver_test.json
```

Onset error for one source model (JSON-format predictions from `zeroshot_runner.py`):

```bash
python detector/scripts/score_predictions.py \
    --predictions results/zeroshot/deepseek-r1-bfcl.jsonl \
    --labels work/test/deepseek-r1-bfcl/final_annotations.jsonl \
    --out results/metrics/deepseek-r1-bfcl.zeroshot.json
```

## 6. (Optional) Regenerate silver labels

The released labels come from two stages, both documented in `annotation_pipeline/README.md`. Both call the DeepSeek API and need `DEEPSEEK_API_KEY`.

The 4-stage pipeline reads per-model annotation packets under `<RAW_ROOT>/BFCL/<source_model>/<input_subdir>/annotation_input_packets.jsonl`, built from raw agent runs by the preprocessing scripts in `annotation_pipeline/`:

```bash
export DEEPSEEK_API_KEY=...
python annotation_pipeline/run_fulltax_pipeline.py --mode run   --dataset BFCL \
    --source-model <source_model> --data-root <RAW_ROOT> --output-root <RAW_ROOT>/annotation_results \
    --workers 16 --resume
python annotation_pipeline/run_fulltax_pipeline.py --mode merge --dataset BFCL \
    --data-root <RAW_ROOT> --output-root <RAW_ROOT>/annotation_results
```

The single-judge re-labeler (v3 5-class prompt) produces the `v2_*` fields, and `build_v2_split.py` turns its output into a stratified train/test split:

```bash
python annotation_pipeline/relabel_single_judge.py \
    --in hf_data/data/train.jsonl --out results/relabel_v2.jsonl --workers 64
python detector/scripts/build_v2_split.py \
    --in results/relabel_v2.jsonl --src hf_data/data/train.jsonl \
    --out-dir work/v2_split --conf high
```
