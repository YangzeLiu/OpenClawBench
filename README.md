# OpenClawBench

An agent can finish a task and still have acted badly along the way, for example by writing to an external system before it had resolved what the user meant, or by pressing on after a tool returned an error. OpenClawBench targets these process-side anomalies. It collects multi-step execution trajectories (thinking, action and reflection at every step) from six open-weight backbone models on BFCL function-calling tasks and labels each trajectory with a five-subtype taxonomy of commitment failures, including the step where harm begins. This repository holds the code behind the paper: the LLM annotation pipeline that produced the silver labels, LoRA fine-tuning of a compact open-weight detector, zero-shot evaluation of any OpenAI-compatible model, and scoring against the silver test split and the human-verified subset. The data itself is hosted on Hugging Face.

- Dataset: https://huggingface.co/datasets/YangzeLiu/OpenClawBench

## Repository layout

| Path | Contents |
|---|---|
| `annotation_pipeline/` | Trajectory preprocessing, the 4-stage silver-label pipeline (DeepSeek-V4-Pro judge), the single-judge v3 re-labeler, prompts and taxonomy files |
| `detector/data/` | Input/label schema, prompt serialization, data loading |
| `detector/train/` | LoRA fine-tuning entry point (`train_lora.py`) |
| `detector/eval/` | Inference for the fine-tuned detector, zero-shot runner, metrics |
| `detector/scripts/` | Data preparation for the public splits, v2 split builder, scoring scripts |
| `detector/configs/` | Training configuration (Gemma 3 12B) |
| `REPRODUCE.md` | Step-by-step reproduction guide |

## Dataset layout

The Hugging Face dataset `YangzeLiu/OpenClawBench` contains:

| File | Rows | Contents |
|---|---|---|
| `data/train.jsonl` | 23,857 | Silver-labeled trajectories |
| `data/test.jsonl` | 2,646 | Silver-labeled held-out trajectories |
| `data/human_verified.jsonl` | 440 | Human labels (`source_model`, `trajectory_id`, `split`, `subset` in {`gold300`, `supp140`}, `is_anomaly`, `l2`) |

Trajectory rows carry `source_model`, `trajectory_id`, `task`, `num_steps`, `steps[...]` and the silver label fields `v2_is_anom`, `v2_l2`, `v2_onset`, `v2_evidence_steps`, `v2_rationale`, `v2_confidence`. Records are joined on `(source_model, trajectory_id)`.

## Quickstart

All commands run from the repository root.

```bash
pip install -r requirements.txt

# 1. Download the dataset into hf_data/
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="YangzeLiu/OpenClawBench", repo_type="dataset", local_dir="hf_data")'

# 2. Split train/test into the per-model layout the detector code reads
python detector/scripts/prepare_public_split.py --split hf_data/data/train.jsonl --out work/train
python detector/scripts/prepare_public_split.py --split hf_data/data/test.jsonl  --out work/test

# 3. Fine-tune the detector (LoRA, 8 GPUs); see REPRODUCE.md for the 2-epoch extension
PYTHONPATH=detector torchrun --nproc_per_node 8 -m train.train_lora \
    --config detector/configs/gemma3_12b.yaml

# 4. Serve the adapter with vLLM and run the detector on each test source model
vllm serve google/gemma-3-12b-it --enable-lora \
    --lora-modules detector=runs/gemma3_12b/checkpoint-1520 --port 8000 &
for d in work/test/*/; do
  m=$(basename "$d")
  python detector/eval/infer_sixfold.py \
      --packet-files "$d/annotation_input_packets.jsonl" --source-models "$m" \
      --api-base http://localhost:8000/v1 --model detector --tokenizer google/gemma-3-12b-it \
      --out "results/predictions/$m.jsonl"
done

# 5. Score against the human-verified labels (or the silver test labels)
python detector/scripts/score_human_verified.py \
    --predictions results/predictions/*.jsonl --labels hf_data/data/human_verified.jsonl
python detector/scripts/score_human_verified.py \
    --predictions results/predictions/*.jsonl --labels hf_data/data/test.jsonl
```

`REPRODUCE.md` covers the training schedule, zero-shot baselines through `detector/eval/zeroshot_runner.py`, onset error, and re-running the annotation pipeline.

## Citation

```bibtex
@inproceedings{liu2026openclawbench,
  title     = {{OpenClawBench}: Benchmarking Process-side Anomalies in Real-world Agent Execution Trajectories},
  author    = {Liu, Yibing and Liu, Yangze and Yin, Xiaolong and Wang, Bin and Zhang, Chong and Yin, Hao and Han, Zhongyi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

Yibing Liu and Yangze Liu contributed equally.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The dataset on Hugging Face is released under CC-BY-4.0.
