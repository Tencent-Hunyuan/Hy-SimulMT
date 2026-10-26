# SimulTrain

**English** | [简体中文](README.zh-CN.md) | [Hy-SimulMT](../README.md)

Continue training a Hy-SimulMT merged checkpoint with the included LlamaFactory integration. The source model is read-only; generated model views, logs and checkpoints are written to your output directory.

## Requirements

- Linux and Python 3.11 or later.
- NVIDIA GPUs and a compatible CUDA/PyTorch environment.
- The supplied dependency profile uses PyTorch 2.9.1, matching CPU torchaudio and Transformers 5.8.0. Install PyTorch and a matching FlashAttention build first; the default training configuration uses `flash_attn: fa2`.
- A complete downloaded Hy-SimulMT checkpoint and training data registered in `dataset_info.json`.

Model releases are listed in the [Hy-SimulMT collection](https://huggingface.co/collections/tencent/hy-simulmt). Model access is pending public release. Keep this environment separate from the vLLM inference environment.

## Install

From `Hy-SimulMT/SimulTrain`:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
bash install.sh
```

The installer creates `.venv` with access to the selected interpreter's existing GPU packages, installs `requirements.txt`, and installs this project and `vendor/LlamaFactory` in editable mode.

To select an existing Python environment:

```bash
PYTHON=/path/to/python bash install.sh
```

## Configure

```bash
cp .env.example .env
```

Set these values in `.env`, using absolute paths for model, data and output locations:

```bash
MERGED_MODEL_PATH=/path/to/Hy-SimulMT-checkpoint
DATASET_DIR=/path/to/train_data
MEDIA_DIR=/path/to/audio
DATASET=dataset_name
OUTPUT_DIR=/path/to/training_output
GPU=0,1,2,3,4,5,6,7
```

- `MERGED_MODEL_PATH`: complete merged checkpoint; never use it as `OUTPUT_DIR`.
- `DATASET_DIR`: directory containing `dataset_info.json`.
- `MEDIA_DIR`: base directory for relative audio paths; may be omitted if audio is under `DATASET_DIR`.
- `DATASET`: registered dataset name.
- `OUTPUT_DIR`: generated model view, logs and saved checkpoints.
- `GPU`: comma-separated visible GPU IDs.

## Data

See [data_demo/dataset_info.json](data_demo/dataset_info.json) and [data_demo/merged_st_demo.json](data_demo/merged_st_demo.json) for the input schema. The example demonstrates audio references, source text and translation. Replace its placeholder audio filename and transcript before training; no audio recording is included with that example.

## Train

```bash
bash train_merged.sh
```

Environment variables override `.env`. Additional `key=value` arguments override the training YAML:

```bash
MAX_STEPS=200 LR=1.0e-5 BATCH_SIZE=1 GRAD_ACCUM=2 \
  bash train_merged.sh save_strategy=no eval_strategy=no
```

The default configuration is [configs/merged_a3b_sft.yaml](configs/merged_a3b_sft.yaml): encoder frozen, language model and adapter trainable, using DeepSpeed ZeRO-3. Logs are written to `OUTPUT_DIR/train.log`.

For direct access to the configurable launcher:

```bash
bash train.sh merged_a3b_sft 0,1 \
  merged_model_path=/path/to/model \
  dataset_dir=/path/to/data media_dir=/path/to/audio \
  dataset=my_dataset output_dir=/path/to/output
```

The launcher checks available GPU memory before training. Distributed settings use `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`.

## Checks

After installation, activate the training environment:

```bash
source .venv/bin/activate
python scripts/preflight.py /path/to/Hy-SimulMT-checkpoint
python scripts/check_protocol.py /path/to/Hy-SimulMT-checkpoint
```

The first command checks the merged checkpoint and audio components; `--skip-encoder-load` skips loading the encoder. The second checks tokenizer, audio placeholders and source/translation labels without loading the full language model.

## Layout

| Directory/file | Purpose |
|---|---|
| `train_merged.sh` | Environment-based training entry point |
| `train.sh` | YAML-based single-node/distributed launcher |
| `s2s_train/` | Model integration, checkpoint loading and training utilities |
| `configs/` | Training configuration |
| `data_demo/` | Dataset format examples |
| `vendor/LlamaFactory/` | Included LlamaFactory source and license |

The `s2s_train` Python namespace and `s2s-train` compatibility entry point are retained. The package also exposes `simultrain` and `simultrain-infer`; deployed inference services are provided by [SimulInfer](../SimulInfer/README.md).
