# SimulTrain

[English](README.md) | **简体中文** | [Hy-SimulMT](../README.zh-CN.md)

使用内置的 LlamaFactory 集成，从 Hy-SimulMT 合并 checkpoint 继续训练。源模型目录只读，生成的模型视图、日志和 checkpoint 写入配置的输出目录。

## 环境要求

- Linux，Python 3.11 及以上。
- NVIDIA GPU，以及兼容的 CUDA／PyTorch 环境。
- 当前依赖配置使用 PyTorch 2.9.1、对应的 CPU torchaudio 和 Transformers 5.8.0。先准备 PyTorch 及匹配的 FlashAttention；默认训练配置使用 `flash_attn: fa2`。
- 完整的 Hy-SimulMT 模型目录，以及包含 `dataset_info.json` 的训练数据目录。

模型入口见 [Hy-SimulMT 模型集合](https://huggingface.co/collections/tencent/hy-simulmt)，模型待公开。训练环境与 vLLM 推理环境分开安装。

## 安装

在 `Hy-SimulMT/SimulTrain` 目录执行：

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
bash install.sh
```

安装脚本创建 `.venv`，复用所选解释器已有的 GPU 依赖，再安装 `requirements.txt`、本项目和 `vendor/LlamaFactory`。

指定已有 Python 环境：

```bash
PYTHON=/path/to/python bash install.sh
```

## 配置

```bash
cp .env.example .env
```

编辑 `.env`，模型、数据和输出使用绝对路径：

```bash
MERGED_MODEL_PATH=/path/to/Hy-SimulMT-checkpoint
DATASET_DIR=/path/to/train_data
MEDIA_DIR=/path/to/audio
DATASET=dataset_name
OUTPUT_DIR=/path/to/training_output
GPU=0,1,2,3,4,5,6,7
```

- `MERGED_MODEL_PATH`：完整合并模型目录，不能与输出目录相同。
- `DATASET_DIR`：包含 `dataset_info.json` 的目录。
- `MEDIA_DIR`：音频相对路径的根目录；音频位于数据目录时可以省略。
- `DATASET`：已注册的数据集名称。
- `OUTPUT_DIR`：模型视图、日志和 checkpoint 输出位置。
- `GPU`：参与训练的 GPU 编号，逗号分隔。

## 数据格式

参考 [dataset_info.json](data_demo/dataset_info.json) 和 [merged_st_demo.json](data_demo/merged_st_demo.json)。示例展示音频引用、源文和译文的组织方式；使用前替换音频占位文件名及对应文本。该示例没有附带录音。

## 启动训练

```bash
bash train_merged.sh
```

命令前的环境变量覆盖 `.env`，命令后的 `key=value` 覆盖 YAML 参数：

```bash
MAX_STEPS=200 LR=1.0e-5 BATCH_SIZE=1 GRAD_ACCUM=2 \
  bash train_merged.sh save_strategy=no eval_strategy=no
```

默认配置是 [configs/merged_a3b_sft.yaml](configs/merged_a3b_sft.yaml)：冻结音频 encoder，训练语言模型和 adapter，使用 DeepSpeed ZeRO-3。日志位于 `OUTPUT_DIR/train.log`。

也可以直接使用配置入口：

```bash
bash train.sh merged_a3b_sft 0,1 \
  merged_model_path=/path/to/model \
  dataset_dir=/path/to/data media_dir=/path/to/audio \
  dataset=my_dataset output_dir=/path/to/output
```

启动前会检查 GPU 空闲显存。多机参数通过 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT` 设置。

## 检查

安装后激活训练环境：

```bash
source .venv/bin/activate
python scripts/preflight.py /path/to/Hy-SimulMT-checkpoint
python scripts/check_protocol.py /path/to/Hy-SimulMT-checkpoint
```

前者检查合并 checkpoint 和音频组件；加 `--skip-encoder-load` 可跳过 encoder 加载。后者检查 tokenizer、音频占位符及源文／译文监督标签，不加载完整语言模型。

## 目录

| 路径 | 内容 |
|---|---|
| `train_merged.sh` | 环境变量配置的训练入口 |
| `train.sh` | YAML 配置及单机／分布式启动 |
| `s2s_train/` | 模型集成、checkpoint 加载及训练工具 |
| `configs/` | 训练配置 |
| `data_demo/` | 数据格式示例 |
| `vendor/LlamaFactory/` | 内置 LlamaFactory 源码和许可证 |

保留 `s2s_train` Python 模块名与 `s2s-train` 兼容入口，同时提供 `simultrain`、`simultrain-infer` 命令。对外推理服务使用 [SimulInfer](../SimulInfer/README.zh-CN.md)。
