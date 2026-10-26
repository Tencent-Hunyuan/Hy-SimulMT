# SimulInfer

[English](README.md) | **简体中文** | [Hy-SimulMT](../README.zh-CN.md)

通过网页、文件上传或实时 WebSocket 使用 Hy-SimulMT 语音翻译。
可选择仅译文模型，或同时输出源文和译文的模型，并按需开启 TTS 译音。

一键安装和启动请先看[项目快速部署](../README.zh-CN.md#部署推理服务)。
本文提供手动部署、不同机器接入、客户端选项及 TTS 的详细用法。

从仓库根目录进入 `SimulInfer/` 后启动：

```bash
cd SimulInfer
bash deploy.sh --install-server --model tencent/Hy-SimulMT-30B-A3B-Interleave
```

需要保存配置时，在当前目录将 `deploy.example.toml` 复制为 `deploy.toml`。

[部署脚本参数](docs/DEPLOYMENT.md)

[配置说明](docs/CONFIGURATION.md) · [音频输入](docs/AUDIO_INPUT.en.md) · [TTS 部署](docs/TTS.zh-CN.md)

## 环境要求

- SimulInfer：Python 3.11 及以上，安装 `[vad]` 依赖以使用默认音频功能。
  可以单独部署在 CPU 机器上。
- 模型服务：vLLM ≥0.31，支持 `HYV3ForCausalLM`，并安装匹配的 PyTorch、CUDA、驱动和音频依赖。
- Hy-SimulMT 模型。

## llama.cpp

前提：支持音频输入的 llama.cpp 服务已在 **8080** 端口加载你的 Hy-SimulMT GGUF 模型。
在 Python 3.11 及以上环境中，从仓库根目录执行：

```bash
cd SimulInfer
python -m pip install .
export MODEL=/path/to/hymt-A3b-tgt_only-1019-Q8_0.gguf
MODEL_TYPE=target_only python serve_llamacpp.py
```

将 `MODEL` 替换为你自己的 GGUF 模型路径，必须与 llama.cpp 服务加载的路径一致。
仅译文模型使用 `MODEL_TYPE=target_only`；源文和译文同出的模型使用 `source_target`。

通过 `serve_llamacpp.py` 启动时，网页打开 **http://127.0.0.1:8094/**。
**8094** 是桥接网页/API端口，**8080** 是其连接的 llama.cpp 模型端口。
另一条 `deploy.sh` / `simulinfer` 启动路径默认使用 **8080** 作为网页/API端口。
文件上传和 WebSocket 接口分别为网页端口下的
`/translate/stream` 和 `/translate/ws`。脚本默认连接 `http://127.0.0.1:8080`；
模型服务在其他地址时设置 `LLAMA_URL`，网页/API 需要其他端口时设置 `PORT`。
远程使用麦克风时，通过 HTTPS 端口转发地址访问。

## 手动部署模型

在已有兼容的 vLLM 环境中获取项目：

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-SimulMT.git Hy-SimulMT
cd Hy-SimulMT/SimulInfer
```

安装模型插件：

```bash
python -m pip install --no-deps --index-url https://pypi.org/simple/ simulst
```

新建 Python 3.12 环境，需要一并安装 vLLM 和音频依赖时执行：

```bash
python -m pip install --index-url https://pypi.org/simple/ 'simulst[server]'
```

使用定制 vLLM 或 CUDA 构建时选第一种方式。缺少音频依赖时，在模型环境安装 `simulst[audio]`。

在模型环境安装 SimulInfer，然后准备模型目录：

```bash
python -m pip install --no-deps .
MODEL_ID="tencent/Hy-SimulMT-30B-A3B-Interleave"
MODEL_DIR="./models/hy-simulmt"
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
simulinfer-prepare-model --model "$MODEL_DIR"
```

已有本地模型时，将 `MODEL_DIR` 设为它的目录并跳过下载。
准备命令会写入当前统一推理模板，并自动备份替换的文件。完成后启动 vLLM：

```bash
vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 --port 8800 --dtype bfloat16 \
  --max-num-seqs 4 --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --enable-chunked-prefill \
  --limit-mm-per-prompt '{"audio":1024}' --chat-template-content-format openai
```

两个命令必须使用同一个 `MODEL_DIR`。根据 GPU 调整资源参数；
若模型服务已在运行，准备完成后需重启对应 vLLM。

## 安装并启动 SimulInfer

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-SimulMT.git Hy-SimulMT
cd Hy-SimulMT/SimulInfer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.2' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[vad]'
simulinfer --init-config config.toml
```

在 `config.toml` 填写模型服务地址：

```toml
[model]
base_url = "http://model-host:8800/v1"
```

启动：

```bash
simulinfer --config config.toml
```

浏览器打开 `http://127.0.0.1:8080/`。需要从其他机器访问时，启动命令加上
`--host 0.0.0.0`。也可安装 `releases/0.5.9/` 中的 wheel。

## 使用网页

1. 选择翻译方向。
2. 上传音频文件，或点击“开始实时对话”并允许麦克风访问。
3. 说话结束后点击“结束并收尾”。
4. 已配置 TTS 时，可以勾选“朗读译文”。

页面自动使用部署的模型，并根据模型能力显示源文，无需选择输出模式。
更换模型后，重启 SimulInfer 并刷新页面即可。

远程使用麦克风需要 HTTPS。通过端口代理访问时，打开完整地址，例如
`https://your-host/proxy/8080/`，并确保代理支持 WebSocket。
服务地址等设置见[配置说明](docs/CONFIGURATION.md)。

## 分块模式

| 选项 | 配置值 | 用途 |
|---|---|---|
| 标准模式（推荐、默认） | `standard` | 每块1600ms，适合日常使用 |
| 短分块模式 | `short_chunk` | **首块必须为1600ms**，后续800ms |

**两种模式的首个输入块都必须是1600ms**，只有后续块可以设为800ms。
在 `config.toml` 中设置；网页跟随服务配置，不提供800ms选项：

```toml
[defaults]
response_mode = "standard"
```

## 标准模式与抗噪模式

- **标准模式**：默认开启，适合日常使用。
- **抗噪模式**：适合环境嘈杂时使用，帮助减少噪声对翻译的干扰。

两种模式都支持文件上传和实时输入。新建抗噪配置：

```bash
simulinfer --init-config noise.toml --profile noise_resistant
# 在 noise.toml 填写模型服务地址后启动
simulinfer --config noise.toml
```

已有配置修改以下设置，然后重启 SimulInfer：

```toml
[inference]
audio_mode = "noise_resistant"
```

改回 `audio_mode = "standard"` 即可恢复标准模式。

## 文件上传与 WebSocket

上传音频文件：

```bash
curl -N http://127.0.0.1:8080/translate/stream \
  -F direction=zh-en -F file=@audio.wav
```

使用实时客户端：

```bash
simulinfer-client audio.wav --url http://127.0.0.1:8080 \
  --transport ws --response-mode standard --realtime
```

自行接入时，连接 `ws://127.0.0.1:8080/translate/ws`，先发送：

```json
{"direction":"zh-en","input_format":"s16le","input_sample_rate":48000,"input_channels":1}
```

收到 `ready` 后发送与声明格式一致的 PCM 二进制音频。结束时发送 `{"eof":true}`，
继续接收直到 `done`。音频格式见[音频输入](docs/AUDIO_INPUT.en.md)，
请求选项见[配置说明](docs/CONFIGURATION.md)。交互式接口文档位于 `/docs`。

## 可选 TTS

在独立模型环境中用 vLLM-Omni 部署 Qwen3-TTS：

```bash
python -m pip install --no-deps /path/to/simulinfer-0.5.9-py3-none-any.whl
simulinfer-tts-omni --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --host 0.0.0.0 --port 8091
```

在 `config.toml` 添加连接配置：

```toml
[tts]
tts_backend = "vllm_omni"
tts_url = "http://127.0.0.1:8091/v1"
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
tts_task_type = "Base"
tts_clone_source = true
tts_history_mode = "paired"
tts_history_window_s = 10
```

此配置自动克隆上传音频或麦克风中的音色，无需另传参考音频或填写文字。
重启 SimulInfer 后，在页面勾选“克隆音色并朗读译文”，或请求设置 `tts=true`。
直接连接已部署的 TTS 服务即可，无需 Docker。更多设置见[TTS 部署](docs/TTS.zh-CN.md)。
