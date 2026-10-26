# Qwen TTS 部署与使用

[English](TTS.en.md)

使用 vLLM-Omni 部署 Qwen3-TTS，然后在 SimulInfer 配置其服务地址即可。
无需 Docker。使用下方启动命令可开启输入音色克隆；已有普通 vLLM-Omni 服务时，
也可使用下方的固定参考音色配置。

## 1. 启动 TTS 服务

在独立 Python 环境中，安装支持 Qwen3-TTS 且版本匹配的 vLLM 和 vLLM-Omni。
PyTorch、CUDA 和驱动按所选版本的安装要求准备。克隆启动器需要 vLLM-Omni 0.31
或提供兼容 Qwen TTS 接口的构建；模型环境中需安装 `soxr` 和 `soundfile`。

```bash
python -m pip install --no-deps /path/to/simulinfer-0.5.9-py3-none-any.whl
simulinfer-tts-omni --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --host 0.0.0.0 --port 8091
```

模型 ID 可换成本地模型目录。需要自定义部署参数时，使用 `--deploy-config`
指定对应版本支持的 Omni 配置文件。

## 2. 连接 SimulInfer

在 `config.toml` 添加：

```toml
[tts]
tts_backend = "vllm_omni"
tts_url = "http://127.0.0.1:8091/v1"
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
tts_task_type = "Base"
tts_sample_rate = 24000
tts_clone_source = true
tts_history_mode = "paired"
tts_history_window_s = 10
tts_speed = 1.0
```

`tts_model` 使用 TTS 服务 `/v1/models` 返回的模型名。
输入音色克隆直接使用上传音频或麦克风，无需额外的参考文件或文字。
开始录音后会先采集一小段音色，之后每10秒更新，并保留前文语音参考。

完整示例见 `configs/tts_vllm_omni.toml`。

## 3. 开启朗读

重启 SimulInfer，在页面勾选“克隆音色并朗读译文”，或命令行加 `--tts`、API 请求设置 `tts=true`。

```bash
simulinfer-client audio.wav --url http://127.0.0.1:8080 \
  --transport ws --tts --tts-output translated.wav
```

## 前文语音参考

Base 模型设置 `tts_history_mode="paired"` 可启用前文语音参考，帮助后续语音衔接。
`tts_history_window_s` 设置大约保留多少秒前文；设为 `tts_history_mode="none"` 可关闭。

## 固定参考音色

要使用固定音色，可连接普通 vLLM-Omni Base 服务，并设置：

```toml
[tts]
tts_backend = "vllm_omni"
tts_url = "http://tts-host:8091/v1"
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
tts_task_type = "Base"
tts_clone_source = false
tts_reference_audio = "/path/to/reference.wav"
tts_reference_text = "与参考音频严格对应的文字。"
tts_history_mode = "paired"
```

参考文件需放在 SimulInfer 所在机器。

## 其他音色和服务

使用 CustomVoice 模型时，填写对应模型名，设置 `tts_task_type="CustomVoice"` 和
模型支持的 `tts_voice`，移除 Base 参考音频及文字字段。

其他服务连接方式见[可选 TTS 后端](TTS_ALTERNATIVES.en.md)。
