# 配置使用说明

## 创建配置

```bash
simulinfer --init-config config.toml
```

修改配置后重启：

```bash
simulinfer --config config.toml
```

## 连接模型

单个模型服务只需填写地址：

```toml
[model]
base_url = "http://model-host:8800/v1"
```

首次部署时，先在模型环境执行 `simulinfer-prepare-model --model /path/to/model`，
再让 vLLM 加载该目录。页面自动显示当前模型支持的内容，无需指定是否输出源文。
更换模型后重启 SimulInfer。

需要指定模型名称时：

```toml
[model]
base_url = "http://model-host:8800/v1"
model = "your-served-model-name"
```

无法自动识别自定义名称时，补充所选模型的 Hugging Face ID 或本地目录：

```toml
checkpoint = "tencent/Hy-SimulMT-30B-A3B-Interleave"
```

## 服务地址与并发

```toml
[server]
host = "0.0.0.0"
port = 8080
max_sessions = 4
timeout_seconds = 900
max_upload_mb = 100
```

| 设置 | 用途 |
|---|---|
| `host` | `127.0.0.1`仅供本机访问；`0.0.0.0`允许远程访问 |
| `port` | 网页和API端口 |
| `max_sessions` | 同时处理的会话数，请根据模型服务容量设置 |
| `timeout_seconds` | 请求超时时间，单位秒 |
| `max_upload_mb` | 上传文件大小上限，单位MB |

## 默认选项

```toml
[defaults]
response_mode = "standard"
vad = true
tts = false
```

- `response_mode`：`standard`为推荐的1600ms模式；`short_chunk`为后续800ms的短分块模式。**两种模式的首块都必须为1600ms**。
- `vad`：是否启用静音检测。
- `tts`：是否默认朗读译文；开启前需配置TTS服务。

## 开启抗噪模式

适合嘈杂环境，支持文件上传和实时输入。新建配置：

```bash
simulinfer --init-config noise.toml --profile noise_resistant
```

或修改已有配置：

```toml
[inference]
audio_mode = "noise_resistant"
```

保持静音检测开启。设置 `audio_mode = "standard"` 可恢复标准模式。

## 连接多个模型服务

将单个 `[model]` 替换为地址列表：

```toml
[[models]]
base_url = "http://model-host:8800/v1"

[[models]]
base_url = "http://model-host:8801/v1"
```

页面默认使用列表中的第一个可用模型。需要指定默认模型时，在已有 `[defaults]` 中
添加 `model_type = "target_only"` 或 `model_type = "source_target"`，分别选择已部署的
仅译文模型或源文＋译文模型。

## 请求选项

文件上传表单与WebSocket首条消息可使用以下选项。省略时使用服务默认值。

| 选项 | 可选值 | 用途 |
|---|---|---|
| `direction` | `zh-en` / `en-zh` | 中译英／英译中，默认中译英 |
| `response_mode` | `standard` / `short_chunk` | 分块模式 |
| `vad` | `true` / `false` | 静音检测 |
| `tts` | `true` / `false` | 朗读译文 |
| `voice` | 字符串 | TTS服务支持的音色名称 |
| `model_type` | `target_only` / `source_target` | 多模型接入时选择模型，单模型无需填写 |

WebSocket音频格式和收发步骤见[音频输入](AUDIO_INPUT.en.md)。
TTS连接和参考音色设置见[TTS部署](TTS.zh-CN.md)。

## 需要鉴权的服务

模型API密钥通过环境变量 `VLLM_API_KEY` 设置；TTS密钥使用 `TTS_API_KEY`。
使用私有Hugging Face模型时设置 `HF_TOKEN`。

## 远程与代理访问

远程使用麦克风需要HTTPS。端口代理需支持WebSocket，请打开包含完整路径的地址，
例如 `https://your-host/proxy/8080/`。
升级安装包或修改配置后，重启 SimulInfer 并刷新页面。

## 配置示例

- `config.example.toml`：默认部署。
- `configs/noise_resistant.toml`：抗噪模式。
- `configs/tts_vllm_omni.toml`：连接vLLM-Omni TTS。
- `configs/tts_openai.toml`：连接其他兼容的TTS服务。

生成的完整配置中保留其他可选参数，日常使用可保持默认值。
