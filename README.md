**English** | [简体中文](README.zh-CN.md)
<p align="center">
  <img src="assets/tencent-hy-logo.png" width="400" alt="Tencent Hy" />
</p>

<h1 align="center">
  <sub><img src="assets/hy-simulmt-logo.png" width="50" height="50" alt="Hy-SimulMT logo"></sub> Hy-SimulMT
</h1>

<p align="center"><b>A Low-Latency, High-Quality Model for Real-Time Simultaneous Translation</b></p>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Tencent%20Hy-ffc107)](https://huggingface.co/collections/tencent/hy-simulmt)
[![ModelScope](https://img.shields.io/badge/ModelScope-Tencent%20Hy-624aff)](https://www.modelscope.cn/collections/Tencent-Hunyuan/Hy-SimulMT)
[![GitHub](https://img.shields.io/badge/GitHub-Hy--SimulMT-181717?logo=github)](https://github.com/Tencent-Hunyuan/Hy-SimulMT)

</div>

## Introduction

**Hy-SimulMT** is a real-time simultaneous translation model developed by Tencent Hunyuan, supporting **bidirectional speech-to-speech translation between Chinese and English**. It continuously receives incoming audio and translates as the speaker talks, producing both translated text and speech. With **real-time voice cloning**, it preserves the speaker's vocal characteristics while conveying the message accurately.

Designed for live communication settings such as meetings, speeches, and interviews, Hy-SimulMT aims to convey information accurately while reducing the disruption that waiting introduces into a conversation. To evaluate this balance between translation quality and response speed, we combine public benchmarks, real-world audio, and comparisons with professional simultaneous interpreters.

In our **RealSI long-form audio evaluation**, Hy-SimulMT achieved a **StreamLAAL of approximately 3 seconds**, the lowest among the systems shown in the comparison. Its GEMBA score exceeded those of the compared open-source models and four of the five commercial APIs.

This repository provides the **SimulInfer inference service** and **SimulTrain training tools**. Speech output and voice cloning require a configured TTS service. You can also try Hy-SimulMT in the **Hy Translate App**.

<p align="center"><b>Scan to download the Hy Translate App</b></p>
<table align="center">
  <tr>
    <th align="center">Android</th>
    <th align="center">iOS</th>
  </tr>
  <tr>
    <td align="center"><img src="assets/android.png" width="180" height="180" alt="Hy Translate App download QR code for Android" /></td>
    <td align="center"><img src="assets/ios.png" width="180" height="180" alt="Hy Translate App download QR code for iOS" /></td>
  </tr>
</table>

## News

- **2026.10.26 | Release**: Hy-SimulMT is released with bidirectional real-time simultaneous translation between Chinese and English and real-time voice cloning. Project and model resources are linked below.

## Results

### Keeping Up Faster, Conveying Meaning More Accurately

The challenge of simultaneous translation is not simply to convert one language into another, but to understand and formulate a translation while the speaker is still talking. Waiting for more information often helps resolve the complete meaning, but also makes listeners wait longer. Speaking earlier, on the other hand, places greater demands on incremental understanding and translation generation.

Hy-SimulMT focuses on this balance between quality and latency: **translating as a continuous audio stream arrives, so that listeners receive information promptly while the original meaning is preserved as accurately as possible.**

We compared Hy-SimulMT with several open-source simultaneous translation models and commercial APIs on the open-source RealSI long-form audio benchmark. The results show that Hy-SimulMT achieves competitive translation quality at lower latency.

<div align="center">
  <img src="assets/realsi_gemba_streamlaal_redstar_logos_v2.png" width="100%" alt="RealSI long-form audio evaluation: GEMBA vs. StreamLAAL" />
</div>

*The horizontal axis shows StreamLAAL, a measure of streaming translation lag, in seconds (lower is better); the vertical axis shows the GEMBA translation quality score (higher is better). StreamLAAL is not the same as the time to first speech output.*

In this evaluation, Hy-SimulMT achieved a **StreamLAAL of approximately 3 seconds**, the lowest among the systems shown. Its GEMBA score exceeded those of the compared open-source models and four of the five commercial APIs.

These results suggest that lower latency does not necessarily require a substantial sacrifice in translation quality. Hy-SimulMT offers a more timely way to access cross-language information during extended speeches, interviews, and meetings.

### From Public Benchmarks to Real-World Settings

#### Translation Quality on Multiple Short-Form Audio Benchmarks

In addition to long-form audio, we evaluated the model on four public short-form audio benchmarks—FLEURS, CoVoST2, ACL6060, and RealSI-Sentence—to examine translation performance under different data conditions.

| API / Model | FLEURS | CoVoST2 | ACL6060 | RealSI-Sentence |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-LiveTranslate | 88.4 | **82.1** | 80.5 | 84.4 |
| **Hy-SimulMT** | **90.3** | **75.5** | **81.2** | **84.7** |
| Gemini3.5-LiveTranslate | **92.4** | 74.4 | **82.6** | 82.2 |
| Confucius4 R2T2 → T3PO | 89.4 | 76.8 | 76.1 | 77.7 |
| Seed-LiveInterpret2.0 | 85.8 | 67.7 | 74.8 | 80.0 |
| SimulS2ST-Omni | 78.6 | 66.6 | 73.1 | 77.4 |
| GPT-Realtime-Translate | 77.7 | 50.8 | 76.4 | 76.7 |
| XunFei-Simult-Interpretation | 73.4 | 57.0 | 57.5 | 70.9 |
| SeamlessStreaming | 72.8 | 60.9 | 49.8 | 56.9 |

*Higher scores are better. The entire Hy-SimulMT row is bolded to highlight our model; other bolded values indicate the highest score in the corresponding column.*

Hy-SimulMT scored higher than the other open-source models in the table on FLEURS, ACL6060, and RealSI-Sentence. Its **84.7** on RealSI-Sentence was the highest among all listed systems. Results across these benchmarks also help us identify the model's strengths and areas for improvement.

#### A Matched Comparison with Professional Simultaneous Interpreters

To further examine performance on real-world content, we collected **18 Chinese and English audio recordings newly appearing online from September 2026 onward, each approximately 6 minutes long**. The recordings cover speeches, interviews, and meetings across ten domains: software and AI, medicine, manufacturing and energy, natural sciences, agriculture and food, business management, law, education, culture and tourism, and sports and daily life.

Alongside Hy-SimulMT, we invited six interpreters with years of practical simultaneous interpreting experience to interpret these recordings. Three held CATTI Level 1 interpreting certificates, and three held CATTI Level 2 interpreting certificates. For each certification level, the highest score among the three interpreters was used for comparison.

<div align="center">
  <img src="assets/hy_vs_human_streamlaal_horizontal_hyblue_en.png" width="80%" alt="StreamLAAL latency comparison between Hy-SimulMT and CATTI Level 1 and Level 2 interpreters" />
</div>

As this small-scale capability assessment shows, Hy-SimulMT achieves translation quality on par with expert human simultaneous interpreters, while often responding faster (with lower latency).

*These findings apply only to the samples, participating interpreters, and evaluation settings used in this test. They do not constitute a comprehensive assessment of speech quality, voice similarity, or other dimensions.*

##### Examples Comparing Hy-SimulMT with Human Interpreters

Beyond the quantitative results, the following four videos illustrate how Hy-SimulMT and human interpreters compare in information delivery, response timing, and phrasing in the target language.

Click the fullscreen button in each player, or select **Watch larger** to open the full-size video.

<table>
  <tr>
    <td align="center" width="50%">
      <p><b>Comparison Example 1</b></p>
      <video src="https://github.com/user-attachments/assets/4bb35f80-d0cb-41f4-b605-62da7a1d7122" width="380" controls></video>
      <p>
        <a href="https://github.com/user-attachments/assets/4bb35f80-d0cb-41f4-b605-62da7a1d7122">Watch larger · 32 seconds</a>
      </p>
    </td>
    <td align="center" width="50%">
      <p><b>Comparison Example 2</b></p>
      <video src="https://github.com/user-attachments/assets/5394f3f7-c555-4309-af7f-2755601ba33e" width="380" controls></video>
      <p>
        <a href="https://github.com/user-attachments/assets/5394f3f7-c555-4309-af7f-2755601ba33e">Watch larger · 38 seconds</a>
      </p>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <p><b>Comparison Example 3</b></p>
      <video src="https://github.com/user-attachments/assets/cf8b73f4-f829-44a9-a9c1-fe1df7b3be7e" width="380" controls></video>
      <p>
        <a href="https://github.com/user-attachments/assets/cf8b73f4-f829-44a9-a9c1-fe1df7b3be7e">Watch larger · 26 seconds</a>
      </p>
    </td>
    <td align="center" width="50%">
      <p><b>Comparison Example 4</b></p>
      <video src="https://github.com/user-attachments/assets/37706426-77a9-4ec8-a4d7-b3edf65a3ac9" width="380" controls></video>
      <p>
        <a href="https://github.com/user-attachments/assets/37706426-77a9-4ec8-a4d7-b3edf65a3ac9">Watch larger · 36 seconds</a>
      </p>
    </td>
  </tr>
</table>



## Beyond Text: Preserving the Speaker's Voice

Cross-language communication conveys not only information, but also the characteristics of a speaker's voice. Hy-SimulMT supports real-time voice cloning, preserving the original speaker's vocal characteristics while generating speech in the target language.

The English-to-Chinese demo features three speakers. The model presents their distinct vocal characteristics, making it easier for listeners to tell who is speaking while following the translation.

*The following two videos are offline replays provided for demonstration; their playback timing should not be interpreted as a live latency measurement.*

### English → Chinese Simultaneous Translation and Voice Cloning

https://github.com/user-attachments/assets/0a56fff0-8144-465d-83d0-7a014e914e07

[Watch the English → Chinese demo (1 minute 39 seconds)](assets/videos/en/en_zh_simultaneous_translation.mp4)

### Chinese → English Simultaneous Translation and Voice Cloning

https://github.com/user-attachments/assets/cf8b0814-1b1d-4c67-ac89-9d9433e94462

[Watch the Chinese → English demo (35 seconds)](assets/videos/en/zh_en_simultaneous_translation.mp4)

From cross-language speeches to multi-speaker interviews, this capability opens up possibilities for more natural and more readily identifiable spoken communication.

## Continuing to Improve Through Real-World Feedback

Simultaneous translation ultimately serves real communication. Beyond benchmark scores, we value professional interpreters' feedback on information delivery, response timing, and how natural the translated speech sounds.

*The following statements are full English translations of the original Chinese remarks.*

> Wang Weiwei, Vice Dean of the School of Interpreting and Translation Studies at Guangdong University of Foreign Studies, said: "In real-world evaluations, Tencent Hunyuan's simultaneous translation produces natural-sounding target-language speech and accurately conveys the main information in ordinary statements. To control latency, the system starts speaking first and makes corrections afterward; these corrections sound similar to an interpreter's self-corrections. This marks a new milestone in China's AI simultaneous interpreting capabilities (automatic speech translation, AST). As the technology continues to be refined, AI simultaneous interpreting is moving from 'assisting understanding' toward 'usable communication' in low-risk settings, providing important technical support for a new paradigm of human–machine collaboration in translation."

> Professor Zhong Weihe, former President of Guangdong University of Foreign Studies, Rector of Macao Millennium College in Macao, China, former Executive Vice President of the Translators Association of China, and Chair of its Interpreting Committee, said: "Simultaneous interpreting is one of the most challenging areas of language services. Systematic evaluations show that Tencent Hunyuan's simultaneous translation model demonstrates notable capabilities in speech recognition stability, simultaneous translation response speed, and the quality of spoken translation output. These research and development achievements reflect the continued progress of AI simultaneous interpreting technology and provide strong technical support for bringing machine simultaneous interpreting into real-world settings."

This feedback also points to a clear application path: helping AI simultaneous translation become useful first in appropriate low-risk settings, while continuing to improve through human–machine collaboration. Professional interpreters remain important in scenarios requiring strict accountability, expert review, or high-stakes decisions.

## Model Links

| Model | Description | Download |
| --- | --- | --- |
| Hy-SimulMT-30B-A3B | Hunyuan simultaneous translation model, translation-only output | [Model](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B) |
| Hy-SimulMT-30B-A3B-FP8 | Hunyuan simultaneous translation model, translation-only output, FP8 quantization | [Model](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-FP8) |
| Hy-SimulMT-30B-A3B-GGUF | Hunyuan simultaneous translation model, translation-only output, GGUF format | [Model](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-GGUF) |
| Hy-SimulMT-30B-A3B-Interleave | Hunyuan simultaneous translation model, source text and translation output | [Model](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave) |
| Hy-SimulMT-30B-A3B-Interleave-FP8 | Hunyuan simultaneous translation model, source text and translation output, FP8 quantization | [Model](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave-FP8) |
| Hy-SimulMT-30B-A3B-Interleave-GGUF | Hunyuan simultaneous translation model, source text and translation output, GGUF format | [Model](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF) |

Model collections: [Hugging Face](https://huggingface.co/collections/tencent/hy-simulmt) | [ModelScope](https://www.modelscope.cn/collections/Tencent-Hunyuan/Hy-SimulMT).

Try Hy-SimulMT in the **Hy Translate App**. Download QR codes are available in the [Introduction](#introduction) above.

From understanding the message to keeping pace with the speaker and delivering more natural speech, Hy-SimulMT aims to reduce the wait in cross-language communication and help information reach listeners more smoothly.

---

## Deploy inference

> **Follow the [SimulInfer README](SimulInfer/README.md) for step-by-step installation, or use the one-command deployment below.** With `--install-server`, the command installs dependencies in isolated deployment environments. Running it again repeats the installation steps and may reinstall SimulInfer or change dependency versions in those environments. If you use a custom vLLM/CUDA environment or need control over dependency versions, follow the step-by-step guide or use the **Existing compatible vLLM environment** option below.

**SimulInfer** provides a web page, audio file uploads, real-time WebSocket and
optional TTS with input voice cloning. The deployed model determines whether
source text is displayed; no output-mode selection is needed.

### vLLM

Prepare a Linux machine with Python 3.12+, Git, an NVIDIA GPU with enough memory
for the selected model, and a CUDA-compatible driver. For an existing model
environment, use vLLM >=0.31 with `HYV3ForCausalLM` support and install `simulst`.

Clone the project and enter the inference directory. Run the deployment commands
below from `SimulInfer/`:

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-SimulMT.git Hy-SimulMT
cd Hy-SimulMT
cd SimulInfer
```

**New environment — install dependencies and start both services:**

```bash
bash deploy.sh --install-server --model tencent/Hy-SimulMT-30B-A3B-Interleave
```

For an accessible local checkpoint, replace the model ID:

```bash
bash deploy.sh --install-server --model /path/to/Hy-SimulMT-30B-A3B-Interleave
```

The command installs dependencies in separate environments, prepares the model,
starts vLLM, then starts the web/API service. The first run downloads dependencies
and, when using a Hugging Face ID, the model. Keep the command running.

**Existing compatible vLLM environment:**

```bash
/path/to/model-env/bin/python -m pip install --no-deps simulst
bash deploy.sh --server-python /path/to/model-env/bin/python --model /path/to/model
```

This uses your existing model environment and creates a separate environment
for the web service. For custom vLLM/CUDA builds, use this option.

**Already running a model API:**

```bash
bash deploy.sh --backend-url http://model-host:8800/v1
```

This starts only SimulInfer. See the [model setup instructions](SimulInfer/README.md#manual-model-deployment)
if the existing model needs its inference template prepared.

#### Open and use

Wait for `Ready: http://127.0.0.1:8080/`, then open that address on the server,
or `http://SERVER_IP:8080/` from another machine. Select a translation direction,
upload audio or start the microphone, then click **Stop and finish** when done.
Remote microphone access requires HTTPS; use your HTTPS port-forwarding URL if
available, including any `/proxy/8080/` prefix.

Upload a file from the command line:

```bash
curl -N http://127.0.0.1:8080/translate/stream \
  -F direction=en-zh -F file=@audio.wav
```

Or open another terminal in `Hy-SimulMT/SimulInfer/` and use the installed client:

```bash
.simulinfer/app-env/bin/simulinfer-client audio.wav --url http://127.0.0.1:8080
```

When TTS is enabled in the deployment configuration, the same command also saves
the returned speech to `outputs/translated.wav`. `/docs` provides API documentation.

#### Configure and stop

Defaults: web/API port **8080**, local model port **8800**. Files are saved under
`SimulInfer/.simulinfer/`, including `config.toml`, `app.log` and `model.log` when a model is started.
Press **Ctrl+C** to stop the services started by the command. An occupied port
causes a startup error; select another port with `--port` or `--model-port`.

For a saved deployment configuration:

```bash
cp deploy.example.toml deploy.toml
# Edit the model, GPU, ports and optional TTS connection.
bash deploy.sh --config deploy.toml
```

Examples:

```bash
bash deploy.sh --install-server --model /path/to/model --gpus 0
bash deploy.sh --install-server --model /path/to/model --gpus 0,1 --tensor-parallel-size 2
bash deploy.sh --backend-url http://model-host:8800/v1 --port 8081 --runtime-dir .simulinfer-alt
```

Use `SIMULINFER_PYTHON=python3.12` before the command if `python3` is not the
Python version you want. Reuse the same deployment command to start it again.
The detailed guide covers [installation and deployment options](SimulInfer/README.md),
[configuration](SimulInfer/docs/CONFIGURATION.md), [audio clients](SimulInfer/docs/AUDIO_INPUT.en.md),
and [TTS setup](SimulInfer/docs/TTS.en.md).

### llama.cpp

We added architecture-specific support for Hy-SimulMT to llama.cpp to enable inference with GGUF models. We have submitted [PR #31198](https://github.com/ggml-org/llama.cpp/pull/31198) to the llama.cpp repository.

> This support has not yet been merged into the official llama.cpp repository. The PR currently supports inference through `llama-mtmd-cli`; `llama-server` support is planned as a follow-up. Before using the server deployment steps below, ensure that your llama.cpp version includes the required Hy-SimulMT server support.

Use this option when an audio-capable llama.cpp server has already loaded your
Hy-SimulMT GGUF model on port **8080**. In a Python 3.11+ environment, run the
following from the repository root:

```bash
cd SimulInfer
python -m pip install .
export MODEL=/path/to/hymt-A3b-tgt_only-1019-Q8_0.gguf
MODEL_TYPE=target_only python serve_llamacpp.py
```

Replace `MODEL` with your own GGUF path, exactly as loaded by the llama.cpp server.
Use `MODEL_TYPE=target_only` for a translation-only model, or `source_target` for
a model that outputs both source text and translation.

For `serve_llamacpp.py`, open **http://127.0.0.1:8094/** for the web page.
This is the bridge web/API port; **8080** is the default llama.cpp model port.
The separate `deploy.sh` / `simulinfer` launch path defaults to web/API port **8080**.
For the llama.cpp bridge, file uploads and WebSocket use
`/translate/stream` and `/translate/ws` on port **8094**. The script connects to
`http://127.0.0.1:8080` by default; set `LLAMA_URL` for a different model server
and `PORT` for a different web/API port. For remote microphone access, use an
HTTPS port-forwarding address.

## Train — SimulTrain

**SimulTrain** continues training from a downloaded Hy-SimulMT checkpoint.
Configure model paths, datasets, GPUs and output directories in `SimulTrain/.env`.

From the repository root:

```bash
cd SimulTrain
```

See the [training guide](SimulTrain/README.md) or [Simplified Chinese version](SimulTrain/README.zh-CN.md) for environment setup, data formats, and training commands.
Use a separate Python environment for training.

## Contributors

Mao Zheng, Bo Lv, Mingrui Sun, Zheng Li, Fangxu Liu, Tao Chen, Bingxin Qu, Shan Yang, rickkan, Guanghua Yu, Decheng Wu, Steve Yves, Hongyu Li

## Citing Hy-SimulMT

If you find Hy-SimulMT useful, please cite:

```bibtex
@misc{hy_simulmt,
  title  = {Hy-SimulMT：A Low-Latency, High-Quality Model for Real-Time Simultaneous Translation},
  author = {Mao Zheng and Bo Lv and Mingrui Sun and Zheng Li and Fangxu Liu and Tao Chen and Bingxin Qu},
  year   = {2026},
  month  = oct,
  url    = {https://hy.tencent.com/model/open-source-model/detail?id=hy-simulmt}
}
```

## Contact Us

If you would like to leave a message for our research and product teams, please contact the Tencent Hunyuan LLM team at [hunyuan_opensource@tencent.com](mailto:hunyuan_opensource@tencent.com).
