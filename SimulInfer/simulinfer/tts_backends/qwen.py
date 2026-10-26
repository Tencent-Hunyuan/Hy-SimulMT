#!/usr/bin/env python3
"""Qwen3-TTS backend for the existing PCM, voice and session protocols."""

import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from transformers import StoppingCriteria, StoppingCriteriaList
from .qwen_acceleration import accelerate_code_predictor

# Operator-provided resources; no training tree, built-in voice or shared filesystem.
MODEL_DIR = None
VOICE_DIR = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "simulinfer" / "voices"
VOICE_META = VOICE_DIR / "qwen_voices.json"
LEGACY_META = VOICE_DIR / "voices.json"
DEFAULT_VOICE = "default"
DEFAULT_WAV = None
DEFAULT_TEXT = ""
AUTO_VOICE_TEXT = "\u81ea\u52a8\u540c\u4f20\u97f3\u8272"
VOICE_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,48}\Z")
LANGUAGES = {
    "": "Chinese", "zh": "Chinese", "en": "English", "ja": "Japanese",
    "ko": "Korean", "de": "German", "fr": "French", "ru": "Russian",
    "pt": "Portuguese", "es": "Spanish", "it": "Italian", "auto": "Auto",
}
VERSION = "qwen-paired-history-v2-portable"
STATE = {}
VOICES = {}
GEN_LOCK = threading.Lock()
VOICE_LOCK = threading.Lock()
SESSION_LOCK = threading.Lock()
SESSIONS = {}
SESSION_TTL = 300.0
STREAM_HOP = 6
DECODE_CONTEXT = 25


def language_name(lang):
    if lang not in LANGUAGES:
        raise ValueError("Unsupported language")
    return LANGUAGES[lang]


def valid_speed(speed):
    speed = float(speed)
    if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    return speed


def get_voice(voice):
    with VOICE_LOCK:
        item = VOICES.get(voice or DEFAULT_VOICE)
    if item is None:
        raise ValueError("Unknown voice")
    return item


def pcm16(audio):
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def wav_header(sr):
    # An unknown-length WAV header is understood by streaming WAV players.
    import struct
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 0x7FFFF024, b"WAVE", b"fmt ",
        16, 1, 1, sr, sr * 2, 2, 16, b"data", 0x7FFFF000,
    )


class Cancelled(Exception):
    pass


class StopOnCancel(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class QwenSession:
    """Keep aligned speech history while refreshing the speaker between chunks."""

    def __init__(self, voice="", window_s=10.0, speed=1.0, max_chars=240):
        window_s = float(window_s)
        if not math.isfinite(window_s) or not 1 <= window_s <= 20:
            raise ValueError("window_s must be between 1 and 20")
        self.voice = voice or DEFAULT_VOICE
        self.anchor = get_voice(voice)["prompt"]
        self.window_frames = max(1, int(window_s * 12))
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")
        self.max_chars = max_chars
        self.speed = valid_speed(speed)
        self.history = []
        self.steps = 0
        self.cancel = threading.Event()
        self.last_metrics = {}
        self.at = time.monotonic()

    def refresh_voice(self, voice=None):
        """Called under GEN_LOCK; never change conditioning during a chunk."""
        selected = self.voice if voice is None else voice or DEFAULT_VOICE
        anchor = get_voice(selected)["prompt"]
        updated = selected != self.voice or anchor is not self.anchor
        self.voice = selected
        self.anchor = anchor
        return updated

    def prompt(self, lang):
        if self.history and self.history[-1][2] != lang:
            self.history.clear()
        if not self.history:
            return self.anchor
        codes = torch.cat([entry[1] for entry in self.history], dim=0)
        separator = " " if lang == "en" else ""
        text = separator.join(entry[0] for entry in self.history)
        return replace(
            self.anchor, ref_code=codes, ref_text=text,
            x_vector_only_mode=False, icl_mode=True,
        )

    def commit(self, text, codes, lang):
        if len(codes) > 30 * 12:
            # A very long last segment is not a safe Base reference; revert to
            # the anchor instead of retaining an unbounded reference pair.
            self.history.clear()
            self.steps += 1
            return
        self.history.append((text, codes.detach().cpu(), lang))
        # Evict whole pairs; cropping audio without its matching transcript
        # would provide a false text/audio example to the next generation.
        while len(self.history) > 1 and (
            sum(len(entry[1]) for entry in self.history) > self.window_frames
            or sum(len(entry[0]) for entry in self.history) > self.max_chars
        ):
            self.history.pop(0)
        self.steps += 1

    def generate(self, text, lang="", trim=True, voice=None):
        text = str(text).strip()
        if not text:
            return
        language = language_name(lang)
        model = STATE["model"]
        self.at = time.monotonic()
        started = time.perf_counter()
        while not GEN_LOCK.acquire(timeout=0.1):
            if self.cancel.is_set():
                raise Cancelled
        handle = None
        try:
            if self.cancel.is_set():
                raise Cancelled
            voice_updated = self.refresh_voice(voice)
            prompt = self.prompt(lang)
            prompt_history_frames = sum(len(entry[1]) for entry in self.history)
            events = queue.Queue()
            frames = []
            emitted = 0
            first_pcm = None
            decode_seconds = 0.0
            generated_samples = 0
            upsample = STATE["upsample"]
            sr = STATE["sr"]
            eos = model.model.config.talker_config.codec_eos_token_id
            left = prompt.ref_code
            if left is not None:
                left = left[-DECODE_CONTEXT:].to(model.device)
            speech_started = not trim

            def decode_new():
                nonlocal emitted, first_pcm, decode_seconds, generated_samples
                nonlocal speech_started
                if len(frames) <= emitted or self.cancel.is_set():
                    return
                all_codes = torch.cat(frames, dim=0)
                start = max(0, emitted - DECODE_CONTEXT)
                codes = all_codes[start:]
                context = emitted - start
                if start == 0 and left is not None:
                    codes = torch.cat([left, codes], dim=0)
                    context += len(left)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    decoded, _ = model.model.speech_tokenizer.decode(
                        [{"audio_codes": codes}]
                    )
                audio = decoded[0][context * upsample:]
                decode_seconds += time.perf_counter() - t0
                emitted = len(frames)
                if trim and not speech_started:
                    active = np.flatnonzero(np.abs(audio) > 0.008)
                    if not len(active):
                        return
                    audio = audio[max(0, int(active[0]) - int(sr * 0.02)):]
                    speech_started = True
                if self.speed != 1.0 and len(audio):
                    import librosa
                    audio = librosa.effects.time_stretch(
                        audio.astype(np.float32), rate=self.speed,
                    )
                if len(audio):
                    if first_pcm is None:
                        first_pcm = time.perf_counter() - started
                    generated_samples += len(audio)
                    events.put(("pcm", pcm16(audio)))

            def on_frame(module, args, output):
                if self.cancel.is_set():
                    raise Cancelled
                code = output.hidden_states[-1]
                if code is None or int(code[0, 0]) == eos:
                    return
                frames.append(code.detach())
                if len(frames) - emitted >= STREAM_HOP:
                    decode_new()

            def worker():
                nonlocal handle
                talker = model.model.talker
                original_generate = talker.generate
                previous_override = talker.__dict__.get("generate")
                had_override = "generate" in talker.__dict__
                try:
                    def checked_generate(*args, **kwargs):
                        # qwen-tts 0.1.1 does not forward arbitrary outer
                        # kwargs into talker.generate. Install cancellation at
                        # the actual generation call and check its final EOS.
                        criteria = list(kwargs.pop("stopping_criteria", ()) or ())
                        kwargs["stopping_criteria"] = StoppingCriteriaList(
                            criteria + [StopOnCancel(self.cancel)]
                        )
                        output = original_generate(*args, **kwargs)
                        if self.cancel.is_set():
                            raise Cancelled
                        require_codec_eos(output.sequences, eos)
                        return output

                    talker.generate = checked_generate
                    handle = model.model.talker.register_forward_hook(on_frame)
                    input_ids = model._tokenize_texts(
                        [model._build_assistant_text(text)]
                    )
                    ref_ids = None
                    if prompt.ref_text and prompt.icl_mode:
                        ref_ids = model._tokenize_texts(
                            [model._build_ref_text(prompt.ref_text)]
                        )
                    with torch.inference_mode():
                        result, _ = model.model.generate(
                            input_ids=input_ids, ref_ids=ref_ids,
                            voice_clone_prompt=model._prompt_items_to_voice_clone_prompt(
                                [prompt]
                            ),
                            languages=[language], non_streaming_mode=False,
                            max_new_tokens=min(1800, max(180, len(text) * 18)),
                            do_sample=True, top_k=50, top_p=1.0, temperature=0.8,
                            subtalker_dosample=True, subtalker_top_k=50,
                            subtalker_top_p=1.0, subtalker_temperature=0.8,
                            repetition_penalty=1.05,
                            pad_token_id=eos,
                            stopping_criteria=StoppingCriteriaList(
                                [StopOnCancel(self.cancel)]
                            ),
                        )
                    if self.cancel.is_set():
                        raise Cancelled
                    # The final non-EOS frame may not have been forwarded yet.
                    codes = result[0]
                    if len(codes) > len(frames):
                        frames.extend(codes[i:i + 1] for i in range(len(frames), len(codes)))
                    decode_new()
                    if not generated_samples:
                        raise RuntimeError("The model generated no audible speech")
                    self.commit(text, codes, lang)
                    elapsed = time.perf_counter() - started
                    self.last_metrics = {
                        "step": self.steps,
                        "voice": self.voice, "voice_updated": voice_updated,
                        "prompt_history_frames": prompt_history_frames,
                        "tokens": len(codes), "history_frames": sum(
                            len(entry[1]) for entry in self.history
                        ),
                        "first_pcm_ms": round((first_pcm or elapsed) * 1000, 1),
                        "compute_ms": round(elapsed * 1000, 1),
                        "llm_ms": round((elapsed - decode_seconds) * 1000, 1),
                        "decode_ms": round(decode_seconds * 1000, 1),
                        "audio_ms": round(generated_samples / sr * 1000, 1),
                        "code_predictor_cuda_graph": "code_graph" in STATE,
                    }
                    print(
                        f"[qwen-tts] voice={self.voice} lang={lang or 'zh'} "
                        f"step={self.steps} {self.last_metrics} text={text[:60]!r}",
                        flush=True,
                    )
                except BaseException as exc:
                    events.put(("error", exc))
                finally:
                    if had_override:
                        talker.generate = previous_override
                    else:
                        delattr(talker, "generate")
                    if handle is not None:
                        handle.remove()
                        handle = None
                    events.put(("end", None))

            thread = threading.Thread(target=worker, name="qwen-generate")
            thread.start()
            completed = False
            try:
                while True:
                    kind, value = events.get()
                    if kind == "end":
                        completed = True
                        break
                    if kind == "error":
                        raise value
                    if not self.cancel.is_set():
                        yield value
            finally:
                if not completed:
                    self.cancel.set()
                thread.join()
        finally:
            self.at = time.monotonic()
            GEN_LOCK.release()


def require_codec_eos(sequences, eos):
    if sequences is None or sequences.ndim != 2 or sequences.shape[1] == 0:
        raise RuntimeError("TTS generation returned no codec sequence")
    if not bool((sequences[:, -1] == eos).all()):
        raise RuntimeError("TTS reached its generation limit without codec EOS; partial audio is not retained as history")


def make_prompt(wav, text, embedding_only=False):
    model = STATE["model"]
    with GEN_LOCK, torch.inference_mode():
        if embedding_only:
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != 24000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            embedding = model.model.extract_speaker_embedding(audio=audio, sr=24000)
            return VoiceClonePromptItem(
                ref_code=None, ref_spk_embedding=embedding,
                x_vector_only_mode=True, icl_mode=False, ref_text=None,
            )
        return model.create_voice_clone_prompt(
            ref_audio=str(wav), ref_text=text, x_vector_only_mode=False,
        )[0]


def save_voices(records):
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=VOICE_DIR, suffix=".json", delete=False,
    ) as f:
        name = f.name
        try:
            json.dump({"voices": records}, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            os.unlink(name)
            raise
    try:
        os.replace(name, VOICE_META)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def restore_voices():
    records = {}
    for path in (LEGACY_META, VOICE_META):
        if not path.exists():
            continue
        try:
            for record in json.loads(path.read_text())["voices"]:
                records[record["voice_id"]] = record
        except Exception as exc:
            print(f"[qwen-tts] cannot read {path}: {exc}", flush=True)
    for vid, source in records.items():
        try:
            if not VOICE_ID_RE.fullmatch(vid) or vid == DEFAULT_VOICE:
                continue
            record = dict(source)
            text = record["text"].rsplit("<|endofprompt|>", 1)[-1].strip()
            embedding_only = bool(record.get("embedding_only")) or text == AUTO_VOICE_TEXT
            record.update(text=text, embedding_only=embedding_only)
            prompt = make_prompt(record["wav"], text, embedding_only)
            VOICES[vid] = {"prompt": prompt, "meta": record}
            print(f"[qwen-tts] restored voice {vid}", flush=True)
        except Exception as exc:
            print(f"[qwen-tts] skipped voice {vid}: {exc}", flush=True)


def register_voice(payload):
    try:
        b64 = payload.get("audio_b64") or ""
        if not isinstance(b64, str):
            raise ValueError("audio_b64 must be a string")
        b64 = b64.split(",", 1)[-1] if b64.startswith("data:") else b64
        raw = base64.b64decode(b64, validate=True)
        if not raw or len(raw) > 30 * 1024 * 1024:
            raise ValueError("Empty or oversized audio")
        text = str(payload.get("text") or "").strip()
        embedding_only = bool(payload.get("embedding_only"))
        if not text and not embedding_only:
            raise ValueError("Reference transcript is required")
        vid = payload.get("voice_id") or "voice_" + uuid.uuid4().hex[:8]
        if not isinstance(vid, str) or not VOICE_ID_RE.fullmatch(vid) or vid == DEFAULT_VOICE:
            raise ValueError("Invalid or reserved voice_id")
        progressive = bool(payload.get("progressive"))
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory(prefix="qwen-voice-") as tmp:
            input_path = Path(tmp) / "input.audio"
            output_path = Path(tmp) / "reference.wav"
            input_path.write_bytes(raw)
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
                 "-i", str(input_path), "-ac", "1", "-ar", "24000",
                 "-t", "31", str(output_path)],
                check=True, capture_output=True, timeout=60,
            )
            audio, sr = sf.read(output_path, dtype="float32")
            duration = len(audio) / sr
            minimum = 2.3 if progressive else 3.0
            if not minimum <= duration <= 30:
                raise ValueError(f"Reference audio must be {minimum}-30 seconds")
            peak = float(np.max(np.abs(audio)))
            if peak < 1e-4:
                raise ValueError("Reference audio is silent")
            gain = 0.7 / peak if peak < 0.5 or peak > 0.95 else 1.0
            sf.write(output_path, audio * gain, sr, subtype="PCM_16")
            prompt = make_prompt(output_path, text, embedding_only)
            VOICE_DIR.mkdir(parents=True, exist_ok=True)
            # Unique files make concurrent replacement and legacy recordings safe.
            wav_path = VOICE_DIR / f"qwen_{vid}_{uuid.uuid4().hex[:8]}.wav"
            os.replace(output_path, wav_path)
        record = {
            "voice_id": vid, "text": text, "wav": str(wav_path),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "embedding_only": embedding_only, "clean_voice": progressive,
        }
        with VOICE_LOCK:
            previous = VOICES.get(vid)
            records = {
                key: value["meta"] for key, value in VOICES.items()
                if key != DEFAULT_VOICE
            }
            records[vid] = record
            save_voices(list(records.values()))
            VOICES[vid] = {"prompt": prompt, "meta": record}
        if previous is not None:
            old_wav = Path(previous["meta"]["wav"])
            # Active sessions hold tensors, so repeated live updates only need
            # the latest stored WAV. Never delete imported legacy recordings.
            if (old_wav.parent == VOICE_DIR and old_wav != wav_path
                    and old_wav.name.startswith(f"qwen_{vid}_")
                    and all(item["wav"] != str(old_wav) for item in records.values())):
                with contextlib.suppress(OSError):
                    old_wav.unlink()
        return {
            "ok": True, "voice_id": vid, "duration": round(duration, 2),
            "peak": round(peak, 4), "gain": round(gain, 2),
            "embedding_only": embedding_only, "clean_voice": progressive,
        }
    except Exception as exc:
        print(f"[qwen-tts] voice registration failed: {exc}", flush=True)
        return {"error": str(exc)}


HEADERS = {
    "X-Accel-Buffering": "no", "Cache-Control": "no-store",
    "Access-Control-Expose-Headers": "X-Sample-Rate",
}


def audio_response(session, text, lang, raw=True, trim=True, voice=None):
    language_name(lang)
    if voice is not None:
        get_voice(voice)
    headers = dict(HEADERS, **{"X-Sample-Rate": str(STATE["sr"])})

    def generate():
        if not raw:
            yield wav_header(STATE["sr"])
        yield from session.generate(text, lang, trim=trim, voice=voice)

    return StreamingResponse(
        generate(), media_type="application/octet-stream" if raw else "audio/wav",
        headers=headers,
    )


def build_app():
    app = FastAPI(title="Qwen3-TTS")

    @app.get("/health")
    @app.get("/tts_info")
    def info():
        return {
            "loaded": "model" in STATE, "model": str(MODEL_DIR),
            "backend": "qwen3-tts", "sample_rate": STATE.get("sr"),
            "spk": DEFAULT_VOICE, "continuity_version": VERSION,
            "streaming_output": True, "streaming_input": "chunked-with-history",
            "live_voice_updates": True, "history_window_s": 10,
            "instruction_control": False, "keepalive_seconds": 0,
            "code_predictor_cuda_graph": "code_graph" in STATE,
        }

    @app.get("/voices")
    def voices():
        with VOICE_LOCK:
            return {
                "default": DEFAULT_VOICE,
                "voices": [
                    {"id": key, "text": value["meta"]["text"][:60],
                     "at": value["meta"]["at"]}
                    for key, value in VOICES.items() if key != DEFAULT_VOICE
                ],
            }

    @app.post("/register_voice")
    def register(payload: dict = Body(...)):
        return register_voice(payload)

    def new_audio(text, lang, speed, instruct, voice, raw, trim):
        try:
            if instruct:
                raise ValueError("This Base model does not support instruction control")
            return audio_response(
                QwenSession(voice=voice, speed=speed), text, lang, raw, bool(trim),
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/tts_pcm")
    def tts_pcm(text: str = "", lang: str = "", speed: float = 1.0,
                instruct: str = "", voice: str = "", trim: int = 1):
        return new_audio(text, lang, speed, instruct, voice, True, trim)

    @app.get("/tts_stream")
    def tts_stream(text: str = "", lang: str = "", speed: float = 1.0,
                   instruct: str = "", voice: str = "", trim: int = 1):
        return new_audio(text, lang, speed, instruct, voice, False, trim)

    @app.post("/tts_stream")
    def tts_stream_post(payload: dict = Body(...)):
        return new_audio(
            payload.get("text", ""), payload.get("lang", ""),
            payload.get("speed", 1), payload.get("instruct", ""),
            payload.get("voice", ""), False, payload.get("trim", 1),
        )

    @app.post("/session_open")
    def session_open(payload: dict = Body(...)):
        try:
            session = QwenSession(
                payload.get("voice", ""), payload.get("window_s", 10),
                payload.get("speed", 1),
                payload.get("history_max_chars", 240),
            )
            sid = uuid.uuid4().hex[:12]
            with SESSION_LOCK:
                for old in [
                    key for key, item in SESSIONS.items()
                    if time.monotonic() - item.at > SESSION_TTL
                ]:
                    SESSIONS.pop(old).cancel.set()
                SESSIONS[sid] = session
            return {
                "sid": sid, "sample_rate": STATE["sr"], "speed": session.speed,
                "window_s": payload.get("window_s", 10),
                "continuity_version": VERSION,
                "history_max_chars": session.max_chars,
                "live_voice_updates": True,
            }
        except ValueError as exc:
            return {"error": str(exc)}

    @app.post("/session_close")
    def session_close(payload: dict = Body(...)):
        with SESSION_LOCK:
            session = SESSIONS.pop(payload.get("sid"), None)
        if session:
            session.cancel.set()
        return {"ok": session is not None}

    @app.get("/session_info")
    def session_info(sid: str):
        with SESSION_LOCK:
            session = SESSIONS.get(sid)
        if session is None:
            return JSONResponse({"error": "session-not-found"}, status_code=404)
        with GEN_LOCK:
            return {"steps": session.steps, "history_pairs": len(session.history),
                    "history_frames": sum(len(e[1]) for e in session.history),
                    "history_seconds": sum(len(e[1]) for e in session.history) / 12,
                    "history_window_s": session.window_frames / 12,
                    "history_max_chars": session.max_chars,
                    "metrics": dict(session.last_metrics)}

    @app.get("/session_say")
    def session_say(sid: str = "", text: str = "", lang: str = "",
                    voice: str | None = None):
        with SESSION_LOCK:
            session = SESSIONS.get(sid)
        if session is None:
            return StreamingResponse(
                iter(()), media_type="application/octet-stream",
                headers={"X-Error": "session-not-found"},
            )
        try:
            return audio_response(session, text, lang, voice=voice)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def websocket_session(ws, typing):
        await ws.accept()
        output = asyncio.Queue()
        incoming = queue.Queue()
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        cancel_reply = threading.Event()
        finish = object()
        benchmark = False
        lang = ""
        session = None
        try:
            if typing:
                opened = json.loads(await ws.receive_text())
                if not isinstance(opened, dict) or opened.get("op") != "open":
                    raise ValueError("First frame must be op=open")
                lang = opened.get("lang")
                if lang not in ("zh", "en"):
                    raise ValueError("lang must be zh or en")
                benchmark = opened.get("benchmark", False)
                if not isinstance(benchmark, bool):
                    raise ValueError("benchmark must be boolean")
                session = QwenSession(voice=opened.get("voice", ""))
                await ws.send_json({
                    "op": "ready", "sample_rate": STATE["sr"],
                    "format": "s16le", "channels": 1, "voice": session.voice,
                    "lang": lang, "benchmark": benchmark,
                })
            else:
                session = QwenSession()
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await ws.send_json({"op": "error", "message": str(exc)})
            await ws.close()
            return

        def emit(kind, value=None):
            loop.call_soon_threadsafe(output.put_nowait, (kind, value))

        def worker():
            try:
                while not cancelled.is_set():
                    item = incoming.get()
                    if item is finish:
                        break
                    seq, text, received = item
                    waited = (time.perf_counter() - received) * 1000
                    if benchmark:
                        emit("text", {"op": "consumed", "seq": seq, "queue_wait_ms": waited})
                    for pcm in session.generate(text, lang):
                        if cancelled.is_set():
                            raise Cancelled
                        emit("pcm", pcm)
                    if benchmark:
                        emit("text", dict(
                            session.last_metrics, op="metric", seq=seq,
                            queue_wait_ms=round(waited, 1),
                        ))
                if not cancelled.is_set():
                    emit("text", {
                        "op": "done", "done": True, "steps": session.steps,
                        "sample_rate": STATE["sr"],
                        "summary": {"steps": session.steps},
                    })
                elif cancel_reply.is_set():
                    emit("text", {
                        "op": "done", "cancelled": True,
                        "sample_rate": STATE["sr"],
                    })
            except Cancelled:
                if cancel_reply.is_set():
                    emit("text", {
                        "op": "done", "cancelled": True,
                        "sample_rate": STATE["sr"],
                    })
            except Exception as exc:
                emit("text", {"op": "error", "message": str(exc)})
            finally:
                emit("end")

        async def receive():
            seen = set()
            finishing = False
            try:
                while True:
                    msg = json.loads(await ws.receive_text())
                    if not isinstance(msg, dict):
                        raise ValueError("Message must be a JSON object")
                    op = msg.get("op") if typing else (
                        "finish" if msg.get("eof") else "append"
                    )
                    if op == "cancel":
                        cancel_reply.set()
                        cancelled.set()
                        session.cancel.set()
                        incoming.put(finish)
                        return
                    if op == "finish":
                        if finishing:
                            raise ValueError("finish may only be sent once")
                        finishing = True
                        incoming.put(finish)
                        continue
                    if op != "append" or finishing:
                        raise ValueError("Invalid operation or append after finish")
                    text = msg.get("text")
                    seq = msg.get("seq") if typing else len(seen) + 1
                    if isinstance(seq, bool) or not isinstance(seq, int) or seq in seen:
                        raise ValueError("seq must be a unique integer")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("text must be a nonempty string")
                    seen.add(seq)
                    incoming.put((seq, text, time.perf_counter()))
                    if typing:
                        await output.put(("text", {"op": "accepted", "seq": seq}))
            except WebSocketDisconnect:
                cancelled.set()
                session.cancel.set()
                incoming.put(finish)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                cancelled.set()
                session.cancel.set()
                incoming.put(finish)
                await output.put(("text", {"op": "error", "message": str(exc)}))

        task = asyncio.create_task(asyncio.to_thread(worker))
        receiver = asyncio.create_task(receive())
        try:
            while True:
                kind, value = await output.get()
                if kind == "end":
                    break
                if kind == "pcm":
                    if not cancelled.is_set():
                        await ws.send_bytes(value)
                else:
                    await ws.send_json(value)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            cancelled.set()
            session.cancel.set()
            incoming.put(finish)
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            await task
            with contextlib.suppress(Exception):
                await ws.close()

    @app.websocket("/typing_tts_ws")
    async def typing_tts_ws(ws: WebSocket):
        await websocket_session(ws, typing=True)

    @app.websocket("/tts_ws")
    async def tts_ws(ws: WebSocket):
        await websocket_session(ws, typing=False)

    return app


def main():
    global MODEL_DIR, VOICE_DIR, VOICE_META, LEGACY_META, DEFAULT_WAV, DEFAULT_TEXT, DEFAULT_VOICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--default-voice", default="default")
    parser.add_argument("--voice-dir", type=Path, default=VOICE_DIR)
    parser.add_argument("--code-graph", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    MODEL_DIR = args.model.expanduser().resolve()
    DEFAULT_WAV = args.reference_audio.expanduser().resolve()
    DEFAULT_TEXT = args.reference_text.strip()
    DEFAULT_VOICE = args.default_voice
    VOICE_DIR = args.voice_dir.expanduser().resolve()
    VOICE_META = VOICE_DIR / "qwen_voices.json"
    LEGACY_META = VOICE_DIR / "voices.json"
    if not MODEL_DIR.is_dir() or not DEFAULT_WAV.is_file() or not DEFAULT_TEXT:
        parser.error("model directory, reference audio and matching transcript are required")
    if not VOICE_ID_RE.fullmatch(DEFAULT_VOICE):
        parser.error("invalid default voice ID")
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    print(f"[qwen-tts] loading {MODEL_DIR}", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_DIR), device_map=f"cuda:{args.gpu}",
        dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True,
    )
    STATE.update(
        model=model, sr=model.model.speech_tokenizer.get_output_sample_rate(),
        upsample=model.model.speech_tokenizer.get_decode_upsample_rate(),
    )
    if args.code_graph:
        try:
            with torch.inference_mode():
                STATE["code_graph"] = accelerate_code_predictor(model)
            print("[qwen-tts] residual codebook CUDA graph enabled", flush=True)
        except Exception as exc:
            print(f"[qwen-tts] CUDA graph unavailable, using original predictor: {exc}", flush=True)
    prompt = make_prompt(DEFAULT_WAV, DEFAULT_TEXT)
    VOICES[DEFAULT_VOICE] = {
        "prompt": prompt,
        "meta": {"text": DEFAULT_TEXT, "wav": str(DEFAULT_WAV), "at": "builtin"},
    }
    restore_voices()
    # Warm the generation and streaming decoder before accepting traffic.
    list(QwenSession(speed=1.1).generate("\u4f60\u597d\u3002", "zh"))
    list(QwenSession().generate("Hello.", "en"))
    print(f"[qwen-tts] ready on {args.port}, sample_rate={STATE['sr']}", flush=True)
    import uvicorn
    uvicorn.run(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
