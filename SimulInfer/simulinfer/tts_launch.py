"""Launch a Qwen3-TTS vLLM-Omni deployment in its own GPU environment."""
import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import sys


def command(args):
    executable = shutil.which('vllm-omni') or shutil.which('vllm') or 'vllm-omni'
    prefix=([sys.executable,'-m','simulinfer.tts_omni_source'] if getattr(args,'clone_source',True) else [executable])
    cmd = [*prefix, 'serve', args.model, '--omni', '--host', args.host, '--port', str(args.port),
           '--gpu-memory-utilization', str(args.gpu_memory_utilization)]
    if args.deploy_config:
        cmd += ['--deploy-config', args.deploy_config]
    if args.trust_remote_code:
        cmd.append('--trust-remote-code')
    return cmd


def check_environment(clone_source=False):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    fields = OpenAICreateSpeechRequest.model_fields
    required = {'ref_audio', 'ref_text', 'task_type', 'stream_format', 'response_format'}
    if clone_source:
        required.update({'extra_params','x_vector_only_mode'})
        from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
        if not callable(getattr(Qwen3TTSAdapter,'build',None)) or not callable(getattr(Qwen3TTSPromptEmbedsBuilder,'put_ref_audio_artifacts',None)):
            raise RuntimeError('This vLLM-Omni build does not support input voice cloning')
    if required - set(fields):
        raise RuntimeError('This vLLM-Omni lacks Qwen reference-audio/raw-stream fields: '
                           + ', '.join(sorted(required - set(fields))))
    return {name: metadata.version(name) for name in ('vllm', 'vllm-omni')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Qwen3-TTS Base/CustomVoice directory or model identifier')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--deploy-config', default='', help='Optional pipeline config supported by your vLLM-Omni release')
    parser.add_argument('--gpu-memory-utilization', type=float, default=.8)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--clone-source',action=argparse.BooleanOptionalAction,default=True)
    args = parser.parse_args(argv)
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error('gpu-memory-utilization must be in (0, 1]')
    local = Path(args.model)
    if local.is_dir():
        config = json.loads((local / 'config.json').read_text())
        if config.get('model_type') != 'qwen3_tts':
            parser.error('This launcher expects a Qwen3-TTS checkpoint')
        if args.clone_source and config.get('tts_model_type') not in (None,'base'):
            parser.error('Input voice cloning requires a Base checkpoint; use --no-clone-source for CustomVoice')
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    try:
        print('TTS environment:', check_environment(args.clone_source), flush=True)
    except Exception as exc:
        parser.error(f'{type(exc).__name__}: {exc}. Install a matching vLLM-Omni/vLLM pair in a separate environment.')
    if args.clone_source:
        os.environ['SIMULINFER_TTS_MODEL']=args.model
        if 'VLLM_PLUGINS' in os.environ:
            plugins=[p.strip() for p in os.environ['VLLM_PLUGINS'].split(',') if p.strip()]
            os.environ['VLLM_PLUGINS']=','.join(dict.fromkeys([*plugins,'simulinfer_tts_source']))
    else:
        os.environ.pop('SIMULINFER_TTS_MODEL',None)
    os.execvp(cmd[0], cmd)


def native_main():
    # Keep the HTTP-only SimulInfer importable without any GPU packages.
    try:
        module = importlib.import_module('simulinfer.tts_backends.qwen')
    except ImportError as exc:
        raise SystemExit(f'{exc}. Install simulinfer[tts-qwen] in a separate GPU environment.') from exc
    module.main()


if __name__ == '__main__':
    sys.exit(main())
