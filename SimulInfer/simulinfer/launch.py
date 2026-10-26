"""Build a reviewable stock vLLM command; no in-process model wrapper."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from importlib import metadata


def logits_api(requested='auto'):
    if requested!='auto':return requested
    setting=os.environ.get('VLLM_USE_V2_MODEL_RUNNER')
    if setting in {'0','1'}:return 'v2' if setting=='1' else 'v1'
    try:
        release=metadata.version('vllm').split('+')[0].split('.')
        return 'v2' if (int(release[0]),int(release[1])) >= (0,31) else 'v1'
    except (metadata.PackageNotFoundError,ValueError,IndexError):
        return 'v1'


def command(args):
    model = Path(args.model_dir).expanduser().resolve()
    config = json.loads((model / 'config.json').read_text())
    if config.get('architectures') != ['HYV3SpeechForCausalLM']:
        raise ValueError('需要simulst_mm导出的HYV3SpeechForCausalLM统一模型目录')
    audio = config.get('audio_config', {})
    for key, expected in {'sample_rate':16000, 'hop_length':160, 'fsk_size':2,
                          'n_window':50, 'max_samples':480000}.items():
        if key in audio and audio[key] != expected:
            raise ValueError(f'SimulInfer音频token预算仅支持{key}={expected}，实际为{audio[key]}')
    if not (model/'preprocessor_config.json').is_file():
        raise ValueError('统一模型缺少preprocessor_config.json')
    from .prepare_model import TEMPLATE
    if (model/'chat_template.jinja').read_bytes()!=TEMPLATE.read_bytes():
        raise ValueError('先执行 simulinfer-prepare-model --model MODEL_DIR，再启动服务')
    template = TEMPLATE
    cmd = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', str(model),
           '--host', args.host, '--port', str(args.port), '--served-model-name', args.model_name,
           '--tensor-parallel-size', str(args.tp), '--dtype', 'bfloat16',
           '--max-model-len', str(args.max_model_len), '--max-num-seqs', str(args.max_num_seqs),
           '--max-num-batched-tokens', str(args.max_num_batched_tokens),
           '--gpu-memory-utilization', str(args.gpu_memory_utilization),
           '--enable-prefix-caching', '--enable-chunked-prefill', '--enforce-eager',
           '--limit-mm-per-prompt', json.dumps({'audio':args.audio_items}),
           '--chat-template', str(template), '--chat-template-content-format', 'openai']
    # N-gram processors are registered through package entry points. Only
    # the optional target-language processor needs an explicit CLI argument.
    if args.no_chinese_target:
        module = 'logits_v2' if logits_api(getattr(args,'logits_api','auto'))=='v2' else 'target_language'
        cmd += ['--logits-processors', f'simulinfer.{module}:NoChineseTargetProcessor']
    return cmd


def main():
    parser=argparse.ArgumentParser(description='打印或启动原生vLLM音频模型API')
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--model-name',default='simulst')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8800)
    parser.add_argument('--tp',type=int,default=1)
    parser.add_argument('--max-model-len',type=int,default=8192)
    parser.add_argument('--max-num-seqs',type=int,default=4)
    parser.add_argument('--max-num-batched-tokens',type=int,default=16384)
    parser.add_argument('--gpu-memory-utilization',type=float,default=.9)
    parser.add_argument('--audio-items',type=int,default=1024)
    parser.add_argument('--no-chinese-target', action=argparse.BooleanOptionalAction, default=True,
                        help='默认屏蔽中译英目标汉字；--no-no-chinese-target关闭')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--logits-api',choices=['auto','v1','v2'],default='auto',
                        help='Match the selected vLLM model runner; upstream >=0.31 defaults to v2')
    args=parser.parse_args()
    if args.max_num_seqs<1 or args.tp<1 or args.audio_items<1:
        parser.error('并发/TP/audio-items必须为正')
    cmd=command(args)
    # Keep the selected processor interface and the actual runner in agreement,
    # including when an explicit CLI override differs from the inherited env.
    runner='1' if logits_api(args.logits_api)=='v2' else '0'
    print(f'VLLM_USE_V2_MODEL_RUNNER={runner} '+shlex.join(cmd),flush=True)
    if not args.dry_run:
        os.environ['VLLM_USE_V2_MODEL_RUNNER']=runner
        os.execv(sys.executable,cmd)


if __name__=='__main__':main()
