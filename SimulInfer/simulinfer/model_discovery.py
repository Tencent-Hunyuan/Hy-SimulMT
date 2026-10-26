"""Discover served model IDs and their declared source/translation format."""
from dataclasses import replace
import json
import os
from pathlib import Path
import re

import httpx

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
SOURCE_TOKEN = '<｜hy_place▁holder▁no▁790｜>'
TARGET_TOKEN = '<｜hy_place▁holder▁no▁791｜>'
# Provider-supplied checkpoint identities, not a heuristic for arbitrary names.
PROVIDED_CHECKPOINTS = {
    'A3b_tgt_only_1019_bf16': 'target_only',
    'A3b_src_tgt_1019_bf16': 'source_target',
    'Hy-SimulMT-30B-A3B': 'target_only',
    'Hy-SimulMT-30B-A3B-Interleave': 'source_target',
}


def declared_format(manifest):
    fmt=manifest.get('source_target_format') if isinstance(manifest,dict) else None
    if not isinstance(fmt,dict) or type(fmt.get('enabled')) is not bool:
        raise ValueError('export_manifest.json缺少有效的source_target_format.enabled')
    if not fmt['enabled']:return 'target_only'
    for field,expected in [('source_token',SOURCE_TOKEN),('target_token',TARGET_TOKEN)]:
        if fmt.get(field)!=expected:
            raise ValueError('模型SRC/TGT标记与当前协议不一致')
    return 'source_target'


def read_local_manifest(path):
    file=Path(path)/'export_manifest.json'
    if not file.is_file():return None
    if file.stat().st_size>MAX_MANIFEST_BYTES:raise ValueError('模型元数据文件超过大小限制')
    return declared_format(json.loads(file.read_text()))


async def checkpoint_format(reference,http):
    """Prefer explicit metadata, with an exact provided-model-name fallback."""
    local=Path(reference).expanduser()
    if local.is_dir():
        mode=read_local_manifest(local)
        if mode:return mode,'manifest'
    # A reference may be a Hugging Face repo ID or an inaccessible server path.
    if re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',reference):
        hf_key=os.environ.get('HF_TOKEN','')
        headers={'Authorization':'Bearer '+hf_key} if hf_key else {}
        url='https://huggingface.co/'+reference+'/resolve/main/export_manifest.json'
        async with http.stream('GET',url,headers=headers,follow_redirects=True) as response:
            if response.is_success:
                buf=bytearray()
                async for chunk in response.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf)>MAX_MANIFEST_BYTES:raise ValueError('模型元数据文件超过大小限制')
                return declared_format(json.loads(buf)),'manifest'
            if response.status_code!=404:
                raise ValueError(f'读取Hugging Face模型元数据失败: HTTP {response.status_code}')
    name=reference.rstrip('/').rsplit('/',1)[-1]
    if name in PROVIDED_CHECKPOINTS:return PROVIDED_CHECKPOINTS[name],'provided_model'
    raise ValueError('无法识别所选模型的输出格式；请在checkpoint中提供模型目录或Hugging Face仓库ID')


async def discover(settings, *, http=None):
    if not settings.model_endpoints:return settings
    own=http is None
    http=http or httpx.AsyncClient(timeout=settings.timeout_seconds,trust_env=False)
    metadata_http=(httpx.AsyncClient(timeout=min(settings.timeout_seconds,30),trust_env=True)
                   if own else http)
    models={}
    try:
        for endpoint in settings.model_endpoints:
            key=os.environ.get(endpoint.api_key_env,'')
            headers={'Authorization':'Bearer '+key} if key else {}
            response=await http.get(endpoint.base_url.rstrip('/')+'/models',headers=headers)
            if response.is_error:raise ValueError(f'获取模型列表失败: HTTP {response.status_code}')
            entries=response.json().get('data',[])
            if not isinstance(entries,list) or not entries:
                raise ValueError('模型服务未返回可用模型')
            if any(not isinstance(entry,dict) for entry in entries):
                raise ValueError('模型服务列表格式无效')
            # A named model disambiguates multi-model endpoints; omission lists all.
            if endpoint.model:
                entries=[e for e in entries if e.get('id')==endpoint.model]
                if len(entries)!=1:raise ValueError('model与服务端返回的模型ID不匹配')
            if endpoint.checkpoint and len(entries)!=1:
                raise ValueError('指定checkpoint时需通过model选择唯一模型')
            for entry in entries:
                model_id=entry.get('id')
                if not isinstance(model_id,str) or not model_id:raise ValueError('模型列表缺少有效ID')
                reference=endpoint.checkpoint or entry.get('root') or model_id
                if not isinstance(reference,str) or not reference:raise ValueError('模型目录/仓库ID无效')
                mode,_=await checkpoint_format(reference,metadata_http)
                if mode in models:
                    raise ValueError('发现多个相同输出类型的模型；请为每种类型选择一个model')
                models[mode]=replace(endpoint,model=model_id)
        return replace(settings,models=models,model_endpoints=())
    finally:
        if own:
            await metadata_http.aclose()
            await http.aclose()
