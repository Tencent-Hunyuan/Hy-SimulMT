"""Write the current inference template to an explicitly selected checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

MARKER = '{#- simulst-mm openai chat branch v2 #}'
TEMPLATE = Path(__file__).parent / 'templates/openai.jinja'
NAMES = ('chat_template.jinja', 'generation_config.json')


def prepare(model):
    root = Path(model).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('模型路径必须是目录')
    config = json.loads((root / 'config.json').read_text())
    if config.get('architectures') != ['HYV3SpeechForCausalLM']:
        raise ValueError('需要包含音频编码器的 Hy-SimulMT 模型目录')
    tokenizer = json.loads((root / 'tokenizer_config.json').read_text())
    ids = {v['content']: int(k) for k, v in tokenizer.get('added_tokens_decoder', {}).items()}
    required = {'bos_token_id': '<｜hy_begin▁of▁sentence｜>',
                'audio_token_id': '<｜hy_place▁holder▁no▁5｜>',
                'turn_end_token_id': '<｜hy_EOT｜>'}
    for field, token in required.items():
        if type(config.get(field)) is not int or ids.get(token) != config[field]:
            raise ValueError(f'模型配置与 tokenizer 不一致: {field}')
    for token in ('<｜hy_User｜>', '<｜hy_Assistant｜>', '<｜hy_place▁holder▁no▁6｜>',
                  '<｜hy_place▁holder▁no▁8｜>', '<｜hy_place▁holder▁no▁790｜>', '<｜hy_place▁holder▁no▁791｜>'):
        if token not in ids:raise ValueError(f'tokenizer 缺少 {token}')
    eos = config.get('eos_token_id')
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(x) is not int or x < 0 for x in eos):
        raise ValueError('模型缺少有效的 eos_token_id')
    payload = {'bos_token_id': config['bos_token_id'],
               'eos_token_id': list(dict.fromkeys([*eos, config['turn_end_token_id']])),
               'temperature': 0.0, 'max_new_tokens': 64}
    wanted = {NAMES[0]: TEMPLATE.read_bytes(),
              NAMES[1]: (json.dumps(payload, indent=2) + '\n').encode()}
    before = {}
    for name in NAMES:
        path = root / name
        if path.is_symlink():raise ValueError(f'目标文件不能是符号链接: {name}')
        before[name] = path.read_bytes() if path.exists() else None
    changed = [n for n in NAMES if before[n] != wanted[n]]
    if not changed:return {'model': str(root), 'changed': [], 'backup': None}
    backup = Path(tempfile.mkdtemp(prefix='simulinfer-assets-backup-', dir=root))
    for name in NAMES:
        if before[name] is None:(backup / (name + '.ABSENT')).touch()
        else:
            shutil.copy2(root / name, backup / name)
            if (backup / name).read_bytes() != before[name]:
                raise RuntimeError('备份期间模型文件发生改变，请重试')
    pending = {}
    replaced = []
    try:
        for name in changed:
            fd, tmp = tempfile.mkstemp(prefix='.' + name + '-', dir=root)
            pending[name] = Path(tmp)
            with os.fdopen(fd, 'wb') as out:
                out.write(wanted[name]);out.flush();os.fsync(out.fileno())
            os.chmod(tmp, ((root / name).stat().st_mode & 0o777) if before[name] is not None else 0o644)
        for name in NAMES:
            path=root / name
            now=path.read_bytes() if path.exists() else None
            if path.is_symlink() or now != before[name]:
                raise RuntimeError('模型文件被其他进程修改，已停止写入')
        for name in changed:
            os.replace(pending[name], root / name);replaced.append(name)
    except BaseException:
        # Recover only our own replacements; never overwrite a later writer.
        for name in replaced:
            path=root / name
            if path.read_bytes() == wanted[name]:
                if before[name] is None:path.unlink()
                else:
                    fd,tmp=tempfile.mkstemp(prefix='.restore-',dir=root)
                    os.close(fd);shutil.copy2(backup / name,tmp);os.replace(tmp,path)
        raise
    finally:
        for tmp in pending.values():tmp.unlink(missing_ok=True)
    return {'model': str(root), 'changed': changed, 'backup': str(backup)}


def main(argv=None):
    parser=argparse.ArgumentParser(description='为所选模型写入当前统一推理模板；替换前自动备份，不修改权重')
    parser.add_argument('--model',required=True,help='将由 vLLM 加载的本地模型目录')
    args=parser.parse_args(argv)
    try:result=prepare(args.model)
    except (ValueError,OSError,RuntimeError) as exc:parser.error(str(exc))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('请让 vLLM 加载以上目录；已运行的 vLLM 需要重启。')


if __name__=='__main__':main()
