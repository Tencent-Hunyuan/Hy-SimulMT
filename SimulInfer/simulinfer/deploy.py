"""Install and start an isolated inference deployment from a source checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = 'tencent/Hy-SimulMT-30B-A3B-Interleave'
DEFAULTS = {
    'model': DEFAULT_MODEL, 'runtime_dir': '.simulinfer', 'host': '0.0.0.0',
    'port': 8080, 'model_port': 8800, 'max_sessions': 4, 'max_num_seqs': 4,
    'tensor_parallel_size': 1, 'gpu_memory_utilization': .9,
    'startup_timeout': 900, 'install_server': False, 'gpus': '',
    'backend_url': '', 'server_python': '', 'app_python': '',
}


def load_options(args):
    path = Path(args.config).expanduser().resolve() if args.config else ROOT / 'deploy.toml'
    data = tomllib.loads(path.read_text()) if path.exists() else {}
    if args.config and not path.exists():raise ValueError(f'Configuration not found: {path}')
    if set(data) - {'deployment', 'inference', 'tts'}:raise ValueError('Use [deployment], optional [inference] and [tts] in deployment config')
    options = dict(DEFAULTS)
    deployment = data.get('deployment', {})
    if not isinstance(deployment, dict) or set(deployment) - set(DEFAULTS):
        raise ValueError('Unknown deployment option')
    options.update(deployment)
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:options[key] = value
    for key in ('port', 'model_port'):
        if type(options[key]) is not int or not 1 <= options[key] <= 65535:
            raise ValueError(f'{key} must be a valid TCP port')
    for key in ('max_sessions', 'max_num_seqs', 'tensor_parallel_size'):
        if type(options[key]) is not int or options[key] < 1:raise ValueError(f'{key} must be positive')
    for key in ('gpu_memory_utilization', 'startup_timeout'):
        value=options[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive')
    if options['gpu_memory_utilization'] > 1:raise ValueError('gpu_memory_utilization must be <= 1')
    if type(options['install_server']) is not bool:raise ValueError('install_server must be true/false')
    if options['max_sessions']>128:raise ValueError('max_sessions must be <= 128')
    for key in ('model', 'runtime_dir', 'host', 'backend_url', 'server_python', 'app_python', 'gpus'):
        if not isinstance(options[key],str):raise ValueError(f'{key} must be a string')
    if not options['model'].strip() or not options['runtime_dir'].strip():raise ValueError('model and runtime_dir cannot be empty')
    base=path.parent if path.exists() else ROOT
    runtime=Path(options['runtime_dir']).expanduser()
    options['runtime_dir'] = str((base/runtime).resolve())
    for key in ('server_python','app_python'):
        value=options[key]
        if value and ('/' in value or value.startswith('.')):
            options[key]=os.path.abspath(base/Path(value).expanduser())
    candidate=Path(options['model']).expanduser()
    if candidate.is_absolute() or str(candidate).startswith('.') or (base/candidate).is_dir():
        options['model']=str((base/candidate).resolve())
    if options['backend_url'] and not options['backend_url'].startswith(('http://','https://')):
        raise ValueError('backend_url must be an HTTP(S) model API URL')
    if not options['backend_url'] and options['port']==options['model_port']:
        raise ValueError('port and model_port must differ')
    if options['backend_url'] and options['install_server']:
        raise ValueError('Choose backend_url or install_server, not both')
    if options['install_server'] and options['server_python']:
        raise ValueError('install_server creates its own environment; omit server_python')
    if not isinstance(data.get('tts',{}),dict):raise ValueError('[tts] must be a table')
    tts=dict(data.get('tts',{}))
    if tts.get('tts_reference_audio'):
        tts['tts_reference_audio']=str((base/Path(tts['tts_reference_audio']).expanduser()).resolve())
    inference=data.get('inference',{})
    if not isinstance(inference,dict):raise ValueError('[inference] must be a table')
    options['inference']=dict(inference)
    return options, tts


def assert_port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:sock.bind((host,port))
        except OSError as exc:raise RuntimeError(f'Port {port} is occupied; choose another port. Existing services were not changed.') from exc


def python_path(value):
    executable=shutil.which(value or sys.executable)
    if not executable:raise ValueError(f'Python executable not found: {value}')
    # Preserve virtualenv symlinks: resolving them would select the base env.
    return os.path.abspath(executable)


def run(command, *, env=None, capture=False):
    return subprocess.run(command,env=env,check=True,text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def ensure_env(directory, python):
    executable=directory/'bin/python'
    if executable.exists():return str(executable)
    if directory.exists():raise ValueError(f'Incomplete environment: {directory}; use a new runtime_dir')
    run([python,'-m','venv',str(directory)])
    return str(executable)


def check_server(python, env):
    code="""import sys
if sys.version_info < (3, 12):
    raise SystemExit('The model environment needs Python 3.12 or newer')
from simulst_mm.plugin import register
register()
import vllm
from importlib.metadata import entry_points
plugins={p.name for p in entry_points(group='vllm.logits_processors')}
if not {'simulinfer_simulst_ngram','simulinfer_st_ngram'} <= plugins:
    raise SystemExit('Install SimulInfer in this model environment and restart vLLM to load no-repeat processors')
print('Model environment is ready')
"""
    run([python,'-c',code],env=env)


def stage_model(source, runtime):
    """Copy metadata, reuse weights, and prepare only our deployment directory."""
    source=Path(source).resolve(strict=True)
    if not (source/'config.json').is_file():raise ValueError(f'Model config.json is missing: {source}')
    fingerprint=hashlib.sha256(str(source).encode())
    fingerprint.update(Path(__file__).with_name('templates').joinpath('openai.jinja').read_bytes())
    for name in ('config.json','tokenizer_config.json','export_manifest.json','model.safetensors.index.json'):
        file=source/name
        if file.is_file():fingerprint.update(file.read_bytes())
    parent=runtime/'models';parent.mkdir(parents=True,exist_ok=True)
    dest=parent/fingerprint.hexdigest()[:16]
    if dest.exists():
        if not (dest/'deployment_source.json').is_file():raise ValueError(f'Incomplete staged model: {dest}')
        if (dest/'chat_template.jinja').read_bytes()!=Path(__file__).with_name('templates').joinpath('openai.jinja').read_bytes():
            raise ValueError(f'Staged model template was modified: {dest}; use a new runtime_dir')
        return dest
    staging=Path(tempfile.mkdtemp(prefix='.preparing-',dir=parent))
    try:
        for file in source.iterdir():
            if file.name.startswith('.') or '-assets-backup-' in file.name:continue
            if file.is_dir() or file.suffix in {'.safetensors','.bin','.pt','.pth'}:
                (staging/file.name).symlink_to(file,target_is_directory=file.is_dir())
            elif file.is_file():shutil.copy2(file,staging/file.name)
        # The script is stdlib-only, so this works before frontend installation.
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('_simulinfer_prepare',Path(__file__).with_name('prepare_model.py'))
        module=module_from_spec(spec);spec.loader.exec_module(module)
        module.prepare(staging)
        (staging/'deployment_source.json').write_text(json.dumps({'source':str(source)},indent=2))
        os.replace(staging,dest)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    return dest


def model_command(options, python, model):
    command = [python,'-m','vllm.entrypoints.cli.main','serve',str(model),
            '--host','127.0.0.1','--port',str(options['model_port']),
            '--served-model-name',Path(options['model']).name,
            '--dtype','bfloat16','--tensor-parallel-size',str(options['tensor_parallel_size']),
            '--max-num-seqs',str(options['max_num_seqs']),
            '--gpu-memory-utilization',str(options['gpu_memory_utilization']),
            '--enable-prefix-caching','--enable-chunked-prefill',
            '--limit-mm-per-prompt','{"audio":1024}','--chat-template-content-format','openai']
    if options.get('inference',{}).get('no_chinese_target'):
        command += ['--logits-processors','simulinfer.logits_plugins:NoChineseTargetProcessor']
    return command


def write_config(path, options, backend, tts):
    def value(v):
        if type(v) is bool:return str(v).lower()
        if type(v) in (str,int,float):return json.dumps(v,ensure_ascii=False)
        raise ValueError('Configuration values must be strings, numbers or booleans')
    sections={'server':{'host':options['host'],'port':options['port'],'max_sessions':options['max_sessions']},
              'model':{'base_url':backend},'defaults':{'tts':bool(tts and tts.get('tts_backend','none')!='none')}}
    if options.get('inference'):sections['inference']=options['inference']
    if tts:sections['tts']=tts
    text='\n\n'.join('['+section+']\n'+'\n'.join(k+' = '+value(v) for k,v in values.items())
                     for section,values in sections.items())+'\n'
    if path.exists() and path.read_text()!=text:
        raise ValueError(f'Existing generated config differs: {path}. Use a new runtime_dir or edit the deployment settings.')
    if not path.exists():
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(text)


def wait_http(url, processes, timeout, *, key_env=''):
    deadline=time.monotonic()+timeout
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers={}
    key=os.environ.get(key_env,'')
    if key:headers['Authorization']='Bearer '+key
    while time.monotonic()<deadline:
        for process in processes:
            if process.poll() is not None:raise RuntimeError('A service stopped during startup; see the deployment logs')
        try:
            with opener.open(urllib.request.Request(url,headers=headers),timeout=2) as response:
                data=json.load(response)
                if response.status==200:return data
        except (OSError,ValueError,urllib.error.URLError):pass
        time.sleep(.5)
    raise RuntimeError(f'Service startup timed out; see logs. Endpoint: {url}')


def stop_owned(processes):
    for process in reversed(processes):
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+15
    for process in reversed(processes):
        try:process.wait(timeout=max(.1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()


def deploy(options, tts, *, no_install=False, dry_run=False):
    runtime=Path(options['runtime_dir'])
    if dry_run:
        print(json.dumps({'runtime_dir':str(runtime),'model':options['model'],
              'backend_url':options['backend_url'] or f"http://127.0.0.1:{options['model_port']}/v1",
              'install_server':options['install_server'],'port':options['port'],
              'tts_enabled':bool(tts)},ensure_ascii=False,indent=2))
        return
    assert_port_free(options['host'],options['port'])
    if not options['backend_url']:assert_port_free('127.0.0.1',options['model_port'])
    runtime.mkdir(parents=True,exist_ok=True)
    lock=runtime/'deploy.lock'
    try:lock_fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError as exc:raise RuntimeError(f'Deployment is locked: {lock}; use another runtime_dir or check the earlier process') from exc
    with os.fdopen(lock_fd,'w') as f:f.write(str(os.getpid()))
    processes=[];logs=[]
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    env.pop('SIMULINFER_CONFIG',None)
    model_env=dict(env)
    if options['gpus']:model_env['CUDA_VISIBLE_DEVICES']=options['gpus']
    def start(command, name, process_env):
        log_path=runtime/(name+'.log')
        file=log_path.open('ab');logs.append(file)
        process=subprocess.Popen(command,env=process_env,stdin=subprocess.DEVNULL,
                                 stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
        print(f'{name} starting; log: {log_path}',flush=True)
        return process
    try:
        model_python=None
        if not options['backend_url']:
            if options['install_server']:
                model_python=python_path(str(runtime/'model-env/bin/python')) if no_install else ensure_env(runtime/'model-env',sys.executable)
                if not no_install:run([model_python,'-m','pip','install','simulst[server]>=0.1.0'],env=model_env)
            else:model_python=python_path(options['server_python'])
            if not no_install:
                run([model_python,'-m','pip','install','--no-deps',str(ROOT)],env=model_env)
            try:check_server(model_python,model_env)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError('Model environment is not ready. Install simulst in a compatible vLLM environment, pass --server-python, or use --install-server.') from exc
        if no_install:
            app_python=python_path(options['app_python'] or str(runtime/'app-env/bin/python'))
        else:
            if options['app_python']:raise ValueError('app_python is for --no-install; automatic installation uses an isolated app-env')
            app_python=ensure_env(runtime/'app-env',sys.executable)
            run([app_python,'-m','pip','install','torch>=2.2','--index-url','https://download.pytorch.org/whl/cpu'],env=env)
            run([app_python,'-m','pip','install',str(ROOT)+'[vad]'],env=env)
        backend=options['backend_url'].rstrip('/')
        config=runtime/'config.toml'
        if not backend:
            model_ref=Path(options['model']).expanduser()
            if not model_ref.is_dir():
                target=runtime/'downloads'/hashlib.sha256(options['model'].encode()).hexdigest()[:16]
                print('Downloading model:',options['model'],flush=True)
                run([model_python,'-c','from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1],local_dir=sys.argv[2])',options['model'],str(target)],env=model_env)
                model_ref=target
            model=stage_model(model_ref,runtime)
            backend=f"http://127.0.0.1:{options['model_port']}/v1"
            # Validate generated app configuration before allocating model resources.
            write_config(config,options,backend,tts)
            run([app_python,'-c','from simulinfer.config import load; import sys; load(sys.argv[1])',str(config)],env=env)
            start(model_command(options,model_python,model),'model',model_env)
        else:
            write_config(config,options,backend,tts)
            run([app_python,'-c','from simulinfer.config import load; import sys; load(sys.argv[1])',str(config)],env=env)
        models=wait_http(backend+'/models',processes,options['startup_timeout'],key_env='VLLM_API_KEY')
        if not models.get('data'):raise RuntimeError('Model endpoint has no available models')
        start([app_python,'-m','simulinfer.app','--config',str(config)],'app',env)
        host='127.0.0.1' if options['host'] in ('0.0.0.0','') else options['host']
        wait_http(f"http://{host}:{options['port']}/health",processes,options['startup_timeout'])
        print(f"Ready: http://{host}:{options['port']}/",flush=True)
        print(f'Config: {config}\nKeep this command running; Ctrl+C stops its services.',flush=True)
        while True:
            if any(p.poll() is not None for p in processes):raise RuntimeError('A service exited; see the deployment logs')
            time.sleep(.5)
    finally:
        stop_owned(processes)
        for file in logs:file.close()
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Install and start the model API and SimulInfer web/API service')
    parser.add_argument('--config')
    for name in ('model','runtime-dir','host','backend-url','server-python','app-python','gpus'):
        parser.add_argument('--'+name)
    for name in ('port','model-port','max-sessions','max-num-seqs','tensor-parallel-size'):
        parser.add_argument('--'+name,type=int)
    parser.add_argument('--gpu-memory-utilization',type=float)
    parser.add_argument('--startup-timeout',type=float)
    parser.add_argument('--install-server',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--no-install',action='store_true',help='Use already installed environments')
    parser.add_argument('--dry-run',action='store_true',help='Print settings without installing or starting anything')
    args=parser.parse_args(argv)
    def interrupted(signum,frame):raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        options,tts=load_options(args)
        deploy(options,tts,no_install=args.no_install,dry_run=args.dry_run)
    except KeyboardInterrupt:
        print('Deployment stopped.',flush=True)
    except (ValueError,OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        print('Deployment failed: '+str(exc),file=sys.stderr)
        return 1
    finally:signal.signal(signal.SIGTERM,previous)
    return 0


if __name__=='__main__':raise SystemExit(main())
