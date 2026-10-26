"""File tasks preserve SSE semantics while each browser HTTP request is short."""
import asyncio
import base64
from dataclasses import replace
import json
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

import simulinfer.app as app_module
from simulinfer.audio import float_wav
from simulinfer.config import load
from simulinfer.file_jobs import FileJobs


def collect(client, token):
    events, cursor = [], 0
    for _ in range(400):
        response = client.get(f'/translate/task/{token}?cursor={cursor}')
        assert response.status_code == 200, response.text
        page = response.json()
        assert page['next_cursor'] == cursor+len(page['events'])
        events.extend(page['events']);cursor = page['next_cursor']
        if page['done'] and cursor == page['total_events']:
            return events
        time.sleep(.005)
    raise AssertionError('file task did not finish')


@pytest.mark.parametrize('mode', ['target_only', 'source_target'])
def test_polling_matches_http_translation_and_tts(mode):
    raw = float_wav(np.full(59200, .1, np.float32))
    with TestClient(app_module.build_app(load(mock=True))) as client:
        assert client.get('/config').json()['file_upload_transport'] == 'polling'
        response = client.post('/translate/stream',files={'file':('test.wav',raw)},
                              data={'model_type':mode,'tts':'true'})
        expected = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith('data:')]
        submitted = client.post('/translate/submit',files={'file':('test.wav',raw)},
                                data={'model_type':mode,'tts':'true'})
        assert submitted.status_code == 200
        assert submitted.headers['cache-control'] == 'no-store'
        actual = collect(client,submitted.json()['task_id'])
        assert actual == expected
        assert actual[-1]['complete'] and actual[-1]['duration_ms'] == 3700
        assert any(base64.b64decode(e['data']) for e in actual if e['event']=='audio')


def test_submit_returns_before_inference_and_cancel_releases_slot(monkeypatch):
    import threading
    started,stopped = threading.Event(),threading.Event()
    async def slow(settings,mode,direction,source,emit,*args):
        started.set()
        try:await asyncio.Event().wait()
        finally:stopped.set()
    monkeypatch.setattr(app_module,'translate',slow)
    raw=float_wav(np.zeros(25600,np.float32))
    with TestClient(app_module.build_app(load(mock=True))) as client:
        response=client.post('/translate/submit',files={'file':('test.wav',raw)})
        assert response.status_code==200 and not stopped.is_set()
        token=response.json()['task_id'];assert started.wait(1)
        assert client.delete(f'/translate/task/{token}').status_code==204
        assert stopped.wait(1)
        rows=collect(client,token)
        assert rows[-1]['event']=='error' and not any(e['event']=='done' for e in rows)
        assert client.get('/health').json()['active_sessions']==0


def test_file_policy_and_error_are_preserved(monkeypatch):
    observed=[]
    original=app_module.session_settings
    def config(*args,**kwargs):
        observed.append(kwargs['transport']);return original(*args,**kwargs)
    async def fail(*args):raise RuntimeError('upstream model failed')
    monkeypatch.setattr(app_module,'session_settings',config)
    monkeypatch.setattr(app_module,'translate',fail)
    with TestClient(app_module.build_app(load(mock=True))) as client:
        result=client.post('/translate/submit',files={'file':('test.wav',float_wav(np.zeros(25600,np.float32)))})
        rows=collect(client,result.json()['task_id'])
        assert rows==[{'event':'error','detail':'upstream model failed'}]
        assert observed==['stream']
        assert client.get('/health').json()['active_sessions']==0


def test_bad_audio_and_unknown_task_fail():
    with TestClient(app_module.build_app(load(mock=True))) as client:
        assert client.post('/translate/submit',files={'file':('empty.wav',b'')}).status_code==400
        assert client.get('/translate/task/missing').status_code==404
        assert client.delete('/translate/task/missing').status_code==404
        result=client.post('/translate/submit',files={'file':('test.wav',float_wav(np.zeros(25600,np.float32)))})
        token=result.json()['task_id']
        assert client.get(f'/translate/task/{token}?cursor=-1').status_code==400
        assert client.get(f'/translate/task/{token}?cursor=999999').status_code==400


@pytest.mark.asyncio
async def test_spooled_pages_are_bounded_and_replayable():
    from simulinfer.file_jobs import Job
    job=Job()
    try:
        expected=[{'event':'audio','data':'x'*10000,'seq':i} for i in range(100)]
        for event in expected:await job.emit(event)
        job.done=True
        first=job.poll(0,max_bytes=40000)
        assert 0<len(first['events'])<100
        assert job.poll(0,max_bytes=40000)==first
        actual=[];cursor=0
        while cursor<len(expected):
            page=job.poll(cursor,max_bytes=40000);actual+=page['events'];cursor=page['next_cursor']
        assert actual==expected
    finally:job.events.close()


@pytest.mark.asyncio
async def test_queue_limits_and_early_cancel_accounting():
    jobs=FileJobs(max_jobs=2,max_pending_bytes=4)
    async def slow(emit):await asyncio.Event().wait()
    try:
        token=jobs.create(4,slow,30)
        with pytest.raises(ValueError):jobs.create(1,slow,30)
        await jobs.cancel(token)
        await asyncio.sleep(0)
        assert jobs.pending_bytes==0
        jobs.create(4,slow,30)
    finally:await jobs.close()
