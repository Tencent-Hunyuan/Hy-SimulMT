"""Public browser assets and the actual browser URL module under a proxy prefix."""
import base64
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from simulinfer.app import build_app
from simulinfer.config import load


@pytest.mark.parametrize('prefix',['','/proxy/8050'])
def test_page_resources_and_config_are_usable_under_mount(prefix):
    child=build_app(load(mock=True))
    app=FastAPI() if prefix else child
    if prefix:app.mount(prefix,child)
    with TestClient(app) as client:
        page=client.get(prefix+'/')
        assert page.status_code==200 and page.headers['cache-control']=='no-store'
        assert 'id="mode"' not in page.text
        assert 'href="./static/' in page.text and 'src="./static/' in page.text
        for asset in ('app.js','urls.js','file-upload.js','recorder.js','style.css'):
            assert client.get(prefix+'/static/'+asset).status_code==200
        response=client.get(prefix+'/config')
        assert response.status_code==200 and response.headers['cache-control']=='no-store'


@pytest.mark.parametrize('origin,prefix,protocol',[
    ('http://localhost:8050','','ws:'),
    ('https://example.test','/proxy/8050','wss:'),
    ('https://example.test','/nested/speech','wss:'),
])
def test_browser_url_module_retains_entire_prefix(origin,prefix,protocol):
    node=shutil.which('node')
    if not node:pytest.skip('Node is required to execute the browser URL module')
    path=Path(__file__).parents[1]/'simulinfer/static/urls.js'
    # Substitute only the module loader URL; execute the shipped JS functions.
    script=path.read_text().replace('import.meta.url',json.dumps(origin+prefix+'/static/urls.js'))
    module='data:text/javascript;base64,'+base64.b64encode(script.encode()).decode()
    code='''import {serviceURL,websocketURL} from %s;
console.log(JSON.stringify({config:String(serviceURL('config')),file:String(serviceURL('translate/stream')),recorder:String(serviceURL('static/recorder.js')),ws:String(websocketURL())}));'''%json.dumps(module)
    result=json.loads(subprocess.check_output([node,'--input-type=module','-e',code],text=True))
    assert result['config']==origin+prefix+'/config'
    assert result['file']==origin+prefix+'/translate/stream'
    assert result['recorder']==origin+prefix+'/static/recorder.js'
    assert result['ws']==origin.replace(origin.split(':')[0]+':',protocol,1)+prefix+'/translate/ws'
