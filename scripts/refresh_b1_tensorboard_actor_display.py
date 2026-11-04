"""Replace only the read-only TensorBoard bridge after its tests pass."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def main():
    import psutil
    control=ROOT/'artifacts/engineering/tensorboard-live-v3'
    proof=control/'actor-display-restart-20261025-v1.json'
    if proof.exists():raise FileExistsError('Do not repeat this display transition')
    checks=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024/tensorboard-full-teacher-BC-v6-tests.xml'
    suites=list(ET.parse(checks).getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=12 or any(
            int(s.get('failures',0))+int(s.get('errors',0)) for s in suites):
        raise ValueError('Require12 passing display checks')
    previous=read(control/'teacher-display-restart-20261025-v1.json')['new_identity']
    process=psutil.Process(previous['pid'])
    if abs(process.create_time()-previous['create_time'])>.01 or read(control/'status.json')['pid']!=process.pid:
        raise ValueError('Live bridge identity differs;no process touched')
    config_path=control/'configuration.json';config=read(config_path)
    config.update(actor_warmup_source='./outputs/engineering/B1-autonomous-dodge-20261024/teacher-full-actor-warmup-v1',
                  export_revision='independent-native-teacher-and-CUDA-BC-v6',updated_utc=utc_now())
    write(config_path,config)
    write(proof,dict(utc=utc_now(),old_identity=previous,tests_passed=12,
        source_sha256=hashlib.sha256((ROOT/'scripts/tensorboard_live.py').read_bytes()).hexdigest(),
        native_or_neural_jobs_modified=False,previous_event_files_and_cursors_preserved=True))
    (control/'STOP').write_text('Replace read-only display bridge after12 checks',encoding='utf-8')
    deadline=time.monotonic()+25
    while process.is_running() and time.monotonic()<deadline:time.sleep(.2)
    if process.is_running():raise RuntimeError('Old display has not exited;no duplicate launched')
    (control/'STOP').unlink()
    env={**os.environ,'TEMP':'./outputs/tmp','TMP':'./outputs/tmp',
         'TMPDIR':'./outputs/tmp','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
    interpreter=Path('./outputs/tools/tensorboard-runtime/Scripts/pythonw.exe')
    with (control/'actor-v6.stdout.log').open('wb') as out,(control/'actor-v6.stderr.log').open('wb') as err:
        child=subprocess.Popen([str(interpreter),str(ROOT/'scripts/tensorboard_live.py'),'--config',str(config_path)],
            cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW,close_fds=True)
    deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        state=read(control/'status.json')
        if state.get('status')=='running' and state['pid']!=previous['pid']:
            current=psutil.Process(state['pid'])
            row=read(proof);row.update(new_identity=dict(pid=current.pid,create_time=current.create_time()),
                launcher_pid=child.pid,verified_utc=utc_now(),URL=state['url'],last_read_error=state['last_read_error'])
            write(proof,row)
            print(json.dumps(row));return
        time.sleep(.3)
    raise RuntimeError('New display is not confirmed;inspect logs without touching training')


if __name__=='__main__':main()
