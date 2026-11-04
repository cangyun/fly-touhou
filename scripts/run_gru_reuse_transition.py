"""One finite dependency: finish v1 tests, then start the sealed GRU reuse test.

No schedule, no training restart, no fly work, no choosing hyperparameters from
held-out outcomes. If v1 already passes G2 there is no need for this extra run.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.experiment import file_sha256
from fly_touhou.native_diagnostic import write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def live(pid,needle):
    try:
        p=psutil.Process(pid)
        return p.is_running() and needle in ' '.join(p.cmdline())
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False


def launch(spec,log):
    job=read(spec)
    project=Path(job['cwd'])
    with log.open('xb') as out:
        subprocess.run([job['python'],
            str(project/'scripts/background_job.py'),'--spec',str(spec)],cwd=project,
            stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True)
    args=parser.parse_args()
    prepared=read(args.preparation)
    run=args.preparation.parent
    state_path=run/'transition_status.json'
    if state_path.exists():raise FileExistsError('Never duplicate a dependency pipeline')
    assert file_sha256(prepared['protocol']['path'])==prepared['protocol']['sha256']
    assert file_sha256(prepared['source_manifest']['path'])==prepared['source_manifest']['sha256']
    parent=Path(prepared['parent_fixed_course'])
    control=Path('./outputs/legacy-build/background_jobs/native-list-diagnostic-suite-v1/status.json')
    state={'schema':'finite-GRU-reuse4-transition-v1','status':'waiting-for-v1-tests',
        'transition_script':str(Path(__file__)),
        'transition_script_sha256':file_sha256(Path(__file__)),
        'pid':__import__('os').getpid(),'preparation':str(args.preparation),
        'fly_work_permitted':False,'restart_existing_processes':False}
    try:
        while True:
            if (run/'stop_before_launch.json').exists() or (parent/'stop_after_current_phase.json').exists():
                state['status']='stopped-before-any-new-launch';return
            job=read(control)
            pipeline=read(parent/'suite/status.json')
            state.update(parent_job_status=job['status'],parent_pipeline_status=pipeline['status'],
                         heartbeat_unix=time.time())
            write_json(state_path,state)
            if job['status'] in ('starting','running'):
                if not (live(job.get('child_pid',0),'run_native_diagnostic_suite.py') or
                        live(job['worker_pid'],'background_job.py')):
                    raise RuntimeError('Parent handles missing without terminal record; preserve and inspect')
                time.sleep(15);continue
            if job['status']!='finished' or job.get('exit_code')!=0 or pipeline['status']!='finished':
                raise RuntimeError('Parent suite failed/stopped; no additional training launched')
            outcome=read(parent/'suite/result.json')
            state['parent_G2_achieved']=outcome['G2_achieved']
            if outcome['G2_achieved']:
                state['status']='v1-GRU-gate-passed-extra-run-not-needed';return
            # Every spec represents a new branch; background_job rejects duplicates.
            for index,spec in enumerate(prepared['training_specs']):
                launch(Path(spec),run/f'launch_seed_{index}.log')
            launch(Path(prepared['suite_spec']),run/'launch_suite.log')
            state.update(status='new-GRU-reuse4-training-and-audited-tests-launched',
                         launched_specs=prepared['training_specs']+[prepared['suite_spec']])
            return
    except BaseException as error:
        state.update(status='failed-preserved-no-restart',error=repr(error));raise
    finally:
        state['updated_unix']=time.time();write_json(state_path,state)
        print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
