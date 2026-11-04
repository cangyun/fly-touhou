"""Exactly three sequential collect/aggregate-fit rounds,then one same3 dev run."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.venv/Scripts/python.exe'
BASE=Path('./outputs/distillation')
ENV=Path('./outputs/engineering/B1-autonomous-dodge-20261024/D95-fresh-Adam-T025-natural30-v3/trial_000_both/config.json')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError('Fresh finite pipeline output required;never resume/retry')
    out.mkdir(parents=True)
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[:2])
    state=dict(schema='fly-finite-DAgger-iterations3-v1',status='running',pid=os.getpid(),create_time=process.create_time(),
        source_script_sha256=sha(__file__),rounds=3,train_seed_masters=[2026102702,2026102703,2026102704],brain_seed=613711,
        fit='512fullbatchAdam.01/noWD/.5old8192+.5allAccumulatedDAgger/noheldselection',
        final_only_development=dict(episodes=3,seed=2027431401,expected_native_seeds=[6517,7487,17506]),
        automatic_retry=False,round_results=[],started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    write(out/'launch.json',state)
    def publish():state['updated_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();write(out/'progress.json',state)
    def run(stage,command,child_output,wall_cap):
        if (out/'STOP').exists():raise RuntimeError('Operator STOP before '+stage)
        child_output.parent.mkdir(parents=True,exist_ok=True)
        log=out/(stage.replace('/','-')+'.stdout.log');err=out/(stage.replace('/','-')+'.stderr.log')
        with log.open('xb') as stdout,err.open('xb') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUTF8':'1'})
            state.update(stage=stage,child_pid=child.pid,child_create_time=psutil.Process(child.pid).create_time(),
                command=[str(x) for x in command],child_output=str(child_output),stdout=str(log),stderr=str(err));publish()
            begun=time.monotonic();stop_time=None
            while child.poll() is None:
                if (out/'STOP').exists() or time.monotonic()-begun>wall_cap:
                    if child_output.exists():(child_output/'STOP').touch(exist_ok=True)
                    stop_time=time.monotonic() if stop_time is None else stop_time
                    if time.monotonic()-stop_time>120:raise RuntimeError('Child did not exit after STOP;no following stage will run')
                time.sleep(.5)
            state['last_child_exit']=child.returncode;state['last_stage_seconds']=time.monotonic()-begun;publish()
            if child.returncode!=0:raise RuntimeError(stage+' failed;pipeline stopped,no retry')
            if stop_time is not None or (out/'STOP').exists():raise RuntimeError('STOP/wall limit ended pipeline after child exit')
    current=BASE/'D95-fly-DN-motor-KL512-DAgger3-v1/candidate.pt'
    current_sha='426968272608fd2ef217712f2c50166ef99ac2f6a102a702c86d11e746b2360c'
    manifests=[BASE/'D95-fly-student-state-DAgger3-v1/data/index.json']
    try:
        if sha(current)!=current_sha:raise ValueError('Initial DAgger1 candidate identity differs')
        for number,seed in enumerate(state['train_seed_masters'],1):
            folder=out/f'round{number:02d}';collect=folder/'collect';fit=folder/'fit'
            state.update(round=number,initial_candidate=str(current),initial_candidate_sha256=current_sha);publish()
            command=[str(PYTHON),'-B',str(ROOT/'scripts/collect_fly_centered_dagger.py'),'--checkpoint',str(current),
                '--checkpoint-sha256',current_sha,'--environment-config',str(ENV),'--output',str(collect),
                '--episodes','3','--seed',str(seed),'--brain-seed','613711','--max-seconds','3600','--execute-cuda']
            run(f'round{number:02d}/collect',command,collect,3900)
            if not (collect/'data/COMMITTED.json').is_file():raise ValueError('Collector exited without complete commit')
            cleanup=list(collect.rglob('native_cleanup.json'))
            if not cleanup or any(json.loads(path.read_text(encoding='utf-8')).get('close_errors') for path in cleanup):
                raise ValueError('Native cleanup missing/failed;no fit will run')
            fresh=collect/'data/index.json'
            command=[str(PYTHON),'-B',str(ROOT/'scripts/refit_fly_DN_motor_KL_DAgger.py'),
                '--old-manifest',str(BASE/'D95-fly-centered512-DN-prefix2048-v1/index.json'),
                '--additional-manifest',str(BASE/'D95-fly-centered512-DN-three-more-episodes-v1/index.json'),
                '--dagger-manifest',str(fresh),'--initial-candidate',str(current),'--initial-candidate-sha256',current_sha,
                '--output',str(fit),'--execute-cuda']
            for prior in manifests:command.extend(['--prior-dagger-manifest',str(prior)])
            run(f'round{number:02d}/fit',command,fit,240)
            report=json.loads((fit/'result.json').read_text(encoding='utf-8'))
            if report.get('status')!='completed' or report.get('optimizer_updates')!=512 or report.get('changed_parameters')!=['brain.action_readout.weight']:
                raise ValueError('Finite motor fit did not complete as declared')
            current=Path(report['candidate']);current_sha=report['candidate_sha256']
            if sha(current)!=current_sha:raise ValueError('Saved round candidate identity differs')
            manifests.append(fresh)
            state['round_results'].append(dict(round=number,seed=seed,collection=str(collect),fit=str(fit),
                accumulated_DAgger_rows=report['DAgger_rows'],candidate=str(current),candidate_sha256=current_sha,metrics=report['metrics']))
            publish()
        final=out/'final-same3-development'
        command=[str(PYTHON),'-B',str(ROOT/'scripts/evaluate_fly_centered_native.py'),'--checkpoint',str(current),
            '--checkpoint-sha256',current_sha,'--environment-config',str(ENV),'--output',str(final),
            '--episodes','3','--seed','2027431401','--brain-seed','613711','--execute-cuda']
        run('final-same3-development',command,final,3900)
        evaluation=json.loads((final/'result.json').read_text(encoding='utf-8'))['evaluation']
        native_seeds=[row['native_seed'] for row in evaluation['episode_results']]
        if sorted(native_seeds)!=[6517,7487,17506]:raise ValueError('Final evaluation did not use the declared same3 development seeds')
        state.update(status='completed',final_candidate=str(current),final_candidate_sha256=current_sha,
            final_development=str(final),final_evaluation=evaluation)
        write(out/'result.json',state);publish()
    except BaseException as error:
        state.update(status='failed_stopped_no_retry',failure=str(error));write(out/'failure.json',state);publish();return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
