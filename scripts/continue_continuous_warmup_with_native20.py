"""One finite handoff: completed supervised job -> frozen original20.

This is a single workflow continuation,not a recurring monitor or retry loop.
The source's actual PID/create-time must exit before any CUDA child is started.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    spec = read_shared_json(a.specification)
    if not a.execute:
        print(json.dumps(dict(status='prepared_not_executed', specification=spec,
            Torch_imported='torch' in sys.modules, GPU_started=False), ensure_ascii=False))
        return
    import psutil
    current = psutil.Process()
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    # Preserve the full affinity for the later original-world launcher.
    control = Path(spec['sequence_control'])
    state_path = control / 'sequence.json'
    if state_path.exists():
        raise FileExistsError('A finite handoff already owns this sequence')
    source = Path(spec['source_output'])
    source_launch = read_shared_json(source / 'launch.json')
    if (source_launch['PID'] != spec['source_PID']
            or abs(source_launch['create_time'] - spec['source_create_time']) >= .01):
        raise ValueError('Source job identity differs')
    started = time.monotonic()
    state = dict(status='waiting_for_actual_source_exit', PID=current.pid,
        create_time=current.create_time(), source_PID=spec['source_PID'],
        source_create_time=spec['source_create_time'], source_output=str(source),
        automatic_retry=False)
    write_json(state_path, state)
    try:
        while True:
            if (control / 'STOP').exists():
                state['status'] = 'cancelled_before_evaluation'
                write_json(state_path, state)
                return
            if time.monotonic() - started > spec['maximum_wait_seconds']:
                raise TimeoutError('Finite handoff wait ended;no evaluation was launched')
            try:
                owner = psutil.Process(spec['source_PID'])
                if abs(owner.create_time() - spec['source_create_time']) >= .01:
                    break  # The original PID owner has exited and been replaced.
                owner.wait(timeout=30)
                break
            except psutil.TimeoutExpired:
                continue
            except psutil.NoSuchProcess:
                break

        result_path = source / 'result.json'
        if not result_path.is_file():
            state.update(status='source_ended_without_complete_result',
                source_failure=read_shared_json(source / 'failure.json')
                    if (source / 'failure.json').is_file() else None)
            write_json(state_path, state)
            return
        result = read_shared_json(result_path)
        if result.get('status') != 'completed' or result.get('complete_epoch') is not True:
            raise RuntimeError('Only the completed full supervised epoch can enter this handoff')
        checkpoint = source / 'final-parameters.pt'
        identity = digest(checkpoint)
        if identity != result['final_parameters']['sha256']:
            raise ValueError('Final parameter bytes differ from the completed source result')
        if (control / 'STOP').exists():
            state['status'] = 'cancelled_before_evaluation'
            write_json(state_path, state)
            return
        if any((p.info['name'] or '').lower() == 'th14.exe'
                for p in psutil.process_iter(['name'])):
            raise RuntimeError('Existing original worlds;do not overlap another job')

        destination = Path(spec['evaluation_control'])
        output = Path(spec['evaluation_output'])
        destination.mkdir(exist_ok=False)
        argv = [str(ROOT / '.venv/Scripts/python.exe'), '-B',
            'scripts/evaluate_continuous_checkpoint.py',
            '--connectome', str(ROOT / 'data/connectome'),
            '--checkpoint', str(checkpoint), '--checkpoint-sha256', identity,
            '--environment-config', spec['environment_config'],
            '--output', str(output), '--max-seconds', str(spec['evaluation_wall_seconds']),
            '--execute-cuda']
        launch_spec = dict(schema='owned-training-with-native-cache-cleanup-v1',
            cwd=str(ROOT), control=str(destination), output=str(output), argv=argv,
            source=str(checkpoint), source_sha256=identity,
            source_result_sha256=digest(result_path),
            native_worlds=20, evaluation_episodes=20, no_new_optimizer=True,
            reason='One completed full-brain supervised epoch;frozen original development20',
            formal_stability_test=False)
        write_json(destination / 'launch-specification.json', launch_spec)
        with (destination / 'supervisor-stdout.log').open('wb') as out, \
                (destination / 'supervisor-stderr.log').open('wb') as err:
            child = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-B',
                'scripts/run_training_with_native_cache_cleanup.py', '--specification',
                str(destination / 'launch-specification.json')], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW)
        owner = psutil.Process(child.pid)
        receipt = dict(launcher_PID=child.pid, launcher_create_time=owner.create_time(),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_sha256=identity, automatic_retry=False)
        write_json(destination / 'dispatch.json', receipt)
        state.update(status='evaluation_dispatched', evaluation_control=str(destination),
            evaluation_output=str(output), dispatch=receipt)
        write_json(state_path, state)
        # The existing finite child supervisor owns evaluation cleanup. This
        # sequence exits here rather than creating another result-polling service.
        config_path = ROOT / 'artifacts/engineering/tensorboard-live-v3/configuration.json'
        config = read_shared_json(config_path)
        config.setdefault('historical_fly_algorithm_sources', {}).update(
            config.get('fly_algorithm_sources', {}))
        config['fly_algorithm_sources'] = {spec['evaluation_label']: str(output)}
        config['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(config_path, config)
    except BaseException as error:
        state.update(status='handoff_error', error=repr(error), automatic_retry=False)
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
