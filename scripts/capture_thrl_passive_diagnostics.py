"""Bounded passive readings from one already-owned TH14 training worker.

No controller, Frida attach, injected hook, action, pause, model or CUDA use.
Never publishes these asynchronous readings as learner observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fly_touhou.atomic_json_io import atomic_write_json
from fly_touhou.thrl_study_io import read_shared_json, utc_now
from fly_touhou.thrl_control import alive
from fly_touhou.native import ProcessMemory, Profile, NativeError
from fly_touhou.hazard_reader import GeometryHazardReader
from fly_touhou.thrl_vector import _filetime


class ReadBudgetExceeded(NativeError):
    pass


class BudgetReadOnlyMemory(ProcessMemory):
    def __init__(self, pid):
        # Do not request write, VM operation, suspend or resume rights.
        try:
            super().__init__(pid, writable=False, can_suspend=False)
        except BaseException:
            # The shared reader can fail module enumeration after OpenProcess
            # but before its normal close() state has been initialized.
            if getattr(self, 'handle', None):
                self.k.CloseHandle(self.handle)
                self.handle = None
            raise
        self.begin_read()

    def begin_read(self, seconds=.05, bytes_limit=16*1024*1024):
        self.deadline = time.perf_counter() + seconds
        self.remaining = bytes_limit
        self.read_bytes = 0

    def read(self, address, size):
        if time.perf_counter() > self.deadline or size > self.remaining:
            raise ReadBudgetExceeded('Passive read budget exceeded; discard this sample')
        self.remaining -= size
        self.read_bytes += size
        return super().read(address, size)


def select_owned_identity(ownership, worker, executable, run):
    if ownership.get('status') != 'active' or not alive(ownership.get('owner')):
        return None
    row = next((row for row in ownership.get('workers', []) if row.get('index') == worker), None)
    if not row or not alive(row.get('worker_identity')):
        return None
    expected=Path(run).resolve()/'native/train'/f'worker{worker}'/'appdata'
    if not row.get('private_appdata') or Path(row['private_appdata']).resolve()!=expected:
        return None
    identity = row.get('native_identity')
    if (not identity or not identity.get('identity_complete') or not alive(identity)
            or Path(identity.get('executable', '')).resolve() != Path(executable).resolve()):
        return None
    return identity


def poll_sleep_seconds(period, tick, started, duration, now):
    return max(0., min(period-(now-tick), duration-(now-started)))


def fingerprint(paths):
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def collect(run, output, *, worker=0, seconds=120., frequency=2., max_samples=600,
            max_raw_bytes=64*1024*1024, read_budget_ms=50.):
    from fly_touhou.thrl_observation_audit import audit_snapshot, aggregate_audits
    from fly_touhou.thrl_collision_telemetry import (new_collision_telemetry,
        snapshot_collision_sample, observe_collision_sample, collision_telemetry_summary,
        record_collision_drop)
    run, output = Path(run).resolve(), Path(output).resolve()
    if not run.is_dir() or output.is_relative_to(run):
        raise ValueError('Use an existing trial and a separate new diagnostic output folder')
    if (not 0 < seconds <= 600 or not 0 < frequency <= 2 or not 1 <= max_samples <= 1200
            or type(worker) is not int or not 0 <= worker < 100
            or not 0 < read_budget_ms <= 250):
        raise ValueError('Passive capture limit: 600 seconds, 2 Hz, 1200 samples, 250 ms read')
    config = read_shared_json(run/'config.json')
    if config.get('training_mode') != 'full-story-v1':
        raise ValueError('Only the explicit full-story training protocol is supported')
    from fly_touhou.thrl_inputs import SEPARATED_TIMER_SCHEMAS
    if config.get('input_schema') not in SEPARATED_TIMER_SCHEMAS:
        raise ValueError('This capture audits the current separated timers-v2 input only')
    source = Path(config['source_runtime'])
    watched = [run/'config.json'] + [source/'src/fly_touhou'/name for name in (
        'thrl_geometry_observation.py', 'thrl_timer_observation.py', 'thrl_model.py', 'thrl_ppo.py')]
    before_hashes = fingerprint(watched)
    local_src=Path(__file__).resolve().parents[1]/'src/fly_touhou'
    observer_files=[Path(__file__).resolve(),Path(config['profile'])] + [local_src/name for name in (
        'thrl_observation_audit.py','thrl_collision_telemetry.py','thrl_geometry_observation.py',
        'thrl_timer_observation.py','thrl_observation.py','native_geometry.py','native.py',
        'hazard_reader.py','resources.py','state.py','thrl_vector.py')]
    observer_hashes=fingerprint(observer_files)
    profile = Profile.load(config['profile'])
    output.mkdir(parents=True, exist_ok=False)
    try:
        if sys.platform == 'win32':
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (psutil.Error, OSError):
        pass
    started = time.perf_counter(); cpu_started = time.process_time()
    audits, episode_states, errors, durations = [], {}, Counter(), []
    memory = reader = identity = None
    last_frame = None; last_emit = 0.; raw_bytes = 0; polls = duplicates = 0
    end_reason = 'duration'
    def make_report(status):
        return {'schema':'th14-passive-diagnostics-v1', 'status':status,
            'source_run':str(run), 'worker':worker, 'updated_utc':utc_now(),
            'wall_seconds':time.perf_counter()-started,
            'observer_cpu_seconds':time.process_time()-cpu_started,
            'frequency_hz_limit':frequency, 'sample_count':len(audits), 'polls':polls,
            'read_budget_ms':read_budget_ms,
            'duplicates_skipped':duplicates, 'errors':dict(errors),
            'uncompressed_raw_bytes':raw_bytes, 'end_reason':end_reason,
            'read_latency_ms':{'mean':sum(durations)/len(durations)*1000 if durations else None,
                'max':max(durations)*1000 if durations else None},
            'observation_audit':aggregate_audits(audits),
            'episodes':[collision_telemetry_summary(state) for state in episode_states.values()],
            'protocol':{'read_only':True,'suspend_rights':False,'injection':False,
                'actions_sent':0,'model_loaded':False,'cuda_used':False,
                'sampling':'one existing worker, wall-time sample; not all policy decisions',
                'consistency':'same-frame and overlapping-field checks; not an atomic checkpoint guarantee',
                'exact_native_event_counters_available':False,
                'snapshot_selection_bias':'duplicates/torn/budgeted reads excluded; one worker only',
                'training_observations_modified':False},
            'source_hashes_before':before_hashes,
            'source_files_unchanged':fingerprint(watched)==before_hashes,
            'source_hash_scope':'listed frozen training files and actual trial config, not every runtime file',
            'observer_source_hashes_before':observer_hashes,
            'observer_source_files_unchanged':fingerprint(observer_files)==observer_hashes}
    try:
        with gzip.open(output/'snapshots.jsonl.gz', 'wt', encoding='utf-8') as stream:
            while time.perf_counter()-started < seconds:
                tick = time.perf_counter(); polls += 1
                if (output/'STOP').exists():
                    end_reason = 'observer-stop-file'; break
                try:
                    ownership = read_shared_json(run/'workers/ownership.json')
                    selected = select_owned_identity(ownership, worker, config['game_exe'], run)
                    if not selected:
                        if not alive(ownership.get('owner')):
                            end_reason='training-owner-ended'; break
                        raise NativeError('Selected worker has no live verified game')
                    if identity != selected:
                        if memory: memory.close()
                        memory = BudgetReadOnlyMemory(selected['pid'])
                        if (not alive(selected) or memory.exe.resolve() != Path(config['game_exe']).resolve()
                                or _filetime(memory.handle) != selected.get('creation_filetime')):
                            raise NativeError('Native process identity changed during read-only open')
                        reader = GeometryHazardReader(memory, profile)
                        identity = selected; last_frame = None
                    memory.begin_read(seconds=read_budget_ms/1000.)
                    frame = reader.global_frame()
                    if frame == last_frame:
                        duplicates += 1
                    else:
                        read_started = time.perf_counter()
                        snapshot = reader.read(retries=1)
                        durations.append(time.perf_counter()-read_started)
                        if not snapshot.player or snapshot.screen != 7 or snapshot.pause != 2:
                            raise NativeError('Not an active measurable gameplay snapshot')
                        audit = audit_snapshot(snapshot, input_schema=config['input_schema'])
                        episode_key = f'{worker}:{identity["pid"]}:{identity["creation_filetime"]}'
                        # Drops for an existing episode are recorded immediately
                        # below, not counted again on the next successful sample.
                        sample = snapshot_collision_sample(snapshot, episode_key=episode_key)
                        state = episode_states.get(episode_key, new_collision_telemetry(expected_frame_stride=None))
                        state = observe_collision_sample(state, sample)
                        row = {'observed_utc':utc_now(), 'source_run':str(run), 'worker':worker,
                            'native_identity':identity, 'episode_key':episode_key,
                            'sampling_scope':'passive-asynchronous-not-policy-rollout',
                            'snapshot':snapshot.to_dict(), 'observation_audit':audit,
                            'collision_sample':sample}
                        payload = json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'
                        size = len(payload.encode('utf-8'))
                        if raw_bytes+size > max_raw_bytes:
                            end_reason='raw-data-budget'; break
                        stream.write(payload); stream.flush()
                        raw_bytes += size; audits.append(audit); episode_states[episode_key]=state
                        last_frame=snapshot.frame
                        if len(audits) >= max_samples:
                            end_reason='sample-budget'; break
                except (OSError, NativeError, ValueError, KeyError, psutil.Error) as error:
                    errors[type(error).__name__+': '+str(error)[:160]] += 1
                    if identity:
                        key=f'{worker}:{identity["pid"]}:{identity["creation_filetime"]}'
                        if key in episode_states:
                            episode_states[key]=record_collision_drop(episode_states[key], reason=type(error).__name__)
                now = time.perf_counter()
                if now-last_emit >= 5:
                    atomic_write_json(output/'report.json', make_report('sampling'))
                    last_emit = now
                time.sleep(poll_sleep_seconds(1/frequency,tick,started,seconds,time.perf_counter()))
    finally:
        if memory: memory.close()
    report = make_report('completed')
    atomic_write_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--seconds',type=float,default=120.)
    parser.add_argument('--frequency',type=float,default=2.)
    parser.add_argument('--read-budget-ms',type=float,default=50.)
    args=parser.parse_args()
    result=collect(args.run,args.output,worker=args.worker,seconds=args.seconds,
        frequency=args.frequency,read_budget_ms=args.read_budget_ms)
    print(json.dumps({key:result[key] for key in ('status','sample_count','errors','wall_seconds',
        'observer_cpu_seconds','source_files_unchanged')},ensure_ascii=False),flush=True)
