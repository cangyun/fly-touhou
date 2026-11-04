"""Run the bounded first engineering learning cycle through existing CLIs.

This supervisor does not implement a learner or native process controller.
It starts only the declared child scripts, retains their results, and stops at
the first failed stage. Existing output directories are never resumed silently.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ('run_m2_shared_teacher.py', 'train_gru_bc.py', 'evaluate_gru_bc.py',
           'collect_gru_dagger.py', 'train_standard_gru.py')
# This single reviewed change wraps pre-stage12 startup with existing hidden
# window/lifecycle diagnostics. It does not change gameplay ticks or inputs.
REVIEWED_STARTUP_SOURCE_CHANGE = {
    'relative_path': 'src/fly_touhou/fixed_start_controller.py',
    'original_sha256': '93221fe938249896be7005792a8d117c6948ef48ff5e776d10a6900c396ced31',
    'current_sha256': 'e2b65a2dcb70e5e33a90aaab8ab71ef7ec4869bdcfcb0aab6fca72c05570f0bc',
    'scope': 'pre-stage12 EarlyHiddenStartupMixin/OwnedLifecycle diagnostics only; recorded input/ticks unchanged',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class StageFailure(RuntimeError):
    pass


class OperatorStop(RuntimeError):
    pass


def training_reuse_evidence(args):
    """Reuse only the completed data stage of an actually terminated cycle."""
    source = getattr(args, 'reuse_training_manifest', None)
    expected_sha = getattr(args, 'reuse_training_manifest_sha256', None)
    if not source and not expected_sha:
        return None
    if not source or not expected_sha:
        raise StageFailure('Training-data reuse needs both manifest path and its explicit SHA256')
    source = Path(source).resolve()
    if sha(source) != expected_sha:
        raise StageFailure('The requested64 training manifest differs from the pinned SHA256')
    if source.parent.name != 'train' or source.parent.parent.name != 'data':
        raise StageFailure('Reuse the completed original cycle data/train/manifest.json')
    origin = source.parent.parent.parent
    result_path, status_path, config_path = [origin/name for name in ('result.json', 'status.json', 'config.json')]
    result, status, config = read(result_path), read(status_path), read(config_path)
    if (result.get('status') not in ('stopped_at_stage_failure', 'operator_stop')
            or not result.get('finished_utc') or status.get('status') != result['status']
            or config.get('seed') != args.seed or config.get('workers') != 2):
        raise StageFailure('The original data cycle needs terminal same-seed evidence before reuse')
    children = status.get('children', [])
    if any(child.get('cuda_child') or not child.get('stage', '').startswith('A_') for child in children):
        raise StageFailure('This data-stage continuation cannot erase previously started BC/PPO costs')
    process_checks = []
    identities = [(status.get('supervisor_pid'), status.get('supervisor_create_time_unix'))]
    identities += [(child.get('pid'), child.get('process_create_time_unix')) for child in children]
    for pid, created in identities:
        if pid is None:
            continue
        if created is None:
            raise StageFailure('An original owned process lacks its recorded creation time')
        try:
            current = psutil.Process(pid)
            alive = current.is_running() and abs(current.create_time()-created) < .001
        except psutil.NoSuchProcess:
            alive = False
        process_checks.append({'pid': pid, 'recorded_create_time_unix': created,
                               'same_owned_process_still_alive': alive})
        if alive:
            raise StageFailure('An original owned handle is still alive; data reuse never restarts it')
    for name, current in (('base_protocol', args.base_protocol), ('teacher_qualification', args.teacher_qualification)):
        original = config.get('evidence', {}).get(name, {})
        if (original.get('sha256') != sha(current)
                or Path(original.get('path', '__missing__')).resolve() != Path(current).resolve()):
            raise StageFailure('Training data belongs to another protocol or qualified teacher')
    if any(config.get('source_hashes', {}).get(name) != sha(ROOT/'scripts'/name) for name in SCRIPTS):
        raise StageFailure('The original data-cycle CLI source provenance changed')
    source_roots = {Path(child['cwd']).resolve() for child in children
                    if child.get('stage') == 'A_train_demonstrations' and child.get('cwd')}
    if len(source_roots) != 1:
        raise StageFailure('The original completed training stage lacks its unique frozen source root')
    original_source_root = next(iter(source_roots))
    manifest = read(source)
    if (manifest.get('schema') != 'native-shared-teacher-demonstrations-v1'
            or manifest.get('successful_demonstrations') != 64
            or manifest.get('requested_demonstrations') != 64
            or manifest.get('demonstration_target_met') is not True
            or manifest.get('protocol_sha256') != sha(args.base_protocol)
            or Path(manifest.get('qualification', '__missing__')).resolve() != Path(args.teacher_qualification).resolve()
            or len(manifest.get('child_manifests', [])) != 2):
        raise StageFailure('Only the original completed64 same-protocol training demonstrations may be reused')
    shard_evidence, compatible_startup_changes = [], []
    for ref in manifest['child_manifests']:
        path = Path(ref['path']).resolve()
        if sha(path) != ref['sha256']:
            raise StageFailure('A retained training shard manifest changed')
        shard, collector = read(path), read(path.parent/'config.json')
        if (shard.get('successful_demonstrations') != 32 or shard.get('requested_demonstrations') != 32
                or not 32 <= shard.get('attempts', 0) <= 40
                or collector.get('network_training') is not False
                or collector.get('task_start_stage_frame') != 12
                or Path(collector.get('qualification', '__missing__')).resolve() != Path(args.teacher_qualification).resolve()
                or not collector.get('source_hashes')):
            raise StageFailure('Original collector provenance does not qualify the retained training shard')
        for relative, expected in collector['source_hashes'].items():
            if sha(original_source_root/relative) != expected:
                raise StageFailure('A retained original frozen collector source changed: '+relative)
            current = sha(ROOT/relative)
            if current != expected:
                allowed = REVIEWED_STARTUP_SOURCE_CHANGE
                if (relative.replace('\\', '/') != allowed['relative_path']
                        or expected != allowed['original_sha256'] or current != allowed['current_sha256']):
                    raise StageFailure('A retained collector observation/teacher source changed: '+relative)
                if allowed not in compatible_startup_changes:
                    compatible_startup_changes.append(dict(allowed))
        shard_evidence.append({'manifest': {'path': str(path), 'sha256': sha(path)},
            'collector_config': {'path': str(path.parent/'config.json'), 'sha256': sha(path.parent/'config.json')}})
    # Reuse the existing loader: real NPZ bytes, full prefixes, start12, 64
    # successes, observation/action fingerprints, and retained attempt limits.
    sys.path.insert(0, str(ROOT/'src'))
    from fly_touhou.gru_imitation import load_demonstrations
    episodes, metadata = load_demonstrations(source, expected_role='train', expected_episodes=64)
    return {'schema': 'engineering-completed-training-data-reuse-v1',
        'manifest': {'path': str(source), 'sha256': sha(source)},
        'origin_cycle': str(origin), 'origin_records': {
            name: {'path': str(path), 'sha256': sha(path)} for name, path in
            (('config', config_path), ('status', status_path), ('result', result_path))},
        'collector_shards': shard_evidence, 'owned_process_checks': process_checks,
        'original_frozen_source_root': str(original_source_root),
        'reviewed_startup_source_changes': compatible_startup_changes,
        'observation_teacher_task_semantics_unchanged': True,
        'successful_demonstrations': len(episodes), 'collection_attempts': metadata['collection_attempts'],
        'all_original_failures_preserved': True, 'new_training_demonstrations_to_collect': 0,
        'prior_BC_epochs_consumed': 0, 'prior_PPO_decisions': 0,
        'new_validation_successes_target': 16, 'new_validation_attempts_cap': 20}


def prerequisites(args):
    if args.workers != 2 or not str(Path(args.output).resolve()).isascii():
        raise ValueError('This cycle uses two workers and a new ASCII output path')
    protocol, teacher, m1 = read(args.base_protocol), read(args.teacher_qualification), read(args.m1_report)
    configuration, soak = m1.get('configuration', {}), m1.get('soak', {})
    waiver_path = getattr(args, 'm1_stability_waiver', None)
    waiver = None
    if waiver_path is not None:
        waiver_path = Path(waiver_path).resolve()
        waiver = read(waiver_path)
        benchmark_path = Path(waiver.get('benchmark_report_path', '__missing__'))
        if not benchmark_path.is_absolute():
            benchmark_path = waiver_path.parent/benchmark_path
        if (waiver.get('schema') != 'engineering-M1-user-skip-soak-v1'
                or waiver.get('stability_test_waived') is not True
                or not waiver.get('user_instruction')
                or waiver.get('selected_workers') != 2 or waiver.get('wait_sleep_ms') != 10
                or waiver.get('no_native_timing') is not True
                or benchmark_path.resolve() != Path(args.m1_report).resolve()
                or waiver.get('benchmark_report_sha256') != sha(args.m1_report)):
            raise StageFailure('The explicit user M1 stability waiver does not match this benchmark')
        if (m1.get('status') != 'passed-bounded-benchmark' or m1.get('mode') != 'bench'
                or 2 not in configuration.get('worlds', [])
                or configuration.get('wait_sleep_ms') != 10
                or configuration.get('no_native_timing') is not True
                or m1.get('source_files_changed_during_run') != []
                or m1.get('owned_cleanup_failures') != [] or not m1.get('finished_utc')):
            raise StageFailure('The user-waived M1 path still needs the passed selected two-worker benchmark')
    elif (m1.get('status') != 'passed-bounded-soak' or m1.get('mode') != 'soak'
          or configuration.get('worlds') != [2] or configuration.get('wait_sleep_ms') != 10
          or configuration.get('no_native_timing') is not True
          or configuration.get('duration_seconds', 0) < 1800 or soak.get('elapsed_seconds', 0) < 1800
          or soak.get('successful_decisions', 0) <= 0
          or soak.get('exact_native_frames') != soak.get('successful_decisions', 0)*4
          or m1.get('source_files_changed_during_run') != []
          or m1.get('owned_cleanup_failures') != [] or not m1.get('finished_utc')):
        raise StageFailure('M1 needs the completed 30-minute two-worker default10ms/no-native-timing soak')
    if m1.get('game_exe', {}).get('sha256') != protocol.get('game_exe_sha256'):
        raise StageFailure('The M1 evidence used another original executable')
    rows = teacher.get('episode_results', [])
    if (teacher.get('teacher_qualified') is not True or teacher.get('episodes') != 20 or len(rows) != 20
            or teacher.get('successes', 0) < 19
            or sum(r.get('horizon_completed_without_hit') is True for r in rows) != teacher.get('successes')
            or teacher.get('critical_sensor_issues_resolved') is not True
            or teacher.get('shared_information') is not True or teacher.get('observation_size') != 1912
            or teacher.get('frame_skip') != 4 or teacher.get('bomb_actions') != 0
            or teacher.get('task_start_stage_frame') != 12
            or any(r.get('native_start_stage_frame') != 12 for r in rows)
            or teacher.get('safety_takeovers') != 0 or teacher.get('desktop_input') is not False
            or teacher.get('protocol_sha256') != sha(args.base_protocol)
            or teacher.get('teacher_source_sha256') != sha(ROOT/'src/fly_touhou/shared_observation_teacher.py')):
        raise StageFailure('M2 common-information teacher has not qualified on this exact protocol')
    evidence = {name: {'path': str(Path(path).resolve()), 'sha256': sha(path)} for name, path in (
        ('base_protocol', args.base_protocol), ('teacher_qualification', args.teacher_qualification),
        ('m1_report', args.m1_report))}
    if waiver is not None:
        evidence['m1_stability_waiver'] = {'path': str(waiver_path), 'sha256': sha(waiver_path),
                                          'receipt': waiver}
    reused = training_reuse_evidence(args)
    if reused is not None:
        evidence['reused_training_data'] = reused
        args._validated_training_reuse = reused
    return evidence


class Supervisor:
    """Small child-process ledger; no timeout causes a live child to restart."""
    def __init__(self, output, *, poll_seconds=.5):
        self.output = Path(output)
        (self.output/'logs').mkdir(parents=True, exist_ok=True)
        self.poll_seconds = poll_seconds
        self.children, self.stages = [], []
        self.stop_requested = False
        self.current_stage = None

    def event(self, kind, **fields):
        with (self.output/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time=now(), kind=kind, **fields), ensure_ascii=False)+'\n')

    def state(self, status='running'):
        write(self.output/'status.json', {'schema': 'bounded-engineering-cycle-status-v1',
            'status': status, 'stage': self.current_stage, 'updated_utc': now(),
            'supervisor_pid': os.getpid(), 'supervisor_create_time_unix': psutil.Process().create_time(),
            'stop_requested': self.stop_requested, 'stages': self.stages, 'children': self.children,
            'automatic_resume_implemented': False, 'automatic_live_child_restarts': 0})

    def check_stop(self):
        self.stop_requested |= (self.output/'STOP').exists()
        return self.stop_requested

    def command(self, script, *arguments):
        return [sys.executable, '-u', str(ROOT/'scripts'/script), *map(str, arguments)]

    def run_group(self, stage, specifications):
        if self.check_stop():
            raise OperatorStop('Root STOP exists before the next stage')
        if sum(bool(spec.get('cuda')) for spec in specifications) > 1:
            raise StageFailure('Only one CUDA policy/learner child may run at once')
        self.current_stage = stage
        active, first_failure = [], None
        self.event('stage_started', stage=stage)
        try:
            for spec in specifications:
                logs = [self.output/'logs'/f'{spec["name"]}.{suffix}.log' for suffix in ('stdout', 'stderr')]
                handles = [path.open('xb') for path in logs]
                record = {'name': spec['name'], 'stage': stage, 'command': spec['command'],
                    'cwd': str(ROOT), 'output': str(spec['output']), 'cuda_child': bool(spec.get('cuda')),
                    'stdout': str(logs[0]), 'stderr': str(logs[1]), 'launch_started_utc': now(),
                    'status': 'starting', 'stop_forwarded': False}
                self.children.append(record)
                try:
                    process = subprocess.Popen(spec['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=handles[0], stderr=handles[1], close_fds=True,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                        env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                except BaseException:
                    for handle in handles:
                        handle.close()
                    record.update(status='launch_failed', finished_utc=now())
                    raise
                try:
                    created = psutil.Process(process.pid).create_time()
                except psutil.NoSuchProcess:
                    created = None
                record.update(pid=process.pid, process_create_time_unix=created, status='running')
                active.append((process, handles, record))
                self.event('child_started', **record)
            while any(process.poll() is None for process, _, _ in active):
                self.check_stop()
                for process, _, record in active:
                    code = process.poll()
                    if code is not None and code != 0 and first_failure is None:
                        first_failure = dict(name=record['name'], exit_code=code)
                    stop_path = Path(record['output'])/'STOP'
                    if (self.stop_requested or first_failure) and code is None and stop_path.parent.is_dir():
                        if not stop_path.exists():
                            stop_path.write_text('Supervisor forwarded STOP; preserve completed work.\n', encoding='utf-8')
                        record['stop_forwarded'] = True
                self.state('draining_children' if self.stop_requested or first_failure else 'running')
                time.sleep(self.poll_seconds)
        except BaseException:
            # Forward STOP to existing owned CLI outputs, then wait for actual
            # termination. Do not delete output, kill native PIDs or relaunch.
            while any(process.poll() is None for process, _, _ in active):
                for process, _, record in active:
                    target = Path(record['output'])/'STOP'
                    if process.poll() is None and target.parent.is_dir():
                        target.touch(exist_ok=True)
                        record['stop_forwarded'] = True
                self.state('draining_children_after_error')
                time.sleep(self.poll_seconds)
            raise
        finally:
            for process, handles, record in active:
                code = process.poll()
                record.update(exit_code=code, status='finished' if code == 0 else 'failed', finished_utc=now())
                for handle in handles:
                    handle.close()
                self.event('child_terminated', name=record['name'], pid=record['pid'], exit_code=code)
            self.state()
        self.check_stop()
        if self.stop_requested:
            raise OperatorStop('Root STOP forwarded and all current children have terminated')
        failures = [r for _, _, r in active if r.get('exit_code') != 0]
        if failures:
            raise StageFailure('Child process failed: '+json.dumps(failures, ensure_ascii=False))
        self.stages.append({'stage': stage, 'status': 'processes_finished', 'finished_utc': now()})
        self.event('stage_processes_finished', stage=stage)

    def single(self, name, script, output, *arguments, cuda=False):
        self.run_group(name, [{'name': name, 'command': self.command(script, *arguments, '--output', output),
                              'output': output, 'cuda': cuda}])


def merge_manifests(paths, output, *, split, per_shard, attempt_cap):
    manifests = [read(path) for path in paths]
    common = ('feature_names', 'observation_version', 'base_protocol_path', 'protocol_sha256',
              'observation_size', 'frames_per_action', 'qualification', 'task_start_stage_frame')
    rows, references = [], []
    for path, manifest in zip(paths, manifests):
        if (manifest.get('split') != split or manifest.get('requested_demonstrations') != per_shard
                or manifest.get('task_start_stage_frame') != 12
                or manifest.get('successful_demonstrations') != per_shard
                or manifest.get('attempts') != len(manifest.get('episodes', []))
                or not per_shard <= manifest['attempts'] <= attempt_cap
                or any(manifest.get(key) != manifests[0].get(key) for key in common)):
            raise StageFailure('A demonstration shard failed its fixed data target or common protocol')
        for entry in manifest['episodes']:
            row = dict(entry)
            if row.get('native_start_stage_frame') != 12:
                raise StageFailure('A collected demonstration used another original reset checkpoint')
            source = Path(row['dataset_path'])
            source = source if source.is_absolute() else Path(path).parent/source
            row.update(dataset_path=str(source.resolve()), dataset_sha256=sha(source))
            rows.append(row)
        references.append({'path': str(Path(path).resolve()), 'sha256': sha(path),
                           'attempts': manifest['attempts'], 'successful_demonstrations': per_shard})
    eligible = [row for row in rows if row.get('eligible_successful_demonstration') is True]
    if len(eligible) != per_shard*len(paths) or len({r['episode_id'] for r in eligible}) != len(eligible):
        raise StageFailure('Merged successful demonstrations contain missing trajectories or duplicate episode identities')
    result = {**{key: manifests[0][key] for key in common},
        'schema': 'native-shared-teacher-demonstrations-v1', 'split': split,
        'successful_only': True, 'all_attempts_retained': True, 'episodes': rows,
        'attempts': len(rows), 'max_attempts': attempt_cap*len(paths),
        'requested_demonstrations': per_shard*len(paths), 'successful_demonstrations': len(eligible),
        'demonstration_target_met': True, 'child_manifests': references,
        'distinct_successful_trajectories': len({r['trajectory_sha256'] for r in eligible}),
        'repeated_training_trajectories_preserved': len(eligible)-len({r['trajectory_sha256'] for r in eligible})}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    return result


def bc_checkpoint(directory, *, expected_cumulative):
    result = read(Path(directory)/'result.json')
    checkpoint = Path(directory)/'bc_actor.pt'
    if (result.get('status') != 'finished' or result.get('epochs_completed_this_run') != 10
            or result.get('cumulative_epochs_consumed') != expected_cumulative
            or result.get('private_critic_unchanged') is not True or not checkpoint.is_file()):
        raise StageFailure('BC stopped early or violated the shared 30-epoch budget')
    return checkpoint


def qualification_passed(path, checkpoint):
    value = read(path)
    groups = value.get('groups', {})
    if (value.get('status') != 'finished' or value.get('episodes_per_mode') != 20
            or value.get('checkpoint_sha256') != sha(checkpoint)
            or value.get('task_start_stage_frame') != 12
            or value.get('teacher_actions') != 0 or value.get('safety_takeovers') != 0
            or value.get('bomb_actions') != 0
            or value.get('actor_unchanged') is not True
            or any(groups.get(mode, {}).get('episodes') != 20 or len(groups[mode].get('episode_results', [])) != 20
                   for mode in ('argmax', 'categorical_sampling'))):
        raise StageFailure('BC qualification is incomplete or refers to another actor')
    successes = [sum(row.get('horizon_completed_without_hit') is True for row in groups[mode]['episode_results'])
                 for mode in ('argmax', 'categorical_sampling')]
    if any(row.get('native_start_stage_frame') != 12 for group in groups.values() for row in group['episode_results']):
        raise StageFailure('BC qualification still contains a different original start checkpoint')
    actual = min(successes) >= 1
    if (actual != value.get('bc_qualified') or successes[0] != value.get('deterministic_successes')
            or successes[1] != value.get('sampling_successes')):
        raise StageFailure('BC qualification boolean disagrees with its actual game results')
    return actual


def execute(args, supervisor):
    out = supervisor.output
    manifests = {}
    reused = getattr(args, '_validated_training_reuse', None)
    if getattr(args, 'reuse_training_manifest', None) is not None:
        if reused is None:
            reused = training_reuse_evidence(args)
        if sha(args.reuse_training_manifest) != args.reuse_training_manifest_sha256:
            raise StageFailure('Reused training manifest changed after preflight')
        manifests['train'] = Path(args.reuse_training_manifest).resolve()
        write(out/'reused_training_data.json', reused)
    for split, target, limit, seed_offset in (('train', 32, 40, 200000), ('validation', 8, 10, 300000)):
        if split == 'train' and reused is not None:
            continue
        paths, specifications = [], []
        for shard in range(2):
            destination = out/f'a_{split}_shard{shard}'
            command = supervisor.command('run_m2_shared_teacher.py', '--mode', 'collect',
                '--protocol', args.base_protocol, '--qualification', args.teacher_qualification,
                '--split', split, '--episodes', target, '--max-attempts', limit,
                '--seed', args.seed+seed_offset+1000*shard, '--output', destination)
            specifications.append({'name': f'a_{split}_shard{shard}', 'command': command,
                                   'output': destination, 'cuda': False})
            paths.append(destination/'manifest.json')
        supervisor.run_group(f'A_{split}_demonstrations', specifications)
        manifests[split] = out/'data'/split/'manifest.json'
        merge_manifests(paths, manifests[split], split=split, per_shard=target, attempt_cap=limit)
    training, validation = read(manifests['train']), read(manifests['validation'])
    fingerprints = [{r['trajectory_sha256'] for r in value['episodes'] if r['eligible_successful_demonstration']}
                    for value in (training, validation)]
    if fingerprints[0] & fingerprints[1]:
        raise StageFailure('Whole demonstration trajectories overlap between training and validation')
    bc_arguments = ['--train-manifest', manifests['train'], '--validation-manifest', manifests['validation'],
        '--teacher-qualification', args.teacher_qualification, '--seed', args.seed,
        '--max-epochs', 10, '--probe-epochs', 3]
    current = out/'b_bc_initial'
    supervisor.single('B_initial_BC', 'train_gru_bc.py', current, *bc_arguments, cuda=True)
    checkpoint = bc_checkpoint(current, expected_cumulative=10)
    dagger_manifests, training_results = [], [current/'result.json']
    for round_index in range(3):
        qualification_dir = out/f'c_bc_qualification_round{round_index}'
        supervisor.single(f'C_qualification_{round_index}', 'evaluate_gru_bc.py', qualification_dir,
            '--checkpoint', checkpoint, '--base-protocol', args.base_protocol, '--workers', args.workers,
            '--seed', args.seed+110000+1000*round_index, cuda=True)
        qualification = qualification_dir/'report.json'
        if qualification_passed(qualification, checkpoint):
            break
        if round_index == 2:
            raise StageFailure('BC remains unqualified after the maximum two DAgger rounds; PPO was not started')
        destination = out/f'c_dagger_round{round_index+1}'
        arguments = ['--checkpoint', checkpoint, '--base-protocol', args.base_protocol,
            '--teacher-qualification', args.teacher_qualification, '--round-index', round_index+1,
            '--episodes', 20, '--seed', args.seed+400000+1000*round_index]
        if dagger_manifests:
            arguments += ['--previous-dagger-manifest', dagger_manifests[-1]]
        supervisor.single(f'C_DAgger_{round_index+1}', 'collect_gru_dagger.py', destination, *arguments, cuda=True)
        manifest_path = destination/'manifest.json'
        manifest = read(manifest_path)
        if manifest.get('attempts') != 20 or manifest.get('completed_round') is not True or len(manifest.get('episodes', [])) != 20:
            raise StageFailure('The DAgger round did not complete its fixed 20 student attempts')
        dagger_manifests.append(manifest_path)
        destination = out/f'c_bc_after_dagger{round_index+1}'
        arguments = [*bc_arguments, '--resume-checkpoint', checkpoint,
                     '--previous-training-results', training_results[-1]]
        for manifest_path in dagger_manifests:
            arguments += ['--dagger-manifest', manifest_path]
        supervisor.single(f'C_resume_BC_{round_index+1}', 'train_gru_bc.py', destination, *arguments, cuda=True)
        checkpoint = bc_checkpoint(destination, expected_cumulative=10*(round_index+2))
        training_results.append(destination/'result.json')
    ppo = out/'d_ppo_first102400'
    ppo_arguments = [
        '--engineering-v3', '--base-protocol', args.base_protocol, '--workers', args.workers,
        '--seed', args.seed, '--steps', 102400, '--features', 'spatial', '--bc-checkpoint', checkpoint,
        '--m1-report', args.m1_report, '--m2-report', args.teacher_qualification,
        '--bc-report', qualification]
    if getattr(args, 'm1_stability_waiver', None) is not None:
        ppo_arguments += ['--m1-stability-waiver', args.m1_stability_waiver]
    supervisor.single('D_PPO_first102400', 'train_standard_gru.py', ppo, *ppo_arguments, cuda=True)
    result = read(ppo/'training_result.json')
    if result.get('status') != 'finished' or result.get('budget_completed') is not True or result.get('decisions') != 102400:
        raise StageFailure('The bounded PPO first segment did not finish its declared new-sample budget')
    return {'status': 'completed_first_cycle', 'PPO_decisions': 102400, 'PPO_output': str(ppo),
        'BC_checkpoint': str(checkpoint), 'BC_checkpoint_sha256': sha(checkpoint),
        'BC_qualification': str(qualification), 'BC_epochs_consumed': 10*len(training_results),
        'DAgger_rounds': len(dagger_manifests), 'automatic_budget_expansion': False,
        'M1_stability_test_waived': getattr(args, 'm1_stability_waiver', None) is not None,
        'M1_soak_passed': getattr(args, 'm1_stability_waiver', None) is None,
        'training_demonstrations_reused': reused is not None,
        'full_engineering_goal_complete': False, 'whole_brain_started': False, 'score_training_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-protocol', type=Path, required=True)
    parser.add_argument('--teacher-qualification', type=Path, required=True)
    parser.add_argument('--m1-report', type=Path, required=True)
    parser.add_argument('--m1-stability-waiver', type=Path,
                        help='Explicit user receipt allowing the passed short benchmark instead of the soak')
    parser.add_argument('--reuse-training-manifest', type=Path,
                        help='Completed original cycle data/train manifest; collect only new validation afterwards')
    parser.add_argument('--reuse-training-manifest-sha256',
                        help='Explicit SHA256 of the completed64 training manifest')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(2,), default=2)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    for field in ('base_protocol', 'teacher_qualification', 'm1_report', 'm1_stability_waiver',
                  'reuse_training_manifest', 'output'):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    supervisor = Supervisor(args.output)
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, lambda *_: setattr(supervisor, 'stop_requested', True))
    code = 0
    try:
        evidence = prerequisites(args)
        config = {'schema': 'bounded-engineering-learning-cycle-v1', 'created_utc': now(),
            'evidence': evidence, 'workers': args.workers, 'seed': args.seed,
            'M1_stability_test_waived': args.m1_stability_waiver is not None,
            'M1_soak_passed': args.m1_stability_waiver is None,
            'M1_evidence_basis': ('user_waived_soak_passed_benchmark' if args.m1_stability_waiver
                                  else 'passed_30minute_soak'),
            'budgets': {'successful_train_demonstrations': 64, 'train_attempts_cap': 80,
                'successful_validation_demonstrations': 16, 'validation_attempts_cap': 20,
                'total_BC_epochs_cap': 30, 'DAgger_rounds_cap': 2, 'student_attempts_per_round': 20,
                'PPO_first_segment_decisions': 102400},
            'source_hashes': {name: sha(ROOT/'scripts'/name) for name in SCRIPTS},
            'automatic_resume_implemented': False, 'native_process_cleanup': 'existing child CLI responsibility'}
        write(args.output/'config.json', config)
        result = {'status': 'validated_only', 'children_started': False} if args.validate_only else execute(args, supervisor)
    except OperatorStop as error:
        result, code = {'status': 'operator_stop', 'reason': str(error)}, 2
    except BaseException as error:
        result, code = {'status': 'stopped_at_stage_failure', 'reason': repr(error)}, 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    write(args.output/'result.json', dict(result, finished_utc=now()))
    supervisor.state(result['status'])
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
