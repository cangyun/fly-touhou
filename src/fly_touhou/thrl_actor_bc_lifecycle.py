"""Read-only main lifecycle gate and non-resumable BC partial checkpoints.

The recognized producer is train_b1_parameter_transfer -> train_trial ->
full-story ready evaluator. All live JSON uses Windows shared-delete reads.
No Torch, process control, native world or automatic job is created on import.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from .thrl_study_io import read_shared_json

SCHEMA = 'B1-offline-BC-main-lifecycle-v2'
PARTIAL_SCHEMA = 'B1-raw-actor-path-BC-partial-analysis-checkpoint-v2'
LIFECYCLE = dict(schema=SCHEMA, partial_every_completed_Adam=100,
    allow_after_verified_normal_main_exit=True, wait_seconds=2.,
    background_terminal_publication_grace_seconds=30., baseline_reuse=False)


class LifecycleStop(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def same_owner(owner, gate):
    return (owner.get('pid') == gate['pid'] and isinstance(owner.get('create_time'), (int, float))
            and abs(owner['create_time'] - gate['create_time']) <= .001)


def live_phase_decision(progress, actual_create_time, gate, now):
    if not math.isfinite(actual_create_time) or abs(actual_create_time - gate['create_time']) > .001:
        raise LifecycleStop('Main PID was reused or its creation identity is uncertain')
    if progress.get('schema') != 'th14-thrl-training-progress-v1':
        raise LifecycleStop('Unknown main progress producer')
    try:
        clock = datetime.fromisoformat(progress['updated_utc'].replace('Z', '+00:00'))
        if clock.tzinfo is None:
            raise ValueError('Timezone required')
        age = now - clock.timestamp()
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise LifecycleStop('Main progress clock is uncertain') from error
    if age < -2:
        raise LifecycleStop('Main progress is in an uncertain future')
    status, stage = progress.get('status'), progress.get('stage')
    if status not in ('training', 'collecting', 'updating', 'development', 'evaluating', 'evaluation', 'finished'):
        raise LifecycleStop('Main failed,stopped or has an unknown lifecycle status')
    if stage == 'collecting' and status != 'finished':
        if age > 30:
            raise LifecycleStop('Main collecting record is stale;GPU permission is uncertain')
        return 'allow', 'main_collecting'
    if stage in ('updating', 'development', 'evaluation', 'evaluating', 'finished'):
        # A verified live busy phase may wait to the finite BC wall budget.
        # It never grants GPU permission, even if the record is old.
        return 'wait', 'main_' + stage
    raise LifecycleStop('Main phase is unknown')


def validate_closed_ownership(ownership, workers, gate):
    if (ownership.get('schema') != 'thrl-native-vector-ownership-v1'
            or ownership.get('status') != 'closed' or not same_owner(ownership.get('owner', {}), gate)
            or ownership.get('close_errors') != [] or ownership.get('failures') != []
            or len(ownership.get('workers', [])) != workers):
        raise LifecycleStop('Closed main native ownership/owner/worker count differs')
    for row in ownership['workers']:
        checks, history = row.get('native_exit_checks', []), row.get('native_history', [])
        if (row.get('worker_alive_after_close') is not False or type(row.get('worker_exitcode')) is not int
                or not history or len(checks) != len(history)
                or any(check.get('same_identity_still_live') is not False for check in checks)):
            raise LifecycleStop('Main native cleanup has missing or unresolved owned process exits')


def validate_terminal_records(records, gate, config, spec, *, final_file_sha, script_sha, failure_present=False):
    """Pure producer-specific completion proof. A dead PID alone never passes."""
    launch, study, training, progress = (records[k] for k in ('launch', 'study', 'training', 'progress'))
    target = config['trial_decisions']
    dev = records['development']
    if (failure_present or not same_owner(launch, gate)
            or config.get('schema') != 'B1-evaluation-parameter-transfer-finite-training-v1'
            or config.get('evaluation_after_training_shutdown') is not True
            or config.get('development_decisions') != [target]
            or config.get('development_episodes') != 20
            or config.get('full_story_evaluation_backend') != 'full-story-ready-parallel-v1'
            or study.get('status') != 'finished' or study.get('training') != training
            or training.get('status') != 'finished' or training.get('decisions') != target
            or training.get('candidate_decisions') != target or training.get('latest_development') != dev
            or training.get('final_sha256') != final_file_sha
            or progress.get('status') != 'finished' or progress.get('stage') != 'finished'
            or progress.get('actual_decisions') != target or progress.get('target_decisions') != target
            or progress.get('completed_development_nodes') != [target]
            or dev.get('episodes') != 20 or len(dev.get('episode_results', [])) != 20
            or any(row.get('outcome') not in ('clear', 'gameover') for row in dev.get('episode_results', []))
            or dev.get('native_seed_split') != 'development' or dev.get('policy_frozen') is not True
            or dev.get('PPO_updates') != 0 or dev.get('continues') != 0
            or dev.get('action_selection') != config.get('full_story_evaluation_action_selection')
            or dev.get('action_temperature') != config['parameters'].get('action_temperature', 1.)):
        raise LifecycleStop('Main normal-completion/training/development/final-checkpoint proof is incomplete')
    for key in ('training_cleanup', 'development_cleanup'):
        if records[key].get('close_errors') != []:
            raise LifecycleStop('Main cleanup reports unresolved errors')
    train_ownership, dev_ownership = records['training_ownership'], records['development_ownership']
    if records['training_cleanup'].get('ownership') != train_ownership:
        raise LifecycleStop('Training cleanup ownership snapshot differs from its committed ownership file')
    validate_closed_ownership(train_ownership, config['workers'], gate)
    validate_closed_ownership(dev_ownership, config['full_story_evaluation_workers'], gate)
    if records['development_cleanup'].get('workers') != config['full_story_evaluation_workers']:
        raise LifecycleStop('Development cleanup worker count differs')
    background = records['background']
    expected_command = [str(Path(spec['python']).resolve()), str(Path(spec['script']).resolve()), *spec['arguments']]
    arguments = spec['arguments']
    if (background.get('schema') != 'fly-touhou-background-status-v1'
            or background.get('spec_sha256') != gate['background_spec_sha256']
            or background.get('script_sha256') != script_sha or background.get('command') != expected_command
            or Path(background.get('cwd', '')).resolve() != Path(spec['cwd']).resolve()
            or '--output' not in arguments
            or Path(arguments[arguments.index('--output')+1]).resolve() != Path(gate['output']).resolve()
            or Path(spec['script']).resolve() != Path(config['source_runtime']).resolve() / 'scripts/train_b1_parameter_transfer.py'):
        raise LifecycleStop('Background completion does not bind the actual main command/output/runtime')
    if background.get('status') == 'running' and background.get('exit_code') is None:
        return 'wait', 'main_background_terminal_publication'
    if (background.get('status') != 'finished' or type(background.get('exit_code')) is not int
            or background['exit_code'] != 0 or not isinstance(background.get('finished_unix'), (int, float))
            or not math.isfinite(background['finished_unix']) or background['finished_unix'] < gate['create_time']):
        raise LifecycleStop('Main launcher did not finish normally with exit0')
    return 'allow', 'main_verified_normal_exit'


class LifecycleGuard:
    def __init__(self, bc_config, contract, output, resource_check):
        self.config, self.contract, self.output = bc_config, contract, Path(output)
        self.resource_check = resource_check
        self.started, self.wait_seconds, self.wait_checks = time.monotonic(), 0., 0
        self.on_wait = self.on_safe_boundary = None
        self._waiting, self._gone_since, self.last_completion = False, None, None
        self.folder = Path(contract['output']).resolve()
        if (sha(self.folder/'launch.json') != contract['launch_sha256']
                or sha(self.folder/'study_config.json') != contract['study_config_sha256']
                or sha(Path(contract['control_directory'])/'background_spec.json') != contract['background_spec_sha256']):
            raise LifecycleStop('Pinned main launch/config/background spec changed')
        self.launch = read_shared_json(self.folder/'launch.json')
        self.main_config = read_shared_json(self.folder/'study_config.json')
        self.spec = read_shared_json(Path(contract['control_directory'])/'background_spec.json')
        if not same_owner(self.launch, contract):
            raise LifecycleStop('Main original launch PID/ctime differs')
        self.progress_path = self.folder/'trial_000_both/progress.json'

    def check(self, *, startup=False):
        self.resource_check(startup=startup, elapsed=time.monotonic()-self.started,
                            stop_exists=(self.output/'STOP').exists())

    def _terminal(self):
        cfg = self.main_config
        trial = self.folder/'trial_000_both'
        dev = trial / ('development_%07d' % cfg['trial_decisions'])
        paths = dict(study=self.folder/'study_result.json', training=trial/'training_result.json',
            progress=self.progress_path, training_cleanup=trial/'native_cleanup.json',
            training_ownership=trial/'workers/ownership.json', development=dev/'report.json',
            development_cleanup=dev/'native_cleanup.json', development_ownership=dev/'workers/ownership.json',
            background=Path(self.contract['control_directory'])/'status.json')
        records = {'launch': self.launch, **{k: read_shared_json(p) for k, p in paths.items()}}
        failed = any(p.exists() for p in (self.folder/'failure.json', trial/'failure.json', dev/'failure.json',
                                        self.folder/'STOP', trial/'STOP'))
        decision = validate_terminal_records(records, self.contract, cfg, self.spec,
            final_file_sha=sha(trial/'final.pt'), script_sha=sha(Path(self.spec['script'])), failure_present=failed)
        if decision[0] == 'allow':
            self.last_completion = dict(utc=utc(), main_PID=self.contract['pid'],
                main_create_time=self.contract['create_time'], reason=decision[1],
                source_file_sha256={k: sha(p) for k, p in paths.items()},
                launch_sha256=self.contract['launch_sha256'], main_process_exited=True)
        return decision

    def main_decision(self):
        import psutil
        try:
            process = psutil.Process(self.contract['pid'])
            actual = process.create_time()
            if not math.isfinite(actual) or abs(actual-self.contract['create_time']) > .001:
                raise LifecycleStop('Main PID reused;stop only this offline run')
            if process.is_running():
                self._gone_since = None
                return live_phase_decision(read_shared_json(self.progress_path), actual, self.contract, time.time())
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            raise LifecycleStop('Main process identity is uncertain') from error
        if self._gone_since is None:
            self._gone_since = time.monotonic()
        if self.last_completion is not None:
            return 'allow', 'main_verified_normal_exit'
        decision = self._terminal()
        if decision[0] == 'wait' and time.monotonic()-self._gone_since > 30:
            raise LifecycleStop('Normal main reports exist but launcher terminal publication did not complete in30seconds')
        return decision

    def before_gpu(self):
        while True:
            self.check()
            try:
                decision, reason = self.main_decision()
            except LifecycleStop:
                raise
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise LifecycleStop('Main lifecycle evidence is missing or uncertain') from error
            if decision == 'allow':
                if self._waiting and self.on_wait:
                    self.on_wait(dict(waiting=False, reason=reason, wait_checks=self.wait_checks,
                        wait_seconds=self.wait_seconds, BC_gpu_block_permitted=True), force=True)
                self._waiting = False
                if self.on_safe_boundary:
                    self.on_safe_boundary()
                return
            first, self._waiting = not self._waiting, True
            self.wait_checks += 1
            if self.on_wait:
                self.on_wait(dict(waiting=True, reason=reason, wait_checks=self.wait_checks,
                    wait_seconds=self.wait_seconds, BC_gpu_block_permitted=False), force=first)
            before = time.monotonic()
            time.sleep(2)
            self.wait_seconds += time.monotonic()-before


def cpu_tree(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(torch, v) for v in value)
    return deepcopy(value)


class PartialWriter:
    """Only used at a qualified normal boundary;never from exception cleanup."""
    def __init__(self, torch, model, optimizer, folder, *, config, arm, scope, cost, provenance, every=100):
        self.torch, self.model, self.optimizer, self.folder = torch, model, optimizer, Path(folder)
        self.config, self.arm, self.scope, self.cost = config, arm, scope, cost
        self.provenance, self.every, self.last_paid, self.committed = provenance, every, 0, []
        if type(every) is not int or every < 1:
            raise ValueError('Positive partial interval required')

    def maybe_save(self):
        cost = self.cost
        paid = cost['paid_BC_optimizer_steps']
        if paid-self.last_paid < self.every or cost.get('complete_train_epoch') is not False:
            return None
        if (type(paid) is not int or paid < 1 or cost.get('pending_uncommitted_legal_labels') != 0
                or cost.get('optimizer_transaction_complete') is not True
                or cost.get('BC_optimizer_step_attempts') != paid
                or any(p.grad is not None for p in self.model.parameters())):
            return None
        payload = dict(schema=PARTIAL_SCHEMA, created_utc=utc(), arm=self.arm, model_config=self.model.model_config(),
            model=cpu_tree(self.torch, self.model.state_dict()), BC_optimizer=cpu_tree(self.torch, self.optimizer.state_dict()),
            scope=deepcopy(self.scope), cost=deepcopy(cost), source_config=deepcopy(self.config),
            lifecycle_provenance=deepcopy(self.provenance), complete_train_epoch=False,
            numerical_resume_supported=False, mid_epoch_resume_supported=False, neural_state_serialized=False,
            gradients_serialized=False, optimizer_state_for_analysis_only=True, native_worlds_started=0,
            deployment_forbidden=True, demonstration_catalogue_eligible=False,
            full_epoch_export_eligible=False, new_PPO_requires_fresh_Adam=True,
            critic_calibrated_for_BC_policy=False)
        path = self.folder / ('partial_%06d.pt' % paid)
        temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('xb') as stream:
                self.torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        row = dict(path=str(path), sha256=sha(path), paid_BC_optimizer_steps=paid, cost=deepcopy(cost),
                   schema=PARTIAL_SCHEMA, full_epoch_export_eligible=False)
        write_json(self.folder/'latest_partial.json', row)
        self.last_paid = paid
        self.committed.append(row)
        return row
