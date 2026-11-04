"""Read native B1 logs into TensorBoard without importing Torch or the game.

Only the dedicated TensorBoard log/control directories are written. Native
JSON/JSONL handles share read, write and delete; an incomplete last JSONL row
is left for the next pass. No checkpoint, action, reward or trainer is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.tensorboard_continuous_v15 import export_continuous_connectome, current_algorithm_jobs

ACTOR_PATH_BC_SCHEMA = 'B1-raw-actor-path-BC-contrast-result-v1'
ACTOR_PATH_BC_LIFECYCLE_SCHEMA = 'B1-raw-actor-path-BC-lifecycle-result-v2'
ACTOR_PATH_BC_PARSER_PATCH = 'lifecycle-v2-and-terminal-priority-v1'
ACTOR_PATH_BC_ARMS = ('full_actor_path', 'gru_actor_control')
MIGRATION_TB_PATCH = 'D95-reference-and-cached-fly-reader-audit-v2'
D95_REFERENCE_SCHEMA = 'D95-fly-migration-paired20-reference-result-v1'
D95_REFERENCE_JOB = 'D95-fly-migration-reference-evaluation-only-v1'
D95_SHA = 'd95fe710068483fc81d8525d89bccf72e61d209c7e1bb442ea66d022b9c1d06c'
FLY_CACHED_SCHEMA = 'fly-cached-complete-connectome-finite-distillation-v1'
FLY_CACHED_PHASES = ('frozen_before_train', 'frozen_before_held', 'train', 'frozen_after_train', 'frozen_after_held')
MIGRATION_SCOPES = {
    'd95_reference': '# D95教师原作参考20局\n\n这是D95神经教师的原作比赛，含自主Bomb；不是果蝇学生在玩原作。'
        '\n\n横轴native是本次原作策略决策，elapsed是实际秒；final只有完整20局的一份快照。'
        '复用的PPO8/20不计入本次新比赛或训练。开发种子结果不等于独立稳定验收。',
    'fly_cached_distillation': '# 果蝇缓存离线蒸馏\n\n读取固定Phi和D95标签，训练感觉/运动接口；不采样原作。'
        '\n\ncost横轴是实际完成Adam更新；elapsed是实际秒；model forwards与label rows都不是原作决策。'
        '\n\nfrozen snapshots是从row0重建的冻结自身历史；完整局与连续开局前缀分开标注，训练中的TBPTT历史近似不冒充冻结验证。'
        '硬脉冲为窗口汇总，不能证明逐事件相同或原作避弹；未知input事件不补0。'
        '\n\nreader_audit单列数据读出行与guard秒数，不能推算或覆盖正式模型forward、Adam或KL。'}


def migration_number(value, name, *, count=False):
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or count and type(value) is not int):
        raise ValueError('Invalid measured migration number: ' + name)
    return value


def migration_sha(path):
    digest = hashlib.sha256()
    with _shared_binary(Path(path)) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def migration_source(folder, kind):
    """Terminal producer file wins; never fill terminal cost/time from progress."""
    folder = Path(folder)
    terminals = [folder / name for name in ('failure.json', 'result.json', 'report.json') if (folder / name).is_file()]
    if len(terminals) > 1:
        raise ValueError('Conflicting producer terminal files; current status is unknown')
    terminal = terminals[0] if terminals else None
    progress = read(folder / 'progress.json') if (folder / 'progress.json').is_file() else {}
    row = read(terminal) if terminal else progress
    if not isinstance(row, dict) or not isinstance(progress, dict):
        raise ValueError('Migration producer JSON must be an object')
    if kind == 'fly_cached_distillation':
        if any(x and x.get('schema') != FLY_CACHED_SCHEMA for x in (row, progress)):
            raise ValueError('Unknown or conflicting cached-fly schema')
        if terminal and terminal.name not in ('failure.json', 'result.json'):
            raise ValueError('Unexpected cached-fly terminal filename')
    else:
        launch = read(folder / 'launch.json') if (folder / 'launch.json').is_file() else {}
        if not isinstance(launch, dict):
            raise ValueError('D95 launch JSON must be an object')
        if row or launch or (folder / 'games/failure.json').is_file():
            if (launch.get('job_type') != D95_REFERENCE_JOB or launch.get('checkpoint_sha256') != D95_SHA
                    or launch.get('optimizer_updates') != 0 or launch.get('native_worlds_max') != 20):
                raise ValueError('D95 reference launch identity differs')
        if progress and (progress.get('job_type') != D95_REFERENCE_JOB
                         or progress.get('policy_checkpoint_sha256') != D95_SHA
                         or progress.get('optimizer_updates') != 0 or 'schema' in progress):
            raise ValueError('Unknown or conflicting D95 reference progress identity')
        if terminal:
            if terminal.name == 'report.json':
                if row.get('schema') != D95_REFERENCE_SCHEMA:
                    raise ValueError('Unknown D95 reference result schema')
            elif (terminal.name != 'failure.json' or 'schema' in row
                  or row.get('status') != 'failed_or_stopped' or row.get('optimizer_updates') != 0
                  or row.get('partial_results_are_not_a_score') is not True or row.get('no_automatic_restart') is not True):
                raise ValueError('Unknown D95 reference failure contract')
        nested_failure = folder / 'games/failure.json'
        if nested_failure.is_file():
            if terminal and terminal.name != 'failure.json':
                raise ValueError('Complete D95 result conflicts with native failure')
            if terminal is None:
                terminal, row = nested_failure, read(nested_failure)
                if row.get('partial_results_are_not_a_score') is not True:
                    raise ValueError('Unknown D95 native failure contract')
    return row, terminal


def parse_d95_reference(folder):
    folder = Path(folder)
    row, terminal = migration_source(folder, 'd95_reference')
    values, snapshots = [], []
    state = dict(source=str(folder), teacher_reference_not_fly_gameplay=True,
                 producer_file=str(terminal.relative_to(folder)) if terminal else 'progress.json',
                 producer_terminal=terminal is not None)
    if not row:
        return dict(state=dict(state, status='waiting_for_progress', phase='unknown'), values=values, snapshots=snapshots)
    if terminal and terminal.name == 'report.json':
        if (row.get('status') != 'completed' or row.get('D95_checkpoint_sha256') != D95_SHA
                or row.get('newly_executed_native_games') != 20 or row.get('reused_PPO_native_games') != 20
                or row.get('optimizer_updates') != 0 or row.get('policy_actions') != 19
                or row.get('autonomous_Bomb') is not True or row.get('continues') != 0
                or row.get('native_frame_horizon', 'missing') is not None
                or row.get('action_selection') != 'greedy' or row.get('action_temperature') != .25
                or row.get('native_seed_split') != 'development' or row.get('fly_student_evaluated') is not False):
            raise ValueError('D95 terminal must be exactly the complete teacher-reference20')
        path = folder / 'games/report.json'
        if (not path.resolve().is_relative_to(folder.resolve())
                or Path(row['D95_report']).resolve() != path.resolve() or migration_sha(path) != row.get('D95_report_sha256')):
            raise ValueError('Immutable D95 complete-game report pointer changed')
        report = read(path)
        if not isinstance(report, dict):
            raise ValueError('D95 complete-game report must be an object')
        rows = report.get('episode_results', [])
        if (report.get('episodes') != 20 or not isinstance(rows, list) or len(rows) != 20 or report.get('policy_frozen') is not True
                or report.get('PPO_updates') != 0 or report.get('native_seed_split') != 'development'
                or report.get('action_selection') != 'greedy' or report.get('action_temperature') != .25
                or any(report.get(k) != 0 for k in ('teacher_actions', 'safety_takeovers', 'continues'))
                or any(not isinstance(r, dict) or r.get('evaluation_game') != i + 1 or r.get('outcome') not in ('clear', 'gameover')
                       or r.get('full_story_episode') is not True for i, r in enumerate(rows))):
            raise ValueError('Partial or incompatible D95 native games are not a final score')
        clears = sum(r['outcome'] == 'clear' for r in rows)
        if report.get('clears') != clears:
            raise ValueError('D95 clear count differs from actual completed rows')
        metrics = [('clears', clears), ('clear_rate', clears / 20)]
        for key in ('mean_native_frames', 'mean_misses', 'mean_bombs_used'):
            value = report.get(key)
            if key == 'mean_native_frames' and value is None:
                frames = [migration_number(r.get('native_frames'), 'native_frames', count=True) for r in rows]
                value = statistics.mean(frames) if all(v is not None for v in frames) else None
            value = migration_number(value, key)
            if value is not None:
                metrics.append((key, value))
        snapshots.append(dict(key='d95_reference_final:' + str(path), fingerprint=row['D95_report_sha256'],
                              values=[('d95_reference/final/' + k, v, 20) for k, v in metrics]))
        timing = report.get('evaluation_timing', {})
        current = dict(completed=20, target=20, workers=20, policy_decisions=timing.get('decisions'),
                       wall_seconds=timing.get('wall_seconds'))
        status, phase = 'completed', 'completed'
    elif terminal:
        current = row
        status = phase = 'failed_or_stopped'
    else:
        current = row.get('evaluation_progress')
        if not isinstance(current, dict) or current.get('target') != 20:
            raise ValueError('D95 requires nested original20 evaluation_progress')
        status, phase = 'evaluating_teacher_reference', row.get('status', 'unknown')
    counts = {key:migration_number(current.get(key), key, count=True) for key in ('completed', 'target', 'workers', 'policy_decisions')}
    if (counts['completed'] is not None and counts['completed'] > 20
            or counts['target'] is not None and counts['target'] != 20
            or counts['workers'] is not None and counts['workers'] != 20):
        raise ValueError('D95 reference population exceeds the single20-game pool')
    seconds = migration_number(current.get('wall_seconds'), 'D95 wall_seconds')
    for axis, step in (('native', counts['policy_decisions']), ('elapsed', int(seconds) if seconds is not None else None)):
        if step is not None:
            values += [('d95_reference/' + axis + '/' + key, value, step) for key, value in counts.items() if value is not None]
    state.update(status=status, phase=phase, completed_games=counts['completed'], policy_decisions=counts['policy_decisions'],
                 elapsed_seconds=seconds, reason=row.get('reason', row.get('error')), native_performance=True)
    return dict(state=state, values=values, snapshots=snapshots)


def fly_reader_audit(folder, row):
    """Bounded known phase/episode paths only; reader yield is not model work."""
    base = Path(folder) / 'reader-audits'
    if not base.is_dir():
        return None, []
    root = Path(folder).resolve()
    if not base.resolve().is_relative_to(root):
        raise ValueError('Reader audit root escapes the current student job')
    cohorts = {}
    for key in ('train', 'heldout'):
        names = row.get(key)
        if (not isinstance(names, list) or len(names) > 30
                or any(not isinstance(name, str) or len(name) != 8 or not name.startswith('world')
                       or not name[5:].isdigit() or not 0 <= int(name[5:]) < 30 for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('Reader audits need explicit original train/heldout episode lists')
        cohorts[key] = names
    if set(cohorts['train']) & set(cohorts['heldout']):
        raise ValueError('Reader audit train and heldout episodes overlap')
    candidates = []
    for phase in FLY_CACHED_PHASES:
        names = cohorts['heldout' if phase.endswith('_held') else 'train']
        for episode in names:
            directory = base / (phase + '-' + episode)
            path = directory / 'RESULT.json'
            if not path.is_file():
                path = directory / 'progress.json'
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError('Known reader audit path escapes the current student job')
            audit = read(path)
            if (not isinstance(audit, dict) or audit.get('schema') != 'fly-v5-sensory9721-full-episode-CPU-consumer-v1'
                    or audit.get('episode') != episode or audit.get('source_start_row') != 0
                    or audit.get('saved_cursor_is_input') is not False or audit.get('synthetic_fixture') is not False
                    or audit.get('bank_identity_sha256') != '1c6006dc7fbbbe5465e7afcef5f43afc8ffebfc4aba8e7833f4ad2b55eebb4f5'):
                raise ValueError('Unknown reader audit schema,episode,row0 or immutable bank identity')
            values = {key:migration_number(audit.get(key), 'reader/' + key, count=True) for key in
                      ('rows_yielded', 'payload_rows_verified', 'payload_rows_read', 'source_rows', 'max_payload_and_yield_rows')}
            if (any(value is None for value in values.values())
                    or not values['rows_yielded'] <= values['payload_rows_verified'] <= values['payload_rows_read']
                    <= values['max_payload_and_yield_rows'] <= values['source_rows']
                    or audit.get('next_source_row_to_yield') != values['rows_yielded']):
                raise ValueError('Reader audit row counts do not describe a continuous verified prefix')
            resources = audit.get('resources')
            if not isinstance(resources, dict):
                raise ValueError('Reader audit guard resources are unknown')
            seconds = migration_number(resources.get('elapsed_seconds'), 'reader guard elapsed_seconds')
            if seconds is None:
                raise ValueError('Reader audit guard elapsed_seconds is unknown')
            candidates.append(dict(phase=phase, episode=episode, status=audit.get('status'),
                producer_file=str(path.relative_to(Path(folder))), reader_RESULT=path.name == 'RESULT.json',
                guard_elapsed_seconds=seconds, **values,
                persisted_cursor_is_lower_bound=audit.get('hard_kill_yield_cursor_is_lower_bound'),
                scope='reader_data_IO_only_not_completed_model_forwards_or_process_NoTorch_evidence'))
    if not candidates:
        return None, []
    # The supplied FitGuard has one run-wide elapsed origin across all phases.
    latest = max(candidates, key=lambda item:item['guard_elapsed_seconds'])
    step = int(latest['guard_elapsed_seconds'])
    values = [('fly_cached/reader_audit/' + key, latest[key], step) for key in
              ('rows_yielded', 'payload_rows_verified', 'guard_elapsed_seconds')]
    return latest, values


def parse_fly_cached_distillation(folder):
    folder = Path(folder)
    row, terminal = migration_source(folder, 'fly_cached_distillation')
    values, snapshots = [], []
    state = dict(source=str(folder), native_performance=False, offline_labels_not_native_decisions=True,
                 producer_file=terminal.name if terminal else 'progress.json', producer_terminal=terminal is not None)
    if not row:
        return dict(state=dict(state, status='waiting_for_progress', phase='unknown'), values=values, snapshots=snapshots)
    status = row.get('status')
    if status not in ('running', 'completed_finite_cached_distillation', 'stopped_finite_budget', 'failed', 'failed_preparation'):
        raise ValueError('Unknown cached-fly producer status')
    if terminal and (terminal.name == 'result.json' and status == 'running'
                     or terminal.name == 'failure.json' and status not in ('failed', 'failed_preparation')):
        raise ValueError('Cached-fly terminal status conflicts with its filename')
    if row.get('native_worlds_started') != 0:
        raise ValueError('Cached-fly is offline; native gameplay must be a separate report')
    if status != 'failed_preparation' and (row.get('teacher_SHA256') != D95_SHA
            or row.get('loss') != 'final19_legal_KL' or row.get('temperature') != 1.
            or row.get('lambda_value') != 0. or row.get('gain_limit') != 0.
            or row.get('batch') != 1 or row.get('substeps') != 100 or row.get('synthetic_fixture') is not False):
        raise ValueError('Unknown or artificial cached-fly learning contract')
    resources = row.get('resources') or {}
    if not isinstance(resources, dict):
        raise ValueError('Cached-fly resources must be an object')
    seconds = migration_number(resources.get('elapsed_seconds'), 'fly elapsed_seconds')
    count_keys = ('optimizer_updates', 'parameter_version', 'backward_passes', 'backward_attempts',
                  'optimizer_step_attempts', 'dropped_pending_rows', 'total_model_decisions', 'total_forward_passes')
    counts = {key:migration_number(row.get(key), key, count=True) for key in count_keys}
    paid = counts['optimizer_updates']
    for axis, step in (('cost', paid), ('elapsed', int(seconds) if seconds is not None else None)):
        if step is not None:
            values += [('fly_cached/' + axis + '/' + key, value, step) for key, value in counts.items() if value is not None]
    latest = row.get('last_completed_update') or {}
    if not isinstance(latest, dict):
        raise ValueError('Cached-fly completed update must be an object')
    if latest:
        update = migration_number(latest.get('optimizer_updates'), 'last completed Adam', count=True)
        kl = migration_number(latest.get('mean_legal_KL'), 'last completed training KL')
        if update is None or paid is None or update != paid:
            raise ValueError('Training KL must belong to the actual last completed Adam update')
        if kl is not None:
            values.append(('fly_cached/train_update/mean_legal_KL_TBPTT_approximate_history', kl, update))
    cuda = row.get('CUDA') or {}
    if not isinstance(cuda, dict):
        raise ValueError('Cached-fly CUDA measurements must be an object')
    for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
        measured = migration_number(cuda.get(key), key, count=True)
        if measured is not None and seconds is not None:
            values.append(('fly_cached/elapsed/CUDA/' + key.replace('_bytes', '_GiB'), measured / 1024**3, int(seconds)))
    costs = row.get('costs') or {}
    if not isinstance(costs, dict) or set(costs) - set(FLY_CACHED_PHASES):
        raise ValueError('Unknown cached-fly cost phase')
    for phase, cost in costs.items():
        if not isinstance(cost, dict):
            raise ValueError('Cached-fly phase cost must be an object')
        for key in ('rows', 'forwards', 'updated_labels', 'label_files_read', 'label_bytes_read'):
            value = migration_number(cost.get(key), phase + '/' + key, count=True)
            if value is not None:
                for axis, step in (('cost', paid), ('elapsed', int(seconds) if seconds is not None else None)):
                    if step is not None:
                        values.append(('fly_cached/' + axis + '/' + phase + '/' + key, value, step))
    scope_states = {}
    scopes = row.get('scopes') or {}
    if not isinstance(scopes, dict):
        raise ValueError('Cached-fly scopes must be an object')
    for phase, scope in scopes.items():
        if phase not in FLY_CACHED_PHASES:
            raise ValueError('Unknown cached-fly replay scope')
        if phase == 'train':
            continue  # TBPTT approximate incoming history is not a frozen-validation curve.
        if not isinstance(scope, dict):
            raise ValueError('Frozen cached-fly scope must be an object')
        if scope.get('own_history_rebuilt_from_row0') is not True or scope.get('parameters_frozen_for_entire_replay') is not True:
            raise ValueError('Frozen fly metrics require rebuilt row0 own history and frozen parameters')
        n = migration_number(scope.get('rows'), 'scope rows', count=True)
        version = migration_number(scope.get('parameter_version_at_end'), 'scope version', count=True)
        records = scope.get('episode_records')
        if n is None or version is None or not isinstance(records, list) or any(not isinstance(r, dict) or type(r.get('complete_episode')) is not bool for r in records):
            raise ValueError('Frozen fly scope population/version is unknown')
        record_rows = [migration_number(r.get('rows'), 'episode rows', count=True) for r in records]
        if any(value is None for value in record_rows) or sum(record_rows) != n:
            raise ValueError('Frozen fly rows differ from original episode-prefix records')
        metrics = scope.get('metrics') or {}
        if not isinstance(metrics, dict):
            raise ValueError('Frozen fly metrics must be an object')
        groups = metrics.get('groups') or {}
        if not isinstance(groups, dict) or any(not isinstance(v, dict) for v in groups.values()):
            raise ValueError('Frozen fly metric groups must be objects')
        if n and groups.get('all', {}).get('rows') != n:
            raise ValueError('Frozen fly KL population differs from scope rows')
        flat = [('rows', n)]
        for group, item in groups.items():
            if group != 'all' and group not in ('stage/' + str(i) for i in range(1, 7)):
                continue
            rows = migration_number(item.get('rows'), 'group rows', count=True)
            correct = migration_number(item.get('correct'), 'group correct', count=True)
            kl_sum = migration_number(item.get('KL_sum'), 'KL_sum')
            kl = migration_number(item.get('mean_legal_KL'), 'mean_legal_KL')
            agreement = migration_number(item.get('argmax_agreement'), 'argmax_agreement')
            if (rows is None or correct is None or correct > rows or kl_sum is None
                    or rows and (kl is None or agreement is None or not math.isclose(kl * rows, kl_sum, rel_tol=1e-6, abs_tol=1e-6)
                                 or not math.isclose(agreement, correct / rows, abs_tol=1e-9))):
                raise ValueError('Inconsistent frozen fly KL/agreement')
            flat.extend((group + '/' + key, value) for key, value in
                        (('rows', rows), ('mean_legal_KL', kl), ('argmax_agreement', agreement)) if value is not None)
        for name in ('teacher_greedy_counts', 'student_greedy_counts'):
            action_counts = metrics.get(name)
            if (not isinstance(action_counts, list) or len(action_counts) != 19
                    or any(migration_number(v, name, count=True) is None for v in action_counts) or sum(action_counts) != n):
                raise ValueError('Frozen fly action counts require the same source19 population')
            flat.append((name + '/Bomb18', action_counts[18]))
        hard = metrics.get('hard_window_counts') or {}
        if not isinstance(hard, dict):
            raise ValueError('Frozen fly hard counts must be an object')
        if hard.get('rows_measured') != n:
            raise ValueError('Hard spike counts must identify their measured row population')
        for key in ('whole_brain_spikes', 'descending_spikes', 'input_hard_events', 'rows_measured',
                    'mean_active_DN', 'mean_active_brain_neurons', 'mean_DN_filtered_rate_Hz'):
            value = migration_number(hard.get(key), 'hard/' + key, count=key in ('whole_brain_spikes', 'descending_spikes', 'input_hard_events', 'rows_measured'))
            if value is not None:
                flat.append(('hard_window/' + key, value))
        complete = n > 0 and bool(records) and all(r['complete_episode'] for r in records)
        scope_states[phase] = dict(rows=n, parameter_version=version, complete_episodes=complete,
                                  scope='empty_population' if n == 0 else 'complete_episode' if complete else 'continuous_row0_prefix', input_hard_events=hard.get('input_hard_events'))
        fingerprint = hashlib.sha256(json.dumps(scope, sort_keys=True, allow_nan=False).encode()).hexdigest()
        snapshots.append(dict(key='fly_cached_snapshot:' + str(folder) + ':' + phase, fingerprint=fingerprint,
            values=[('fly_cached/frozen_snapshots/' + phase + '/' + key, value, version) for key, value in flat]))
    state.update(status=status if terminal or status == 'running' else 'awaiting_terminal_commit',
                 phase=status if terminal else row.get('phase', 'unknown'), producer_phase=row.get('phase'),
                 reason=row.get('reason'), elapsed_seconds=seconds, optimizer_updates=paid,
                 total_model_forwards=counts['total_forward_passes'], frozen_scopes=scope_states,
                 unvalidated_after_update=row.get('unvalidated_after_update'))
    process = row.get('process') or {}
    if not isinstance(process, dict):
        raise ValueError('Cached-fly process identity must be an object')
    state['process'] = {key:process[key] for key in ('pid', 'create_time') if process.get(key) is not None}
    state['reader_audit'] = None
    if terminal is None:
        try:
            state['reader_audit'], reader_values = fly_reader_audit(folder, row)
            values.extend(reader_values)
        except (OSError, ValueError, KeyError) as error:
            # Optional IO telemetry cannot replace or invalidate producer costs.
            state['reader_audit'] = dict(status='read_error', reason=str(error), data_IO_only=True)
    return dict(state=state, values=values, snapshots=snapshots)


def active_migration_sources(config):
    active = config.get('active_migration_only', False)
    if type(active) is not bool:
        raise ValueError('active_migration_only must be boolean')
    if active and any(config.get(key) for key in ('training_source', 'validation_source', 'teacher_source',
            'actor_warmup_source', 'actor_development_source', 'actor_path_bc_source', 'reference_reports')):
        raise ValueError('Active migration view must not export canceled or historical jobs as active sources')
    paths = {role:Path(config[key]).resolve() if config.get(key) else None for role, key in
             (('d95_reference', 'd95_reference_source'), ('fly_cached_distillation', 'fly_cached_distillation_source'))}
    if paths['d95_reference'] is not None and paths['d95_reference'] == paths['fly_cached_distillation']:
        raise ValueError('Teacher reference and offline student need distinct job roots')
    return active, paths
ACTOR_PATH_BC_SCOPE = (
    '# 离线完整actor路径BC对照\n\n'
    'label行是离线监督标签，source行是轨迹重放读取行；均不是新原作决策或比赛。\n\n'
    'cost横轴是各arm已付费Adam步，elapsed横轴是本作业经过秒数。'
    '未知的selected/model步数不补0，也不把paid当作选中模型步数。\n\n'
    'heldout只显示initial/final两份完整冻结自历史快照，不是逐batch留出验证。'
    'CE/top1不是原作避弹成绩；Bomb计数是标签与argmax，不是原作Bomb使用或决死救援。')


def actor_path_bc_number(value, name, *, count=False):
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or count and type(value) is not int):
        raise ValueError('Invalid measured actor-path BC number: ' + name)
    return value


def actor_path_bc_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError('Expected actor-path BC JSON object: ' + name)
    return value


def actor_path_bc_heldout_metrics(report):
    """Complete immutable classification snapshots; no trainer/model import."""
    actor_path_bc_mapping(report, 'heldout')
    actor_path_bc_mapping(report.get('by_stage'), 'heldout stages')
    if (report.get('complete_episodes') != 6 or report.get('full_own_history_from_row0') is not True
            or report.get('parameter_frozen_during_evaluation') is not True
            or report.get('native_performance') is not False
            or report.get('Bomb_counts_are_labels_and_argmax_not_executed_game_events') is not True
            or set(report.get('by_stage', {})) != {str(i) for i in range(1, 7)}):
        raise ValueError('Actor-path BC heldout requires a complete frozen six-episode classification snapshot')
    count_keys = ('rows', 'legal_labels', 'correct', 'teacher_Bomb_labels', 'policy_argmax_Bomb', 'correct_Bomb')
    groups = [('overall', report.get('overall', {}))] + [
        ('stage' + str(i), report['by_stage'][str(i)]) for i in range(1, 7)]
    values = []
    for group, row in groups:
        actor_path_bc_mapping(row, 'heldout ' + group)
        if any(actor_path_bc_number(row.get(k), k, count=True) is None for k in count_keys):
            raise ValueError('Complete heldout counts must be known')
        n = row['legal_labels']
        if (n > row['rows'] or row['correct'] > n or row['teacher_Bomb_labels'] > n
                or row['policy_argmax_Bomb'] > n
                or row['correct_Bomb'] > min(row['teacher_Bomb_labels'], row['policy_argmax_Bomb'], row['correct'])):
            raise ValueError('Inconsistent heldout label/argmax counts')
        ce_sum = actor_path_bc_number(row.get('CE_sum'), 'CE_sum')
        if ce_sum is None:
            raise ValueError('Complete heldout CE sum must be known')
        ce, top = row.get('hard_CE'), row.get('top1_agreement')
        if n:
            if (actor_path_bc_number(ce, 'hard_CE') is None
                    or actor_path_bc_number(top, 'top1_agreement') is None or top > 1
                    or not math.isclose(ce * n, ce_sum, rel_tol=1e-6, abs_tol=1e-6)
                    or not math.isclose(top, row['correct'] / n, abs_tol=1e-9)):
                raise ValueError('Inconsistent complete heldout CE/top1')
        elif ce is not None or top is not None or ce_sum != 0:
            raise ValueError('Empty legal-label strata have unknown CE/top1, not zero scores')
        for key in ('rows', 'legal_labels', 'hard_CE', 'top1_agreement',
                    'teacher_Bomb_labels', 'policy_argmax_Bomb', 'correct_Bomb'):
            if row.get(key) is not None:
                values.append((group + '/' + key, row[key]))
    overall = report['overall']
    if (overall['legal_labels'] <= 0
            or any(sum(row[key] for row in report['by_stage'].values()) != overall[key] for key in count_keys)
            or not math.isclose(sum(row['CE_sum'] for row in report['by_stage'].values()), overall['CE_sum'], rel_tol=1e-6)):
        raise ValueError('Stage metrics do not sum to the complete heldout population')
    return values


def actor_path_bc_layout_groups():
    charts = [('主线门累计等待秒：横轴本作业秒', ['actor_path_bc/elapsed/main_gate_wait_seconds'])]
    for arm in ACTOR_PATH_BC_ARMS:
        p = 'actor_path_bc/' + arm
        snapshots = p + '/heldout_snapshots/'
        charts += [
            (arm + '：paid/selected/model步数，横轴paid Adam', [p + '/cost/' + k for k in
                ('paid_BC_optimizer_steps', 'selected_BC_optimizer_steps', 'model_BC_optimizer_steps')]),
            (arm + '：Adam进度，横轴本作业秒', [p + '/elapsed/' + k for k in
                ('paid_BC_optimizer_steps', 'selected_BC_optimizer_steps', 'model_BC_optimizer_steps')]),
            (arm + '：离线训练label/source行，横轴本作业秒', [p + '/elapsed/' + k for k in
                ('paid_BC_training_labels', 'train_source_rows_completed', 'pending_uncommitted_legal_labels')]),
            (arm + '：留出重放行进度，不是batch验证', [p + '/elapsed/' + k for k in
                ('baseline_heldout_source_rows_completed', 'final_heldout_source_rows_completed')]),
            (arm + '：完整heldout CE，仅initial/final快照',
                [snapshots + s + '/overall/hard_CE' for s in ('initial', 'final')]),
            (arm + '：完整heldout top1，仅initial/final快照',
                [snapshots + s + '/overall/top1_agreement' for s in ('initial', 'final')]),
            (arm + '：六面CE，仅两份快照',
                [snapshots + s + '/stage' + str(i) + '/hard_CE' for s in ('initial', 'final') for i in range(1, 7)]),
            (arm + '：Bomb标签/argmax计数，不是游戏事件',
                [snapshots + s + '/overall/' + k for s in ('initial', 'final')
                 for k in ('teacher_Bomb_labels', 'policy_argmax_Bomb')]),
        ]
    return [('08 离线完整actor路径BC：原作效果未由此验证', charts)]


def ongoing_stage_survival_metrics(worlds):
    """Current-world lower bounds, never a cumulative passage/test success rate.

    Whole-game Miss/Bomb counters are required together. Missing counters are
    unknown, and a new native game starts its own current-world counters.
    Zero Miss alone does not imply zero contact or zero deathbomb rescue.
    """
    def valid_number(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)

    advanced = [row for row in worlds
                if valid_number(row.get('stage')) and row['stage'] >= 2]
    known = [row for row in advanced
             if valid_number(row.get('misses')) and valid_number(row.get('bombs_used'))]
    return dict(
        current_worlds_past_stage1=len(advanced),
        current_worlds_past_stage1_with_known_Miss_Bomb=len(known),
        current_worlds_past_stage1_unknown_Miss_Bomb=len(advanced) - len(known),
        current_worlds_past_stage1_no_Miss=sum(row['misses'] == 0 for row in known),
        current_worlds_past_stage1_no_Miss_or_Bomb=sum(
            row['misses'] == 0 and row['bombs_used'] == 0 for row in known))


class RunExport:
    def __init__(self, source, logdir, control, *, actor_path_bc=False, migration_kind=None):
        if migration_kind is not None and (migration_kind not in MIGRATION_SCOPES or actor_path_bc):
            raise ValueError('Unknown or ambiguous migration exporter role')
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        self.source = Path(source).resolve()
        self.name = self.source.name
        key = hashlib.sha256(str(self.source).encode()).hexdigest()[:12]
        self.cursor_path = control / ('cursor-' + key + '.json')
        self.state = read(self.cursor_path) if self.cursor_path.exists() else dict(
            source=str(self.source), offsets={}, last_steps={}, reports={}, recent_episodes=[])
        if self.state['source'] != str(self.source):
            raise ValueError('TensorBoard cursor belongs to a different source')
        self.state.setdefault('episode_count',0)
        self.writer = EventFileWriter(str(logdir / self.name), max_queue_size=100, flush_secs=5)
        self.new_scalars = 0
        self._last_hardware_sample=0.
        self._event_wall_time=None;self._trial_wall_origin=None
        self._actor_path_bc_export = actor_path_bc
        self._migration_kind = migration_kind
        progress_path=self.source/'trial_000_both/progress.json'
        if progress_path.exists():
            row=read(progress_path)
            if row.get('updated_utc') and isinstance(row.get('wall_seconds'),(int,float)):
                self._trial_wall_origin=datetime.fromisoformat(row['updated_utc'].replace('Z','+00:00')).timestamp()-row['wall_seconds']
        self.custom_layout()
        if migration_kind is not None:
            self.text('about/scope', MIGRATION_SCOPES[migration_kind], 0)
            return
        if actor_path_bc:
            self.text('about/scope', ACTOR_PATH_BC_SCOPE, 0)
            return
        self.text('about/scope',
            '# B1 实时训练\n\n来源：`' + str(self.source) + '`\n\n'
            '- train：持续变化策略的训练数据，不能当独立验收。\n'
            '- eval/development：按报告区分禁 Bomb 首次碰撞诊断与允许 Bomb 的完整六面；前者触顶表示截尾。\n'
            '- eval/test：冻结模型的新种子测试；自然整局与禁 Bomb 分开。\n'
            '- live 的横轴为本轮累计采样决策；loss 的横轴为本轮累计优化决策，恢复训练保留此前进度。\n'
            '- 恢复前后分别显示运行曲线；原始局部采样计数与实际付费采样下界另行保留。\n'
            '- 测试重启后的横轴接续此前已确认的采样下界；本次采样另列，速度只按本次计算。\n'
            '- train/episode 与 train/recent_games 的横轴为已完成游戏编号，包含同一批结束的每局。\n'
            '- live/current_worlds_past_stage1：当前已到第二面或之后的实例；无Miss与无Miss且无Bomb分开。'
            '这些计数会随新局重置，不是固定策略通关率，也不能把无Miss解释为零碰撞。\n'
            '- proxy 为当前运动预测指标，不等于原作实际避弹成绩；硬件利用率是整张显卡。\n'
            '- 所有原作日志只读；本页面不启动、停止或控制训练。', 0)

    def custom_layout(self):
        from tensorboard.plugins.custom_scalar import layout_pb2,metadata
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        groups=[
            ('01 实际避弹与完整游戏',[
                ('开发：禁Bomb首次碰撞或截尾帧数',['eval/development/mean_first_hit_or_cap_frames','eval/development/median_first_hit_or_cap_frames']),
                ('开发：允许Bomb完整六面通关率',['eval/development/full_story/clear_rate']),
                ('开发：完整六面平均原作帧数',['eval/development/full_story/mean_native_frames']),
                ('独立禁Bomb：新旧模型最高概率动作',['eval/test/noBomb/trained_greedy/mean_capped_frames','eval/test/noBomb/initial_greedy/mean_capped_frames','eval/test/noBomb/uniform18/mean_capped_frames']),
                ('独立完整六面：新旧模型通关率',['eval/test/full_story/initial/clear_rate','eval/test/full_story/final/clear_rate']),
                ('测试中父模型：尚未结束的实例所在面数',['eval/runtime/full_story_initial/current_worlds_in_stage_'+str(i) for i in range(1,7)]),
                ('测试中新模型：尚未结束的实例所在面数',['eval/runtime/full_story_final/current_worlds_in_stage_'+str(i) for i in range(1,7)]),
                ('训练游戏：最近20局中弹与Bomb',['train/recent_games/mean_misses','train/recent_games/mean_bombs_used']),
                ('训练游戏：首次碰撞（允许Bomb）',['train/episode/first_collision_frames_Bomb_allowed']),
                ('当前已过第一面：无Miss、无Miss且无Bomb（训练中实例，非通关率）',[
                    'live/current_worlds_past_stage1',
                    'live/current_worlds_past_stage1_no_Miss',
                    'live/current_worlds_past_stage1_no_Miss_or_Bomb']),
                ('当前已过第一面：Miss/Bomb计数未知的实例',[
                    'live/current_worlds_past_stage1_unknown_Miss_Bomb']),
            ]),
            ('02 PPO与价值学习',[
                ('Actor损失与总损失',['train/optimization/policy_loss','train/optimization/total_loss']),
                ('三个奖励分支的价值损失',['train/value_loss/survival','train/value_loss/combat','train/value_loss/resources']),
                ('更新后的价值解释率',['train/explained_variance_after/survival','train/explained_variance_after/combat','train/explained_variance_after/resources']),
                ('KL与裁剪比例',['train/optimization/approx_kl','train/optimization/clip_fraction']),
                ('动作熵',['train/optimization/entropy']),
                ('实际采样与PPO概率的温度',['train/optimization/action_temperature']),
                ('本批平均奖励分量',['train/reward_scale/mean/survival','train/reward_scale/mean/combat','train/reward_scale/mean/resources']),
            ]),
            ('03 教师、移动与决死',[
                ('教师CE、目标熵与KL',['train/guidance/teacher_mean_CE','train/guidance/teacher_mean_target_entropy','train/guidance/teacher_mean_KL']),
                ('参考可用比例',['train/guidance/teacher_usable_fraction','train/guidance/teacher_unusable_reference_fraction']),
                ('动作输出梯度：辅助项相对PPO',['train/output_gradients/online_teacher/L2_relative_to_PPO_actor','train/output_gradients/movement_guidance/L2_relative_to_PPO_actor','train/output_gradients/deathbomb/L2_relative_to_PPO_actor']),
                ('匹配无保护移动：每千次实际碰撞',['train/guidance/observed_collisions_per_1000_matched_movements']),
                ('几何预测危险方向概率（代理指标）',['train/guidance/proxy_unsafe_movement_probability']),
                ('决死机会中的Bomb选择和概率',['train/guidance/deathbomb_behavior_Bomb_selection_rate','train/guidance/deathbomb_Bomb_probability']),
            ]),
            ('04 速度、硬件与进度',[
                ('采样与完整一组训练速度',['train/throughput/sampling_decisions_per_second','train/throughput/sampling_and_update_decisions_per_second','live/decisions_per_second']),
                ('每组耗时：采样、Adam、记忆重热',['train/throughput/collect_seconds','train/throughput/learn_seconds','train/throughput/recurrent_refresh_seconds']),
                ('GPU整卡实时利用率（本运行经过秒数）',['hardware_live/system_gpu_utilization_percent','hardware_live/system_gpu_memory_utilization_percent']),
                ('训练器CUDA内存MiB',['live/cuda_allocated_mb','live/cuda_reserved_mb']),
                ('累计采样和优化决策',['live/candidate_decisions','live/candidate_optimized_decisions','live/candidate_paid_decisions_lower_bound']),
                ('当前30实例所在面数',['live/worlds_in_stage_'+str(i) for i in range(1,7)]),
                ('完整测试累计决策速度',['eval/runtime/full_story_initial/mean_decisions_per_second','eval/runtime/full_story_final/mean_decisions_per_second']),
            ]),
            ('05 完整教师数据采集：不是神经网络成绩',[
                ('原作动作、已记录和已落盘样本',['teacher_data/acknowledged_native_actions','teacher_data/recorded_samples','teacher_data/committed_samples']),
                ('已完成自然局与目标局数',['teacher_data/complete_native_episodes','teacher_data/target_native_episodes']),
                ('教师已结束局的终局计数',['teacher_data/clear_episodes','teacher_data/gameover_episodes']),
                ('教师当前实例所在面数',['teacher_data/current_worlds_in_stage_'+str(i) for i in range(1,7)]),
                ('教师采集速度：累计动作/秒',['teacher_data/mean_native_actions_per_second']),
                ('采集盘当前可用GiB',['teacher_data/free_storage_GiB']),
            ]),
            ('06 完整轨迹模仿预训练：原作效果另行验证',[
                ('整局分组：训练与留出轨迹CE',['bc/epoch/train/hard_teacher_CE','bc/epoch/heldout/hard_teacher_CE']),
                ('整局分组：与实际教师动作的一致率',['bc/epoch/train/executed_action_top1_agreement','bc/epoch/heldout/executed_action_top1_agreement']),
                ('留出：移动、原地与风险代理动作一致率（仅BC前后）',['bc/strata/teacher_moves/action_agreement','bc/strata/teacher_stays/action_agreement','bc/strata/frozen_risk_critical_proxy/action_agreement']),
                ('留出：六面分关卡动作一致率（仅BC前后）',['bc/strata/stage'+str(i)+'/action_agreement' for i in range(1,7)]),
                ('每轮平均训练损失',['bc/epoch/training_minibatch_CE']),
                ('输出裁剪边界上的教师动作比例',['bc/epoch/train/teacher_action_at_policy_logit_limit_fraction','bc/epoch/heldout/teacher_action_at_policy_logit_limit_fraction']),
                ('实际付费与最后选中的Adam步数',['bc/selection/paid_BC_optimizer_steps','bc/selection/selected_BC_optimizer_steps']),
                ('实时训练步数与遍数（本阶段经过秒数）',['bc/live/BC_paid_optimizer_steps','bc/live/epoch','bc/live/target_epochs']),
                ('实时CUDA内存MiB（本阶段经过秒数）',['bc/live/cuda_allocated_mb','bc/live/cuda_reserved_mb']),
                ('当前编码或记忆重热步骤完成百分比',['bc/phase/percent_complete']),
                ('当前编码或记忆重热：已处理与目标行数',['bc/phase/completed_rows','bc/phase/total_rows']),
            ]),
            ('07 预训练后原作避弹开发：不是正式验收',[
                ('新旧模型已完成局数：各20局',['bc_dev/runtime/trained_greedy_completed','bc_dev/runtime/initial_greedy_completed']),
                ('总计完成局数与40局目标',['bc_dev/runtime/completed','bc_dev/runtime/target']),
                ('完整20局：首次碰撞或32768帧截止的均值',['bc_dev/final/trained_greedy/mean_capped_frames','bc_dev/final/initial_greedy/mean_capped_frames']),
                ('完整20局：触顶截尾比例',['bc_dev/final/trained_greedy/censored_fraction','bc_dev/final/initial_greedy/censored_fraction']),
                ('配对均值差与描述性区间',['bc_dev/final/paired_frame_gain','bc_dev/final/paired_lower95','bc_dev/final/paired_upper95']),
            ]),
        ]
        if getattr(self, '_actor_path_bc_export', False):
            groups = actor_path_bc_layout_groups()
        if getattr(self, '_migration_kind', None) == 'd95_reference':
            groups = [('D95教师原作参考：不是果蝇实战', [
                ('原作决策横轴：本次完整20局进度', ['d95_reference/native/completed', 'd95_reference/native/target']),
                ('实际秒横轴：原作决策', ['d95_reference/elapsed/policy_decisions']),
                ('完整20局单次快照', ['d95_reference/final/clear_rate', 'd95_reference/final/mean_misses', 'd95_reference/final/mean_bombs_used'])])]
        elif getattr(self, '_migration_kind', None) == 'fly_cached_distillation':
            groups = [('果蝇离线缓存蒸馏：不是原作比赛', [
                ('Adam横轴：模型forward/已完成更新', ['fly_cached/cost/total_forward_passes', 'fly_cached/cost/optimizer_updates']),
                ('实际秒横轴：Adam更新', ['fly_cached/elapsed/optimizer_updates']),
                ('数据读出：横轴reader guard秒数；不是模型forward',
                 ['fly_cached/reader_audit/rows_yielded', 'fly_cached/reader_audit/payload_rows_verified']),
                ('训练更新KL：TBPTT近似历史，不能代替冻结重放', ['fly_cached/train_update/mean_legal_KL_TBPTT_approximate_history']),
                ('冻结自身历史KL：横轴参数版本，仅before/after快照',
                 ['fly_cached/frozen_snapshots/' + phase + '/all/mean_legal_KL' for phase in FLY_CACHED_PHASES if phase != 'train']),
                ('硬DN窗口脉冲汇总：不证明逐事件相同',
                 ['fly_cached/frozen_snapshots/' + phase + '/hard_window/descending_spikes' for phase in FLY_CACHED_PHASES if phase != 'train'])])]
        layout=layout_pb2.Layout(category=[layout_pb2.Category(title=title,
            chart=[layout_pb2.Chart(title=name,multiline=layout_pb2.MultilineChartContent(tag=tags))
                   for name,tags in charts]) for title,charts in groups])
        self.writer.add_event(Event(wall_time=time.time(),step=0,summary=Summary(value=[Summary.Value(
            tag=metadata.CONFIG_SUMMARY_TAG,metadata=metadata.create_summary_metadata(),
            tensor=TensorProto(dtype=7,string_val=[layout.SerializeToString()]))])))

    def scalar(self, tag, value, step):
        if value is None or isinstance(value, (str, dict, list, tuple)):
            return
        value = float(value)
        if not math.isfinite(value):
            return
        step = int(step)
        if step <= self.state['last_steps'].get(tag, -1):
            return
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        self.writer.add_event(Event(wall_time=getattr(self,'_event_wall_time',None) or time.time(), step=step,
            summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)])))
        self.state['last_steps'][tag] = step
        self.new_scalars += 1

    def text(self, tag, value, step):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        tensor = TensorProto(dtype=7, string_val=[value.encode('utf-8')],
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
        metadata = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name='text'))
        self.writer.add_event(Event(wall_time=time.time(), step=int(step),
            summary=Summary(value=[Summary.Value(tag=tag, tensor=tensor, metadata=metadata)])))

    def migration(self, kind, folder):
        """Only current explicit job roles; no model/trainer imports or pt reads."""
        if kind not in MIGRATION_SCOPES:
            raise ValueError('Unknown migration exporter role')
        try:
            parsed = (parse_d95_reference(folder) if kind == 'd95_reference'
                      else parse_fly_cached_distillation(folder))
            values = list(parsed['values'])
            updates = {}
            for snapshot in parsed['snapshots']:
                previous = self.state['reports'].get(snapshot['key'])
                if previous is not None and previous != snapshot['fingerprint']:
                    raise ValueError('Published frozen migration snapshot changed')
                if previous is None:
                    values.extend(snapshot['values'])
                    updates[snapshot['key']] = snapshot['fingerprint']
        except (OSError, ValueError, KeyError) as error:
            self.state[kind] = dict(status='read_error', phase='unknown', source=str(folder),
                                    reason=str(error), producer_status='unknown')
            if self.state.get(kind + '_error') != str(error):
                self.text('about/' + kind + '_phase', MIGRATION_SCOPES[kind]
                          + '\n\n读取异常，当前状态未知：' + str(error), self.state.get(kind + '_text_step', 0))
                self.state[kind + '_error'] = str(error)
            raise
        for tag, value, step in values:
            self.scalar(tag, value, step)
        self.state['reports'].update(updates)
        state = parsed['state']
        step = int(state['elapsed_seconds']) if state.get('elapsed_seconds') is not None else self.state.get(kind + '_text_step', 0)
        audit = state.get('reader_audit') or {}
        phase = [state.get(key) for key in ('status', 'phase', 'producer_file', 'reason')]
        phase += [audit.get(key) for key in ('phase', 'episode', 'producer_file')]
        if self.state.get(kind + '_phase') != phase:
            self.text('about/' + kind + '_phase', MIGRATION_SCOPES[kind] + '\n\n当前producer状态：'
                      + json.dumps(state, ensure_ascii=False), step)
        self.state[kind + '_phase'] = phase
        self.state[kind + '_text_step'] = step
        self.state.pop(kind + '_error', None)
        self.state[kind] = state

    def lines(self, path):
        path = Path(path)
        if not path.exists():
            return
        key = str(path)
        offset = self.state['offsets'].get(key, 0)
        with _shared_binary(path) as stream:
            stream.seek(0, 2)
            if stream.tell() < offset:
                raise ValueError('Append-only training log shrank; do not silently mix histories')
            stream.seek(offset)
            data = stream.read(8 * 1024 * 1024)
        complete = data[:data.rfind(b'\n') + 1]
        for line in complete.splitlines(keepends=True):
            if line.strip():
                yield json.loads(line)
            offset += len(line)
            self.state['offsets'][key] = offset

    @staticmethod
    def decision_offset(trial):
        path=trial/'config.json'
        return read(path).get('candidate_decision_offset',0) if path.exists() else 0

    def updates(self, trial):
        offset=self.decision_offset(trial)
        for row in self.lines(trial / 'updates.jsonl'):
            origin=getattr(self,'_trial_wall_origin',None)
            self._event_wall_time=origin+row['wall_seconds'] if origin is not None and 'wall_seconds' in row else None
            step = row['decisions']+offset
            metrics = row['metrics']
            for key in ('total_loss', 'policy_loss', 'entropy', 'approx_kl', 'clip_fraction',
                        'grad_norm', 'entropy_coeff', 'max_post_epoch_kl', 'epochs_completed',
                        'sample_passes', 'custom_update_rejections', 'action_temperature'):
                self.scalar('train/optimization/' + key, metrics.get(key), step)
            for key, value in (('rollouts', row['updates']), ('Adam_steps', row['adam'])):
                self.scalar('train/counters/' + key, value, step)
            for group in ('value_loss', 'explained_variance_before'):
                for index, value in enumerate(metrics.get(group, [])):
                    self.scalar('train/' + group + '/' + ('survival', 'combat', 'resources')[index], value, step)
            for group in ('reward_scale','return_scale','advantage_scale'):
                for statistic,values in metrics.get(group,{}).items():
                    for index,value in enumerate(values):
                        self.scalar('train/'+group+'/'+statistic+'/'+('survival','combat','resources')[index],value,step)
            post = metrics.get('post_epoch_diagnostics', [])
            if post:
                for index, value in enumerate(post[-1].get('explained_variance', [])):
                    self.scalar('train/explained_variance_after/' + ('survival', 'combat', 'resources')[index], value, step)
            timing = row.get('latest_rollout_timing', {})
            for key in ('collect_seconds', 'learn_seconds', 'recurrent_refresh_seconds',
                        'sampling_decisions_per_second', 'sampling_and_update_decisions_per_second'):
                self.scalar('train/throughput/' + key, timing.get(key), step)
            reference = metrics.get('online_teacher_guidance', {})
            for key,tag in [('mean_CE','teacher_mean_CE'),('mean_target_entropy','teacher_mean_target_entropy'),
                            ('mean_teacher_to_policy_KL','teacher_mean_KL'),('coefficient','teacher_coefficient'),
                            ('teacher_execution_overrides','teacher_execution_overrides')]:
                self.scalar('train/guidance/'+tag,reference.get(key),step)
            count = reference.get('states', 0)
            if count:
                self.scalar('train/guidance/teacher_usable_fraction', reference.get('usable_reference_states', count) / count, step)
                self.scalar('train/guidance/teacher_unusable_reference_fraction',
                    reference.get('unusable_reference_states',reference.get('unknown_laser_reference_states',0))/count,step)
                if 'unknown_laser_reference_states' in reference:
                    self.scalar('train/guidance/teacher_unknown_laser_fraction',reference['unknown_laser_reference_states']/count,step)
            movement = metrics.get('movement_guidance', {})
            for key in ('observed_collision_actions', 'observed_four_tick_transitions',
                        'dense_loss_mean', 'collision_CE_mean'):
                self.scalar('train/guidance/' + key, movement.get(key), step)
            matched=movement.get('observed_four_tick_transitions',0)
            if matched:
                self.scalar('train/guidance/observed_collisions_per_1000_matched_movements',
                    1000*movement.get('observed_collision_actions',0)/matched,step)
            self.scalar('train/guidance/proxy_unsafe_movement_probability',
                movement.get('post_update_dodge_policy', {}).get('predicted_unsafe4_movement_mass'), step)
            for key, value in movement.get('dense', {}).get('abstentions', {}).items():
                self.scalar('train/guidance/auxiliary_abstentions/' + key, value, step)
            bomb = metrics.get('deathbomb_auxiliary', {}).get('post_update_policy', {})
            self.scalar('train/guidance/deathbomb_Bomb_probability', bomb.get('Bomb_probability'), step)
            auxiliary=metrics.get('deathbomb_auxiliary',{})
            for key in ('unique_usable_decisions','behavior_Bomb_selection_rate','weighted_loss_mean_per_minibatch'):
                self.scalar('train/guidance/deathbomb_'+key,auxiliary.get(key),step)
            for component,values in metrics.get('actor_output_loss_gradients',{}).get('components',{}).items():
                for key,value in values.items():
                    self.scalar('train/output_gradients/'+component+'/'+key,value,step)
        self._event_wall_time=None

    def episodes(self, trial):
        offset=self.decision_offset(trial)
        for row in self.lines(trial / 'training_episodes.jsonl'):
            self.state['episode_count']+=1
            step = self.state['episode_count']
            self.scalar('train/episode/source_sampled_decisions',row['at_decisions']+offset,step)
            values = {key: row.get(key) for key in ('native_frames', 'stage', 'misses', 'bombs_used')}
            values['clear'] = int(row.get('outcome') == 'clear')
            for key, value in values.items():
                self.scalar('train/episode/' + key, value, step)
            for key in ('reward_vector','raw_reward_vector'):
                for index,value in enumerate(row.get(key,[])):
                    self.scalar('train/episode/'+key+'/'+('survival','combat','resources')[index],value,step)
            diag = row.get('reward_diagnostics', {})
            for key in ('confirmed_deathbombs', 'ordinary_or_unconfirmed_bombs', 'vulnerable_frames'):
                self.scalar('train/episode/' + key, diag.get(key), step)
            movement = row.get('movement_diagnostics', {})
            self.scalar('train/episode/first_collision_frames_Bomb_allowed',
                movement.get('first_collision_observed_frames'), step)
            for key in ('vulnerable_edge_fraction','observed_collision_events','vulnerable_movement_decisions'):
                self.scalar('train/episode/'+key,movement.get(key),step)
            self.state['recent_episodes'].append(values)
            self.state['recent_episodes'] = self.state['recent_episodes'][-20:]
            recent = self.state['recent_episodes']
            self.scalar('train/recent_games/window_size', len(recent), step)
            for key in values:
                available = [x[key] for x in recent if x.get(key) is not None]
                if available:
                    self.scalar('train/recent_games/mean_' + key, statistics.mean(available), step)

    def hardware(self,step):
        if time.monotonic()-self._last_hardware_sample<30:return
        self._last_hardware_sample=time.monotonic()
        import subprocess
        try:
            result=subprocess.run(['nvidia-smi','-i','0',
                '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:return
            values=result.stdout.strip().splitlines()[0].split(',')
            tags=('system_gpu_utilization_percent','system_gpu_memory_utilization_percent',
                'system_gpu_used_MiB','system_gpu_total_MiB','system_gpu_temperature_C','system_gpu_power_W')
            origin=self.state.setdefault('hardware_wall_origin_unix',time.time())
            seconds=max(0,int(time.time()-origin))
            for tag,value in zip(tags,values):
                try:
                    self.scalar('hardware/'+tag,float(value.strip()),step)
                    self.scalar('hardware_live/'+tag,float(value.strip()),seconds)
                except ValueError:pass
        except (OSError,subprocess.SubprocessError,IndexError):pass

    def live(self, trial):
        path = trial / 'progress.json'
        if not path.exists():
            return None
        progress = read(path)
        step = progress.get('candidate_decisions',progress.get('actual_decisions',0)+self.decision_offset(trial))
        self.scalar('live/candidate_decisions',step,step)
        for key in ('candidate_optimized_decisions', 'candidate_paid_decisions_lower_bound',
                    'actual_decisions', 'target_decisions', 'candidate_target_decisions',
                    'unoptimized_sampled_decisions', 'episodes', 'clear_episodes',
                    'decisions_per_second', 'cuda_allocated_mb', 'cuda_reserved_mb', 'action_temperature'):
            self.scalar('live/' + key, progress.get(key), step)
        worlds = progress.get('current_environments', [])
        if worlds:
            counts = Counter(x.get('stage', 0) for x in worlds)
            self.scalar('live/current_worlds', len(worlds), step)
            self.scalar('live/max_stage', max(counts), step)
            for stage in range(1, 7):
                self.scalar('live/worlds_in_stage_' + str(stage), counts.get(stage, 0), step)
            for key in ('lives', 'misses', 'bombs_used', 'power'):
                values = [x[key] for x in worlds if x.get(key) is not None]
                if values:
                    self.scalar('live/worlds_mean_' + key, statistics.mean(values), step)
            for key, value in ongoing_stage_survival_metrics(worlds).items():
                self.scalar('live/' + key, value, step)
        status = progress.get('status', '') + '/' + progress.get('stage', '')
        if self.state.get('phase') != status:
            self.text('about/current_phase', status + '\n\n' + progress.get('updated_utc', ''), step)
            self.state['phase'] = status
        self.state['latest_source_utc'] = progress.get('updated_utc')
        self.state['latest_sampled_decisions'] = step
        self.state['latest_optimized_decisions'] = progress.get('candidate_optimized_decisions', 0)
        self.state['max_live_stage'] = max((x.get('stage', 0) for x in worlds), default=None)
        self.hardware(step)
        return progress

    def diagnostic_report(self, path, step, prefix):
        if not path.exists():
            return
        report = read(path)
        identity = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        if self.state['reports'].get(str(path)) == identity:
            return
        rows = report.get('episode_results', [])
        if not rows:
            return
        if report.get('training_mode') == 'full-story-v1' and report.get('policy_frozen') is True:
            protocol = report.get('episode_protocol', {})
            episodes = report.get('episodes')
            clears = sum(row.get('outcome') == 'clear' for row in rows)
            if (report.get('native_seed_split') != 'development'
                    or protocol.get('first_hit_terminal') is not False
                    or protocol.get('native_frame_horizon') is not None
                    or report.get('continues') != 0
                    or episodes != len(rows) or report.get('clears') != clears
                    or any(row.get('outcome') not in ('clear', 'gameover')
                           or row.get('native_frames', 0) <= 0 for row in rows)):
                raise ValueError('Development full-game metrics need complete natural development episodes')
            values = dict(episodes=len(rows), clears=clears, clear_rate=clears / len(rows),
                mean_native_frames=statistics.mean(row['native_frames'] for row in rows))
            for key in ('mean_max_stage', 'mean_misses', 'mean_bombs_used'):
                values[key] = report.get(key)
            for key, value in values.items():
                self.scalar(prefix + '/full_story/' + key, value, step)
            self.text(prefix + '/scope',
                f'{len(rows)}局完整原作开发评估；允许Bomb，从开局至原作Game Over或六面通关；'
                '无帧截断、无续关；不是正式稳定通关验收。', step)
            self.state['reports'][str(path)] = identity
            return
        cap = report.get('diagnostic_native_frame_cap', 16384)
        frames = [min(row['native_frames'], row.get('diagnostic_native_frame_cap', cap)) for row in rows]
        self.scalar(prefix + '/mean_first_hit_or_cap_frames', statistics.mean(frames), step)
        self.scalar(prefix + '/median_first_hit_or_cap_frames', statistics.median(frames), step)
        self.scalar(prefix + '/noBomb_stage1_pass_fraction', sum(x['stage'] >= 2 for x in rows) / len(rows), step)
        self.scalar(prefix + '/censored_fraction', sum(bool(x.get('right_censored_at_cap')) for x in rows) / len(rows), step)
        self.scalar(prefix + '/observed_collision_fraction', sum(bool(x.get('first_collision_observed')) for x in rows) / len(rows), step)
        self.scalar(prefix + '/games', len(rows), step)
        self.text(prefix + '/scope', f'{len(rows)}局；禁Bomb；首次碰撞或{cap}帧停止。触顶是截尾，不是死亡或通关。', step)
        self.state['reports'][str(path)] = identity

    def frozen_validation(self, folder, optimized):
        if folder is None:
            return
        folder = Path(folder)
        recovery_path=folder/'recovery_protocol.json'
        recovery=read(recovery_path) if recovery_path.exists() else {}
        path = folder / 'no_Bomb_fresh100/report.json'
        if path.exists():
            report = read(path)
            for mode, values in report.get('policies', {}).items():
                for key, value in values.items():
                    self.scalar('eval/test/noBomb/' + mode + '/' + key, value, optimized)
        for mode in ('initial', 'final'):
            path = folder / ('full_story_' + mode) / 'report.json'
            if path.exists():
                report = read(path)
                for key in ('episodes', 'clears', 'clear_rate', 'mean_native_frames',
                            'mean_max_stage', 'mean_misses', 'mean_bombs_used'):
                    self.scalar('eval/test/full_story/' + mode + '/' + key, report.get(key), optimized)
        self.state['validation_runtime'] = {}
        for name in ('no_Bomb_fresh100', 'full_story_initial', 'full_story_final'):
            path = folder / name / 'progress.json'
            result_path = path.parent / 'report.json'
            if not path.exists() and not result_path.exists():
                continue
            progress = read(path) if path.exists() else {}
            result = read(result_path) if result_path.exists() else {}
            complete = result.get('status') == 'completed' or (
                name.startswith('full_story_') and result.get('policy_frozen') is True
                and result.get('episodes') == 20)
            # The final atomic report can be newer than the throttled progress
            # file. It also retires that file's stale active-world snapshot.
            done = (result.get('completed', result.get('episodes', 0)) if complete
                    else progress.get('completed', 0))
            prefix = 'eval/runtime/' + name
            self.scalar(prefix + '/completed_games', done, done)
            self.state['validation_runtime'][name] = dict(
                status='completed' if complete else 'running', completed_games=done)
            if complete:
                continue
            local_step = progress.get('policy_decisions', progress.get('decisions', 0))
            prior = recovery.get('prior_full_story_acknowledged_decisions_lower_bound',0) if name=='full_story_initial' else 0
            if type(prior) is not int or prior<0:
                raise ValueError('Nonnegative original evaluation sampling lower bound required')
            step = prior + local_step
            self.scalar(prefix + '/sampled_decisions', step, step)
            if prior:
                self.scalar(prefix + '/new_attempt_decisions', local_step, step)
                self.scalar(prefix + '/prior_sampling_lower_bound', prior, step)
            seconds = progress.get('wall_seconds')
            if isinstance(seconds, (int, float)) and seconds > 0:
                self.scalar(prefix + '/mean_decisions_per_second', local_step / seconds, step)
            worlds = progress.get('current_environments', [])
            measured = [row for row in worlds if type(row.get('stage')) is int
                        and 1 <= row['stage'] <= 6]
            if measured:
                counts = Counter(row['stage'] for row in measured)
                for stage in range(1, 7):
                    self.scalar(prefix + '/current_worlds_in_stage_' + str(stage), counts[stage], step)
                self.scalar(prefix + '/current_observed_worlds', len(measured), step)
                self.scalar(prefix + '/current_max_stage', max(counts), step)
                frames = [row['episode_frames'] for row in measured
                          if isinstance(row.get('episode_frames'), (int, float))]
                if frames:
                    self.scalar(prefix + '/current_mean_episode_frames', statistics.mean(frames), step)
                self.state['validation_runtime'][name].update(
                    current_observed_worlds=len(measured), current_stage_counts=dict(counts),
                    sampled_decisions=step,new_attempt_decisions=local_step,
                    prior_sampling_lower_bound=prior)

    def teacher_collection(self, folder):
        folder=Path(folder);path=folder/'progress.json'
        if not path.is_file():return
        row=read(path)
        if row.get('neural_model_test') is not False or row.get('optimizer_updates')!=0:
            raise ValueError('Rule-teacher data collection must not be presented as a neural test')
        step=row['acknowledged_native_actions']
        for key in ('acknowledged_native_actions','recorded_samples','committed_samples',
                    'complete_native_episodes','target_native_episodes','active_workers'):
            self.scalar('teacher_data/'+key,row.get(key),step)
        outcomes=row.get('outcomes',{})
        self.scalar('teacher_data/clear_episodes',outcomes.get('clear',0),step)
        self.scalar('teacher_data/gameover_episodes',outcomes.get('gameover',0),step)
        seconds=row.get('wall_seconds',0)
        if seconds>0:self.scalar('teacher_data/mean_native_actions_per_second',step/seconds,step)
        import shutil
        self.scalar('teacher_data/free_storage_GiB',shutil.disk_usage(folder).free/1024**3,step)
        counts=Counter(x['stage'] for x in row.get('current_environments',[]) if x.get('stage') in range(1,7))
        for stage in range(1,7):self.scalar('teacher_data/current_worlds_in_stage_'+str(stage),counts[stage],step)
        status='failed' if (folder/'failure.json').is_file() else row.get('status','teacher_collection')
        self.state['teacher_collection']=dict(status=status,source=str(folder),
            updated_utc=row.get('updated_utc'),acknowledged_native_actions=step,
            recorded_samples=row.get('recorded_samples'),committed_samples=row.get('committed_samples'),
            complete_native_episodes=row.get('complete_native_episodes'),target_native_episodes=row.get('target_native_episodes'),
            current_stage_counts=dict(counts),neural_model_test=False,optimizer_updates=0)
        self.text('about/teacher_collection',
            '规则教师完整原作轨迹采集；不是神经网络成绩。每局从开局运行到自然终局，保留19动作和自主Bomb。'
            '\n\n记录与落盘样本的差值是等待压缩的缓存，不是丢失数据。\n\n状态：'+status,step)

    def actor_warmup(self, folder):
        """Separate imitation metrics from PPO and from native performance."""
        folder=Path(folder)
        config_path=folder/'config.json'
        if not config_path.is_file():
            if (folder/'progress.json').is_file():
                raise ValueError('Imitation progress requires its declared training role')
            self.state['actor_warmup']=dict(status='waiting_for_source',source=str(folder))
            return
        config=read(config_path)
        if (config.get('neural_training_role')!='temporary-full-teacher-actor-warmup'
                or config.get('numerical_qualification_only') or config.get('deployment_forbidden')
                or config.get('PPO_updates')!=0 or config.get('native_worlds')!=0):
            raise ValueError('Only real CUDA actor imitation belongs in the pretraining run')
        progress=read(folder/'progress.json') if (folder/'progress.json').is_file() else {}
        if progress and (progress.get('CUDA_required') is not True
                or progress.get('PPO_updates')!=0 or progress.get('native_worlds')!=0):
            raise ValueError('Actor warmup must not be represented as native PPO training')
        metrics=('hard_teacher_CE','executed_action_top1_agreement',
                 'teacher_action_at_policy_logit_limit_fraction','legal_labels')
        def epoch_metrics(row,step):
            for split in ('train','heldout'):
                for key in metrics:
                    self.scalar('bc/epoch/'+split+'/'+key,row.get(split,{}).get(key),step)
        initial=folder/'initial_fit.json'
        if initial.is_file():epoch_metrics(read(initial),0)
        for row in self.lines(folder/'fit_epochs.jsonl'):
            step=row['epoch']
            epoch_metrics(row,step)
            for key in ('training_minibatch_CE','paid_BC_optimizer_steps','BC_examples_seen'):
                self.scalar('bc/epoch/'+key,row.get(key),step)
        # Wall time continues to move during one-time encoder caching and
        # recurrent context refresh, when the Adam counter legitimately stays put.
        seconds=int(progress.get('wall_seconds',0))
        completed,total=progress.get('completed'),progress.get('total')
        if completed is not None or total is not None:
            if type(completed) is not int or type(total) is not int or not 0<=completed<=total or total<=0:
                raise ValueError('Measured preparation progress must be bounded original row counts')
            self.scalar('bc/phase/completed_rows',completed,seconds)
            self.scalar('bc/phase/total_rows',total,seconds)
            self.scalar('bc/phase/percent_complete',100*completed/total,seconds)
        for key in ('epoch','target_epochs','BC_paid_optimizer_steps','BC_examples_seen',
                    'unique_native_teacher_rows','trainable_parameters','total_parameters',
                    'cuda_allocated_mb','cuda_reserved_mb','latest_training_CE'):
            self.scalar('bc/live/'+key,progress.get(key),seconds)
        report=read(folder/'fit_report.json') if (folder/'fit_report.json').is_file() else {}
        result=read(folder/'result.json') if (folder/'result.json').is_file() else {}
        if result and (result.get('numerical_qualification_only') or result.get('deployment_forbidden')):
            raise ValueError('Numerical smoke results cannot be displayed as policy learning')
        selection=result or report
        if selection:
            step=result.get('epochs_executed',len(report.get('epoch_history',[])))
            for key in ('paid_BC_optimizer_steps','selected_BC_optimizer_steps','best_epoch','selected_epoch'):
                self.scalar('bc/selection/'+key,selection.get(key),step)
            self.scalar('bc/selection/selected_heldout_CE',result.get('selected_heldout',{}).get('hard_teacher_CE'),step)
        status=('failed' if (folder/'failure.json').is_file() else
                'completed' if result.get('status')=='completed' else progress.get('status','starting'))
        phase=progress.get('phase',progress.get('stage','starting'))
        if self.state.get('actor_warmup',{}).get('phase')!=phase:
            self.text('about/actor_warmup',
                '完整原作轨迹的CUDA模仿预训练。横轴 epoch 为遍数，live 为本阶段经过秒数。'
                '\n\n留出数据按整局种子划分；动作一致率和CE不能代替原作避弹成绩。'
                '\n\n此阶段暂时训练GRU和动作头，之后仍需完整策略优化与新种子原作测试。'
                '\n\n当前步骤：'+str(phase),seconds)
        self.state['actor_warmup']=dict(status=status,phase=phase,source=str(folder),
            epoch=progress.get('epoch'),target_epochs=progress.get('target_epochs'),
            paid_BC_optimizer_steps=progress.get('BC_paid_optimizer_steps'),
            preparation_completed_rows=completed,preparation_total_rows=total,
            selected_epoch=result.get('selected_epoch',report.get('best_epoch')),
            updated_utc=progress.get('updated_utc'),native_efficacy_not_claimed=True)

    def actor_path_bc(self, folder):
        """Read only new-schema JSON; never inspect checkpoint/model tensors."""
        folder = Path(folder)
        try:
            terminals = [folder / name for name in ('failure.json', 'result.json') if (folder / name).is_file()]
            if len(terminals) > 1:
                raise ValueError('Conflicting producer failure.json and result.json; terminal status is unknown')
            terminal = terminals[0] if terminals else None
            progress = read(folder / 'progress.json') if (folder / 'progress.json').is_file() else {}
            row = read(terminal) if terminal is not None else progress
            actor_path_bc_mapping(progress, 'progress')
            actor_path_bc_mapping(row, 'result/progress')
            recorded = self.state.get('actor_path_bc_terminal')
            if recorded is not None and terminal is None:
                raise ValueError('Recorded producer terminal disappeared; fixed output directory cannot be reused')
            if not row:
                if terminal is not None:
                    raise ValueError('Producer terminal JSON is empty; terminal status is unknown')
                self.state['actor_path_bc'] = dict(status='waiting_for_source', source=str(folder), offline_labels=True)
                return
            for data in (progress, row):
                if data and (data.get('schema') not in (ACTOR_PATH_BC_SCHEMA, ACTOR_PATH_BC_LIFECYCLE_SCHEMA)
                             or type(data.get('actual_original_worlds_started')) is not int
                             or data['actual_original_worlds_started'] != 0
                             or data.get('native_performance') is True
                             or data.get('numerical_qualification_only') is True):
                    raise ValueError('Require new-schema offline actor-path BC, not native/PPO/qualification scores')
            if progress and progress['schema'] != row['schema']:
                raise ValueError('Actor-path BC progress/result schemas must identify the same source version')
            last_progress_seconds = progress.get('wall_seconds')
            if (isinstance(last_progress_seconds, (int, float)) and not isinstance(last_progress_seconds, bool)
                    and math.isfinite(last_progress_seconds) and last_progress_seconds >= 0):
                # Text can replace an older b52 cursor's last phase at that
                # measured point; terminal numeric elapsed remains unknown.
                self.state['actor_path_bc_text_step'] = max(int(last_progress_seconds), self.state.get('actor_path_bc_text_step', 0))
            if row.get('reason') is not None and not isinstance(row['reason'], str):
                raise ValueError('Producer terminal reason must be text or unknown')
            if terminal is not None:
                allowed = (('stopped', 'failed') if terminal.name == 'failure.json' else
                           ('completed', 'completed_finite_BC_contrast',
                            'completed_finite_BC_contrast_lifecycle_v2', 'CPU_inspect_only'))
                if row.get('status') not in allowed:
                    raise ValueError('Producer terminal JSON does not declare a terminal status')
                identity = dict(source=str(folder.resolve()), file=terminal.name, schema=row['schema'],
                    sha256=hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest())
                if recorded is not None and recorded != identity:
                    raise ValueError('Published producer terminal changed; fixed output directory cannot be reused')
                self.state['actor_path_bc_terminal'] = identity
                # A later metric error must not leave the last running state active.
                self.state['actor_path_bc'] = dict(status=row['status'], phase=row['status'],
                    source=str(folder), reason=row.get('reason'), producer_file=terminal.name,
                    producer_terminal=True, arms=None, offline_labels_not_native_decisions=True)
        except (OSError, ValueError, KeyError) as problem:
            previous = self.state.get('actor_path_bc') or {}
            last_seconds = previous.get('elapsed_seconds')
            text_step = (int(last_seconds) if isinstance(last_seconds, (int, float))
                         and not isinstance(last_seconds, bool) and math.isfinite(last_seconds) and last_seconds >= 0 else 0)
            text_step = max(text_step, self.state.get('actor_path_bc_text_step', 0))
            self.state['actor_path_bc'] = dict(status='read_error', phase='unknown', producer_status='unknown',
                source=str(folder), reason=str(problem), arms=None, offline_labels_not_native_decisions=True)
            self.state.pop('actor_path_bc_phase_key', None)
            if self.state.get('actor_path_bc_read_error') != str(problem):
                self.text('about/actor_path_bc_phase', ACTOR_PATH_BC_SCOPE
                          + '\n\n读取异常；当前producer状态未知，不能沿用训练中。\n\n' + str(problem), text_step)
                self.state['actor_path_bc_read_error'] = str(problem)
                self.state['actor_path_bc_text_step'] = text_step
            raise
        lifecycle_v2 = row['schema'] == ACTOR_PATH_BC_LIFECYCLE_SCHEMA
        arms = row.get('arms', {})
        if not isinstance(arms, dict) or set(arms) - set(ACTOR_PATH_BC_ARMS):
            raise ValueError('Unknown actor-path BC arm')
        requested = row.get('requested_arms') if lifecycle_v2 else None
        if requested is not None and (not isinstance(requested, list) or not requested
                or any(arm not in ACTOR_PATH_BC_ARMS for arm in requested)
                or len(set(requested)) != len(requested) or set(arms) - set(requested)):
            raise ValueError('Lifecycle requested arms must be an explicit full/control subset')
        completion = row.get('verified_main_normal_completion', progress.get('verified_main_normal_completion')) if lifecycle_v2 else None
        if isinstance(completion, dict):
            if completion.get('main_process_exited') is not True:
                raise ValueError('Lifecycle normal-completion record must explicitly confirm main exit')
            completion = True
        elif completion is not None and type(completion) is not bool:
            raise ValueError('Lifecycle normal-completion flag must be known boolean or a confirmed record')
        # Terminal run_state.utc is creation time, not an exit/elapsed clock.
        timing = row if terminal is not None else progress
        seconds = actor_path_bc_number(timing.get('wall_seconds'), 'wall_seconds')
        elapsed_step = int(seconds) if seconds is not None else None
        waiting = timing.get('waiting_for_main', {}) or {}
        actor_path_bc_mapping(waiting, 'waiting_for_main')
        wait_seconds = actor_path_bc_number(timing.get('main_gate_wait_seconds', waiting.get('wait_seconds')), 'main_gate_wait_seconds')
        waiting_flag = waiting.get('waiting')
        if waiting_flag is not None and type(waiting_flag) is not bool:
            raise ValueError('Measured main gate waiting state must be boolean or unknown')
        values, texts, report_updates, arm_states = [], [], {}, {}
        if elapsed_step is not None:
            for key, value in (('main_gate_wait_seconds', wait_seconds),
                               ('waiting_for_main', int(waiting_flag) if waiting_flag is not None else None),
                               ('main_gate_wait_checks', actor_path_bc_number(waiting.get('wait_checks'), 'wait_checks', count=True))):
                if value is not None:
                    values.append(('actor_path_bc/elapsed/' + key, value, elapsed_step))
            if completion is not None:
                values.append(('actor_path_bc/elapsed/verified_main_normal_completion', int(completion), elapsed_step))
        count_keys = ('paid_BC_optimizer_steps', 'selected_BC_optimizer_steps', 'paid_BC_training_labels',
                      'pending_uncommitted_legal_labels', 'BC_optimizer_step_attempts',
                      'train_source_rows_started', 'train_source_rows_completed',
                      'baseline_heldout_source_rows_started', 'baseline_heldout_source_rows_completed',
                      'final_heldout_source_rows_started', 'final_heldout_source_rows_completed',
                      'train_rows_with_approximate_incoming_history')
        for arm, data in arms.items():
            actor_path_bc_mapping(data, 'arm')
            prefix = 'actor_path_bc/' + arm
            cost = dict(actor_path_bc_mapping(data.get('cost') or {}, 'cost'))
            selection_path = folder / arm / 'selection.json'
            selection = ((data.get('selection') or {}) if terminal is not None else
                         read(selection_path) if selection_path.is_file() else (data.get('selection') or {}))
            actor_path_bc_mapping(selection, 'selection')
            if selection:
                if selection.get('native_performance') is not False or selection.get('selected') not in ('initial', 'final'):
                    raise ValueError('BC selection is an offline initial/final choice')
                selection_cost = actor_path_bc_mapping(selection.get('cost') or {}, 'selection cost')
                if terminal is None:
                    cost.update(selection_cost)
            for key in ('native_worlds_started', 'PPO_optimizer_steps'):
                if key in cost and actor_path_bc_number(cost[key], key, count=True) != 0:
                    raise ValueError('Actor-path BC cost must not contain native/PPO work')
            counts = {key: actor_path_bc_number(cost.get(key), key, count=True) for key in count_keys}
            paid, selected = counts['paid_BC_optimizer_steps'], counts['selected_BC_optimizer_steps']
            if paid is not None and selected is not None and selected > paid:
                raise ValueError('Selected BC steps cannot exceed paid BC steps')
            explicit_models = [actor_path_bc_number(v, 'model_BC_optimizer_steps', count=True)
                               for v in (data.get('model_BC_optimizer_steps'), cost.get('model_BC_optimizer_steps'),
                                         selection.get('model_BC_optimizer_steps') if terminal is None else None) if v is not None]
            if len(set(explicit_models)) > 1:
                raise ValueError('Model BC step fields disagree')
            model_steps = explicit_models[0] if explicit_models else None
            model_source = 'explicit_JSON' if explicit_models else None
            if selection and selected is not None:
                if (selection['selected'] == 'initial' and selected != 0
                        or selection['selected'] == 'final' and paid is not None and selected != paid):
                    raise ValueError('Selected checkpoint choice and measured BC step cost disagree')
                if model_steps is not None and model_steps != selected:
                    raise ValueError('Model and completed selection steps disagree')
                model_steps, model_source = selected, 'completed_selection.cost.selected_BC_optimizer_steps'
            if paid is not None and model_steps is not None and model_steps > paid:
                raise ValueError('Model BC steps cannot exceed paid cost')
            counts['model_BC_optimizer_steps'] = model_steps
            for key, value in counts.items():
                if value is None:
                    continue
                if paid is not None:
                    values.append((prefix + '/cost/' + key, value, paid))
                if elapsed_step is not None:
                    values.append((prefix + '/elapsed/' + key, value, elapsed_step))
            for snapshot in ('initial', 'final'):
                path = folder / arm / (snapshot + '_heldout.json')
                if not path.is_file() or snapshot == 'final' and paid is None:
                    continue
                metrics = read(path)
                flattened = actor_path_bc_heldout_metrics(metrics)
                fingerprint = hashlib.sha256(json.dumps(metrics, sort_keys=True, allow_nan=False).encode()).hexdigest()
                key = 'actor_path_bc_snapshot:' + str(path)
                previous = self.state['reports'].get(key)
                if previous is not None and previous != fingerprint:
                    raise ValueError('Published complete heldout snapshot changed; do not draw a batch-validation curve')
                if previous is None:
                    step = 0 if snapshot == 'initial' else paid
                    values.extend((prefix + '/heldout_snapshots/' + snapshot + '/' + tag, value, step)
                                  for tag, value in flattened)
                    texts.append((prefix + '/heldout_snapshots/' + snapshot + '/scope',
                                  '完整6局冻结自历史的' + snapshot + '离线分类快照；横轴paid Adam。'
                                  '不是逐batch留出验证或原作成绩。\n\n来源：`' + str(path) + '`', step))
                    report_updates[key] = fingerprint
            arm_states[arm] = dict(status=data.get('status'), **counts, model_steps_source=model_source,
                                   offline_labels_not_native_decisions=True)
            partials = data.get('partial_checkpoints') if lifecycle_v2 else None
            if partials is not None:
                if not isinstance(partials, list) or any(not isinstance(item, dict) for item in partials):
                    raise ValueError('Lifecycle partial checkpoints must be committed JSON records or unknown')
                arm_states[arm]['committed_partial_checkpoint_count'] = len(partials)
                if elapsed_step is not None:
                    values.append((prefix + '/elapsed/committed_partial_checkpoint_count', len(partials), elapsed_step))
        evaluation = timing.get('evaluation_progress') or {}
        actor_path_bc_mapping(evaluation, 'evaluation_progress')
        current_arm = row.get('current_arm')
        if current_arm is not None and current_arm not in ACTOR_PATH_BC_ARMS:
            raise ValueError('Unknown current actor-path BC arm')
        if requested is not None and current_arm is not None and current_arm not in requested:
            raise ValueError('Current lifecycle arm was not requested')
        if evaluation and current_arm in arms:
            role = evaluation.get('role')
            if role not in ('baseline_heldout', 'final_heldout') or evaluation.get('full_own_history_from_row0') is not True:
                raise ValueError('Heldout replay progress is not a complete metric snapshot')
            completed = actor_path_bc_number(evaluation.get('source_rows_completed'), 'evaluation source rows', count=True)
            total = actor_path_bc_number(evaluation.get('source_rows_total'), 'evaluation total rows', count=True)
            if completed is not None and total is not None and completed > total:
                raise ValueError('Heldout replay rows exceed the declared original population')
            if elapsed_step is not None:
                p = 'actor_path_bc/' + current_arm + '/elapsed/evaluation_progress/' + role
                for key, value in (('source_rows_completed', completed), ('source_rows_total', total)):
                    if value is not None:
                        values.append((p + '/' + key, value, elapsed_step))
        phase = row['status'] if terminal is not None and row['status'] in ('stopped', 'failed') else row.get('stage', row.get('status', 'unknown'))
        if any(v is not None and not isinstance(v, str) for v in (phase, row.get('status'), waiting.get('reason'))):
            raise ValueError('Actor-path BC phase/status/wait reason must be text or unknown')
        phase_key = (row.get('status'), phase, current_arm, waiting_flag, waiting.get('reason'),
                     terminal.name if terminal is not None else None, row.get('reason'))
        phase_text_step = max(elapsed_step or 0, self.state.get('actor_path_bc_text_step', 0))
        if self.state.get('actor_path_bc_phase_key') != list(phase_key):
            texts.append(('about/actor_path_bc_phase', ACTOR_PATH_BC_SCOPE + '\n\n当前阶段：' + str(phase)
                          + '；arm：' + str(current_arm) + '；状态：' + str(row.get('status'))
                          + '；主线门：' + str(waiting.get('reason', 'unknown'))
                          + '；producer文件：' + (terminal.name if terminal is not None else 'progress.json')
                          + '；终局原因：' + str(row.get('reason', 'unknown'))
                          + ('；终局elapsed未知，文本挂在最后已读时间点。' if terminal is not None and seconds is None else ''),
                          phase_text_step))
        # Validate all JSON before emitting any new snapshot/cost values.
        for tag, value, step in values:
            self.scalar(tag, value, step)
        for tag, value, step in texts:
            self.text(tag, value, step)
        self.state['reports'].update(report_updates)
        self.state['actor_path_bc_phase_key'] = list(phase_key)
        self.state['actor_path_bc_text_step'] = phase_text_step
        self.state.pop('actor_path_bc_read_error', None)
        self.state['actor_path_bc'] = dict(status=row.get('status'), phase=phase, source=str(folder),
            producer_stage=row.get('stage'), reason=row.get('reason'),
            producer_file=terminal.name if terminal is not None else 'progress.json', producer_terminal=terminal is not None,
            current_arm=current_arm, updated_utc=timing.get('updated_utc'),
            elapsed_seconds=seconds, main_gate_wait_seconds=wait_seconds, waiting_for_main=waiting_flag,
            arms=arm_states, offline_labels_not_native_decisions=True,
            heldout_only_initial_and_final_complete_snapshots=True, native_performance=False)
        if lifecycle_v2:
            if requested is not None:
                self.state['actor_path_bc']['requested_arms'] = list(requested)
            if completion is not None:
                self.state['actor_path_bc']['verified_main_normal_completion'] = completion

    def actor_development(self,folder):
        """Only a complete20+20 report publishes aggregate development scores."""
        folder=Path(folder);path=folder/'movement40/progress.json'
        if not path.is_file():return
        identity=read(folder/'launch.json')
        if identity.get('optimizer_updates')!=0:raise ValueError('BC development is frozen-model inference')
        progress=read(path);done=progress['completed'];target=progress['target'];counts=progress.get('by_policy',{})
        names=('trained_greedy','initial_greedy')
        if (target!=40 or type(done) is not int or not 0<=done<=40 or set(counts)!=set(names)
                or any(type(counts[name]) is not int or not 0<=counts[name]<=20 for name in names)
                or sum(counts.values())!=done):
            raise ValueError('BC movement development must contain only the declared20+20 games')
        step=progress['decisions']
        failed=(folder/'failure.json').is_file() or (folder/'movement40/failure.json').is_file()
        result=read(folder/'report.json') if (folder/'report.json').is_file() else {}
        completed=not failed and result.get('status')=='completed'
        if completed:
            if (result.get('schema')!='B1-full-natural-teacher-BC-original-movement-development-v1'
                    or result.get('native_games')!=40 or result.get('episodes_per_policy')!=20
                    or result.get('native_seed_split')!='development' or result.get('formal_test_seeds_used') is not False
                    or result.get('neural_optimizer_updates')!=0):
                raise ValueError('BC development cannot be relabeled as training or formal test')
            if any(result['policies'][name]['episodes']!=20 for name in names):
                raise ValueError('Wait for all20 games before reporting a BC development mean')
            terminal=read(folder/'movement40/report.json')
            if terminal.get('completed')!=40:raise ValueError('Sealed complete development rows are missing')
            for name in names:
                metrics=result['policies'][name]
                self.scalar('bc_dev/final/'+name+'/mean_capped_frames',metrics['mean_capped_frames'],20)
                self.scalar('bc_dev/final/'+name+'/censored_fraction',metrics['right_censored']/20,20)
            paired=result['paired'];ci=paired['descriptive_paired_bootstrap95']
            self.scalar('bc_dev/final/paired_frame_gain',paired['mean_capped_frame_difference'],20)
            self.scalar('bc_dev/final/paired_lower95',ci[0],20)
            self.scalar('bc_dev/final/paired_upper95',ci[1],20)
            step=max(step,terminal['decisions']);done=40;counts={name:20 for name in names}
        self.scalar('bc_dev/runtime/completed',done,step)
        self.scalar('bc_dev/runtime/target',target,step)
        for name in names:self.scalar('bc_dev/runtime/'+name+'_completed',counts[name],step)
        seconds=progress.get('wall_seconds',0)
        if seconds>0:self.scalar('bc_dev/runtime/decisions_per_second',step/seconds,step)
        status='failed' if failed else 'completed' if completed else 'running'
        if self.state.get('BC_development',{}).get('status')!=status:
            self.text('about/BC_development',
                '冻结预训练前后的B1，在相同20个开发种子上进行禁Bomb首次碰撞诊断。'
                '\n\n完整20局后才发布均值；32768帧触顶表示截尾。此处不是六面通关或正式验收。'
                '\n\n当前状态：'+status,step)
        self.state['BC_development']=dict(status=status,source=str(folder),completed=done,target=40,
            by_policy=counts,latest_source_utc=result.get('utc') if completed else progress.get('utc'),formal_test=False,optimizer_updates=0)

    def full_story_reference(self, item):
        path = Path(item['report'])
        if not path.exists():
            return
        report = read(path)
        if report.get('policy_frozen') is not True or report.get('continues') != 0:
            raise ValueError('Reference must be a frozen no-continue full-game report')
        prefix = 'reference/frozen_parent/' + item['name']
        for key in ('episodes', 'clears', 'clear_rate', 'mean_native_frames',
                    'mean_max_stage', 'mean_misses', 'mean_bombs_used'):
            self.scalar(prefix + '/' + key, report.get(key), 0)
        identity = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        if self.state['reports'].get(str(path)) != identity:
            self.text(prefix + '/scope', '继承模型的已完成冻结对照；不是本轮新训练的验收结果。\n\n来源：`' + str(path) + '`', 0)
            self.state['reports'][str(path)] = identity

    def tick(self, validation=None, reference_reports=()):
        trial = self.source / 'trial_000_both'
        failure_path = self.source / 'failure.json'
        training_failed = failure_path.exists()
        self.updates(trial)
        self.episodes(trial)
        progress = self.live(trial)
        initial = trial / 'initial_avoidance_development'
        if not initial.is_dir():
            initial = self.source / 'initial_avoidance_development'
        initial_progress = initial / 'progress.json'
        if initial_progress.exists():
            baseline = read(initial_progress)
            done = baseline.get('completed', 0)
            self.scalar('eval/runtime/initial_avoidance/completed_games', done, done)
            self.scalar('eval/runtime/initial_avoidance/target_games', baseline.get('target'), done)
            if progress is None:
                if not training_failed and self.state.get('phase') != 'initial_avoidance_development':
                    self.text('about/current_phase', '本轮训练前禁 Bomb 基线；尚未开始 PPO 更新。', done)
                if not training_failed:
                    self.state['phase'] = 'initial_avoidance_development'
                self.state['latest_source_utc'] = read(self.source / 'study_status.json').get('updated_utc')
                self.state['latest_optimized_decisions'] = 0
        self.diagnostic_report(initial / 'report.json', 0, 'eval/development')
        for path in sorted(trial.glob('development_*/report.json')):
            try:
                step = int(path.parent.name.split('_')[-1])
            except ValueError:
                continue
            self.diagnostic_report(path, step, 'eval/development')
        self.frozen_validation(validation, self.state.get('latest_optimized_decisions', 0))
        for item in reference_reports:
            self.full_story_reference(item)
        self.state['trainer_status'] = 'failed' if training_failed else self.state.get('phase')
        self.scalar('live/training_failed', int(training_failed), self.state.get('latest_sampled_decisions') or 0)
        if training_failed and self.state.get('phase') != 'failed':
            failure = read(failure_path)
            self.text('about/current_phase', '训练已退出；日志服务仍在运行。\n\n来源：`' + str(failure_path)
                      + '`\n\n```\n' + str(failure.get('traceback', failure))[-6000:] + '\n```',
                      self.state.get('latest_sampled_decisions') or 0)
            self.state['phase'] = 'failed'
        self.writer.flush()
        write(self.cursor_path, self.state)
        return progress

    def close(self):
        self.writer.flush()
        self.writer.close()


def resolve_sources(config):
    if config.get('training_source'):
        source = Path(config['training_source']).resolve()
        if config.get('follow_current_training'):
            phase = read(config['goal_state']).get('current_native_phase', {})
            if phase.get('output'):
                current = Path(phase['output']).resolve()
                if ((current / 'trial_000_both').is_dir()
                        or (current / 'effective_study_config.json').is_file()):
                    source = current
        if not source.is_dir() or not (source / 'study_config.json').is_file():
            raise FileNotFoundError('Pinned current study is not initialized')
        # An explicitly pinned study must never attach another run's tests.
        validation=config.get('validation_source') if source==Path(config['training_source']).resolve() else None
        validation=config.get('validation_sources',{}).get(str(source),validation)
        return source, validation
    goal = read(config['goal_state'])
    phase = goal.get('current_native_phase', {})
    candidates = [phase.get('output'), goal.get('output'), config['fallback_training']]
    source = next((Path(x) for x in candidates if x and (Path(x) / 'trial_000_both').is_dir()), None)
    if source is None:
        raise FileNotFoundError('No actual training directory found')
    queue = goal.get('following_frozen_comparison', {})
    validation = queue.get('output', config.get('fallback_validation'))
    return source.resolve(), validation


def acquire_mutex():
    if os.name != 'nt':
        return None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel.CreateMutexW(None, False, 'Local\\fly-touhou-tensorboard-live-v1')
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        kernel.CloseHandle(ctypes.c_void_p(handle))
        return False
    return handle


def fly_policy_probability_metrics(teacher, student, legal):
    """Same-trajectory behavior KL and an in-window fitted constant reference."""
    import numpy as np
    p = np.asarray(teacher, dtype=np.float64).reshape(-1, 19)
    mask = np.asarray(legal, dtype=bool).reshape(-1, 19)
    if not mask[:, :18].all():
        return {}  # The analytical reference is specific to inventory-only Bomb masking.
    p = np.where(mask, p, 0.)
    p /= p.sum(axis=1, keepdims=True)
    available = mask[:, 18]
    bomb = p[available, 18].mean() if available.any() else 0.
    movement = p[:, :18].sum(axis=0)
    movement /= movement.sum()
    constant = np.zeros_like(p)
    constant[:, :18] = movement[None, :] * np.where(available, 1. - bomb, 1.)[:, None]
    constant[available, 18] = bomb
    def kl(q):
        return float(np.sum(p * (np.log(np.maximum(p, 1e-30)) -
                                 np.log(np.maximum(q, 1e-30))), axis=1).mean())
    result = {'best_same_window_constant_KL': kl(constant)}
    if student is not None:
        q = np.asarray(student, dtype=np.float64).reshape(-1, 19)
        q = np.where(mask, q, 0.)
        q /= q.sum(axis=1, keepdims=True)
        result.update(teacher_behavior_KL=kl(q),
                      KL_minus_same_window_constant=kl(q) - result['best_same_window_constant_KL'],
                      greedy_teacher_agreement=float((q.argmax(1) == p.argmax(1)).mean()),
                      student_probability_variance_over_rollout=float(q.var(axis=0).mean()))
    return result


def fly_term_mean(terms, key):
    """Mean-valued losses use real sample counts; old row-less logs keep mean."""
    selected = [term for term in terms if isinstance(term.get(key), (int, float))]
    if not selected:
        return None
    weights = [term.get('real_rows') for term in selected]
    if (key not in ('start', 'group', 'real_rows', 'gradient_loss_weight')
            and all(type(w) in (int, float) and math.isfinite(w) and w > 0 for w in weights)):
        return sum(term[key] * weight for term, weight in zip(selected, weights)) / sum(weights)
    return sum(term[key] for term in selected) / len(selected)


def _fly_population_stats(weighted):
    """Pool JSON population summaries without reading neural NPZ or models."""
    measured = [(stats, weight) for stats, weight in weighted
                if isinstance(stats, dict) and weight > 0
                and all(isinstance(stats.get(k), (int, float)) and math.isfinite(stats[k])
                        for k in ('mean', 'std', 'min', 'max'))]
    if not measured:
        return {}
    total = sum(weight for _, weight in measured)
    mean = sum(stats['mean'] * weight for stats, weight in measured) / total
    second = sum((stats['std']**2 + stats['mean']**2) * weight for stats, weight in measured) / total
    return dict(mean=mean, std=math.sqrt(max(0., second - mean**2)),
                min=min(stats['min'] for stats, _ in measured), max=max(stats['max'] for stats, _ in measured))


def export_fly_neural_observability(exporter, label, root, prefix):
    """Independent rollout axis/cursor; diagnostic work never adds PPO costs.

    Reads only aggregate report.json. A snapshot's count window PRECEDES its
    input; newly reset lanes with no previous window remain missing, not zero.
    """
    cursor = exporter.state.setdefault('fly_neural_observability_cursors', {}).setdefault(label, dict(last_rollout=-1))
    paths = [path for path in (root / 'neural-diagnostics').glob('rollout-*/report.json')
             if path.parent.name.rsplit('-', 1)[-1].isdigit()]
    for path in sorted(paths, key=lambda p: int(p.parent.name.rsplit('-', 1)[-1])):
        number = int(path.parent.name.rsplit('-', 1)[-1])
        if number <= cursor['last_rollout']:
            continue
        try:
            report = read(path)
        except (OSError, ValueError) as error:
            cursor['read_error'] = str(error)
            break  # Do not let a later report advance past an unread earlier one.
        tags = prefix + 'neural_observability/'
        emit = lambda name, value: exporter.scalar(tags + name, value, number)
        emit('flags/completed_read_only', report.get('status') == 'completed_read_only')
        activity = report.get('activity', {})
        if activity.get('schema') == 'fly-visual-neural-observability-v1':
            emit('activity/available', activity.get('status') in ('available', 'snapshot_available'))
            replay = activity.get('replay', {})
            if replay:
                emit('activity/replay_verified', replay.get('status') == 'replay_verified'
                     and replay.get('actual_trace_claimed') is True)
                for key in ('NPZ_bytes', 'sparse_uncompressed_bytes'):
                    emit('overhead/activity/replay/' + key, replay.get(key))
            for key in ('elapsed_seconds', 'CPU_seconds', 'extra_model_calls', 'extra_neural_lane_forwards',
                        'extra_neural_microstep_lane_updates', 'extra_state_initializations', 'summary_json_bytes'):
                emit('overhead/activity/' + key, activity.get(key))
            snapshot = activity.get('snapshot', {})
            batches = snapshot.get('batches', [])
            if batches and activity.get('status') in ('available', 'snapshot_available'):
                emit('snapshot/count_window_ms', snapshot.get('count_window_ms'))
                emit('snapshot/preceding_window_may_use_previous_policy_theta',
                     snapshot.get('preceding_window_may_use_previous_policy_theta'))
                emit('snapshot/real_rows', sum(b.get('real_lanes', 0) for b in batches))
                emit('snapshot/previous_window_available_rows',
                     sum(b.get('previous_20ms_window_available_rows', 0) for b in batches))
                emit('overhead/activity/CPU_snapshot_payload_bytes', snapshot.get('CPU_snapshot_payload_bytes'))
                for name in ('VP_input_ports', 'DN_output_ports', 'remaining_intermediate'):
                    groups = [(b, b.get('populations', {}).get(name, {})) for b in batches]
                    base = 'populations/' + name + '/'
                    for field, previous in (('membrane_mv', False), ('preceding_window_firing_hz', True),
                                            ('gain_multiplier', False)):
                        summaries = []
                        for batch, population in groups:
                            lanes = batch.get('previous_20ms_window_available_rows' if previous else 'real_lanes', 0)
                            summaries.append((population.get(field), lanes * population.get('neurons', 0)))
                        for statistic, value in _fly_population_stats(summaries).items():
                            emit(base + field + '/' + statistic, value)
                    for field, previous in (('near_threshold_fraction', False), ('preceding_window_active_fraction', True)):
                        weighted = [(population.get(field),
                                     batch.get('previous_20ms_window_available_rows' if previous else 'real_lanes', 0)
                                     * population.get('neurons', 0)) for batch, population in groups]
                        measured = [(value, weight) for value, weight in weighted
                                    if isinstance(value, (int, float)) and math.isfinite(value) and weight > 0]
                        if measured:
                            emit(base + field, sum(v*w for v, w in measured) / sum(w for _, w in measured))
                dn = [(b.get('DN_current_EMA_rate_hz'), b.get('real_lanes', 0)
                       * b.get('populations', {}).get('DN_output_ports', {}).get('neurons', 0)) for b in batches]
                for statistic, value in _fly_population_stats(dn).items():
                    emit('DN_current_EMA_rate_hz/' + statistic, value)
        horizon = report.get('credit_horizon', {})
        if horizon.get('schema') == 'fly-visual-readonly-credit-horizon-T4-T16-v1':
            if type(horizon.get('verified')) is bool:
                emit('credit_horizon/verified', horizon['verified'])
            for parameter, measures in horizon.get('parameter_gradients', {}).items():
                for key in ('T4_norm', 'T16_norm', 'difference_norm', 'difference_max_abs',
                            'cosine', 'relative_difference_vs_T4'):
                    emit('credit_horizon/gradients/' + parameter + '/' + key, measures.get(key))
            pair = horizon.get('pair', {})
            for key in ('loss_terms_max_abs_error', 'raw_max_abs_error', 'logp_max_abs_error'):
                emit('credit_horizon/pair/' + key, pair.get(key))
            for key in ('exact', 'verified'):
                emit('credit_horizon/pair/following_state_' + key, pair.get('following_state', {}).get(key))
            for key in ('elapsed_seconds', 'extra_neural_lane_forwards', 'extra_backwards_completed'):
                emit('overhead/credit_horizon/' + key, horizon.get(key))
            for arm in ('T4', 'T16'):
                for key in ('base_neural_lane_forwards', 'checkpoint_recompute_neural_lane_forwards',
                            'backward_attempts', 'backwards_completed'):
                    emit('overhead/credit_horizon/' + arm + '/' + key, horizon.get('costs', {}).get(arm, {}).get(key))
        cursor.update(last_rollout=number, status=report.get('status'), source=str(path), read_error=None)


def export_fly_kickstarting(exporter, label, root, row):
    """Incremental JSON-only reader; never import the trainer or open its checkpoints."""
    prefix = 'fly_iterations/' + label + '/'
    cursors = exporter.state.setdefault('fly_online_cursors', {})
    cursor = cursors.setdefault(label, dict(last_rollout=0, episodes=0, recent=[]))
    for path in sorted((root / 'traces').glob('update-*.json'), key=lambda p: int(p.stem.rsplit('-', 1)[-1])):
        number = int(path.stem.split('-')[-1])
        if number <= cursor['last_rollout']:
            continue
        report = read(path)
        costs = report.get('costs', {})
        decisions = int(costs.get('student_actions_executed', 0))
        if not decisions:
            continue
        learning = report.get('learning', {})
        for key in ('completed_rollouts', 'student_actions_executed', 'Adam_steps',
                    'backward_passes', 'actor_replay_neural_lane_forwards',
                    'checkpoint_recompute_neural_lane_forwards'):
            exporter.scalar(prefix + 'counts/' + key, costs.get(key), decisions)
        exporter.scalar(prefix + 'time/elapsed_seconds', report.get('elapsed_seconds'), decisions)
        elapsed = report.get('elapsed_seconds', 0)
        if elapsed:
            exporter.scalar(prefix + 'time/student_decisions_per_second', decisions / elapsed, decisions)
        terms = learning.get('terms', [])
        if terms:
            for key in terms[0]:
                exporter.scalar(prefix + 'learning/' + key, fly_term_mean(terms, key), decisions)
        for key in ('teacher_coefficient', 'gamma', 'gae_lambda', 'epoch_count', 'tbptt',
                    'old_logp_replay_max_abs_error'):
            exporter.scalar(prefix + 'settings/' + key, learning.get(key), decisions)
        metadata = report.get('rollout_metadata', {})
        for source_key, tag in (('teacher_temperature', 'teacher_target_temperature'),
                                ('action_temperature', 'student_action_temperature')):
            if metadata.get(source_key) is not None:
                cursor[tag] = metadata[source_key]
                exporter.scalar(prefix + 'settings/' + tag, metadata[source_key], decisions)
        graph = learning.get('training_graph', {})
        if graph:
            cursor['training_graph_status'] = graph.get('status')
            exporter.scalar(prefix + 'settings/training_CUDA_graph_enabled',
                            str(graph.get('status', '')).startswith('enabled'), decisions)
        for key in ('training_graph_extra_neural_lane_forwards', 'training_graph_extra_backwards_completed'):
            exporter.scalar(prefix + 'counts/' + key, costs.get(key), decisions)
        observation = {}
        observation_path = root / 'traces' / ('rollout-%03d.json' % (number - 1))
        if observation_path.exists():
            observation = read(observation_path)
            components = [entry for step in observation.get('reward_components', []) for entry in step]
            eligible = sum(entry.get('vulnerable_progress', 0.) >= .0009999 for entry in components)
            cursor['full_vulnerable_reward_steps'] = cursor.get('full_vulnerable_reward_steps', 0) + eligible
            exporter.scalar(prefix + 'counts/full_vulnerable_reward_steps',
                            cursor['full_vulnerable_reward_steps'], decisions)
            if components:
                exporter.scalar(prefix + 'changing_policy_rollout/full_vulnerable_reward_fraction',
                                eligible / len(components), decisions)
        probability_path = observation_path.with_suffix('.npz')
        if probability_path.exists():
            import numpy as np
            with _shared_binary(probability_path) as stream, np.load(stream, allow_pickle=False) as archive:
                if {'teacher_probs', 'legal_actions'} <= set(archive.files):
                    metrics = fly_policy_probability_metrics(archive['teacher_probs'],
                        archive['action_probs'] if 'action_probs' in archive.files else None,
                        archive['legal_actions'])
                    for key, value in metrics.items():
                        exporter.scalar(prefix + 'behavior/' + key, value, decisions)
                    infos = [entry for step in observation.get('step_info', []) for entry in step]
                    if infos and len(infos) == archive['teacher_probs'].reshape(-1, 19).shape[0]:
                        known = np.asarray([entry.get('protected_decision') in (0, 1) for entry in infos])
                        vulnerable = np.asarray([entry.get('protected_decision') == 0 for entry in infos])
                        exporter.scalar(prefix + 'behavior/unprotected/known_protection_rows',
                                        int(known.sum()), decisions)
                        exporter.scalar(prefix + 'behavior/unprotected/rows', int(vulnerable.sum()), decisions)
                        if known.any():
                            exporter.scalar(prefix + 'behavior/protected_fraction',
                                            1. - vulnerable.sum() / known.sum(), decisions)
                        if vulnerable.any() and 'action_probs' in archive.files:
                            split_metrics = fly_policy_probability_metrics(
                                archive['teacher_probs'].reshape(-1, 19)[vulnerable],
                                archive['action_probs'].reshape(-1, 19)[vulnerable],
                                archive['legal_actions'].reshape(-1, 19)[vulnerable])
                            for key, value in split_metrics.items():
                                exporter.scalar(prefix + 'behavior/unprotected/' + key, value, decisions)
        gradients = learning.get('gradient_diagnostics', [])
        for parameter in ('recurrent_gain_logits', 'input_encoder.weight', 'input_encoder.bias',
                          'action_readout.weight'):
            values = [entry.get(parameter, {}).get('RMS') for entry in gradients]
            values = [value for value in values if isinstance(value, (float, int))]
            if values:
                exporter.scalar(prefix + 'gradient_RMS/' + parameter, sum(values) / len(values), decisions)
        for item in report.get('completed_native_episodes', []):
            game = item.get('summary', item)
            cursor['episodes'] += 1
            game_step = cursor['episodes']
            compact = {key: game.get(key) for key in ('native_frames', 'misses', 'bombs_used', 'stage')}
            compact['clear'] = int(game.get('outcome') == 'clear')
            compact['first_collision_frames'] = game.get('movement_diagnostics', {}).get('first_collision_observed_frames')
            rewards = game.get('reward_vector', [])
            compact['episode_return'] = sum(rewards) if rewards else None
            for key, value in compact.items():
                exporter.scalar(prefix + 'changing_policy_games/' + key, value, game_step)
            cursor['recent'].append(compact)
            cursor['recent'] = cursor['recent'][-20:]
        exporter.scalar(prefix + 'counts/completed_training_games', cursor['episodes'], decisions)
        for key in ('native_frames', 'misses', 'bombs_used', 'stage', 'clear',
                    'first_collision_frames', 'episode_return'):
            values = [game[key] for game in cursor['recent'] if isinstance(game.get(key), (float, int))]
            if values:
                exporter.scalar(prefix + 'changing_policy_recent20/' + key, sum(values) / len(values), decisions)
        cursor['last_rollout'] = number
    export_fly_neural_observability(exporter, label, root, prefix)
    evaluation = row.get('development', row.get('evaluation', {}))
    if isinstance(evaluation, dict) and evaluation:
        step = max(1, int(evaluation.get('newly_executed_decisions', evaluation.get('policy_decisions', 0))))
        for key in ('episodes', 'completed', 'clears', 'clear_rate', 'mean_native_frames',
                    'mean_max_stage', 'mean_misses', 'mean_bombs_used'):
            exporter.scalar(prefix + 'frozen_development/' + key, evaluation.get(key), step)
        hits = [game.get('movement_diagnostics', {}).get('first_collision_observed_frames')
                for game in evaluation.get('episode_results', [])]
        hits = [float(value) for value in hits if value is not None]
        if hits:
            exporter.scalar(prefix + 'frozen_development/first_observed_collision_mean_frames',
                            sum(hits) / len(hits), step)
    exporter.state.setdefault('fly_online_jobs', {})[label] = dict(
        source=str(root), phase=row.get('status', row.get('phase')),
        student_decisions=row.get('costs', {}).get('student_actions_executed'),
        Adam_steps=row.get('actual_Adam_steps', row.get('actual_new_Adam_steps', row.get('costs', {}).get('Adam_steps'))),
        completed_training_games=cursor['episodes'],
        teacher_target_temperature=cursor.get('teacher_target_temperature'),
        student_action_temperature=cursor.get('student_action_temperature'),
        training_graph_status=cursor.get('training_graph_status'),
        development_episodes=evaluation.get('episodes', evaluation.get('completed')) if evaluation else None,
        development_clears=evaluation.get('clears') if evaluation else None)


def export_fly_algorithm_iterations(exporter, sources):
    """Read lightweight algorithm reports while native/learning jobs run."""
    exporter.state['algorithm_sources'] = dict(sources)
    for label, folder in sources.items():
        root = Path(folder)
        if export_continuous_connectome(exporter, label, root, read_json=read):
            continue
        result = root / 'result.json'
        progress = root / 'progress.json'
        if not result.exists() and not progress.exists():
            continue
        row = read(result if result.exists() else progress)
        known=exporter.state.get('fly_online_jobs',{}).get(label,{})
        if (row.get('schema')=='native-compact129-rule-teacher-episode-v1'
                or row.get('phase')=='teacher_playing_original' or known.get('family')=='native_rule_teacher'):
            failure=root/'failure.json'
            if failure.exists():row=read(failure)
            n=int(row.get('decisions',row.get('rows',known.get('teacher_decisions',0))))
            summary=row.get('episode_summary',{})
            prefix='fly_iterations/'+label+'/rule_teacher_collection/'
            for key in ('stage','stage_frame','misses','bombs_used','seconds'):
                exporter.scalar(prefix+key,row.get(key,summary.get(key)),n)
            exporter.scalar(prefix+'teacher_decisions',n,n)
            exporter.state.setdefault('fly_online_jobs',{})[label]=dict(source=str(root),
                family='native_rule_teacher',phase='failed' if failure.exists() else row.get('phase',row.get('status')),
                producer_terminal=result.exists() or failure.exists(),teacher_controlled=True,
                teacher_decisions=n,PPO_updates=0,new_student_actions=0,
                stage=row.get('stage'),misses=row.get('misses',summary.get('misses')),
                bombs_used=row.get('bombs_used',summary.get('bombs_used')),outcome=summary.get('outcome'),
                CUDA_started=False,read_error=row.get('error'),
                scope='Rule-teacher data collection, not student learning or a fly-model clear.')
            continue
        if ('completed_rollouts' in row.get('costs', {})
                and any(k in row for k in ('actual_Adam_steps','actual_new_Adam_steps'))):
            export_fly_kickstarting(exporter, label, root, row)
            continue
        step = int(row.get('optimizer_updates', 0))
        prefix = 'fly_iterations/' + label + '/'
        if step:
            exporter.scalar(prefix + 'optimizer_updates', step, step)
            update = row.get('last_completed_update', {})
            current_loss = row.get('balanced_loss', row.get('loss'))
            if isinstance(current_loss, str):
                current_loss = update.get('balanced_KL')
            exporter.scalar(prefix + 'training/current_KL', current_loss, step)
            for field in ('old_mean_KL', 'new_mean_KL', 'balanced_KL'):
                exporter.scalar(prefix + 'training/window/' + field, update.get(field), step)
            for parameter, rms in row.get('last_gradient_RMS', {}).items():
                exporter.scalar(prefix + 'gradient_RMS/' + parameter, rms, step)
        for role, scores in row.get('metrics', {}).items():
            if isinstance(scores, dict):
                exporter.scalar(prefix + role + '/KL', scores.get('mean_T1_legal_KL'), max(1, step))
                exporter.scalar(prefix + role + '/action_agreement', scores.get('greedy_accuracy'), max(1, step))
        evaluation = row.get('evaluation', row.get('evaluation_progress', {}))
        if isinstance(evaluation, dict) and evaluation:
            decisions = int(evaluation.get('policy_decisions', evaluation.get('newly_executed_decisions', 0)))
            native_step = max(1, decisions)
            for key in ('completed', 'episodes', 'clears', 'mean_native_frames', 'mean_misses', 'mean_bombs_used'):
                exporter.scalar(prefix + 'native/' + key, evaluation.get(key), native_step)
            exporter.scalar(prefix + 'native/decisions', decisions, native_step)
            games = evaluation.get('episode_results', [])
            hits = [game.get('movement_diagnostics', {}).get('first_collision_observed_frames') for game in games]
            hits = [float(value) for value in hits if value is not None]
            if hits:
                exporter.scalar(prefix + 'native/first_observed_collision_mean_frames', sum(hits)/len(hits), native_step)
        exporter.state['algorithm_sources'] = dict(sources)


def main(args):
    config = read(args.config)
    control = Path(config['control_directory'])
    logdir = Path(config['log_directory'])
    control.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    lock = acquire_mutex()
    if lock is False:
        if args.open_browser:
            import webbrowser
            webbrowser.open('http://127.0.0.1:' + str(config.get('port', 6006)) + '/')
        print('Existing TensorBoard bridge already owns this workspace; no duplicate started.')
        return
    exporter = None
    teacher_exporter = None
    actor_exporter = None
    development_exporter = None
    actor_path_exporter = None
    migration_exporters = {}
    algorithm_exporter = None
    url = None
    try:
        if not args.once:
            from tensorboard import program
            board = program.TensorBoard()
            board.configure(argv=['tensorboard', '--logdir=' + str(logdir), '--host=127.0.0.1',
                '--port=' + str(config.get('port', 6006)), '--reload_interval=5', '--load_fast=false'])
            url = board.launch()
            if args.open_browser:
                import webbrowser
                webbrowser.open(url)
        while True:
            config = read(args.config)
            migration_only, migration_paths = active_migration_sources(config)
            if migration_only:
                source, validation = None, None
                for old in (exporter, teacher_exporter, actor_exporter, development_exporter, actor_path_exporter):
                    if old is not None:
                        old.close()
                exporter = teacher_exporter = actor_exporter = development_exporter = actor_path_exporter = None
            else:
                source, validation = resolve_sources(config)
                if exporter is None or exporter.source != source:
                    if exporter is not None:
                        exporter.close()
                    exporter = RunExport(source, logdir, control)
            teacher_source=config.get('teacher_source')
            if teacher_source and (teacher_exporter is None or teacher_exporter.source!=Path(teacher_source).resolve()):
                if teacher_exporter is not None:teacher_exporter.close()
                teacher_exporter=RunExport(teacher_source,logdir,control)
                teacher_exporter.text('about/scope',
                    '# 完整教师数据\n\n此运行只记录规则教师在原作中的示范，不更新神经网络。'
                    '\n\n通关、Game Over与采样数均属于教师数据；神经模型成绩显示在独立测试组。',1)
            actor_source=config.get('actor_warmup_source')
            if actor_source and Path(actor_source).is_dir() and (
                    actor_exporter is None or actor_exporter.source!=Path(actor_source).resolve()):
                if actor_exporter is not None:actor_exporter.close()
                actor_exporter=RunExport(actor_source,logdir,control)
                actor_exporter.text('about/scope',
                    '# 完整轨迹模仿预训练\n\n独立显示BC损失、整局留出验证、实际和选中优化步数。'
                    '\n\n本阶段不采样原作；模仿质量不等于实际避弹或通关成绩。',1)
            development_source=config.get('actor_development_source')
            if development_source and Path(development_source).is_dir() and (
                    development_exporter is None or development_exporter.source!=Path(development_source).resolve()):
                if development_exporter is not None:development_exporter.close()
                development_exporter=RunExport(development_source,logdir,control)
                development_exporter.text('about/scope',
                    '# B1原作避弹开发\n\n新旧模型各20局；禁Bomb；首次碰撞或32768帧截止。'
                    '\n\n使用开发种子，不更新模型；正式测试在其他运行中显示。',1)
            actor_path_source = config.get('actor_path_bc_source')
            actor_path_path = Path(actor_path_source).resolve() if actor_path_source else None
            if actor_path_exporter is not None and actor_path_exporter.source != actor_path_path:
                actor_path_exporter.close()
                actor_path_exporter = None
            if actor_path_path is not None and actor_path_path.is_dir() and actor_path_exporter is None:
                actor_path_exporter = RunExport(actor_path_path, logdir, control, actor_path_bc=True)
            for role, path in migration_paths.items():
                current = migration_exporters.get(role)
                if current is not None and current.source != path:
                    current.close()
                    migration_exporters.pop(role)
                if path is not None and path.is_dir() and role not in migration_exporters:
                    migration_exporters[role] = RunExport(path, logdir, control, migration_kind=role)
            error = None
            try:
                if exporter is not None:
                    exporter.tick(validation, config.get('reference_reports', ()))
                if teacher_source and teacher_exporter is not None:
                    teacher_exporter.teacher_collection(teacher_source)
                    teacher_exporter.hardware(teacher_exporter.state.get('teacher_collection',{}).get('acknowledged_native_actions',0))
                    teacher_exporter.writer.flush()
                    write(teacher_exporter.cursor_path,teacher_exporter.state)
                if actor_source and actor_exporter is not None and actor_exporter.source==Path(actor_source).resolve():
                    actor_exporter.actor_warmup(actor_source)
                    actor_exporter.hardware(int(time.time()))
                    actor_exporter.writer.flush()
                    write(actor_exporter.cursor_path,actor_exporter.state)
                if development_source and development_exporter is not None and development_exporter.source==Path(development_source).resolve():
                    development_exporter.actor_development(development_source)
                    development_exporter.hardware(int(time.time()))
                    development_exporter.writer.flush()
                    write(development_exporter.cursor_path,development_exporter.state)
                if actor_path_source and actor_path_exporter is not None:
                    actor_path_exporter.actor_path_bc(actor_path_path)
                    actor_path_exporter.writer.flush()
                    write(actor_path_exporter.cursor_path, actor_path_exporter.state)
            except (OSError, ValueError, KeyError) as problem:
                # A transient read/report publication never changes training.
                error = repr(problem)
            for role, current in migration_exporters.items():
                try:
                    current.migration(role, migration_paths[role])
                    current.writer.flush()
                    write(current.cursor_path, current.state)
                except (OSError, ValueError, KeyError) as problem:
                    error = ((error + '; ') if error else '') + role + ': ' + repr(problem)
            algorithm_sources = config.get('fly_algorithm_sources', {})
            if algorithm_sources:
                try:
                    if algorithm_exporter is None:
                        algorithm_exporter = RunExport(config['fly_algorithm_source_directory'], logdir, control)
                        algorithm_exporter.text('about/algorithm_scope',
                            'Fruitfly transfer: offline imitation and student-controlled on-policy learning. '
                            'Learning curves, changing-policy training games and frozen development games are separate.', 1)
                    algorithm_exporter.compact_story_dodge_settings=config.get('compact_story_dodge_diagnostics',{})
                    export_fly_algorithm_iterations(algorithm_exporter, algorithm_sources)
                    algorithm_exporter.writer.flush()
                    write(algorithm_exporter.cursor_path, algorithm_exporter.state)
                except (OSError, ValueError, KeyError, TypeError) as problem:
                    error = ((error + '; ') if error else '') + 'fly_algorithms: ' + repr(problem)
            old_state = exporter.state if exporter is not None else {}
            state = dict(schema='fly-touhou-live-tensorboard-v1', status='running' if not error else 'read_retry',
                utc=utc_now(), pid=os.getpid(), url=url, training_source=str(source) if source is not None else None, validation_source=validation,
                log_directory=str(logdir), latest_source_utc=old_state.get('latest_source_utc'),
                sampled_decisions=old_state.get('latest_sampled_decisions'),
                optimized_decisions=old_state.get('latest_optimized_decisions'),
                max_live_stage=old_state.get('max_live_stage'), scalar_events_this_process=(
                    (exporter.new_scalars if exporter else 0) +
                    (algorithm_exporter.new_scalars if algorithm_exporter else 0)),
                trainer_status=old_state.get('trainer_status'),
                validation_runtime=old_state.get('validation_runtime', {}),
                teacher_collection=teacher_exporter.state.get('teacher_collection') if teacher_exporter else None,
                actor_warmup=actor_exporter.state.get('actor_warmup') if actor_exporter else None,
                BC_development=development_exporter.state.get('BC_development') if development_exporter else None,
                trainer_modified=False, native_instances_started=0, GPU_used=False, last_read_error=error)
            if actor_path_source:
                state['actor_path_bc'] = (actor_path_exporter.state.get('actor_path_bc') if actor_path_exporter
                                         else dict(status='waiting_for_source', source=str(actor_path_path)))
            if migration_only or any(migration_paths.values()):
                state['active_migration_only'] = migration_only
                state['migration_parser_patch'] = MIGRATION_TB_PATCH
                state['active_jobs'] = {role: (migration_exporters[role].state.get(role) if role in migration_exporters
                    else dict(status='prepared_not_started', phase='not_started', source=str(path)))
                    for role, path in migration_paths.items() if path is not None}
                if algorithm_exporter is not None:
                    state['online_learning_jobs'] = current_algorithm_jobs(algorithm_exporter.state, algorithm_sources)
            write(control / 'status.json', state)
            if args.once:
                if error:
                    raise RuntimeError(error)
                print(json.dumps(state, ensure_ascii=False))
                break
            if (control / 'STOP').exists():
                write(control / 'status.json', {**state, 'status': 'stopped', 'utc': utc_now()})
                break
            time.sleep(max(2, float(config.get('refresh_seconds', 5))))
    except BaseException:
        write(control / 'failure.json', dict(utc=utc_now(), traceback=traceback.format_exc(), trainer_modified=False))
        raise
    finally:
        if exporter is not None:
            exporter.close()
        if teacher_exporter is not None:
            teacher_exporter.close()
        if actor_exporter is not None:
            actor_exporter.close()
        if development_exporter is not None:
            development_exporter.close()
        if actor_path_exporter is not None:
            actor_path_exporter.close()
        for current in migration_exporters.values():
            current.close()
        if algorithm_exporter is not None:
            algorithm_exporter.close()
        if lock and os.name == 'nt':
            ctypes.WinDLL('kernel32', use_last_error=True).CloseHandle(ctypes.c_void_p(lock))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--open-browser', action='store_true', help='Only the user-clicked shortcut opens a browser')
    main(parser.parse_args())
