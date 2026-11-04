"""Read RNG through the existing owned controller at its input checkpoints.

Default: validate files only. No RNG Frida session or initialization hook exists
in this entry. --execute-owned explicitly starts a fresh, bounded owned process.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fly_touhou.native_rng_observer import (PINNED_SHA256, IMAGE_BASE,
    pe_code_bytes, sha256_file, validate_profile)

PROTOCOL_SHA256 = '1300d3738e3d7febe017c0d80c5d0177ef6748c7e015a0424111c86c8254a6bc'
RNG_PROFILE_SHA256 = 'bfe0eeab53547ff2c433132ea2161b90e3f25d76c45fdb0d47c7c2380d71cef1'
NATIVE_PROFILE_SHA256 = '659b55a03ff77ad39980c9776e42c4b713116a26e403d4301d35828dedd89c8d'
RNG_NAMES = ('visual_seed', 'visual_counter', 'gameplay_seed', 'gameplay_counter')
RAW_NAMES = ('raw_input', 'raw_previous_input', 'raw_repeat_input',
             'raw_pressed_input', 'raw_released_input')


def checkpoint_layout(profile, pristine):
    fields = profile['state_fields']
    start = fields['visual_seed']['rva']
    expected = ((0, 'u16'), (4, 'u32'), (8, 'u16'), (12, 'u32'))
    for name, (offset, kind) in zip(RNG_NAMES, expected):
        if fields[name] != {'rva': start + offset, 'kind': kind}:
            raise ValueError('RNG block layout differs: ' + name)
    # The pinned A3 imm32 instruction stores EAX to this native timestamp.
    # Derive its RVA from validated pristine code, not a guessed profile key.
    code = pe_code_bytes(pristine, profile['init_before']['rva'], 5)
    if code[0] != 0xA3:
        raise ValueError('Expected the validated timestamp-copy instruction')
    timestamp_rva = struct.unpack_from('<I', code, 1)[0] - profile['image_base']
    if not 0 < timestamp_rva < 0x100000:
        raise ValueError('Timestamp RVA outside the pinned data range')
    raw_start = fields['raw_input']['rva']
    for index, name in enumerate(RAW_NAMES):
        if fields[name] != {'rva': raw_start + 4 * index, 'kind': 'u32'}:
            raise ValueError('Raw input block layout differs: ' + name)
    return {'rng_rva': start, 'rng_size': 16, 'rng_format': '<H2xIH2xI',
            'timestamp_rva': timestamp_rva, 'timestamp_kind': 'u32',
            'timestamp_address_source': 'pinned pristine init_before A3 immediate',
            'raw_input_rva': raw_start, 'raw_input_size': 20}


def decode_rng_block(data):
    if len(data) != 16:
        raise ValueError('RNG block must contain exactly 16 bytes')
    return dict(zip(RNG_NAMES, struct.unpack('<H2xIH2xI', data)))


def validate_inputs(protocol_path, rng_path, static_path):
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('Protocol SHA differs from the reviewed protocol')
    if sha256_file(rng_path) != RNG_PROFILE_SHA256:
        raise ValueError('RNG profile SHA differs from the reviewed profile')
    validation = validate_profile(rng_path, static_path)
    protocol = json.loads(Path(protocol_path).read_text(encoding='utf-8'))
    profile = json.loads(Path(rng_path).read_text(encoding='utf-8'))
    if protocol['game_exe_sha256'] != PINNED_SHA256:
        raise ValueError('Protocol executable identity differs')
    native_path = Path(protocol['profile']).resolve(strict=True)
    if sha256_file(native_path) != NATIVE_PROFILE_SHA256:
        raise ValueError('Native controller profile SHA differs from the reviewed profile')
    native = json.loads(native_path.read_text(encoding='utf-8'))
    if PINNED_SHA256 not in native['sha256']:
        raise ValueError('Native controller profile has a different identity')
    if sha256_file(protocol['game_exe']) != PINNED_SHA256:
        raise ValueError('Actual executable SHA differs')
    layout = checkpoint_layout(profile, Path(static_path).read_bytes())
    for native_key, rng_key in (('gameplay_rng_rva', 'gameplay_seed'),
            ('gameplay_rng_counter_rva', 'gameplay_counter'),
            ('visual_rng_rva', 'visual_seed'), ('visual_rng_counter_rva', 'visual_counter')):
        if native_key in native['hooks'] and native['hooks'][native_key] != profile['state_fields'][rng_key]['rva']:
            raise ValueError('Native/RNG profile field disagreement: ' + native_key)
    return protocol, profile, {'static_validation': validation, 'layout': layout,
        'protocol_sha256': PROTOCOL_SHA256, 'rng_profile_sha256': RNG_PROFILE_SHA256,
        'native_profile': str(native_path), 'native_profile_sha256': sha256_file(native_path),
        'actual_executable_sha256': PINNED_SHA256}


def checkpoint_sample(controller, profile, layout, label, snapshot=None):
    checkpoint = getattr(controller, '_checkpoint', None)
    memory = getattr(controller, '_memory', None)
    if checkpoint is None or memory is None or memory.pid != controller.pid:
        raise RuntimeError('Read requires this owned controller\'s stopped input checkpoint')
    ready = getattr(controller, '_hook_ready', None) or {}
    if ready.get('arch') != 'ia32' or Path(ready.get('module', '')).resolve() != controller.executable.resolve():
        raise RuntimeError('Existing hook readiness did not identify the owned executable')
    base = memory.base  # Existing ProcessMemory Toolhelp module enumeration.
    if not isinstance(base, int) or not 0x10000 <= base < 0x100000000:
        raise RuntimeError('Existing module enumeration returned an invalid base')
    if 'base' in ready and int(str(ready['base']), 0) != base:
        raise RuntimeError('Existing hook and ProcessMemory module bases disagree')
    if base + max(layout['rng_rva'] + 16, layout['timestamp_rva'] + 4) > 0x100000000:
        raise RuntimeError('RNG read exceeds the target address range')
    observed_at = time.monotonic()
    raw = memory.read(base + layout['rng_rva'], 16)
    rng = decode_rng_block(raw)
    rng['initialization_timestamp'] = struct.unpack('<I', memory.read(base + layout['timestamp_rva'], 4))[0]
    input_raw = memory.read(base + layout['raw_input_rva'], 20)
    snapshot = snapshot if snapshot is not None else controller.read()
    pointers = {}
    for name, item in profile['pointer_fields'].items():
        pointer = memory.integer(base + item['pointer_rva'])
        pointers[name] = {'pointer': pointer, 'value': memory.integer(pointer + item['offset']) if pointer else None}
    return {'label': label, 'pid': controller.pid, 'checkpoint': dict(checkpoint),
        'host_monotonic_seconds': observed_at, 'module_base': base,
        'module_base_source': 'existing ProcessMemory._module_base Toolhelp enumeration',
        'hook_ready': ready, 'rng': rng, 'rng_block_hex': raw.hex(),
        'rng_block_single_read': True, 'timestamp_separate_read': True,
        'raw_input': dict(zip(RAW_NAMES, struct.unpack('<5I', input_raw))),
        'raw_input_block_hex': input_raw.hex(), 'pointer_context': pointers,
        'screen': snapshot.screen, 'pause': snapshot.pause, 'frame': snapshot.frame,
        'stage_frame': snapshot.stage_frame, 'stage': snapshot.stage,
        'difficulty': snapshot.difficulty, 'character': snapshot.character,
        'subshot': snapshot.subshot, 'has_player': snapshot.player is not None,
        'initial_copy_before': 'unknown; no initialization hook',
        'initial_copy_after': 'unknown; no initialization hook',
        'init_before_hits': None, 'init_after_hits': None}


class ProbeStop(RuntimeError):
    pass


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--protocol', type=Path, required=True)
    result.add_argument('--rng-profile', type=Path, required=True)
    result.add_argument('--static-copy', type=Path, required=True)
    result.add_argument('--output', type=Path)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--execute-owned', action='store_true')
    result.add_argument('--attempts', type=int, choices=(1, 2), default=2)
    result.add_argument('--decisions-per-attempt', type=int, choices=range(1, 17), default=16)
    result.add_argument('--wall-cap', type=float, default=120.)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not math.isfinite(args.wall_cap) or not 1 <= args.wall_cap <= 120:
        raise ValueError('Wall cap must be finite and within 1..120 seconds')
    protocol, profile, validation = validate_inputs(args.protocol, args.rng_profile, args.static_copy)
    report = {'schema': 'native-rng-existing-checkpoints-v1', 'status': 'validated_not_executed',
        'target_executed': False, 'GPU_used': False, 'seed_written': False,
        'added_rng_sessions': 0, 'added_rng_interceptors': 0, 'validation': validation,
        'configuration': {'attempts': args.attempts, 'decisions_per_attempt': args.decisions_per_attempt,
            'ticks_per_decision': 4, 'mask': 9, 'difficulty': 3, 'single_wait_timeout_seconds': 5.,
            'wall_cap_between_operations_seconds': args.wall_cap, 'strict_watchdog': False,
            'desktop_input': False, 'resource_override': False, 'model': None},
        'initial_copy_before_after': 'unknown', 'counter_reset_semantics': 'unknown',
        'runtime_validation_completed': False, 'samples': [], 'operations': [], 'lifecycle': [],
        'task_decisions_requested': 0, 'task_decisions_completed': 0,
        'task_input_ticks_requested': 0, 'task_input_ticks_acknowledged': 0,
        'menu_input_ticks_requested': 0, 'menu_input_ticks_acknowledged': 0,
        'limitations': ['Checkpoint reads cannot observe the two initialization copy sites.',
            'Input-poll ticks are not necessarily identical to logic-frame deltas.',
            'Counter reset and native retry semantics need real observations.',
            'No controlled-seed, trajectory-repeatability, speed or stability result.']}
    if not args.execute_owned:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        raise ValueError('Explicit owned execution requires an exclusive --output')
    output = args.output.resolve()
    private = output / 'game_userdata'
    if not str(private).isascii() or len(str(private)) > 210:
        raise ValueError('Private APPDATA must be ASCII and shorter than 211 characters')
    output.mkdir(parents=True, exist_ok=False)
    from fly_touhou.atomic_json_io import atomic_write_json
    started = time.monotonic()
    controller = None
    context = {'phase': 'startup', 'task': False, 'decision': None}
    report['status'] = 'starting_explicit_owned_probe'

    def persist(event=None):
        if event is not None:
            with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        atomic_write_json(output / 'report.json', report)

    def budget():
        if time.monotonic() - started >= args.wall_cap or (output / 'STOP').exists():
            raise ProbeStop('Wall cap/STOP observed between existing controller operations')

    def capture(label, snapshot=None):
        sample = checkpoint_sample(controller, profile, validation['layout'], label, snapshot)
        report['samples'].append(sample)
        persist({'type': 'sample', **sample})
        return sample

    def lifecycle(label):
        if controller is None:
            return
        evidence = {'label': label, 'pid': controller.pid,
            'last_checkpoint': dict(controller._checkpoint) if controller._checkpoint else None,
            'fine_start_audit': controller.fine_start_audit(),
            'parallel_startup': controller.parallel_startup_audit()}
        report['lifecycle'].append(evidence)
        persist({'type': 'lifecycle', **evidence})

    persist()
    try:
        from fly_touhou.fixed_start_controller import HiddenFineStartDialogueGeometryController
        controller = HiddenFineStartDialogueGeometryController(protocol['game_exe'], protocol['profile'],
            private_appdata=private, worker_name='m1b-rng-checkpoints', timeout=5.)
        original_start, original_advance, original_close = controller.start, controller.advance, controller.close

        def observed_start():
            budget()
            snapshot = original_start()
            report['target_executed'] = True
            capture('first-start-checkpoint', snapshot)  # Persist before ensure_run.
            return snapshot

        def observed_advance(mask, ticks):
            budget()
            op = {'phase': context['phase'], 'task': context['task'], 'decision': context['decision'],
                'pid': controller.pid, 'mask': mask, 'requested_input_ticks': ticks,
                'actual_input_ticks': None, 'status': 'requested',
                'actual_native_logic_ticks': None,
                'checkpoint_before': dict(controller._checkpoint) if controller._checkpoint else None}
            before = report['samples'][-1] if report['samples'] else None
            report['operations'].append(op)
            key = 'task' if context['task'] else 'menu'
            report[key + '_input_ticks_requested'] += ticks
            persist({'type': 'advance-request', **op})
            snapshot = original_advance(mask, ticks)
            op.update(status='returned', actual_input_ticks=ticks,
                actual_ticks_source='existing synchronous controller completion contract; not logic frames',
                checkpoint_after=dict(controller._checkpoint) if controller._checkpoint else None,
                frame=snapshot.frame, stage_frame=snapshot.stage_frame, screen=snapshot.screen)
            if before and before['pid'] == controller.pid:
                op['global_frame_delta'] = snapshot.frame - before['frame']
                op['stage_frame_delta'] = snapshot.stage_frame - before['stage_frame']
            report[key + '_input_ticks_acknowledged'] += ticks
            capture(context['phase'] + '-checkpoint', snapshot)
            return snapshot

        def observed_close():
            try:
                lifecycle('before-owned-close')
            except BaseException as error:
                report.setdefault('lifecycle_errors', []).append({'phase': 'before-close', 'error': repr(error)})
            try:
                return original_close()
            finally:
                try:
                    lifecycle('after-owned-close')
                except BaseException as error:
                    report.setdefault('lifecycle_errors', []).append({'phase': 'after-close', 'error': repr(error)})

        # Instance-only host observers. Existing class/hook sources and methods
        # remain unchanged; fallback __init__ retains these instance attributes.
        controller.start, controller.advance, controller.close = observed_start, observed_advance, observed_close
        controller.start()
        context['phase'] = 'initial-menu'
        controller.ensure_run(3)
        capture('after-ensure-run')
        lifecycle('after-ensure-run')
        for attempt in range(args.attempts):
            budget()
            if attempt:
                context.update(phase='native-retry', task=False, decision=None)
                capture('before-retry')
                lifecycle('before-retry')
                previous_pid = controller.pid
                controller.restart()
                capture('after-retry')
                report['retry_process_identity'] = {'before_pid': previous_pid, 'after_pid': controller.pid,
                    'same_process': previous_pid == controller.pid,
                    'kind': 'same-process-native-retry' if previous_pid == controller.pid else 'fresh-process-fallback'}
                lifecycle('after-retry')
            context.update(phase=f'attempt-{attempt + 1}', task=True)
            capture(context['phase'] + '-start')
            for decision in range(args.decisions_per_attempt):
                context['decision'] = decision + 1
                report['task_decisions_requested'] += 1
                snapshot = controller.advance(9, 4)
                report['task_decisions_completed'] += 1
                persist()
                if snapshot.screen != 7:
                    raise ProbeStop('Native scene exited before all requested task decisions')
            context.update(task=False, decision=None)
        report['status'] = 'bounded_observation_finished'
        report['runtime_validation_completed'] = True
    except ProbeStop as error:
        report.update(status='bounded_stop', error=repr(error), traceback=traceback.format_exc())
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    finally:
        if controller is not None:
            report['target_executed'] = bool(report['target_executed'] or controller.pid is not None)
            report['last_owned_pid'] = controller.pid
            try:
                lifecycle('finally-before-close')
                if controller._checkpoint is not None and controller._memory is not None and not controller._closed:
                    capture('finally-last-checkpoint')
            except BaseException as error:
                report['final_capture_error'] = {'error': repr(error), 'traceback': traceback.format_exc()}
            try:
                controller.close()
            except BaseException as error:
                report['cleanup_error'] = {'error': repr(error), 'traceback': traceback.format_exc()}
                report['status'] = 'failed'
            report['native_detach_events'] = list(controller.native_detach_events)
            report['startup_visibility_events'] = list(controller.startup_visibility_events)
        report['wall_seconds'] = time.monotonic() - started
        persist()
    print(json.dumps({'status': report['status'], 'task_decisions_completed': report['task_decisions_completed'],
        'counter_reset_semantics': 'unknown; audit recorded checkpoints'}, ensure_ascii=False))
    return 0 if report['status'] == 'bounded_observation_finished' else 2


if __name__ == '__main__':
    raise SystemExit(main())
