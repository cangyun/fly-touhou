"""Opt-in single-tick menu checkpoints for the stage12 short task.

The original controller's normal menu inputs are unchanged, but the return to
live play is inspected after every requested hook tick instead of after a whole
8/16-tick batch. No game state or resource is written. Alignment to the declared
task start remains the environment's responsibility.
"""
from __future__ import annotations

from collections import deque
from types import MethodType

from .geometry_environment import (GeometryHazardDialogueResourceGameController,
                                   GeometryHazardResourceGameController,
                                   GeometryObservationCampaignEnv)
from .native import NativeError
from .native_performance import (EarlyHiddenStartupMixin, OwnedLifecycleAuditMixin,
                                 owned_exit_status)


def raw_stage_timer(controller):
    """Read diagnostic bytes at the existing timer address without mutation."""
    try:
        reader = controller.reader
        thread = reader.pointer('game_thread', 'game_thread_pointer')
        offset = controller.profile.offsets['game_thread']['zGameThread_stage_timer']
        raw = controller._memory.read(thread + offset, 16) if thread else b''
        return {'game_thread_pointer': thread, 'timer_offset': offset,
                'timer_bytes_hex': raw.hex(),
                'timer_first_u32': int.from_bytes(raw[:4], 'little') if raw else None}
    except Exception as error:
        return {'available': False, 'error': repr(error)}


class FineStartControllerMixin:
    """Observe native menu/retry transitions at each tick, only when opted in."""
    def __init__(self, *args, **kwargs):
        state = getattr(self, '_startup_retry_state', None)
        self._startup_retry_state = state if state and state['active'] else None
        self._fine_constructor_args = args
        self._fine_constructor_kwargs = dict(kwargs)
        self.fine_start_trace = deque(maxlen=64)
        self.fine_start_ticks = 0
        self.fine_start_operation = None
        super().__init__(*args, **kwargs)

    def startup_recovery_audit(self):
        state = getattr(self, '_startup_retry_state', None)
        return ({'max_attempts': 3, 'attempts': list(state['attempts']),
                 'reached_playing': state['reached_playing'], 'active': state['active'],
                 'successful_identity': state.get('successful_identity')}
                if state else {'max_attempts': 3, 'attempts': [], 'reached_playing': False, 'active': False})

    def _startup_owned_identity(self):
        # THRLVector's existing spawn observer already guards this exact child.
        identity = getattr(getattr(self, '_device', None), 'identity', None)
        if identity and identity.get('pid') == self.pid and identity.get('identity_complete'):
            return dict(identity)
        captured = getattr(self, '_spawn_identity', None)
        if captured and captured.get('pid') == self.pid and captured.get('identity_complete'):
            return dict(captured)
        from .thrl_vector import _identity
        memory = getattr(self, '_memory', None)
        return _identity(self.pid, owned_handle=getattr(memory, 'handle', None))

    def _startup_cleanup(self, identity):
        from .thrl_vector import _identity, _same_identity, _terminate_native
        self.close()  # existing exact sessions / guarded kill / owned Job close
        if not identity or not identity.get('identity_complete'):
            raise NativeError('Startup cleanup lacks a recorded exact native identity; no retry')
        current = _identity(identity['pid'])
        action = None
        if _same_identity(identity, current):
            action = _terminate_native(identity)
            current = _identity(identity['pid'])
        gone = (current.get('identity_complete') and not _same_identity(identity, current)) or (
            not current.get('identity_complete') and 'NoSuchProcess' in current.get('error', ''))
        if not gone:
            raise NativeError('Startup exact native cleanup could not be confirmed; no retry')
        return {'identity': identity, 'current_identity': current, 'exact_exit_confirmed': True,
                'fallback_cleanup': action}

    def _startup_retryable(self, error):
        event = getattr(error, 'native_event', {})
        if event.get('pid') != getattr(self, 'pid', None):
            return False
        return ((event.get('kind') == 'native-process-detached' and event.get('reason') == 'process-terminated')
                or (event.get('kind') == 'native-checkpoint-timeout' and event.get('expected') == 'checkpoint'))

    def _startup_guard(self, operation, callback, difficulty=None):
        state = self._startup_retry_state
        if state is None:
            state = self._startup_retry_state = {'active': True, 'busy': False,
                'reached_playing': False, 'attempts': []}
        if state.get('busy') or not state['active']:
            return callback()
        state['busy'] = True
        need_start = False
        try:
            while True:
                try:
                    if need_start:
                        self.start()  # nested guard shares this same three-attempt budget
                    snapshot = callback()
                    if self._playing(snapshot):
                        state['reached_playing'], state['active'] = True, False
                        state['successful_identity'] = self._startup_owned_identity() if getattr(self, 'pid', None) else None
                    return snapshot
                except NativeError as error:
                    if state['reached_playing'] or not self._startup_retryable(error):
                        raise
                    record = {'attempt': len(state['attempts'])+1, 'operation': operation,
                        'error': repr(error), 'native_event': dict(error.native_event),
                        'native_identity': self._startup_owned_identity(),
                        'native_detach_events': list(getattr(self, 'native_detach_events', ())),
                        'parallel_startup_events': list(getattr(self, 'parallel_startup_events', ())),
                        'requested_menu_ticks': self.fine_start_ticks,
                        'last_checkpoints': list(self.fine_start_trace)}
                    state['attempts'].append(record)
                    try:
                        record['cleanup'] = self._startup_cleanup(record['native_identity'])
                    except BaseException as cleanup_error:
                        record['cleanup_error'] = repr(cleanup_error)
                        raise
                    if len(state['attempts']) >= 3:
                        error.startup_recovery = self.startup_recovery_audit()
                        raise
                    # Use the same concrete class and original constructor args;
                    # existing event-reader/mutex/performance mixins are retained.
                    args, kwargs = self._fine_constructor_args, dict(self._fine_constructor_kwargs)
                    self.__init__(*args, **kwargs)
                    if operation == 'start':
                        need_start = False
                    else:
                        need_start = True
                        callback = lambda: self._ensure_run_once(difficulty)
        finally:
            state['busy'] = False

    def start(self):
        return self._startup_guard('start', lambda: super(FineStartControllerMixin, self).start())

    def _fine_begin(self, operation):
        self.fine_start_trace.clear()
        self.fine_start_ticks = 0
        self.fine_start_operation = operation
        self._fine_record(self.read(), mask=None)

    def _fine_record(self, snapshot, mask):
        self.fine_start_trace.append({
            'tick': self.fine_start_ticks, 'mask': mask,
            'frame': snapshot.frame, 'stage_frame': snapshot.stage_frame,
            'stage': snapshot.stage, 'screen': snapshot.screen,
            'pause': snapshot.pause, 'has_player': snapshot.player is not None,
            'menu_page': snapshot.extra.get('menu_page'),
            'menu_substate': snapshot.extra.get('menu_substate'),
            'menu_selection': snapshot.extra.get('menu_selection')})

    def fine_start_audit(self):
        detach_events = list(getattr(self, 'native_detach_events', ()))
        exit_status = {'source': 'current-owned-handle', **owned_exit_status(self)}
        if not exit_status.get('available'):
            # reset() closes the owned memory handle before its outer failure
            # logger runs. Preserve the status captured at the native detach.
            for event in reversed(detach_events):
                captured = event.get('owned_exit_status', {})
                if captured.get('available'):
                    exit_status = {'source': 'native-detach-before-handle-close', **captured}
                    break
        return {'kind': 'opt-in-native-menu-single-tick-checkpoints-v1',
                'pid': getattr(self, 'pid', None),
                'operation': self.fine_start_operation,
                'requested_menu_ticks': self.fine_start_ticks,
                'last_checkpoints': list(self.fine_start_trace),
                'raw_stage_timer': raw_stage_timer(self),
                'native_detach_events': detach_events,
                'startup_visibility_events': list(getattr(self, 'startup_visibility_events', ())),
                'startup_recovery': self.startup_recovery_audit(),
                'owned_exit_status': exit_status,
                'desktop_input': False, 'gameplay_or_resource_writes': False}

    def _fine_segment(self, mask, ticks, stop):
        snapshot = self.read()
        for _ in range(ticks):
            snapshot = self.advance(mask, 1)
            self.fine_start_ticks += 1
            self._fine_record(snapshot, mask)
            if stop(snapshot):
                return snapshot, True
        return snapshot, False

    def _fine_finish(self, snapshot, difficulty):
        state = self._startup_retry_state
        if state and self._playing(snapshot):
            state['reached_playing'], state['active'] = True, False
        if difficulty is not None and snapshot.difficulty != difficulty:
            raise NativeError(f'Original menu selected difficulty {snapshot.difficulty}, expected {difficulty}')
        self.audit_windows()
        return self.read()

    def ensure_run(self, difficulty=None):
        return self._startup_guard('menu', lambda: self._ensure_run_once(difficulty), difficulty)

    def _ensure_run_once(self, difficulty=None):
        if difficulty is not None and difficulty not in (0, 1, 2, 3):
            raise ValueError('Normal mode difficulty must be Easy0/Normal1/Hard2/Lunatic3')
        self._target_difficulty = difficulty
        self._fine_begin('initial-menu')
        snapshot = self.read()
        for _ in range(180):
            if self._playing(snapshot):
                return self._fine_finish(snapshot, difficulty)
            if difficulty is not None and snapshot.extra.get('menu_page') == 5:
                if snapshot.extra.get('menu_substate') != self.hooks.get('menu_ready_substate', 2):
                    segments = ((0, 4),)
                else:
                    selection = snapshot.extra.get('menu_selection')
                    if selection not in range(4):
                        raise NativeError(f'Unexpected native difficulty selection: {selection}')
                    segments = (((0x20 if selection < difficulty else 0x10, 4), (0, 4))
                                if selection != difficulty else ((1, 4), (0, 4), (0, 8)))
            else:
                segments = ((1, 4), (0, 4), (0, 8))
            for mask, ticks in segments:
                snapshot, ready = self._fine_segment(mask, ticks, self._playing)
                if ready:
                    return self._fine_finish(snapshot, difficulty)
        raise NativeError(f'Cannot enter original game through single-tick menu input: '
                          f'screen={snapshot.screen}, pause={snapshot.pause}, '
                          f'stage_frame={snapshot.stage_frame}')

    def restart(self):
        self._startup_retry_state = None  # a completed-episode reset boundary
        return self._startup_guard('restart', self._restart_once, self._target_difficulty)

    def _restart_once(self):
        self._fine_begin('native-retry')
        before = self.read()
        difficulty = self._target_difficulty
        if before.screen == 7 and before.pause in (0, 2) and before.lives >= 0:
            if before.pause == 2:
                self._fine_segment(0x100, 4, lambda _: False)
                self._fine_segment(0, 4, lambda _: False)
            ready = False
            for _ in range(32):
                menu = self._memory.integer(self._memory.base + self.hooks['retry_pointer_rva'])
                if menu and self._memory.integer(menu + self.hooks['retry_state_offset']) == self.hooks['retry_ready_state']:
                    ready = True
                    break
                self._fine_segment(0, 1, lambda _: False)
            if ready:
                # A lower stage timer distinguishes a new run from accidentally
                # dismissing the pause menu and returning to the previous run.
                new_run = lambda s: (self._playing(s) and s.stage == 1
                                     and s.stage_frame < before.stage_frame)
                for mask, ticks in ((self.hooks['retry_mask'], 4), (0, 4), (0, 128)):
                    snapshot, restarted = self._fine_segment(mask, ticks, new_run)
                    if restarted:
                        self._milestones.clear()
                        return self._fine_finish(snapshot, difficulty)
        # Preserve the original explicit fresh-process fallback. The mixin is
        # part of the concrete class, so reinitializing cannot lose fine startup.
        self.close()
        args, kwargs = self._fine_constructor_args, dict(self._fine_constructor_kwargs)
        self.__init__(*args, **kwargs)
        self.start()
        return self.ensure_run(difficulty=difficulty)


class FineStartGeometryController(FineStartControllerMixin, GeometryHazardResourceGameController):
    pass


class FineStartDialogueGeometryController(FineStartControllerMixin, GeometryHazardDialogueResourceGameController):
    pass


class HiddenFineStartGeometryController(EarlyHiddenStartupMixin,
                                       OwnedLifecycleAuditMixin, FineStartGeometryController):
    """Explicit owned-window startup audit and native detach diagnostics."""


class HiddenFineStartDialogueGeometryController(EarlyHiddenStartupMixin,
                                               OwnedLifecycleAuditMixin,
                                               FineStartDialogueGeometryController):
    """The same fine input sequence with visibility checks at menu ticks."""


def _create_fine_controller(env):
    if env.hide_window:
        kind = (HiddenFineStartDialogueGeometryController if env.dialogue_skip
                else HiddenFineStartGeometryController)
    else:
        kind = FineStartDialogueGeometryController if env.dialogue_skip else FineStartGeometryController
    return kind(env.executable, env.profile_path, timeout=env.timeout,
                private_appdata=env.private_appdata,
                worker_name=env.window_title or env.private_appdata.name)


def enable_fine_start(env):
    """Install before lazy reset; legacy environments keep their own factory."""
    if not isinstance(env, GeometryObservationCampaignEnv) or env.controller is not None:
        raise ValueError('Fine startup requires an unstarted geometry observation environment')
    env._create_controller = MethodType(_create_fine_controller, env)
    return env
