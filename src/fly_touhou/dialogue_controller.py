"""Process-local native dialogue input edges; gameplay actions remain unmodified.

TH14's MsgVM arms held-Z fast-forward only after a fresh Z/Ctrl edge. The
campaign action space holds SHOT continuously, so release SHOT for one requested
input tick on entry to a native dialogue object, then deliver the requested mask.
No extra game ticks, desktop input, movement, Bomb or resource writes are added.
"""
from __future__ import annotations

from dataclasses import replace

from .frida_controller import OriginalGameController
from .resource_environment import ResourceGameController


class DialogueEdgeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dialogue_object = 0
        self._dialogue_release_pending = False
        self.dialogue_input_events = []

    def _dialogue_identity(self, snapshot):
        if not snapshot.dialogue:
            return 0
        gui = self.reader.pointer('gui', 'gui_pointer')
        return self._memory.integer(gui + self.profile.offsets['gui']['zGui_dialogue']) if gui else 0

    def read(self):
        snapshot = super().read()
        return replace(snapshot, extra={**snapshot.extra,
            'native_dialogue_object': self._dialogue_identity(snapshot),
            'dialogue_input_automation': {
                'kind': 'fresh-native-Z-edge-on-dialogue-entry',
                'release_events': len(self.dialogue_input_events),
                'extra_game_ticks': 0, 'desktop_input': False}})

    def advance(self, mask, ticks):
        # Validate before considering the split, as OriginalGameController does.
        if not 1 <= int(ticks) <= 36000 or not 0 <= int(mask) <= 0xffffff:
            raise ValueError('Input mask or requested ticks out of bounds')
        before = self.read()
        active = bool(before.screen == 7 and before.pause == 2 and before.dialogue)
        identity = before.extra['native_dialogue_object']
        if not active:
            self._dialogue_object = 0
            self._dialogue_release_pending = False
        elif identity != self._dialogue_object:
            self._dialogue_object = identity
            self._dialogue_release_pending = True
        if active and self._dialogue_release_pending and int(mask) & 1:
            self._dialogue_release_pending = False
            snapshot = super().advance(int(mask) & ~1, 1)
            if int(ticks) > 1:
                snapshot = super().advance(int(mask), int(ticks) - 1)
            self.dialogue_input_events.append({
                'frame': before.frame, 'stage': before.stage,
                'stage_frame': before.stage_frame, 'native_dialogue': True,
                'native_dialogue_object': identity,
                'requested_mask': int(mask), 'requested_ticks': int(ticks),
                'release_mask': int(mask) & ~1, 'release_ticks': 1,
                'repress_in_same_call': int(ticks) > 1,
                'elapsed_frames': snapshot.frame - before.frame,
                'movement_and_bomb_bits_unchanged': True,
                'extra_game_ticks': 0})
            return self.read()
        return super().advance(mask, ticks)


class DialogueGameController(DialogueEdgeMixin, OriginalGameController):
    pass


class DialogueResourceGameController(DialogueEdgeMixin, ResourceGameController):
    pass


def label_owned_game_windows(controller, title):
    """Queue bounded title messages without blocking the game's input driver.

    The native main thread waits for Python at every hook checkpoint. A
    synchronous SetWindowText here deadlocks that pair, so a daemon worker sends
    timeout-bounded messages while the policy continues its ordinary steps.
    """
    import ctypes
    from ctypes import wintypes
    import threading
    import time
    import win32gui
    import win32process
    audit = controller.audit_windows()
    windows = [w for w in audit['windows'] if w['visible']]
    status = {'pid': controller.pid, 'requested_title': str(title), 'status': 'queued',
              'changed': [], 'errors': [], 'focus_or_desktop_input_requested': False}
    controller.window_label_status = status

    def worker():
        send = ctypes.WinDLL('user32', use_last_error=True).SendMessageTimeoutW
        send.argtypes = [wintypes.HWND, wintypes.UINT, ctypes.c_size_t, ctypes.c_ssize_t,
                         wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
        send.restype = ctypes.c_ssize_t
        title_buffer = ctypes.create_unicode_buffer(str(title))
        for window in windows:
            hwnd = window['hwnd']
            try:
                for _ in range(16):
                    if not win32gui.IsWindow(hwnd) or win32process.GetWindowThreadProcessId(hwnd)[1] != controller.pid:
                        break
                    result = ctypes.c_size_t()
                    delivered = send(hwnd, 0x000C, 0, ctypes.cast(title_buffer, ctypes.c_void_p).value,
                                     0x0001 | 0x0002, 250, ctypes.byref(result))
                    if delivered:
                        status['changed'].append({'hwnd': hwnd, 'title': str(title)})
                        break
                    time.sleep(0.01)
            except Exception as error:
                status['errors'].append(str(error))
        status['status'] = 'delivered' if status['changed'] else 'timed_out_or_closed'
        try:
            controller.audit_windows()
        except Exception:
            pass

    threading.Thread(target=worker, name='owned-game-window-label', daemon=True).start()
    return dict(status)
