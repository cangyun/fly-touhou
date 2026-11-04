"""Hide only windows belonging to an explicit owned native game controller."""
import ctypes
from ctypes import wintypes
import time


def hide_owned_windows(controller):
    import win32process
    if not controller.pid or controller._job is None:
        raise ValueError('Explicit owned game process required')
    hide = ctypes.WinDLL('user32', use_last_error=True).ShowWindowAsync
    hide.argtypes, hide.restype = [wintypes.HWND, ctypes.c_int], wintypes.BOOL
    audit = controller.audit_windows()
    requested = []
    for window in audit['windows']:
        hwnd = window['hwnd']
        if win32process.GetWindowThreadProcessId(hwnd)[1] != controller.pid:
            raise RuntimeError('Owned window identity changed')
        if window['visible']:
            hide(hwnd, 0)
            requested.append(hwnd)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        audit = controller.audit_windows()
        if audit['all_hidden']:
            return {'requested': requested, 'audit': audit, 'desktop_input': False,
                    'rendering_or_frame_timing_changed': False}
        time.sleep(.01)
    raise RuntimeError('Owned native window did not become hidden')
