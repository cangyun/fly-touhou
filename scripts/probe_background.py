"""Read-only desktop audit around our hidden original-game child."""
import ctypes
import json
from pathlib import Path
import time

from fly_touhou.frida_controller import OriginalGameController

root = Path(__file__).resolve().parents[1]
user = ctypes.WinDLL('user32')
user.GetForegroundWindow.restype = ctypes.c_void_p
user.IsWindowVisible.argtypes = [ctypes.c_void_p]
user.IsWindowVisible.restype = ctypes.c_int
audit = {'foreground_before': user.GetForegroundWindow(), 'source': 'native-process', 'events': []}
controller = OriginalGameController(root / 'games/th14_trial/extracted/app/th14.exe',
                                    root / 'profiles/th14_trial_0.01b.json', timeout=20)
try:
    started = time.time()
    snapshot = controller.start()
    print('START', controller.pid, snapshot.screen, snapshot.pause, snapshot.frame, flush=True)
    audit['pid'] = controller.pid
    audit['startup_seconds'] = time.time() - started
    for i in range(20):
        audit['events'].append(snapshot.to_dict())
        if snapshot.screen == 7 and snapshot.pause == 2 and snapshot.player:
            break
        snapshot = controller.tap(1, 4)
        snapshot = controller.advance(0, 24)
        print('MENU', i, snapshot.screen, snapshot.pause, snapshot.frame,
              snapshot.stage_frame, snapshot.player, flush=True)
    for mask in (1, 1 | 8 | 64, 1 | 8 | 128, 1 | 8) + (1 | 8,) * 4:
        before = snapshot
        snapshot = controller.advance(mask, 60)
        print('GAME', hex(mask), snapshot.frame, snapshot.stage_frame, snapshot.player,
              len(snapshot.bullets), len(snapshot.enemies), flush=True)
        audit['events'].append(snapshot.to_dict())
    print('RESTART', controller.restart().stage_frame, flush=True)
    windows = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    @callback_type
    def each(hwnd, param):
        owner = ctypes.c_uint32()
        user.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(owner))
        if owner.value == controller.pid:
            windows.append({'hwnd': hwnd, 'visible': bool(user.IsWindowVisible(hwnd))})
        return 1
    user.EnumWindows(each, 0)
    audit['windows'] = windows
    audit['foreground_after'] = user.GetForegroundWindow()
    audit['foreground_unchanged'] = audit['foreground_before'] == audit['foreground_after']
except Exception as exc:
    audit['error'] = repr(exc)
    raise
finally:
    audit['foreground_after'] = user.GetForegroundWindow()
    audit['foreground_unchanged'] = audit['foreground_before'] == audit['foreground_after']
    print('AUDIT', {k:v for k,v in audit.items() if k != 'events'}, flush=True)
    controller.close()
    (root / 'artifacts/background_probe.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
