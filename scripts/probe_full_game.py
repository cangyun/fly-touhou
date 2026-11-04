"""Version/IPC/runtime evidence for the user's original TH14 v1.00b."""
from pathlib import Path
from datetime import datetime, timezone
import json
import argparse
import hashlib

from fly_touhou.frida_controller import OriginalGameController

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exe', type=Path, default=Path(
    r'games/th14/th14.exe'))
parser.add_argument('--profile', type=Path, default=ROOT/'profiles/th14_1.00b_original.json')
options = parser.parse_args()
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output = ROOT/'artifacts/full_game/runtime'/stamp
output.mkdir(parents=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
evidence = {'started_utc': stamp, 'executable': str(options.exe),
    'exe_sha256_before': sha(options.exe),
    'original_directory_entries_before': sorted(p.name for p in options.exe.parent.iterdir()),
    'snapshots': [], 'source': 'native-process', 'desktop_input': False}
controller = OriginalGameController(options.exe, options.profile, timeout=25)
try:
    snapshot = controller.start()
    evidence['pid'] = controller.pid
    snapshot = controller.ensure_run()
    evidence['snapshots'].append(snapshot.to_dict())
    print('FULL START', controller.pid, snapshot.stage, snapshot.difficulty,
          snapshot.character, snapshot.stage_frame, flush=True)
    for mask, ticks in ((1|8|64, 20), (1|8|128, 20), (1|8, 60)) + ((1|8, 60),)*8:
        before = snapshot
        snapshot = controller.advance(mask, ticks)
        evidence['snapshots'].append(snapshot.to_dict())
        print('FULL STEP', hex(mask), snapshot.frame-before.frame,
              snapshot.stage_frame, snapshot.player, len(snapshot.bullets), flush=True)
    before_pid, before_frame = controller.pid, snapshot.stage_frame
    snapshot = controller.restart()
    evidence['snapshots'].append(snapshot.to_dict())
    evidence['restart'] = {'same_process': before_pid == controller.pid,
        'before_stage_frame': before_frame, 'after_stage_frame': snapshot.stage_frame,
        'playing': bool(snapshot.player and snapshot.pause == 2 and snapshot.screen == 7)}
    evidence['window_audit'] = controller.audit_windows()
    evidence['private_appdata'] = str(controller.private_appdata)
    evidence['private_game_files'] = [str(p.relative_to(controller.private_appdata))
                                    for p in controller.private_appdata.rglob('*') if p.is_file()]
    evidence['passed'] = evidence['restart']['playing'] and snapshot.stage_frame < before_frame
except BaseException as error:
    evidence['error'] = repr(error)
    raise
finally:
    controller.close()
    evidence['exe_sha256_after'] = sha(options.exe)
    evidence['original_directory_entries_after'] = sorted(p.name for p in options.exe.parent.iterdir())
    evidence['original_directory_entries_unchanged'] = (
        evidence['original_directory_entries_before'] == evidence['original_directory_entries_after'])
    (output/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print('EVIDENCE', output, flush=True)
