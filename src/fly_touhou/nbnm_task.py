"""Native attack boundaries and first-contact task termination."""
import re
from .nbnm_sensation import current_attack


def attack_key(name):
    # Keep helper routines belonging to the SAME attack in that attack.
    match = re.match(r'^(M?Boss)(Card)?(\d+)(?:\D.*)?$', name or '')
    return (match[1], bool(match[2]), int(match[3])) if match else None


class TaskTracker:
    def __init__(self, task):
        self.task = dict(task)
        if task.get('mode') not in ('full_story', 'attack', 'stage'):
            raise ValueError('NBNM task mode must be full_story, stage or attack')
        if task['mode'] != 'full_story' and not 1 <= task.get('stage',0) <= 6:
            raise ValueError('A native stage is required')
        if task['mode'] == 'attack' and attack_key(task.get('attack')) is None:
            raise ValueError('Use the actual measured primary Boss ECL attack name')
        self.active = False
        self.reason = None

    def begin(self, snapshot):
        events = snapshot.extra['thrl_events']
        if snapshot.player is None or snapshot.player.state != 1 or snapshot.player.invulnerable_frames:
            raise ValueError('Task entry must be controllable and unprotected')
        if self.task['mode'] != 'full_story' and snapshot.stage != self.task['stage']:
            raise ValueError('Prefix did not reach the requested stage')
        if self.task['mode'] == 'attack' and attack_key(current_attack(snapshot)) != attack_key(self.task['attack']):
            raise ValueError('Prefix did not reach the requested attack')
        self.start_events = dict(misses=events['misses'], bombs_used=events['bombs_used'])
        if self.task['mode'] == 'full_story' and any(self.start_events.values()):
            raise ValueError('Full NBNM must begin before any Miss or Bomb')
        self.start_frame = snapshot.frame
        self.active = True
        self.reason = None

    def terminal(self, snapshot, native_end=0):
        if not self.active: return native_end
        e = snapshot.extra['thrl_events']
        if e['bombs_used'] > self.start_events['bombs_used']:
            self.reason = 'Bomb-used'; return 1
        if e['misses'] > self.start_events['misses'] or (snapshot.player is not None and snapshot.player.state == 4):
            self.reason = 'first-native-contact'; return 1
        if native_end == 1:
            self.reason = 'native-gameover'; return 1
        if native_end == 2:
            self.reason = 'native-six-stage-clear'; return 2
        if self.task['mode'] != 'full_story' and snapshot.stage > self.task['stage']:
            self.reason = 'native-stage-complete'; return 2
        if self.task['mode'] == 'attack':
            key = attack_key(current_attack(snapshot))
            roles = e.get('ecl_roles', {})
            if roles.get('observed') is True and key is not None and key != attack_key(self.task['attack']):
                self.reason = 'native-next-attack'; return 2
        return 0

    def report(self, snapshot, *, terminal):
        e = snapshot.extra['thrl_events']
        return dict(task=dict(self.task), task_native_frames=max(0, snapshot.frame-self.start_frame),
            task_misses=e['misses']-self.start_events['misses'],
            task_Bombs=e['bombs_used']-self.start_events['bombs_used'],
            nb_nm_contact=self.reason == 'first-native-contact',
            nb_nm_reason=self.reason, nb_nm_terminal=bool(terminal),
            nbnm_success=bool(terminal == 2),
            full_game_NBNM=bool(terminal == 2 and self.task['mode'] == 'full_story'))
