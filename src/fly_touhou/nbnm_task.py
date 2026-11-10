"""Native attack boundaries and first-contact task termination."""
import re
from .nbnm_sensation import current_attack


def attack_key(name):
    # Keep helper routines belonging to the SAME attack in that attack.
    match = re.match(r'^(M?Boss[AB]?)(Card)?(\d+)(?:\D.*)?$', name or '')
    return (match[1], bool(match[2]), int(match[3])) if match else None


def entry_matches(entry_task, task):
    """Reuse a real Story entry by its start, independently of its later goal."""
    if task.get('mode') != 'chain':
        return entry_task == task
    return (entry_task.get('stage') == task['stage'] and
            (attack_key(entry_task.get('attack')) == attack_key(task.get('attack'))
             if task.get('attack') else entry_task.get('mode') == 'stage'))


def fresh_opening(task):
    return task['mode']=='full_story' or (task['mode'] in ('chain','stage') and
        task.get('stage')==1 and not task.get('attack'))


class TaskTracker:
    def __init__(self, task):
        self.task = dict(task)
        if task.get('mode') not in ('full_story', 'attack', 'stage', 'chain'):
            raise ValueError('NBNM task mode must be full_story, stage, attack or chain')
        if task['mode'] != 'full_story' and not 1 <= task.get('stage',0) <= 6:
            raise ValueError('A native stage is required')
        if task['mode'] == 'attack' and attack_key(task.get('attack')) is None:
            raise ValueError('Use the actual measured primary Boss ECL attack name')
        if task['mode'] == 'chain':
            if not task['stage'] <= task.get('end_stage', 0) <= 6:
                raise ValueError('A forward chain requires start_stage <= end_stage <= 6')
            for field in ('attack', 'end_attack'):
                if task.get(field) and attack_key(task[field]) is None:
                    raise ValueError('Use a measured native attack for ' + field)
        self.active = False
        self.reason = None

    def begin(self, snapshot):
        events = snapshot.extra['thrl_events']
        if snapshot.player is None or snapshot.player.state != 1 or snapshot.player.invulnerable_frames:
            raise ValueError('Task entry must be controllable and unprotected')
        if self.task['mode'] != 'full_story' and snapshot.stage != self.task['stage']:
            raise ValueError('Prefix did not reach the requested stage')
        if self.task.get('attack') and attack_key(current_attack(snapshot)) != attack_key(self.task['attack']):
            raise ValueError('Prefix did not reach the requested attack')
        self.start_events = dict(misses=events['misses'], bombs_used=events['bombs_used'])
        if self.task['mode'] == 'full_story' and any(self.start_events.values()):
            raise ValueError('Full NBNM must begin before any Miss or Bomb')
        self.start_frame = snapshot.frame
        self.active = True
        self.reason = None
        self.transitions = []
        self.last_attack = (snapshot.stage, attack_key(current_attack(snapshot)))
        self.end_seen = False
        self._observe_attack(snapshot)

    def _observe_attack(self, snapshot):
        key = attack_key(current_attack(snapshot))
        observed = snapshot.extra['thrl_events'].get('ecl_roles', {}).get('observed') is True
        current = (snapshot.stage, key)
        if observed and key is not None:
            if current != self.last_attack:
                self.transitions.append(dict(frame=snapshot.frame, stage=snapshot.stage,
                    attack=current_attack(snapshot)))
                self.last_attack = current
            if (snapshot.stage == self.task.get('end_stage') and
                    key == attack_key(self.task.get('end_attack'))):
                self.end_seen = True
        return key, observed

    def terminal(self, snapshot, native_end=0, *, allow_task_transition=True):
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
        if self.task['mode'] == 'chain':
            key, observed = self._observe_attack(snapshot)
            end_stage = self.task['end_stage']
            # No reset/reward at intermediate cards or stages. Leaving the
            # requested last attack (or last stage) is the ONLY chain success.
            finished = snapshot.stage > end_stage
            finished |= bool(self.task.get('end_attack') and self.end_seen and observed
                and key is not None and snapshot.stage == end_stage
                and key != attack_key(self.task['end_attack']))
            if allow_task_transition and finished:
                self.reason = 'native-chain-complete'; return 2
            return 0
        if allow_task_transition and self.task['mode'] != 'full_story' and snapshot.stage > self.task['stage']:
            self.reason = 'native-stage-complete'; return 2
        if allow_task_transition and self.task['mode'] == 'attack':
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
            chain_transitions=list(getattr(self, 'transitions', [])),
            full_game_NBNM=bool(terminal == 2 and self.task['mode'] == 'full_story'))
