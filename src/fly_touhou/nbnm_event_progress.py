"""Bounded native milestone rewards; no HP, clock, position or forecast credit.

Attack order comes from the measured Lunatic Story entry catalog. Stage four's
two routes share event slots. Road boundaries are the actual first midboss and
main-boss attacks; a missing role, animation helper or HP reset is not an event.
"""
from .nbnm_task import attack_key
from .nbnm_sensation import current_attack

REVISION = 'native-events-v1'
CONFIRMATION_REVISION = 'th14-native-attack-interrupt-v1'
SCHEMA = 'th14-nbnm-event-progress-v1'
ATTACKS = {
    1: ('MBoss1', 'MBossCard1', 'Boss1', 'BossCard1', 'Boss2', 'BossCard2'),
    2: ('MBoss1', 'MBossCard1', 'Boss1', 'BossCard1', 'Boss2', 'BossCard2', 'Boss3', 'BossCard3'),
    3: ('MBoss1', 'MBoss2', 'MBossCard1', 'Boss1', 'BossCard1', 'Boss2', 'BossCard2', 'Boss3', 'BossCard3'),
    4: ('MBoss1', 'MBoss2', 'Boss1', 'BossCard1', 'Boss2', 'BossCard2', 'Boss3', 'BossCard3'),
    5: ('MBoss1', 'MBossCard1', 'Boss1', 'BossCard1', 'Boss2', 'BossCard2', 'Boss3', 'BossCard3', 'BossCard4'),
    6: ('MBoss1', 'MBoss2', 'Boss1', 'BossCard1', 'Boss2', 'BossCard2', 'Boss3', 'BossCard3',
        'Boss4', 'BossCard4', 'Boss5', 'BossCard5', 'Boss6', 'BossCard6'),
}


def canonical_attack(stage, name):
    key = attack_key(name)
    if key is None:
        return None
    role, spell, number = key
    if stage == 4:
        role = 'MBoss' if role.startswith('MBoss') else 'Boss'
    result = role + ('Card' if spell else '') + str(number)
    return result if result in ATTACKS.get(stage, ()) else None


def event_unit(stage):
    # Each stage is 1/6 of the global budget: attacks + two roads + one clear.
    return 1. / (6 * (len(ATTACKS[stage]) + 3))


class EventProgress:
    def __init__(self, task, confirmation_revision=None):
        if confirmation_revision not in (None, CONFIRMATION_REVISION):
            raise ValueError('Unknown native event confirmation revision')
        self.confirmation_revision = confirmation_revision
        self.task = dict(task)
        self.stage = None
        self.last_attack = None
        self.entered = set()
        self.road_before_entered = False
        self.midboss_entered = False
        self.credited = set()
        self.events = []
        self.attempts = {}
        self.total = 0.
        self.native_sequence = 0
        self.owner = None

    def begin(self, snapshot):
        self.__init__(self.task, self.confirmation_revision)
        self._enter_stage(snapshot)
        self.native_sequence = snapshot.extra.get('thrl_events', {}).get('attack_event_sequence', 0)

    def _enter_stage(self, snapshot):
        self.stage = snapshot.stage
        self.last_attack = canonical_attack(self.stage, current_attack(snapshot))
        self.entered = {self.last_attack} if self.last_attack else set()
        self.road_before_entered = self.last_attack is None
        self.midboss_entered = bool(self.last_attack and self.last_attack.startswith('MBoss'))
        self._remember_owner(snapshot)
        if self.last_attack:
            self.attempts[f's{self.stage}/{self.last_attack}'] = dict(stage=self.stage,
                attack=self.last_attack, entry_frame=snapshot.frame)

    def _in_scope(self, stage, kind, attack):
        mode = self.task['mode']
        if mode == 'full_story':
            return True
        if mode == 'attack':
            return (stage == self.task['stage'] and kind in ('nonspell', 'spell')
                    and attack == canonical_attack(stage, self.task['attack']))
        if not self.task['stage'] <= stage <= self.task.get('end_stage', self.task['stage']):
            return False
        if mode == 'chain' and self.task.get('end_attack') and stage == self.task['end_stage']:
            end = canonical_attack(stage, self.task['end_attack'])
            if kind == 'stage_clear':
                return False
            if attack and ATTACKS[stage].index(attack) > ATTACKS[stage].index(end):
                return False
        return True

    def _emit(self, stage, kind, slot, snapshot, confirmation, attack=None, event_frame=None):
        key = f's{stage}/{slot}'
        if key in self.credited or not self._in_scope(stage, kind, attack):
            return
        self.credited.add(key)
        value = event_unit(stage)
        self.total += value
        self.events.append(dict(key=key, kind=kind, stage=stage, attack=attack,
            frame=snapshot.frame if event_frame is None else event_frame,
            observed_frame=snapshot.frame, progress=value, confirmation=confirmation))

    def _complete_attack(self, name, snapshot, confirmation, event_frame=None):
        if name is not None and name in self.entered:
            self._emit(self.stage, 'spell' if 'Card' in name else 'nonspell', name,
                snapshot, confirmation, attack=name, event_frame=event_frame)

    def _remember_owner(self, snapshot):
        roles = snapshot.extra.get('thrl_events', {}).get('ecl_roles', {})
        if roles.get('observed') is not True:
            return
        for group in ('main_bosses', 'midbosses'):
            for row in roles.get(group, ()):
                name = row.get('ecl_sub_name') or ''
                if group == 'main_bosses' and name.startswith('MBoss'):
                    name = name[1:]
                if self.last_attack and canonical_attack(self.stage, name) == self.last_attack:
                    self.owner = row.get('id')

    def _confirm_native_interrupts(self, snapshot):
        if not self.confirmation_revision:
            return
        native = snapshot.extra.get('thrl_events', {})
        if native.get('attack_event_revision') != self.confirmation_revision:
            raise ValueError('Native attack confirmation hook is not active')
        for row in native.get('attack_transitions', ()):
            if row['sequence'] <= self.native_sequence:
                continue
            self.native_sequence = row['sequence']
            if row['stage'] != self.stage or self.owner is None or row['owner'] != self.owner or not self.last_attack:
                continue
            target = row['target']
            next_attack = canonical_attack(self.stage, target)
            forward = next_attack and ATTACKS[self.stage].index(next_attack) > ATTACKS[self.stage].index(self.last_attack)
            stem = attack_key(self.last_attack)[0]
            endings = (stem+'Dead', stem+'Escape')
            if self.stage == 4:
                endings += (stem+'BDead', stem+'BEscape', stem+'ADead', stem+'AEscape')
            if forward or target in endings:
                self._complete_attack(self.last_attack, snapshot,
                    'native-phase-interrupt:'+target, event_frame=row['frame'])

    def _finish_stage(self, snapshot, confirmation):
        if self.stage not in ATTACKS:
            return
        # A stage clear confirms the final attack, not unobserved skipped attacks.
        if self.last_attack == ATTACKS[self.stage][-1]:
            self._complete_attack(self.last_attack, snapshot, confirmation)
        self._emit(self.stage, 'stage_clear', 'clear', snapshot, confirmation)

    def advance(self, snapshot, *, terminal=0, reason=None):
        before = len(self.events)
        if terminal == 1:
            # Contact/Bomb wins over an attack/stage change on the same sample.
            return 0., []
        self._confirm_native_interrupts(snapshot)
        observed = snapshot.extra.get('thrl_events', {}).get('ecl_roles', {}).get('observed') is True
        if observed and snapshot.stage in ATTACKS:
            if snapshot.stage > self.stage:
                self._finish_stage(snapshot, 'native-next-stage')
                if terminal != 2:
                    self._enter_stage(snapshot)
            elif snapshot.stage == self.stage:
                name = canonical_attack(self.stage, current_attack(snapshot))
                order = ATTACKS[self.stage]
                forward = name is not None and (self.last_attack is None
                    or order.index(name) > order.index(self.last_attack))
                if forward:
                    self._complete_attack(self.last_attack, snapshot, 'native-next-attack')
                    if name == order[0] and self.road_before_entered:
                        self._emit(self.stage, 'road', 'road-before-midboss', snapshot, 'native-first-midboss')
                    if name == 'Boss1' and self.midboss_entered:
                        self._emit(self.stage, 'road', 'road-after-midboss', snapshot, 'native-first-main-boss')
                    self.entered.add(name)
                    self.last_attack = name
                    self.midboss_entered |= name.startswith('MBoss')
                    if terminal != 2 and self._in_scope(self.stage,'spell' if 'Card' in name else 'nonspell',name):
                        self.attempts.setdefault(f's{self.stage}/{name}',dict(stage=self.stage,attack=name,entry_frame=snapshot.frame))
                self._remember_owner(snapshot)
        if terminal == 2:
            mode = self.task['mode']
            if mode == 'attack':
                name = canonical_attack(self.task['stage'], self.task['attack'])
                self._complete_attack(name, snapshot, reason or 'native-task-complete')
            elif mode == 'chain' and self.task.get('end_attack'):
                name = canonical_attack(self.task['end_stage'], self.task['end_attack'])
                self._complete_attack(name, snapshot, reason or 'native-chain-complete')
            else:
                self._finish_stage(snapshot, reason or 'native-stage-complete')
        emitted = self.events[before:]
        return sum(e['progress'] for e in emitted), emitted

    def report(self):
        counts = {kind: 0 for kind in ('road', 'nonspell', 'spell', 'stage_clear')}
        for event in self.events:
            counts[event['kind']] += 1
        return dict(schema=SCHEMA, progress=self.total, counts=counts, events=list(self.events),
            confirmation_revision=self.confirmation_revision,
            attack_attempts=[dict(row,key=key,confirmed_complete=key in self.credited) for key,row in self.attempts.items()],
            scope='native events after task entry; no credit for prefix', hp_used=False)


def summarize_evaluation_events(episodes):
    """Conditional phase outcomes of frozen full games, not Spell Practice."""
    attacks = {}
    stages = {}
    for episode in episodes:
        progress = episode.get('native_event_progress') or {}
        for row in progress.get('attack_attempts', ()):
            count = attacks.setdefault(row['key'],dict(attempts=0,confirmed_completions=0))
            count['attempts'] += 1
            count['confirmed_completions'] += int(row['confirmed_complete'])
        cleared = {r['stage'] for r in progress.get('events', ()) if r['kind']=='stage_clear'}
        for stage in episode.get('stage_history', ()):
            count = stages.setdefault(str(stage),dict(entries=0,clears=0))
            count['entries'] += 1; count['clears'] += int(stage in cleared)
    return dict(attacks=attacks, stages=stages,
        scope='Conditional outcomes within these frozen games; native interrupt when enabled, otherwise next attack/task completion. No independent card-practice qualification.')
