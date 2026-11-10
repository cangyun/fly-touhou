"""Native TH14 Spell Practice: menu entry and measured capture results.

The game executes its own spell scripts. No prior-stage replay, ECL jump,
resource edit, teacher action, or desktop input is used for entry.
"""
from pathlib import Path
from functools import partial
import shutil
import time

from .native import NativeError
from .nbnm_sensation import current_attack
from .nbnm_task import attack_key

ENTRY_MODE = 'native_spell_practice'


def enabled(nbnm):
    return nbnm.get('entry_mode', 'story_prefix') == ENTRY_MODE


def validate_task(task):
    key = attack_key(task.get('attack'))
    if (task.get('mode') != 'attack' or not key or not key[1]
            or not 1 <= task.get('stage', 0) <= 6
            or type(task.get('spell_index')) is not int or task['spell_index'] < 0
            or type(task.get('spell_id')) is not int or task['spell_id'] < 0):
        raise ValueError('Native Spell Practice requires stage, measured card ECL name, spell_index and spell_id')


def copy_score(source, private_appdata):
    """Copy an existing user's unlock record into this private game only."""
    target = Path(private_appdata) / 'ShanghaiAlice/th14/scoreth14.dat'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def spell_state(controller):
    offsets = controller.profile.offsets['spell_card']
    memory = controller._memory
    pointer = memory.integer(memory.base + offsets['spell_card_pointer'])
    if not pointer:
        return None
    return {name: memory.integer(pointer + offsets['zSpellCard_' + name], signed=True)
            for name in ('id', 'bonus')}


def bind_controller(controller, task):
    # Keep the existing hidden event-aware controller and all sensor patches.
    controller.ensure_run = partial(enter, controller, task=task)
    original_read = controller.read

    def read():
        snapshot = original_read()
        if snapshot.screen == 7 and snapshot.extra.get('run_mode') == 2:
            snapshot.extra['native_spell_practice'] = spell_state(controller)
        return snapshot
    controller.read = read
    return controller


def enter(controller, *, task, difficulty=3):
    validate_task(task)
    started = time.monotonic()
    snapshot = controller.read()
    visits = {}
    for _ in range(320):
        if snapshot.screen == 7 and snapshot.player is not None and snapshot.player.state == 1:
            # Native intro/appearance is not a training trajectory. Stop on
            # the first card-script frame, before its first policy action.
            for _ in range(600):
                key = attack_key(current_attack(snapshot))
                if key and key[1]:
                    if (key != attack_key(task['attack']) or snapshot.stage != task['stage']
                            or (spell_state(controller) or {}).get('id') != task['spell_id']):
                        raise NativeError('Native Spell Practice selected a different card')
                    controller.spell_practice_entry = dict(entry_mode=ENTRY_MODE,
                        stage=snapshot.stage, attack=current_attack(snapshot),
                        spell_id=task['spell_id'], spell_index=task['spell_index'],
                        stage_frame=snapshot.stage_frame, preceding_stages_played=False,
                        policy_history='zero recurrent state at first native card frame',
                        elapsed_seconds=time.monotonic()-started)
                    return snapshot
                snapshot = controller.advance(0, 1)
            raise NativeError('Native spell prelude did not reach a card script')
        extra = snapshot.extra
        page, substate, selection = (extra.get(k) for k in ('menu_page','menu_substate','menu_selection'))
        if substate == 2:
            key = (page, selection)
            visits[key] = visits.get(key, 0) + 1
            if visits[key] > 16:
                raise NativeError('Spell Practice selection is locked or unavailable in the supplied score record')
        if page == 1 and substate == 2:
            mask = 1 if selection == 3 else 0x20
        elif page == 17 and substate == 2:
            mask = 1 if selection == task['stage']-1 else 0x20
        elif page == 18 and substate == 2:
            mask = 1 if selection == task['spell_index'] else 0x20
        elif page == 19 and substate == 2:
            # TH14's own menu table; also documented in thprac_th14.cpp,
            # THGuiSP::State. Read only the four actual difficulty entries.
            memory = controller._memory
            menu = memory.integer(memory.base + controller.hooks['menu_pointer_rva'])
            selected_id = memory.integer(menu + 0x5d70 + 4*difficulty, signed=True)
            if selected_id != task['spell_id']:
                raise NativeError('Requested difficulty/card is not unlocked in the supplied score record')
            mask = 1 if selection == difficulty else 0x20
        elif substate == 2 or page in (None, 0):
            mask = 1
        else:
            mask = 0
        snapshot = controller.tap(mask, 4) if mask else controller.advance(0, 8)
    raise NativeError('Native Spell Practice menu entry did not complete')


def result(snapshot, task):
    """1=failure, 2=confirmed capture; never equate disappearance with capture."""
    if not (snapshot.extra.get('run_mode') == 2 and snapshot.pause == 1
            and snapshot.extra.get('pause_menu_state') == 30
            and snapshot.extra.get('pause_menu_result') == 1):
        return 0
    spell = snapshot.extra.get('native_spell_practice') or {}
    if spell.get('id') != task['spell_id']:
        raise NativeError('Native result belongs to a different spell')
    # Positive native capture bonus was observed together with the actual
    # score award and result menu. A timeout/failed capture is not NBNM credit.
    return 2 if spell.get('bonus', 0) > 0 else 1
