"""TH14's own Stage Practice entry, through private process-local menu input."""
from functools import partial
from .native import NativeError

ENTRY_MODE = 'native_stage_practice'
PREFIX_MODE = 'native_stage_prefix'


def enabled(settings):
    return settings.get('entry_mode') in (ENTRY_MODE, PREFIX_MODE)


def bind_controller(controller, task, *, entry_frame=6):
    controller.ensure_run = partial(enter, controller, stage=task['stage'],entry_frame=entry_frame)
    return controller


def enter(controller, *, stage, difficulty=3,entry_frame=6):
    if not 1 <= stage <= 6:
        raise ValueError('Stage Practice covers the six Story stages')
    from .nbnm_practice_access import unlock,restore
    unlock(controller);controller._fine_begin('native-stage-practice');snapshot=controller.read()
    def playing(s):
        return (s.screen==7 and s.pause==2 and s.player is not None and s.player.state==1
                and s.stage_frame>=entry_frame and not s.player.invulnerable_frames)
    visits={}
    for _ in range(320):
        if playing(snapshot):
            if (snapshot.stage!=stage or snapshot.difficulty!=difficulty
                    or snapshot.extra.get('run_mode')!=1):
                raise NativeError('Native Stage Practice selected a different stage/difficulty')
            restore(controller);return controller._fine_finish(snapshot,difficulty)
        e=snapshot.extra;page=e.get('menu_page');sub=e.get('menu_substate');selection=e.get('menu_selection')
        if sub==2:
            key=(page,selection);visits[key]=visits.get(key,0)+1
            if visits[key]>18:raise NativeError('Requested native stage is locked or unavailable: '+str(key))
            target={1:2,5:difficulty,6:0,7:0,8:stage-1}.get(page)
            mask=1 if target is None or selection==target else 0x20
        else:mask=1 if page in (None,0) else 0
        for key,ticks in (((mask,4),(0,4)) if mask else ((0,8),)):
            snapshot,ready=controller._fine_segment(key,ticks,playing)
            if ready:break
    raise NativeError('Native Stage Practice entry did not finish')


def result(snapshot):
    # The actual clear menu signature is measured by the all-stage preparer.
    # Game Over/contact is checked first by NBNMEnv.
    return bool(snapshot.extra.get('run_mode')==1 and snapshot.pause==1
                and snapshot.extra.get('pause_menu_state')==31
                and snapshot.extra.get('pause_menu_result')==1
                and not snapshot.extra.get('native_gameover'))
