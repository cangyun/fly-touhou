from types import SimpleNamespace
import pytest
from fly_touhou.fixed_start_controller import FineStartControllerMixin
from fly_touhou.native import NativeError
from fly_touhou import nbnm_spell_practice, nbnm_stage_practice


class Driver:
    def __init__(self):
        self.pid = 123
    def start(self):
        return self.read()
    def read(self):
        return SimpleNamespace(playing=False,screen=1,extra={})
    def _playing(self,snapshot):
        return snapshot.playing


class Controller(FineStartControllerMixin, Driver):
    def _startup_owned_identity(self):
        return dict(pid=self.pid,identity_complete=True)
    def _startup_cleanup(self,identity):
        return dict(exact_exit_confirmed=True)
    def _ensure_run_once(self,difficulty):
        raise AssertionError('A practice retry must not fall back to Story mode')


def detached(controller):
    error=NativeError('terminated before policy sampling')
    error.native_event=dict(kind='native-process-detached',reason='process-terminated',pid=controller.pid)
    return error


@pytest.mark.parametrize('module',[nbnm_spell_practice,nbnm_stage_practice])
def test_practice_retries_same_entry_after_menu_exit_and_stops_retrying_after_play(monkeypatch,module):
    calls=[]
    def enter(controller,**kwargs):
        calls.append(kwargs)
        if len(calls)!=2:raise detached(controller)
        return SimpleNamespace(playing=True)
    monkeypatch.setattr(module,'enter',enter)
    controller=Controller()
    task=dict(mode='attack',stage=4,attack='BossCard2',spell_index=1,spell_id=47)
    module.bind_controller(controller,task)
    assert controller.ensure_run(difficulty=3).playing
    assert calls[0]==calls[1]
    assert controller.startup_recovery_audit()['reached_playing']
    assert len(controller.startup_recovery_audit()['attempts'])==1
    with pytest.raises(NativeError):controller.ensure_run(difficulty=3)
    assert len(calls)==3


@pytest.mark.parametrize('module',[nbnm_spell_practice,nbnm_stage_practice])
def test_practice_startup_recovery_has_three_attempt_limit(monkeypatch,module):
    calls=[]
    def enter(controller,**kwargs):
        calls.append(kwargs)
        raise detached(controller)
    monkeypatch.setattr(module,'enter',enter)
    controller=Controller()
    module.bind_controller(controller,dict(stage=3))
    with pytest.raises(NativeError):controller.ensure_run(difficulty=3)
    assert len(calls)==3
    assert len(controller.startup_recovery_audit()['attempts'])==3


@pytest.mark.parametrize('module',[nbnm_spell_practice,nbnm_stage_practice])
def test_practice_does_not_retry_wrong_card_or_mapping_failure(monkeypatch,module):
    calls=[]
    def enter(controller,**kwargs):
        calls.append(kwargs)
        raise NativeError('Native practice selected a different card')
    monkeypatch.setattr(module,'enter',enter)
    controller=Controller()
    module.bind_controller(controller,dict(stage=2))
    with pytest.raises(NativeError):controller.ensure_run(difficulty=3)
    assert len(calls)==1


def test_practice_unlock_script_is_recreated_for_recovered_native_session():
    from fly_touhou.nbnm_practice_access import unlock,restore
    class Script:
        def __init__(self):
            self.exports_sync=self;self.applied=0;self.restored=0;self.unloaded=0
        def load(self):pass
        def apply(self):self.applied+=1
        def restore(self):self.restored+=1
        def unload(self):self.unloaded+=1
    class Session:
        def __init__(self):self.scripts=[]
        def create_script(self,source):
            script=Script();self.scripts.append(script);return script
    controller=SimpleNamespace(executable='th14.exe',_session=Session())
    first=controller._session
    unlock(controller);unlock(controller)
    assert len(first.scripts)==1
    controller._session=Session()
    unlock(controller);restore(controller)
    assert len(controller._session.scripts)==1
    assert first.scripts[0].restored==0
    assert controller._session.scripts[0].applied==1
    assert controller._session.scripts[0].restored==1
