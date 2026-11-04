"""Real TH14 rendering and live frozen-policy inference; no desktop input."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import hashlib
import os
import time

from .thrl_demo_control import digest, validate_selection, CATALOGUE_V2
from .thrl_durable_records import write_json as write
from .thrl_study_io import read_shared_json as read, utc_now


class OwnedWindowPresenter:
    """Present only the verified child HWND, without activating/capturing input."""
    def __init__(self, identity):
        from .thrl_control import alive
        if os.name != 'nt' or not alive(identity):
            raise RuntimeError('A verified live Windows child is required')
        self.identity = identity
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        u = self.user
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowThreadProcessId.restype = wintypes.DWORD
        u.IsWindowVisible.argtypes = [wintypes.HWND];u.IsWindowVisible.restype = wintypes.BOOL
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.GetWindowRect.restype = wintypes.BOOL
        u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.GetClientRect.restype = wintypes.BOOL
        u.ShowWindowAsync.argtypes = [wintypes.HWND, ctypes.c_int];u.ShowWindowAsync.restype = wintypes.BOOL
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, wintypes.UINT];u.SetWindowPos.restype = wintypes.BOOL
        u.GetForegroundWindow.argtypes = [];u.GetForegroundWindow.restype = wintypes.HWND
        self.callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        u.EnumWindows.argtypes = [self.callback_type, wintypes.LPARAM];u.EnumWindows.restype = wintypes.BOOL
        self.hwnd = None
        self.present_calls = 0

    def find(self):
        from .thrl_control import alive
        if not alive(self.identity):
            raise RuntimeError('The owned demonstration process exited')
        windows = []
        def collect(hwnd, _):
            pid = wintypes.DWORD();self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == self.identity['pid']:
                rect = wintypes.RECT()
                if self.user.GetClientRect(hwnd, ctypes.byref(rect)):
                    windows.append(((rect.right-rect.left)*(rect.bottom-rect.top), hwnd))
            return True
        callback = self.callback_type(collect)
        self.user.EnumWindows(callback, 0)
        if not windows:
            raise RuntimeError('The original renderer did not create an owned window')
        self.hwnd = max(windows)[1]
        return self.hwnd

    def show(self):
        self.find()
        before = self.user.GetForegroundWindow()
        rect = wintypes.RECT();self.user.GetWindowRect(self.hwnd, ctypes.byref(rect))
        if not self.user.IsWindowVisible(self.hwnd) or rect.left < -1000:
            # Existing in-child guards keep startup hidden. This call is made
            # by the host, affects only this HWND, and never activates it.
            self.user.SetWindowPos(self.hwnd, wintypes.HWND(-2), 60, 60, 0, 0, 0x1|0x10|0x40|0x4000)
            self.user.ShowWindowAsync(self.hwnd, 4)  # SW_SHOWNOACTIVATE
            self.present_calls += 1
        self.user.GetWindowRect(self.hwnd, ctypes.byref(rect))
        return dict(hwnd=int(self.hwnd), visible=bool(self.user.IsWindowVisible(self.hwnd)),
            rectangle=[rect.left, rect.top, rect.right, rect.bottom],
            host_foreground_before=int(before or 0),
            host_foreground_after=int(self.user.GetForegroundWindow() or 0),
            presentation_requests=self.present_calls, desktop_input=False,
            real_Direct3D=True, headless_renderer=False)

    def capture(self, path):
        """Capture this owned client only, for renderer qualification."""
        from PIL import Image
        u = self.user;g = ctypes.WinDLL('gdi32', use_last_error=True)
        u.GetDC.argtypes=[wintypes.HWND];u.GetDC.restype=wintypes.HDC
        u.ReleaseDC.argtypes=[wintypes.HWND,wintypes.HDC];u.ReleaseDC.restype=ctypes.c_int
        u.PrintWindow.argtypes=[wintypes.HWND,wintypes.HDC,wintypes.UINT];u.PrintWindow.restype=wintypes.BOOL
        g.CreateCompatibleDC.argtypes=[wintypes.HDC];g.CreateCompatibleDC.restype=wintypes.HDC
        g.CreateCompatibleBitmap.argtypes=[wintypes.HDC,ctypes.c_int,ctypes.c_int];g.CreateCompatibleBitmap.restype=wintypes.HBITMAP
        g.SelectObject.argtypes=[wintypes.HDC,wintypes.HGDIOBJ];g.SelectObject.restype=wintypes.HGDIOBJ
        g.GetBitmapBits.argtypes=[wintypes.HBITMAP,wintypes.LONG,ctypes.c_void_p];g.GetBitmapBits.restype=wintypes.LONG
        g.DeleteObject.argtypes=[wintypes.HGDIOBJ];g.DeleteObject.restype=wintypes.BOOL
        g.DeleteDC.argtypes=[wintypes.HDC];g.DeleteDC.restype=wintypes.BOOL
        rect=wintypes.RECT();u.GetClientRect(self.hwnd,ctypes.byref(rect))
        width,height=rect.right,rect.bottom
        dc=u.GetDC(self.hwnd);memory=g.CreateCompatibleDC(dc)
        bitmap=g.CreateCompatibleBitmap(dc,width,height);old=g.SelectObject(memory,bitmap)
        try:
            ok=u.PrintWindow(self.hwnd,memory,3)
            data=ctypes.create_string_buffer(width*height*4)
            count=g.GetBitmapBits(bitmap,len(data),data)
            if not ok or count!=len(data):raise RuntimeError('Owned renderer capture failed')
            Image.frombuffer('RGB',(width,height),data.raw,'raw','BGRX',0,1).save(path)
        finally:
            g.SelectObject(memory,old);g.DeleteObject(bitmap);g.DeleteDC(memory);u.ReleaseDC(self.hwnd,dc)


def visible_environment(options):
    from .thrl_full_story_evaluation import FullStoryEvaluationEnv
    from .thrl_native_events import THRLDialogueGeometryController
    class VisibleDemoEnv(FullStoryEvaluationEnv):
        def _create_controller(self):
            return THRLDialogueGeometryController(self.executable,self.profile_path,timeout=self.timeout,
                private_appdata=self.private_appdata,worker_name='B1 REAL NEURAL DEMO',
                checkpoint_wait_sleep_ms=10,checkpoint_timing=False,headless=False,
                presentation_guard=True,binary_reader=self.binary_reader,native_seed=self.native_seed)
    return VisibleDemoEnv(**{**options,'headless':False,'hide_window':False})


def resume_owned_pause(env):
    """Resume only an observed native pause menu; do not run a neural step."""
    snapshot=env.snapshot
    if snapshot.pause==2:return None
    if snapshot.screen!=7 or snapshot.pause!=0 or snapshot.lives<0:
        raise RuntimeError('Unexpected demonstration screen; refusing a menu action')
    controller=env.controller;hooks=controller.hooks;memory=controller._memory
    before_stage=snapshot.stage;before_miss=snapshot.extra['thrl_events']['misses']
    ticks=0
    for index in range(64):
        snapshot=controller.read()
        if snapshot.pause==2:break
        menu=memory.integer(memory.base+hooks['retry_pointer_rva'])
        if not menu:raise RuntimeError('Pause pointer is absent; no menu action issued')
        state=memory.integer(menu+hooks['pause_menu_state_offset'])
        selection=memory.integer(menu+hooks['pause_menu_selection_offset'])
        if state==3 and selection!=0:
            raise RuntimeError('Pause selection is not Return to Game; no confirmation issued')
        if state in (14,15,23):raise RuntimeError('A terminal/continue menu cannot be resumed')
        controller.advance(0 if index<4 else 1,1);ticks+=1
    snapshot=controller.read()
    if snapshot.pause!=2:raise RuntimeError('Native pause did not resume within64 menu-only input ticks')
    if snapshot.stage!=before_stage or snapshot.extra['thrl_events']['misses']!=before_miss:
        raise RuntimeError('A resume operation unexpectedly changed stage or lives')
    env.snapshot=snapshot
    observation=env.encoder.encode(snapshot)
    return observation,env._info(),ticks


def demonstration_environment_role(plan, selected):
    """A development clear must regenerate its development-partition opening."""
    role = plan.get('environment_role', 'test')
    expected = ('development' if plan.get('schema') == CATALOGUE_V2
                and selected.get('evidence_scope') == 'development' else 'test')
    if role != expected:
        raise ValueError('The demonstration role differs from its recorded clear evidence')
    return role


def run(plan_path, runtime):
    import numpy as np
    import psutil
    import torch
    from .thrl_control import process_identity
    from .thrl_pair_policy import model_from_config
    from .thrl_inputs import forward_observation
    from .thrl_policy_actions import policy_logits
    from train_thrl_native import make_env_kwargs
    plan=read(plan_path);output=Path(plan['output']);runtime=Path(runtime)
    if digest(runtime/'source_manifest.json')!=plan['source_manifest_sha256']:
        raise ValueError('Frozen demonstration source manifest changed')
    for item in read(runtime/'source_manifest.json')['files']:
        if digest(runtime/item['path'])!=item['sha256']:
            raise ValueError('Frozen demonstration source changed:'+item['path'])
    selected=validate_selection(plan,plan['selected_model'])
    environment_role=demonstration_environment_role(plan,selected)
    if not plan['known_seed_demonstration'] or plan['formal_test']:
        raise ValueError('This interface is an explicitly selected-seed demonstration')
    cfg=read(plan['environment_config'])
    if (digest(plan['environment_config'])!=plan['environment_config_sha256']
            or digest(cfg['game_exe'])!=plan['game_exe_sha256']
            or digest(cfg['profile'])!=plan['profile_sha256']):
        raise ValueError('Original game, profile or inference configuration changed')
    if sum((p.info['name'] or '').lower()=='th14.exe' for p in psutil.process_iter(['name']))>=30:
        raise RuntimeError('The thirty-original-world budget is already occupied')
    if not torch.cuda.is_available():raise RuntimeError('CUDA inference is required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    payload=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    model=model_from_config(**payload['model_config']).cuda();model.load_state_dict(payload['model'],strict=True);model.eval()
    if any(not torch.equal(v.detach().cpu(),payload['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('Actual CUDA inference weights differ from the selected policy')
    identity=process_identity(os.getpid());begun=time.monotonic();decisions=0;action_hash=hashlib.sha256()
    base=dict(schema='B1-real-neural-demo-status-v1',worker_identity=identity,model=plan['selected_model'],
        model_label=selected['label'],checkpoint_sha256=selected['checkpoint_sha256'],
        known_seed_demonstration=True,formal_test=False,native_seed=plan['known_native_seed'],
        evidence_scope=selected.get('evidence_scope','formal'),
        evaluation_episodes=selected.get('evaluation_episodes',selected.get('formal_test_episodes')),
        evaluation_clears=selected.get('evaluation_clears',selected.get('formal_test_clears')),
        stable_completion_claimed=False,
        desktop_input=False,teacher_actions=0,safety_takeovers=0,optimizer_updates=0,
        recorded_action_playback=False,action_selection='greedy',real_Direct3D=True)
    write(output/'launch.json',{**base,'utc':utc_now()})
    env=None;presenter=None;info={};last_write=0.;paused=False;menu_ticks=0
    def publish(status,**extra):
        write(output/'status.json',{**base,'status':status,'updated_utc':utc_now(),
            'policy_decisions':decisions,'menu_only_input_ticks':menu_ticks,'wall_seconds':time.monotonic()-begun,
            'game':{k:info.get(k) for k in ('pid','stage','stage_frame','lives','bombs','misses',
                'bombs_used','power','episode_frames')},**extra})
    publish('starting')
    try:
        options=make_env_kwargs(cfg,output,1,environment_role)[0]
        env=visible_environment(options)
        observation,info=env.reset(options={'evaluation_seed':plan['environment_seed']})
        if env.native_seed!=plan['known_native_seed']:
            raise ValueError('The demonstration opening does not match its labelled seed')
        presenter=OwnedWindowPresenter(process_identity(env.controller.pid))
        presentation=presenter.show();write(output/'presentation.json',presentation)
        hidden=torch.zeros(1,1,model.gru_hidden_size,device='cuda:0')
        with (output/'live_neural_actions.bin').open('wb') as tape,torch.inference_mode():
            while True:
                if (output/'STOP').exists():publish('stopped',reason='operator_stop');break
                if (output/'PAUSE').exists():
                    if not paused:publish('paused');paused=True
                    time.sleep(.1);continue
                paused=False
                if env.snapshot.pause!=2:
                    publish('menu_resume')
                    resumed=resume_owned_pause(env)
                    if resumed is not None:
                        observation,info,released=resumed;menu_ticks+=released
                        write(output/'menu_resume.json',dict(utc=utc_now(),menu_only_input_ticks=menu_ticks,
                            gameplay_action_overrides=0,teacher_actions=0,desktop_input=False))
                    continue
                logits,_,hidden=forward_observation(torch,model,{k:v[None] for k,v in observation.items()},hidden)
                probabilities=torch.softmax(policy_logits(logits),dim=-1)[0].cpu().numpy()
                action=int(np.argmax(probabilities))
                observation,_,terminated,truncated,info=env.step(action)
                decisions+=1;action_hash.update(bytes([action]));tape.write(bytes([action]))
                if truncated:raise RuntimeError('A visible full-story demonstration unexpectedly truncated')
                now=time.monotonic()
                if now-last_write>=1 or terminated:
                    presentation=presenter.show()
                    tape.flush();os.fsync(tape.fileno())
                    publish('running',window=presentation,last_action=action,
                        Bomb_probability=float(probabilities[18]),inference_calls=decisions)
                    last_write=now
                    if decisions>=64 and not (output/'renderer.png').exists():
                        try:presenter.capture(output/'renderer.png')
                        except Exception as error:write(output/'capture_error.json',dict(error=repr(error)))
                if terminated:
                    summary=info['episode_summary']
                    unchanged=all(torch.equal(v.detach().cpu(),payload['model'][k]) for k,v in model.state_dict().items())
                    report={**base,'status':'completed','utc':utc_now(),'episode_summary':summary,
                        'inference_calls':decisions,'action_sha256':action_hash.hexdigest(),
                        'menu_only_input_ticks':menu_ticks,
                        'checkpoint_unchanged':unchanged and digest(selected['checkpoint'])==selected['checkpoint_sha256'],
                        'demonstration_only_not_new_test_evidence':True}
                    write(output/'report.json',report)
                    publish('completed',outcome=summary['outcome'],inference_calls=decisions)
                    try:presenter.capture(output/'terminal-renderer.png')
                    except Exception as error:write(output/'terminal_capture_error.json',dict(error=repr(error)))
                    # Keep the original terminal scene visible for inspection.
                    hold_until=time.monotonic()+600
                    while time.monotonic()<hold_until and not (output/'STOP').exists():time.sleep(.1)
                    break
    finally:
        if env is not None:env.close()
        write(output/'cleanup.json',dict(utc=utc_now(),only_owned_child_closed=True,
            desktop_input=False,other_jobs_modified=False))
