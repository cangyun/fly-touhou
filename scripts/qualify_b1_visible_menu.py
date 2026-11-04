"""Bounded real-renderer focus/input test, hidden throughout; not a policy trial."""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))


def main(args):
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.thrl_durable_records import write_json as write
    from fly_touhou.thrl_demo_control import digest
    from fly_touhou.thrl_visible_demo import visible_environment,OwnedWindowPresenter
    from train_thrl_native import make_env_kwargs
    catalogue=read(args.catalogue)
    if digest(ROOT/'source_manifest.json')!=catalogue['source_manifest_sha256']:
        raise ValueError('Qualified menu source differs from the catalogue')
    for row in read(ROOT/'source_manifest.json')['files']:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Frozen source file changed')
    worlds_before=sum((p.info['name'] or '').lower()=='th14.exe' for p in psutil.process_iter(['name']))
    if worlds_before>=30:raise RuntimeError('No original-world slot is available')
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=read(catalogue['environment_config'])
    if digest(catalogue['environment_config'])!=catalogue['environment_config_sha256']:
        raise ValueError('Original environment changed')
    user=ctypes.WinDLL('user32');user.GetForegroundWindow.restype=wintypes.HWND
    foreground_before=int(user.GetForegroundWindow() or 0)
    env=visible_environment(make_env_kwargs(cfg,args.output,1,'test')[0])
    identity=None
    try:
        observation,info=env.reset(options={'evaluation_seed':catalogue['known_environment_seed']})
        controller=env.controller;identity=dict(pid=controller.pid,create_time=psutil.Process(controller.pid).create_time())
        if env.snapshot.pause!=2 or env.snapshot.lives!=2:raise RuntimeError('Fresh original opening was not verified')
        presenter=OwnedWindowPresenter(identity);hwnd=presenter.find()
        guard_before=dict(controller._script.exports_sync.demopresentationaudit())
        input_probe=dict(controller._script.exports_sync.demoinputguardprobe())
        if input_probe['values']!=[0,0]:raise RuntimeError('Owned keyboard-query guards did not return neutral input')
        user.PostMessageW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
        user.PostMessageW.restype=wintypes.BOOL
        # Target this owned HWND only. No physical keyboard/focus operation.
        for message in (0x0006,0x0008,0x001c):
            if not user.PostMessageW(hwnd,message,0,0):raise RuntimeError('Owned focus-message test could not be posted')
        controller.advance(0,8);env.snapshot=controller.read()
        observation=env.encoder.encode(env.snapshot)
        if (env.snapshot.pause!=2 or env.snapshot.stage!=1 or env.snapshot.lives!=2
                or env.snapshot.extra['thrl_events']['misses']!=0
                or env.snapshot.extra['thrl_events']['bombs_used']!=0):
            raise RuntimeError('Menu recovery changed gameplay resources or scene')
        guard=dict(controller._script.exports_sync.demopresentationaudit())
        if (guard['keyboard_queries_blocked']-guard_before['keyboard_queries_blocked']<2
                or guard['focus_loss_messages_normalized']-guard_before['focus_loss_messages_normalized']<2):
            raise RuntimeError('The owned presentation guard did not process its qualification messages')
        visibility=env.snapshot.extra.get('native_visibility') or {}
        if visibility.get('on_screen_visible_checkpoints',0)!=0:
            raise RuntimeError('The hidden interface qualification appeared on screen')
        write(args.output/'report.json',dict(status='passed',utc=utc_now(),source_runtime=str(ROOT),
            source_manifest_sha256=catalogue['source_manifest_sha256'],actual_native_identity=identity,
            qualification_input_ticks=8,real_Direct3D=True,headless_fake_renderer=False,
            native_pause_resume_exercised=False,focus_loss_and_keyboard_guards_exercised=True,
            original_worlds_before=worlds_before,original_worlds_max_observed=worlds_before+1,
            desktop_input=False,native_policy_trial=False,neural_inference_calls=0,optimizer_updates=0,
            foreground_before=foreground_before,foreground_after=int(user.GetForegroundWindow() or 0),
            native_visibility=visibility,presentation_guard=guard,
            observation_shapes={k:list(v.shape) for k,v in observation.items()}))
        print('Real-renderer focus/input guards passed; no model or game-resource update')
    finally:
        env.close()
        live=False
        if identity:
            from fly_touhou.thrl_control import alive
            live=alive(identity)
        write(args.output/'cleanup.json',dict(utc=utc_now(),owned_child_exited=not live,other_jobs_modified=False))
        if live:raise RuntimeError('The bounded menu diagnostic did not close its owned original process')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--catalogue',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
