"""One owned TH14 startup and short native-input smoke from an explicit runtime.

Run only after authorization to launch the original game. No training, desktop
input, activation, desktop switching or automatic startup retries. Pair with
observe_th14_focus.py for independent transient-foreground event evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback


def snapshot_record(state):
    return {'frame':state.frame,'stage_frame':state.stage_frame,'stage':state.stage,
        'lives':state.lives,'player_present':state.player is not None,
        'execution':state.extra.get('native_execution'),
        'visibility':state.extra.get('native_visibility'),'native_focus':state.extra.get('native_focus'),
        'native_desktop_isolation':state.extra.get('native_desktop_isolation')}


def check_native_state(state,*,expected_native=None):
    data=snapshot_record(state);execution=data['execution'] or {};focus=data['native_focus'] or {}
    fake=execution.get('render_backend')=='fake-com-v1'
    if fake:
        native_hash=execution.get('fake_d3d9_sha256','')
        native_path=execution.get('fake_d3d9_path','')
        if (execution.get('schema')!='th14-fake-com-uncapped-v1'
                or execution.get('requested_device')!='host-memory-fake-com'
                or execution.get('fake_d3d9_loaded') is not True
                or execution.get('system_create_device_called') is not False
                or execution.get('unsupported_getters')!=0
                or not execution.get('device_created') or not execution.get('window_changes_disabled')
                or len(native_hash)!=64 or any(c not in '0123456789abcdef' for c in native_hash)
                or not native_path or not Path(native_path).is_absolute()):
            raise RuntimeError('No confirmed software D3D9 COM device and binary identity')
        if expected_native and (Path(native_path).resolve()!=Path(expected_native['path']).resolve()
                or native_hash!=expected_native['sha256']):
            raise RuntimeError('Loaded software D3D9 binary differs from the selected runtime')
        if not data['native_desktop_isolation']:
            raise RuntimeError('Software D3D9 qualification requires private desktop isolation')
    else:
        # Preserve the old NULLREF report contract; never reinterpret its flags
        # or a new/unknown backend as evidence for the software COM backend.
        if execution.get('render_backend') not in (None,'nullref-v1') or execution.get('schema') not in (None,'th14-nullref-uncapped-v1'):
            raise RuntimeError('Unknown native rendering audit schema')
        if not execution.get('device_created') or not execution.get('window_changes_disabled'):
            raise RuntimeError('No confirmed NULLREF device with window changes disabled')
        if execution.get('behavior_flags',0)&0x800==0:
            raise RuntimeError('D3DCREATE_NOWINDOWCHANGES not recorded in actual CreateDevice flags')
    if focus.get('schema')!='th14-no-activation-v1' or not focus.get('foreground_samples'):
        raise RuntimeError('Missing native foreground diagnostics')
    isolation=data['native_desktop_isolation']
    if isolation:
        name=isolation.get('desktop_name')
        if (not name or isolation.get('main_thread_desktop')!=name or not isolation.get('resumed')
                or isolation.get('input_desktop_before_resume')==name or isolation.get('desktop_switched')
                or isolation.get('default_desktop_fallback')
                or focus.get('isolated_desktop_name')!=name):
            raise RuntimeError('Private non-input desktop identity/isolation was not confirmed')
        # Activity within the private desktop is legitimate. Only an independent
        # event observer on the user's input desktop can establish no leakage.
    elif any(focus.get(name,0) for name in ('owned_foreground_samples','activation_messages',
                                           'focus_messages','app_activation_messages')):
        raise RuntimeError('Owned game reported activation/focus evidence; consult external observer')
    if not isolation and (data['visibility'] or {}).get('on_screen_visible_checkpoints',0):
        raise RuntimeError('Owned game appeared on screen')
    return data


def probe(controller,*,decisions,identity_reader,same_identity,cleanup_seconds=5.,expected_native=None):
    """Reuse native controller lifecycle; dependencies permit no-game unit tests."""
    report={'schema':'th14-focus-candidate-smoke-v1','status':'started',
        'desktop_input':False,'training_started':False,'automatic_startup_retries':False,
        'independent_foreground_event_observer_required':True,'trace':[]}
    identity=None;started=time.monotonic()
    try:
        controller.start()
        identity=dict(controller._spawn_identity)
        report['owned_identity']=identity
        if not identity.get('identity_complete'):raise RuntimeError('No owned native process identity')
        first=controller.ensure_run(3)
        report['initial']=check_native_state(first,expected_native=expected_native)
        report['startup_seconds']=time.monotonic()-started
        for index in range(decisions):
            state=controller.advance(1|8,4)  # original SHOT+SLOW; process-local only
            data=snapshot_record(state);data.update(index=index,action_mask=9,native_ticks=4)
            report['trace'].append(data)
            check_native_state(state,expected_native=expected_native)
            if state.frame!=first.frame+4*(index+1):raise RuntimeError('Native frame progression differs from requested ticks')
        report.update(status='passed_runtime_checks',decisions=decisions,
            frames_advanced=state.frame-first.frame,final=snapshot_record(state))
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        checkpoint=getattr(controller,'_checkpoint',None) or {}
        report['failure_checkpoint_focus']=checkpoint.get('native_focus')
        report['failure_execution']=getattr(controller,'_execution_audit',None)
    finally:
        identity=identity or getattr(controller,'_spawn_identity',None)
        if identity:report['owned_identity']=dict(identity)
        try:controller.close()
        except BaseException as error:
            report.update(status='failed',cleanup_error=repr(error))
        if identity and identity.get('identity_complete'):
            deadline=time.monotonic()+cleanup_seconds
            while True:
                current=identity_reader(identity['pid'])
                gone=(current.get('identity_complete') and not same_identity(identity,current)) or (
                    not current.get('identity_complete') and 'NoSuchProcess' in current.get('error',''))
                if gone or time.monotonic()>=deadline:break
                time.sleep(.05)
            report['cleanup']={'owned_identity':identity,'observed_identity_after_close':current,
                               'exact_owned_exit_confirmed':gone}
            if not gone:report.update(status='failed',cleanup_error='Owned native exit could not be confirmed')
        else:
            report.update(status='failed',cleanup={'exact_owned_exit_confirmed':False,
                'note':'No complete owned identity was available; no unrelated PID was terminated.'})
        report['elapsed_seconds']=time.monotonic()-started
    return report


def source_hashes(runtime):
    names=('src/fly_touhou/responsive_controller.py','src/fly_touhou/isolated_desktop_process.py',
        'src/fly_touhou/thrl_native_events.py','src/fly_touhou/hooks/th14_responsive.js',
        'src/fly_touhou/hooks/th14_headless.js','native/th14_fake_d3d9.dll','native/th14_fake_d3d9.dll.sha256')
    return {name:hashlib.sha256((runtime/name).read_bytes()).hexdigest() for name in names if (runtime/name).exists()}


def load_controller(runtime,*,workspace_source=False):
    runtime=Path(runtime).resolve()
    if not workspace_source:
        manifest=json.loads((runtime/'source_manifest.json').read_text('utf-8'))
        for row in manifest['files']:
            path=(runtime/row['path']).resolve()
            if not path.is_relative_to(runtime) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Frozen runtime file identity mismatch: '+row['path'])
    if any(name=='fly_touhou' or name.startswith('fly_touhou.') for name in sys.modules):
        raise RuntimeError('Use a fresh Python process to prevent mixed runtime imports')
    sys.path.insert(0,str(runtime/'src'))
    module=importlib.import_module('fly_touhou.thrl_native_events')
    vector=importlib.import_module('fly_touhou.thrl_vector')
    for name,loaded in sys.modules.copy().items():
        if name.startswith('fly_touhou.') and getattr(loaded,'__file__',None):
            if not Path(loaded.__file__).resolve().is_relative_to(runtime/'src'):
                raise RuntimeError('Mixed source runtime import: '+name)
    class OneAttemptController(module.HiddenTHRLDialogueGeometryController):
        def _startup_retryable(self,error):return False
    return OneAttemptController,vector._identity,vector._same_identity


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--decisions',type=int,default=16)
    parser.add_argument('--workspace-source',action='store_true',help='Explicit bounded qualification of unfrozen workspace sources')
    args=parser.parse_args()
    if not 1<=args.decisions<=256:parser.error('Use 1..256 bounded decisions')
    if args.output.exists():raise FileExistsError('Keep earlier qualification results')
    runtime=args.runtime.resolve();factory,identity_reader,same_identity=load_controller(runtime,workspace_source=args.workspace_source)
    hashes=source_hashes(runtime)
    dll=runtime/'native/th14_fake_d3d9.dll'
    expected_native=({'path':str(dll),'sha256':hashes['native/th14_fake_d3d9.dll']}
        if dll.exists() else None)
    args.output.mkdir(parents=True,exist_ok=False)
    controller=factory(args.executable,runtime/'profiles/th14_1.00b_original.json',
        private_appdata=args.output/'userdata',worker_name='focus-candidate-single-smoke',
        timeout=25,headless=True,binary_reader=True,native_seed=12345)
    result=probe(controller,decisions=args.decisions,identity_reader=identity_reader,same_identity=same_identity,
        expected_native=expected_native)
    result.update(runtime=str(runtime),workspace_source=args.workspace_source,
        runtime_manifest_sha256=(None if args.workspace_source else hashlib.sha256((runtime/'source_manifest.json').read_bytes()).hexdigest()),
        source_hashes=hashes,
        controller_source=str(Path(sys.modules['fly_touhou.thrl_native_events'].__file__).resolve()))
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({key:result.get(key) for key in ('status','owned_identity','decisions','frames_advanced','cleanup')},ensure_ascii=False))
    if result['status']!='passed_runtime_checks':raise SystemExit(1)


if __name__=='__main__':main()
