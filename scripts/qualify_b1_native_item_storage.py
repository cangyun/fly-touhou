"""One idle-only original-world same-checkpoint item-reader qualification.

Replays a sealed original19 action prefix, compares two readers at paused
checkpoints, and closes its owned world. It is not a game/skill evaluation.
No training, CUDA model, focus/keyboard/mouse operation or frame-rule change.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,statistics,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_readers(legacy,candidate,fingerprint,*,repeats=3,clock=time.perf_counter):
    before=fingerprint();samples={'legacy':[],'candidate':[]};reference=None
    for iteration in range(repeats):
        for name,reader in ([('legacy',legacy),('candidate',candidate)] if iteration%2 else [('candidate',candidate),('legacy',legacy)]):
            started=clock();value=reader._resource_items();samples[name].append(clock()-started)
            serialized=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
            if reference is None:reference=serialized
            elif serialized!=reference:raise ValueError('Same-checkpoint item extraction differs')
    if fingerprint()!=before:raise ValueError('Native checkpoint/frame/RNG changed during read qualification')
    return dict(native_item_count=value[1],decoded_items=len(value[0]),unknown_kinds=value[2],
        item_results_exact=True,checkpoint_frame_and_RNG_unchanged=True,
        read_median_seconds={k:statistics.median(v) for k,v in samples.items()})


def run(a):
    import psutil
    from fly_touhou.thrl_control import alive
    from fly_touhou.thrl_full_story_env import THRLFullStoryEnv
    from fly_touhou.thrl_job_paths import configure_native_temp
    from train_thrl_native import make_env_kwargs
    from probe_checkpoint_snapshot_reuse import HardDeadline
    plan=read(a.plan);out=Path(plan['output'])
    if Path(plan['source_runtime']).resolve()!=ROOT.resolve() or sha(ROOT/'source_manifest.json')!=plan['source_manifest_sha256']:
        raise ValueError('Declared immutable qualification runtime changed')
    # Fail before launch rather than overlap any current evaluation/training.
    status=read(Path(plan['after_control'])/'status.json')
    parent=Path(plan['after_output']);report=read(parent/'report.json')
    if (status.get('status')!='finished' or status.get('exit_code')!=0 or alive(plan['after_identity'])
            or report.get('status')!='completed' or report.get('native_games')!=20 or report.get('reused_native_games')!=20
            or report.get('pins',{}).get('greedy',{}).get('checkpoint_sha256')!=plan['checkpoint_sha256']
            or report.get('pins',{}).get('greedy',{}).get('action_selection')!='greedy'
            or read(parent/'greedy/native_cleanup.json').get('close_errors')!=[]):
        raise RuntimeError('Exact greedy predecessor must be completed, exited and closed before item qualification')
    if any('th14' in (p.info['name'] or '').lower() for p in psutil.process_iter(['name'])):
        raise RuntimeError('Other original worlds live;no qualification world started')
    if sha(plan['environment_config'])!=plan['environment_config_sha256']:raise ValueError('Environment pin differs')
    if sha(plan['action_receipt'])!=plan['action_receipt_sha256']:raise ValueError('Action receipt pin differs')
    receipt=read(plan['action_receipt']);tape=Path(receipt['action_tape']['path'])
    if (sha(tape)!=receipt['action_tape']['sha256'] or receipt.get('schema')!='B1-full19-native-observed-event-tail-v1'
            or receipt.get('checkpoint_sha256')!=plan['action_checkpoint_sha256']
            or len(tape.read_bytes())!=receipt['acknowledged_decisions']):raise ValueError('Sealed original action prefix differs')
    baseline=Path(plan['baseline_runtime'])/'src/fly_touhou/resources.py'
    if sha(baseline)!=plan['baseline_resource_sha256']:raise ValueError('Legacy extraction source pin differs')
    spec=importlib.util.spec_from_file_location('fly_touhou._native_item_legacy',baseline)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg=read(plan['environment_config']);out.mkdir(parents=True,exist_ok=False);configure_native_temp(out)
    write(out/'launch.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()))
    os.environ['FLY_TOUHOU_DESKTOP_NAME']='fly-touhou-job-item-read-qualification-'+str(os.getpid())
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    env=None;owned=None;samples=[];started=time.monotonic();failure=None
    timer=HardDeadline(240,out/'hard_deadline.json');timer.start()
    def stop():
        if (out/'STOP').exists() or time.monotonic()-started>180:raise RuntimeError('Qualification stop or work deadline')
    try:
        kwargs=make_env_kwargs(cfg,out,1,'test')[0]
        if kwargs.get('wall_time_limit_seconds') is not None or 'online_teacher_revision' in kwargs:raise ValueError('Unexpected teacher/horizon')
        env=THRLFullStoryEnv(**kwargs);_,info=env.reset(seed=receipt['evaluation_seed'])
        c=env.controller;owned=dict(pid=c.pid,create_time=psutil.Process(c.pid).create_time())
        if info['full_story_native_seed']!=receipt['native_seed']:raise ValueError('Native seed differs from acknowledged prefix')
        legacy=object.__new__(old.ResourceNativeReader);legacy.mem=c._memory;legacy.o=c.reader.o
        if not callable(getattr(c._memory,'read_into',None)):raise ValueError('Candidate native buffer method not deployed')
        def fingerprint():
            return (c.pid,c._native_callback_generation,c._checkpoint['sequence'],c.reader.global_frame(),
                c._memory.read(c._memory.base+0xdb508,16).hex())
        actions=tape.read_bytes()[:1024]
        for position,action in enumerate(actions):
            stop()
            if position%16==0:
                row=compare_readers(legacy,c.reader,fingerprint)
                samples.append(dict(row,decision=position,checkpoint=c._checkpoint['sequence']))
            _,_,terminated,truncated,_=env.step(action)
            if truncated:raise ValueError('Unexpected native truncation')
            if terminated:break
        if len(samples)<32 or not any(row['native_item_count']>0 for row in samples):
            raise ValueError('Insufficient real active-item checkpoints for qualification')
    except BaseException:
        failure=traceback.format_exc()
    finally:
        errors=[]
        if env is not None:
            try:env.close()
            except BaseException as error:errors.append(repr(error))
        if owned is not None:
            for _ in range(20):
                if not alive(owned):break
                time.sleep(.1)
            if alive(owned):errors.append('Owned original process identity still live after close')
        write(out/'cleanup.json',dict(closed=not errors,close_errors=errors,owned_identity=owned))
        timer.finish()
    if failure or errors:
        write(out/'failure.json',dict(traceback=failure,close_errors=errors,partial_results_are_not_qualification=True))
        raise RuntimeError('Native item qualification failed;see preserved failure/cleanup')
    active=[row for row in samples if row['native_item_count']>0]
    result=dict(schema='B1-native-same-checkpoint-item-storage-qualification-v1',status='qualified-native-reading-only',
        utc=utc_now(),samples=samples,nonempty_checkpoints=len(active),native_worlds=1,CUDA_model_initialized=False,
        PPO_updates=0,skill_games_added=0,exact_field_results=True,checkpoint_frame_and_RNG_unchanged=True,
        median_active_read_seconds={name:statistics.median(row['read_median_seconds'][name] for row in active) for name in ['legacy','candidate']},
        complete_game_throughput_not_measured=True,plan=plan,source_manifest_sha256=sha(ROOT/'source_manifest.json'))
    write(out/'report.json',result)
    print(json.dumps({k:result[k] for k in ['status','nonempty_checkpoints','median_active_read_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);run(p.parse_args())
