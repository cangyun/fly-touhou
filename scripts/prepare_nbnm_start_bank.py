"""Collect real attack entries with the rule teacher; never train a network.

Snapshots are rebuilt by original input replay. No external game's memory or
resources are copied. A short recent prefix is replayed with current sensory
code and current neural weights when training later starts.
"""
from pathlib import Path
import argparse, json, sys, time
from collections import deque
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


class FixedNativeSeed:
    """Select a declared native opening through the existing reset path."""
    def __init__(self,seed):self.seed=int(seed)
    def integers(self,*args,**kwargs):return self.seed


def continued_record(template,receipt,ticks):
    from copy import deepcopy
    result=deepcopy(template);shift=0
    for transition,actual in zip(result['transitions'],receipt['barriers'],strict=True):
        transition['before']['ticks']+=shift
        shift+=actual['actual_ticks']-actual['recorded_ticks']
        transition['after']['ticks']+=shift
    result.update(ticks=ticks,status='recording',raw_input_segments=[])
    return result


def capture(args):
    config,folder,tasks,split,seed,maximum=args[:6]
    resume=args[6] if len(args)>6 else None
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.compact_frozen_prefix import start
    from fly_touhou.nbnm_sensation import current_attack
    from fly_touhou.nbnm_task import attack_key
    from fly_touhou.native_replay import semantic_state
    from fly_touhou.thrl_durable_records import write_json
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    configure_generated_caches();temporary=configure_native_temp(folder)
    config=dict(config,seed=seed)
    options=story_kwargs(config,folder,1,split,teacher=True)[0]
    # make_env_kwargs normally disables online labels in development. Here
    # the teacher only creates the prefix; no development network is trained.
    from fly_touhou.resource_teacher_v10 import REVISION
    options['online_teacher_revision']=REVISION
    options.pop('nbnm',None)
    options['stage6_prefix_capture']=str(folder/'full-prefix.json')
    env=factory(options);history=deque(maxlen=64);entries=[];begun=time.monotonic();seen=set();skipped=[]
    try:
        if resume is not None:env._full_story_rng=FixedNativeSeed(resume['native_seed'])
        obs,info=env.reset();record=env._frozen_prefix
        if resume is not None:
            from fly_touhou.nbnm_prefix import restore_entry
            template=json.loads(Path(resume['prefix']).read_text(encoding='utf-8-sig'))
            obs,info,_=restore_entry(env,resume)
            record=continued_record(template,env._nbnm_entry_receipt,env._native_input_ticks)
            env._frozen_prefix=record
            env._teacher.reset();info=env._label_current(env._info())
        for decision in range(maximum):
            s=env.snapshot
            for task in tasks:
                if task in [r['task'] for r in entries]:continue
                key=(task['stage'],task['attack'])
                if key in seen:continue
                if s.stage==task['stage'] and attack_key(current_attack(s))==attack_key(task['attack']):
                    seen.add(key)
                    if s.player.state!=1 or s.player.invulnerable_frames or len(history)<16:
                        skipped.append(dict(task=task,reason='first attack observation was protected/uncontrollable or lacked warm-up'))
                        continue
                    # Do not put stage-loading or uneven input intervals inside
                    # recurrent burn-in. Prefix replay handles earlier loading.
                    usable=[]
                    for row in reversed(history):
                        if row['snapshot'].stage!=s.stage or sum(t for _,t in row['commands'])!=4:break
                        usable.append(row)
                    usable=list(reversed(usable))
                    if len(usable)<16:
                        skipped.append(dict(task=task,reason='loading inside recurrent warm-up'));continue
                    first=usable[0]
                    name=f's{s.stage}-{task["attack"]}'
                    prefix_path=folder/(name+'-prefix.json')
                    prefix=dict(record,schema='frozen-SNN-actual-prefix-commands-v1',status='nbnm_warmup',
                        raw_input_segments=list(env.controller._input_tape[:first['segment']]),
                        ticks=first['ticks'],landing=semantic_state(first['snapshot']),
                        transitions=[r for r in record['transitions'] if r['after']['ticks']<=first['ticks']],
                        native_events={k:first['snapshot'].extra['thrl_events'][k] for k in ('misses','bombs_used','point_items')},
                        teacher_actions=first['decision'],optimizer_updates=0,
                        prefix_was_played_live_by_frozen_network=False)
                    write_json(prefix_path,prefix)
                    entries.append(dict(task=task,split=split,native_seed=env.native_seed,prefix=str(prefix_path),
                        warmup_commands=[dict(action=r['action'],commands=r['commands']) for r in usable],
                        landing_stage_frame=s.stage_frame,landing_power=s.power,landing_x=s.player.x,landing_y=s.player.y,
                        prefix_Misses=s.extra['thrl_events']['misses'],prefix_Bombs=s.extra['thrl_events']['bombs_used']))
                    write_json(folder/'entries.json',entries)
            if len(seen)==len(tasks):break
            before=dict(snapshot=s,segment=len(env.controller._input_tape),ticks=record['ticks'],decision=decision)
            action=int(info['teacher_label']['action'])
            obs,reward,done,cut,info=env.step(action)
            before.update(action=action,commands=list(env.controller._input_tape[before['segment']:]))
            history.append(before)
            if decision%128==0:
                write_json(folder/'progress.json',dict(phase='preparing_native_prefixes',decisions=decision,
                    stage=env.snapshot.stage,stage_frame=env.snapshot.stage_frame,attack=current_attack(env.snapshot),
                    entries=len(entries),elapsed_seconds=time.monotonic()-begun,network_training=False))
            if done:break
        write_json(folder/'result.json',dict(entries=entries,skipped=skipped,elapsed_seconds=time.monotonic()-begun,network_training=False))
        return entries
    except Exception as error:
        import traceback
        write_json(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),network_training=False))
        raise
    finally:
        env.close();cleanup_native_temp(temporary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',type=int,default=3);p.add_argument('--attack',default='BossCard1')
    p.add_argument('--train-seeds',type=int,default=4);p.add_argument('--development-seeds',type=int,default=2)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--max-decisions',type=int,default=40000)
    p.add_argument('--seed',type=int,default=2026110901)
    p.add_argument('--resume-bank',type=Path,help='Extend existing earlier attack entries using real input replay')
    p.add_argument('--tasks',type=Path,help='JSON array of native attack tasks to capture during each run')
    p.add_argument('--allow-partial-bank',action='store_true',help='Keep only whole, unprotected attack entries; report unavailable tasks')
    p.add_argument('--log-file',type=Path)
    a=p.parse_args()
    if a.log_file:
        a.log_file.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout=sys.stderr=a.log_file.open('a',encoding='utf-8',buffering=1)
    config=json.loads(a.config.read_text(encoding='utf-8-sig'));config.pop('nbnm',None)
    task=dict(mode='attack',stage=a.stage,attack=a.attack)
    tasks=json.loads(a.tasks.read_text(encoding='utf-8-sig')) if a.tasks else [task]
    a.output.mkdir(parents=True,exist_ok=True)
    jobs=[(config,str(a.output/f'{split}-{i:02d}'),tasks,split,a.seed+1009*i+(0 if split=='train' else 200000),a.max_decisions)
        for split,count in [('train',a.train_seeds),('development',a.development_seeds)] for i in range(count)]
    if a.resume_bank:
        bank=json.loads(a.resume_bank.read_text(encoding='utf-8-sig'));jobs=[]
        for i,entry in enumerate(bank['entries']):
            entry=dict(entry);path=Path(entry['prefix'])
            if not path.is_absolute():path=a.resume_bank.parent/path
            entry['prefix']=str(path.resolve())
            jobs.append((config,str(a.output/f'{entry["split"]}-{entry["native_seed"]}'),tasks,
                entry['split'],a.seed+1009*i,a.max_decisions,entry))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from fly_touhou.nbnm_prefix import SCHEMA
    from fly_touhou.thrl_durable_records import write_json
    import psutil
    process=psutil.Process();begun=time.monotonic()
    entries=[];failures=[];completed=0
    def progress(phase):
        write_json(a.output/'progress.json',dict(phase=phase,PID=process.pid,create_time=process.create_time(),
            completed_jobs=completed,total_jobs=len(jobs),entries=len(entries),failed_jobs=len(failures),
            elapsed_seconds=time.monotonic()-begun,network_training=False))
    progress('preparing_native_prefixes')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(capture,job) for job in jobs]
        for future in as_completed(futures):
            try:entries.extend(future.result())
            except Exception as error:failures.append(repr(error))
            completed+=1
            write_json(a.output/'manifest.json',dict(schema=SCHEMA,entries=entries,network_training=False,
                expected_entries=len(jobs)*len(tasks),failures=failures))
            print(dict(entries=len(entries),failed_jobs=len(failures)),flush=True)
            progress('preparing_native_prefixes')
    progress('complete' if not failures else 'complete_with_failures')
    if len(entries)!=len(jobs)*len(tasks) and not a.allow_partial_bank:
        raise RuntimeError('Not all requested seed entries reached the task unprotected; inspect per-prefix results')


if __name__=='__main__':main()
