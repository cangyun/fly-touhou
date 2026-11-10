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


def capture(args):
    config,folder,tasks,split,seed,maximum=args
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
    env=factory(options);history=deque(maxlen=64);entries=[];begun=time.monotonic()
    try:
        obs,info=env.reset();record=env._frozen_prefix
        for decision in range(maximum):
            s=env.snapshot
            for task in tasks:
                if task in [r['task'] for r in entries]:continue
                if (s.stage==task['stage'] and attack_key(current_attack(s))==attack_key(task['attack'])
                        and s.player.state==1 and s.player.invulnerable_frames==0 and len(history)>=16):
                    # Do not put stage-loading or uneven input intervals inside
                    # recurrent burn-in. Prefix replay handles earlier loading.
                    usable=[]
                    for row in reversed(history):
                        if row['snapshot'].stage!=s.stage or sum(t for _,t in row['commands'])!=4:break
                        usable.append(row)
                    usable=list(reversed(usable))
                    if len(usable)<16:continue
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
            if len(entries)==len(tasks):break
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
        write_json(folder/'result.json',dict(entries=entries,elapsed_seconds=time.monotonic()-begun,network_training=False))
        return entries
    except Exception as error:
        write_json(folder/'failure.json',dict(error=repr(error),network_training=False))
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
    a=p.parse_args();config=json.loads(a.config.read_text(encoding='utf-8-sig'));config.pop('nbnm',None)
    task=dict(mode='attack',stage=a.stage,attack=a.attack)
    a.output.mkdir(parents=True,exist_ok=True)
    jobs=[(config,str(a.output/f'{split}-{i:02d}'),[task],split,a.seed+1009*i+(0 if split=='train' else 200000),a.max_decisions)
        for split,count in [('train',a.train_seeds),('development',a.development_seeds)] for i in range(count)]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from fly_touhou.nbnm_prefix import SCHEMA
    from fly_touhou.thrl_durable_records import write_json
    entries=[];failures=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(capture,job) for job in jobs]
        for future in as_completed(futures):
            try:entries.extend(future.result())
            except Exception as error:failures.append(repr(error))
            write_json(a.output/'manifest.json',dict(schema=SCHEMA,entries=entries,network_training=False,failures=failures))
            print(dict(entries=len(entries),failed_jobs=len(failures)),flush=True)
    if len(entries)!=len(jobs):raise RuntimeError('Not all requested seed entries reached the task; inspect per-prefix results')


if __name__=='__main__':main()
