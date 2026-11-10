"""Capture actual attack starts inside native Stage Practice, without training.

The prefix controller may use its finite native resources; none of its data
are student PPO/NBNM evidence. Training begins unprotected at the measured
attack. Every prefix starts in the original stage menu, not an ECL warp.
"""
from pathlib import Path
import argparse,json,os,sys,time,traceback
from collections import deque
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def capture(job):
    spec,stage,seed,split,folder=job;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-nbnm-stage-bank-'+str(os.getpid()))
    from fly_touhou.nbnm_run import read,environment
    from fly_touhou.compact_full_story import CompactFullStoryEnv,story_kwargs
    from fly_touhou.thrl_native_env import THRLNativeEnv
    from fly_touhou.nbnm_stage_practice import bind_controller
    from fly_touhou.nbnm_spell_practice import copy_score
    from fly_touhou.nbnm_task import attack_key
    from fly_touhou.nbnm_sensation import current_attack
    from fly_touhou.compact_frozen_prefix import start
    from fly_touhou.native_replay import semantic_state
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    configure_generated_caches();tmp=configure_native_temp(folder)

    class PrefixEnv(CompactFullStoryEnv):
        def _create_controller(self):return bind_controller(super()._create_controller(),dict(stage=stage))
        def _validate_initial_run(self):self._verify_native_run(self.snapshot)
        def _verify_native_run(self,s):
            if s.extra.get('run_mode')!=1 or s.stage!=stage or s.difficulty!=3 or s.character!=0 or s.subshot!=0:
                raise ValueError('Expected original Lunatic ReimuA Stage Practice')
        def _native_end(self,s):
            if s.extra.get('native_gameover') or s.lives<0:return 1
            # Diagnostic exit only, not a training success classification.
            if s.pause==1 or s.screen!=7:return 2
            return 0

    cfg=environment(spec,evaluation=True);cfg.pop('nbnm',None)
    options=story_kwargs(cfg,folder,1,split,teacher=True)[0];options.pop('online_teacher_revision',None)
    from fly_touhou.resource_teacher_v10 import REVISION
    env=PrefixEnv(**options,teacher_revision=REVISION)
    env.native_seed=seed;env.private_appdata=folder/'appdata';copy_score(spec['nbnm']['score_template'],env.private_appdata)
    entries=[];history=deque(maxlen=64);seen=set();phases=[];last=None;begun=time.monotonic();first_seen={}
    requested={attack_key(x) for x in spec.get('stage_bank_stop_after_attacks',[])}
    try:
        obs,info=THRLNativeEnv.reset(env)
        env.stage6_prefix_capture=str(folder/'full-prefix.json');start(env)
        # Suppress the stage6-only recorder's automatic stop; record every
        # actual input ourselves throughout this one native practice stage.
        env._frozen_prefix['status']='stage-bank-recording'
        info=env._label_current(env._info());record=env._frozen_prefix
        for decision in range(18000):
            s=env.snapshot;name=current_attack(s);key=attack_key(name)
            if key!=last:
                phases.append(dict(attack=name,stage_frame=s.stage_frame,frame=s.frame,player_state=s.player.state,
                    protected=s.player.invulnerable_frames));last=key
                write_json(folder/'phases.json',phases)
            if key is not None:first_seen.setdefault(key,s.stage_frame)
            if key is not None and key not in seen and s.player.state==1 and not s.player.invulnerable_frames:
                usable=list(history)
                if len(usable)>=16 and all(sum(t for _,t in r['commands'])==4 for r in usable):
                    # The entry may occur after a protected introduction. Its
                    # offset is explicit; zero-offset Spell Practice remains
                    # the primary spell-start task in the catalog.
                    first=usable[0];task=dict(mode='attack',stage=stage,attack=key[0]+('Card' if key[1] else '')+str(key[2]))
                    path=folder/(task['attack']+'-prefix.json')
                    prefix=dict(schema='frozen-SNN-actual-prefix-commands-v1',status='nbnm_warmup',
                        start=record['start'],raw_input_segments=list(env.controller._input_tape[:first['segment']]),
                        transitions=[],ticks=first['ticks'],native_seed=seed,landing=semantic_state(first['snapshot']),
                        native_events={k:first['snapshot'].extra['thrl_events'][k] for k in ('misses','bombs_used','point_items')},
                        teacher_actions=first['decision'],optimizer_updates=0,prefix_was_played_live_by_frozen_network=False,
                        entry_protocol='native_stage_practice')
                    write_json(path,prefix)
                    entries.append(dict(task=task,split=split,native_seed=seed,prefix=str(path),
                        warmup_commands=[dict(action=r['action'],commands=r['commands']) for r in usable],
                        landing_stage_frame=s.stage_frame,landing_power=s.power,landing_x=s.player.x,landing_y=s.player.y,
                        attack_first_observed_stage_frame=first_seen[key],
                        target_attack_preplayed_frames=s.stage_frame-first_seen[key],
                        prefix_Misses=s.extra['thrl_events']['misses'],prefix_Bombs=s.extra['thrl_events']['bombs_used'],
                        entry_protocol='native_stage_practice'))
                    seen.add(key);write_json(folder/'entries.json',entries)
                    if requested and requested<=seen:break
            before=dict(snapshot=s,segment=len(env.controller._input_tape),ticks=env._native_input_ticks,decision=decision)
            action=int(info['teacher_label']['action'])
            obs,reward,done,cut,info=env.step(action)
            before.update(action=action,commands=list(env.controller._input_tape[before['segment']:]))
            history.append(before)
            if decision%128==0:
                write_json(folder/'progress.json',dict(decisions=decision,stage=stage,stage_frame=env.snapshot.stage_frame,
                    attack=current_attack(env.snapshot),entries=len(entries),network_training=False))
            if done:break
        s=env.snapshot
        ending=dict(screen=s.screen,pause=s.pause,lives=s.lives,bombs=s.bombs,stage_frame=s.stage_frame,
            menu_state=s.extra.get('pause_menu_state'),menu_result=s.extra.get('pause_menu_result'),
            milestones=s.extra.get('native_milestones'),native_gameover=s.extra.get('native_gameover'))
        result=dict(stage=stage,native_seed=seed,split=split,entries=entries,phases=phases,ending=ending,
            elapsed_seconds=time.monotonic()-begun,network_training=False)
        write_json(folder/'report.json',result);return result
    except BaseException:
        write_json(folder/'failure.json',dict(error=traceback.format_exc(),network_training=False));raise
    finally:env.close();cleanup_native_temp(tmp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stages',type=int,nargs='+',default=list(range(1,7)))
    p.add_argument('--train-seeds',type=int,nargs='*',default=[50000,50001])
    p.add_argument('--development-seeds',type=int,nargs='*',default=[50016])
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--retry-missing-from',type=Path,help='Reuse completed jobs and retry only missing native attempts')
    a=p.parse_args()
    from fly_touhou.nbnm_run import read
    from fly_touhou.nbnm_prefix import SCHEMA
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_training_mode import seed_split
    spec=read(a.config);a.output.mkdir(parents=True,exist_ok=True);reports=[]
    jobs=[]
    for split,seeds in [('train',a.train_seeds),('development',a.development_seeds)]:
        for seed in seeds:
            if seed_split(seed)!=split:raise ValueError('Incorrect native seed partition')
            for stage in a.stages:
                name=f's{stage}-{split}-{seed}'
                prior=a.retry_missing_from/name/'report.json' if a.retry_missing_from else None
                if prior and prior.exists():reports.append(read(prior))
                else:jobs.append((spec,stage,seed,split,str(a.output/name)))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        failures=[];futures=[pool.submit(capture,j) for j in jobs]
        for future in as_completed(futures):
            try:reports.append(future.result())
            except Exception as error:failures.append(str(error))
            write_json(a.output/'manifest.json',dict(schema=SCHEMA,entries=[e for r in reports for e in r['entries']],
                entry_protocol='native_stage_practice',network_training=False))
            write_json(a.output/'progress.json',dict(completed=len(reports),total=len(jobs),failures=failures,network_training=False))
    write_json(a.output/'report.json',dict(reports=reports,failures=failures,network_training=False))
    if failures:raise RuntimeError('Some native entry attempts failed; see retained per-job failures')


if __name__=='__main__':main()
