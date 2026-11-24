"""Exercise the real NBNM adapter in native Spell Practice without training."""
from pathlib import Path
import argparse,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher',choices=('geometry','resource','idle'),default='resource')
    p.add_argument('--seed',type=int,default=4747)
    p.add_argument('--repeat',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-spell-adapter-'+str(os.getpid()))
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    from fly_touhou.nbnm_run import read,environment
    from fly_touhou.compact_full_story import factory,story_kwargs,CompactFullStoryEnv
    from fly_touhou.thrl_durable_records import write_json
    configure_generated_caches();temporary=configure_native_temp(a.output)
    spec=read(a.config);cfg=environment(spec,evaluation=True)
    env=None;reports=[];entries=[];started=time.monotonic()
    try:
        options=story_kwargs(cfg,a.output,1,'development',teacher=False)[0]
        if a.teacher!='idle':
            from fly_touhou.resource_teacher_v10 import REVISION
            options['online_teacher_revision']=REVISION
        if a.teacher=='resource':options['full_native_teacher_support']=True
        env=factory(options)
        if a.teacher=='resource':
            # Same strong existing rule teacher; probe only. Never calls PPO.
            import types
            env._label_current=types.MethodType(CompactFullStoryEnv._label_current,env)
        for episode in range(a.repeat):
            obs,info=env.reset(options=dict(native_seed=a.seed))
            entries.append(dict(features=obs['features'].tolist(),
                stage=env.snapshot.stage,stage_frame=env.snapshot.stage_frame,
                run_mode=env.snapshot.extra['run_mode'],native_seed=info['native_seed'],
                native_entry=info['native_spell_practice'],Bombs=env.snapshot.bombs))
            write_json(a.output/'entries.json',entries)
            for decision in range(5000):
                if a.teacher=='idle':action=0
                else:
                    label=info['teacher_label']
                    action=max(range(18),key=lambda k:label['probabilities'][k])
                obs,reward,done,truncated,info=env.step(action)
                if decision%128==0:
                    write_json(a.output/'progress.json',dict(episode=episode,decision=decision,
                        stage_frame=env.snapshot.stage_frame,elapsed_seconds=time.monotonic()-started,
                        network_training=False))
                if done:
                    reports.append(info['episode_summary']);break
                if truncated:raise RuntimeError('Unexpected practice truncation')
            else:raise RuntimeError('Qualification limit reached without native terminal')
            write_json(a.output/'report.json',dict(episodes=reports,network_training=False,
                repeat_entry_features_identical=all(r['features']==entries[0]['features'] for r in entries),
                elapsed_seconds=time.monotonic()-started))
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc()));raise
    finally:
        if env:env.close()
        cleanup_native_temp(temporary)


if __name__=='__main__':main()
