"""Continue an original-game live world across the opt-in EX sensor migration."""
from pathlib import Path
import argparse,sys,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.nbnm_run import runtime,read,environment,task_settings
    from fly_touhou.compact_full_story import story_kwargs,factory
    from fly_touhou.nbnm_continuation import export_world,restore_world
    from fly_touhou.nbnm_geometry_revision import REVISION
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    temp=runtime(args.output);spec=read(args.config);env=None
    def create(new):
        cfg=environment(spec,evaluation=True);cfg['nbnm']=task_settings(spec,spec['tasks']['s1_BossCard1'])
        cfg['nbnm'].update(teacher_mode='none',demonstration_route_teacher={'enabled':False},live_route_planner={'enabled':False})
        if new:
            cfg['nbnm']['native_rule_geometry']=dict(revision=REVISION,horizon=24)
            profile=read(cfg['profile']);profile.setdefault('hooks',{}).update(native_ex_program=REVISION,native_ex_program_horizon=24)
            path=args.output/'new-profile.json';write_json(path,profile);cfg['profile']=str(path)
        options=story_kwargs(cfg,args.output/('new' if new else 'old'),1,'development',teacher=False)[0]
        options.pop('online_teacher_revision',None)
        return factory(options)
    try:
        env=create(False);env.reset(options={'native_seed':60006})
        for _ in range(64):
            _,_,done,truncated,_=env.step(0)
            if done or truncated:raise RuntimeError('Probe ended before the planned continuation landing')
        capsule=export_world(env);env.close();env=None
        env=create(True);_,info=restore_world(env,capsule)
        migration=info['native_continuation']
        # The next decision also uses the new geometry, not the old cached
        # observation from the checkpoint.
        _,_,done,truncated,_=env.step(0)
        write_json(args.output/'report.json',dict(status='complete',migration=migration,
            next_frame=env.snapshot.frame,next_decision_terminal=bool(done or truncated),
            training_started=False,optimizer_updates=0))
    except BaseException:
        write_json(args.output/'failure.json',dict(error=traceback.format_exc()));raise
    finally:
        if env is not None:env.close()
        cleanup_native_temp(temp)


if __name__=='__main__':main()
