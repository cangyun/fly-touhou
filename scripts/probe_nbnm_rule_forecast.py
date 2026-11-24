"""Small native emission comparison. Does not train or alter any saved policy."""
from pathlib import Path
import argparse,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decisions',type=int,default=240);p.add_argument('--horizon',type=int,default=24)
    p.add_argument('--task',default=None,help='Declared native practice task; default original full-story opening')
    p.add_argument('--future-inputs',action='store_true',help='Also exercise425-column sensing, without training')
    p.add_argument('--catalog',type=Path)
    args=p.parse_args()
    from fly_touhou.nbnm_run import read,runtime,environment,task_settings
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.native_shot_rules import forecast_emissions,capture_enemy_states
    import numpy as np
    args.output.mkdir(parents=True,exist_ok=False);temp=runtime(args.output);env=None
    spec=read(args.config);cfg=environment(spec,evaluation=True)
    if args.task:cfg['nbnm']=task_settings(spec,spec['tasks'][args.task])
    else:
        cfg['nbnm'].update(task={'mode':'full_story'})
        cfg['nbnm'].pop('entry_mode',None)
    cfg['nbnm'].update(route_prior={'enabled':False},future_rules={'enabled':False})
    if args.future_inputs:
        if args.catalog is None:p.error('--future-inputs requires --catalog')
        cfg['nbnm']['future_rules']=dict(enabled=True,catalog=str(args.catalog),horizon=args.horizon)
    started=time.monotonic();queries=0;predicted=0;unknown={};births=[];all_predictions=[];known_ids=set()
    try:
        env=factory(story_kwargs(cfg,args.output/'native',1,'train',teacher=False)[0])
        obs,info=env.reset(options={'native_seed':50000})
        sensor_shape=list(obs['features'].shape)
        for decision in range(args.decisions):
            s=env.snapshot;r=s.extra.get('resources',{})
            for b in s.extra.get('hazard_metadata',{}).get('bullets',[]):
                if b['id'] not in known_ids:
                    births.append(dict(frame=s.frame,stage_frame=s.stage_frame,**b))
                known_ids.add(b['id'])
            if decision%4==0:
                states=capture_enemy_states(env.controller.reader,s)
                path=np.tile([s.player.x,s.player.y],(1,args.horizon+1,1));rng=s.extra['native_rng_state']
                began=time.monotonic()
                result=forecast_emissions(env.controller.reader.mem,states,path,seed=rng['gameplay_seed'],
                    counter=rng['gameplay_counter'],horizon=args.horizon)[0]
                queries+=1;predicted+=len(result['events'])
                all_predictions.extend(dict(frame=s.frame+e['tick'],query_frame=s.frame,**e) for e in result['events'])
                for u in result['unknown']:unknown[u['reason']]=unknown.get(u['reason'],0)+1
                append_jsonl(args.output/'forecasts.jsonl',dict(frame=s.frame,stage_frame=s.stage_frame,
                    enemies=len(states),seconds=time.monotonic()-began,**result))
                write_json(args.output/'progress.json',dict(decision=decision,frame=s.frame,queries=queries,
                    predicted_emissions=predicted,unknown=unknown))
            obs,reward,done,truncated,info=env.step(0)
            if done or truncated:break
        write_json(args.output/'report.json',dict(status='complete',decisions=decision+1,queries=queries,
            predicted_emissions=predicted,unknown=unknown,seconds=time.monotonic()-started,
            native_frame=env.snapshot.frame,training_updates=0))
        report=read(args.output/'report.json');report.update(task=cfg['nbnm']['task'],sensor_shape=sensor_shape,
            future_inputs=bool(args.future_inputs),final_future=(env.encoder.last_result.get('future_rules') if args.future_inputs else None))
        write_json(args.output/'report.json',report)
        write_json(args.output/'native_births.json',births)
        write_json(args.output/'predicted_births.json',all_predictions)
    except BaseException:
        write_json(args.output/'failure.json',dict(error=traceback.format_exc()));raise
    finally:
        if env is not None:env.close()
        cleanup_native_temp(temp)


if __name__=='__main__':main()
