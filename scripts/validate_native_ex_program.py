"""Compare opt-in EX forecasts to a separate, unmodified original TH14 run."""
from pathlib import Path
from collections import Counter,defaultdict,deque
import argparse,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--decisions',type=int,default=360)
    p.add_argument('--split',choices=('train','development'),default='development')
    p.add_argument('--seed',type=int,default=60006)
    p.add_argument('--invulnerable-fixture',action='store_true',help='Physics validation only, never NBNM evidence')
    p.add_argument('--long-hints',action='store_true')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--benchmark-reader',action='store_true')
    p.add_argument('--profile-encoding',action='store_true')
    p.add_argument('--continue-after-contact',action='store_true',
        help='Physics probe only: use original lives, never report NBNM success')
    p.add_argument('--greedy-dodge',action='store_true',help='Probe motion after the first emission; no network/learning')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from fly_touhou.nbnm_run import runtime,environment,read,task_settings
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.native_bullet_program import REVISION,predict_program
    from fly_touhou.native_bullet_release import predict_bullet_release
    temp=runtime(a.output);spec=read(a.config);env=None;counts=Counter();reasons=Counter();bad=[];timings=[];task_rows=[]
    start=time.monotonic();sample_count=0;benchmarks=[];laser_checks=Counter();laser_bad=[]
    try:
        for name in a.tasks:
            cfg=environment(spec,evaluation=True);cfg['nbnm']=task_settings(spec,spec['tasks'][name])
            profile=read(cfg['profile']);profile.setdefault('hooks',{})['native_ex_program']=REVISION
            if a.long_hints:profile['hooks']['native_ex_program_horizon']=96
            profile_path=a.output/(name+'-profile.json');write_json(profile_path,profile);cfg['profile']=str(profile_path)
            cfg['nbnm'].update(demonstration_route_teacher={'enabled':False},live_route_planner={'enabled':False})
            options=story_kwargs(cfg,a.output/name,1,a.split,teacher=False)[0]
            options.pop('online_teacher_revision',None)
            env=factory(options)
            placeholder={k:np.zeros(space.shape,space.dtype) for k,space in env.observation_space.spaces.items()}
            def skip(snapshot):
                env._last_measurable_frame=snapshot.frame;env._last_measurable_observation=placeholder
                return placeholder
            original_encode=env._encode
            env._encode=skip
            obs,_=env.reset(options={'native_seed':a.seed})
            if a.invulnerable_fixture:
                import struct
                from fly_touhou.native import ProcessMemory
                with_memory=ProcessMemory(env.controller.reader.mem.pid,writable=True)
                # zTimer stores both an integer and a float clock. Updating
                # only the integer is overwritten on the next native tick.
                with_memory.write(with_memory.integer(with_memory.base+0xdb67c)+99012,struct.pack('<if',1000000,1000000.))
                with_memory.close()
            env.encoder.last_result=None;history=deque();task_samples=0;laser_history=deque()
            conditional_path=(np.repeat([[env.snapshot.player.x,env.snapshot.player.y]],97,axis=0)
                if a.invulnerable_fixture and a.continue_after_contact and not a.greedy_dodge else None)
            for decision in range(a.decisions):
                snapshot=env.snapshot;frame=snapshot.frame
                rows=snapshot.extra.get('hazard_metadata',{}).get('bullets',[])
                by_id={r['id']:r for r in rows}
                lines=snapshot.extra.get('native_line_lasers') or {}
                curves=snapshot.extra.get('native_curve_lasers') or {}
                actual_lines={r['id']:r for r in lines.get('lines',())}
                actual_curves={r['id']:r for r in curves.get('curves',())}
                for prior_frame,line_predictions,curve_predictions in laser_history:
                    delta=frame-prior_frame
                    if delta not in (4,12,24,48,96):continue
                    for raw,forecast in line_predictions:
                        current=actual_lines.get(raw['id'])
                        if current is None or current['age']!=raw['age']+delta or not forecast['known_by_frame'][delta]:continue
                        error=max(float(np.max(np.abs(np.asarray(forecast['origin'][delta])-current['origin']))),
                            abs(forecast['length'][delta]-current['length']))
                        laser_checks[f'line_{delta}']+=1
                        if error>.005:
                            laser_checks[f'line_outside{delta}']+=1
                            if len(laser_bad)<20:laser_bad.append(dict(task=name,horizon=delta,error=error,kind='line',raw=raw,actual=current))
                    for raw,forecast in curve_predictions:
                        current=actual_curves.get(raw['id'])
                        if current is None or len(current['nodes'])!=len(raw['nodes']) or abs(current['curve_time']-raw['curve_time']-delta)>.001:continue
                        error=float(np.max(np.abs(np.asarray(forecast[delta])[:,:2]-np.asarray(current['nodes'])[:,:2])))
                        laser_checks[f'curve_{delta}']+=1
                        if error>.005:
                            laser_checks[f'curve_outside{delta}']+=1
                            if len(laser_bad)<20:laser_bad.append(dict(task=name,horizon=delta,error=error,kind='curve',raw=raw,actual=current))
                while laser_history and frame-laser_history[0][0]>=96:laser_history.popleft()
                for origin, predictions in history:
                    h=frame-origin
                    if h not in (4,12,24,48,96):continue
                    for raw,old,new in predictions:
                        current=by_id.get(raw['id'])
                        if current is None or current['native_timer']!=raw['native_timer']+h or current['native_state'] not in (1,2):continue
                        for label,forecast in (('old',old),('new',new)):
                            if not forecast['known_by_frame'][h]:counts[f'{label}_unknown{h}']+=1;continue
                            counts[f'{label}_known{h}']+=1
                            if not forecast.get('collision_enabled_by_frame',[True]*97)[h]:continue
                            if label=='new' and forecast['mode']=='native_ex_program':
                                predicted_size=raw['native_scale']*raw['native_hitbox_value']*forecast['collision_scale_ratio_by_frame'][h]
                                actual_size=current['native_scale']*current['native_hitbox_value']
                                counts[f'new_size_checks{h}']+=1
                                if actual_size>predicted_size+.002:
                                    counts[f'new_size_under{h}']+=1
                                    if len(bad)<30:bad.append(dict(task=name,frame=origin,horizon=h,
                                        size_under=actual_size-predicted_size,raw=raw,actual=current,prediction=forecast))
                            bounds=np.asarray(forecast['bounds'][h]);actual=np.asarray(current['native_position'])
                            delta=bounds[1]-bounds[0];norm=float(delta@delta)
                            t=np.clip(float((actual-bounds[0])@delta)/norm,0,1) if norm else 0.
                            error=float(np.linalg.norm(actual-bounds[0]-t*delta))-forecast.get('extra_radii',[0.]*97)[h]
                            if error>.002:
                                counts[f'{label}_outside{h}']+=1
                                if label=='new' and len(bad)<30:bad.append(dict(task=name,frame=origin,horizon=h,error=error,raw=raw,actual=current,prediction=forecast))
                while history and frame-history[0][0]>=96:history.popleft()
                if decision%4==0:
                    from fly_touhou.native_line_laser_paths import predict_native_line
                    from fly_touhou.native_curve_motion_paths import motion_covered,predicted_curve_nodes
                    line_predictions=[(r,predict_native_line(r,game_speed=lines['game_speed'],horizon=96)) for r in lines.get('lines',())]
                    curve_predictions=[(r,predicted_curve_nodes(r,np.arange(97))) for r in curves.get('curves',()) if motion_covered(r,96)]
                    laser_history.append((frame,line_predictions,curve_predictions))
                    groups=Counter();samples=[]
                    for raw in rows:
                        key=(raw['native_state'],raw.get('native_ex_active_flags'),raw.get('native_ex_current_record_hex'))
                        if groups[key]>=2 or len(samples)>=48:continue
                        groups[key]+=1
                        old_raw={k:v for k,v in raw.items() if k not in ('native_ex_program_revision','native_ex_program_state_hex','native_ex_program_hex')}
                        began=time.perf_counter()
                        new=predict_program(raw,96,player_path=conditional_path) if conditional_path is not None else None
                        if new is None:new=predict_bullet_release(raw,96)
                        timings.append(time.perf_counter()-began)
                        old=predict_bullet_release(old_raw,96);reasons[new['reason']]+=1
                        samples.append((raw,old,new));sample_count+=1;task_samples+=1
                        append_jsonl(a.output/'samples.jsonl',dict(task=name,frame=frame,raw=raw))
                    history.append((frame,samples))
                if (a.long_hints or a.benchmark) and decision in (32,33,64,65):
                    import copy
                    baseline=copy.deepcopy(env.encoder)
                    from fly_touhou.nbnm_geometry_revision import legacy_snapshot
                    legacy=legacy_snapshot(snapshot)
                    baseline.native_rule_geometry=None
                    old_begin=time.perf_counter();baseline.encode(legacy);old_elapsed=time.perf_counter()-old_begin
                    old_result=baseline.last_result
                    if a.profile_encoding and decision==64:
                        import cProfile,pickle
                        with (a.output/(name+'-encoding.pkl')).open('wb') as file:
                            pickle.dump((copy.deepcopy(env.encoder),snapshot),file)
                        profiler=cProfile.Profile();profiler.enable()
                    began=time.perf_counter();original_encode(snapshot);elapsed=time.perf_counter()-began
                    if a.profile_encoding and decision==64:
                        profiler.disable();profiler.dump_stats(str(a.output/(name+'-encoding.prof')))
                    result=env.encoder.last_result
                    reads={}
                    if a.benchmark_reader:
                        reader=env.controller.reader;hooks=reader.profile.hooks
                        revision=hooks.pop('native_ex_program')
                        try:
                            began=time.perf_counter();reader.read();reads['legacy_seconds']=time.perf_counter()-began
                        finally:hooks['native_ex_program']=revision
                        began=time.perf_counter();reader.read();reads['rules_seconds']=time.perf_counter()-began
                    benchmarks.append(dict(task=name,decision=decision,bullets=len(rows),lasers=len(actual_lines)+len(actual_curves),
                        reader_timings=reads,
                        encode_seconds=elapsed,baseline_encode_seconds=old_elapsed,hints=result.get('rule_long_hints'),feature_width=len(result['features']),
                        old_trusted=np.asarray(old_result['details']['trusted']).tolist(),new_trusted=np.asarray(result['details']['trusted']).tolist(),
                        changed_input_columns=int(np.count_nonzero(np.abs(np.asarray(old_result['features'])-np.asarray(result['features']))>1e-5))))
                    append_jsonl(a.output/'benchmarks.jsonl',benchmarks[-1])
                    env.encoder.last_result=None
                if decision%32==0:write_json(a.output/'progress.json',dict(task=name,decision=decision,samples=sample_count,counts=dict(counts),seconds=time.monotonic()-start))
                if a.continue_after_contact:
                    mask=1
                    if a.greedy_dodge:
                        from fly_touhou.background import action_mask
                        from fly_touhou.thrl_native_env import SOURCE19_TO_NATIVE36
                        from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
                        env.encoder._nbnm_cache=None
                        original_encode(snapshot)
                        margins=np.asarray(env.encoder.last_result['details']['margins'])
                        v=MovementSafetyTeacher().velocities
                        end=np.clip(np.array([snapshot.player.x,snapshot.player.y])+v*12,[-184,32],[184,432])
                        score=np.clip(margins[:2].min(0),-64,64)-.03*np.linalg.norm(end-[0,360],axis=1)
                        mask=action_mask(SOURCE19_TO_NATIVE36[int(score.argmax())])
                    env.snapshot=env._check_snapshot(env.controller.advance(mask,4))
                    if a.invulnerable_fixture and env.snapshot.player is not None and env.snapshot.player.invulnerable_frames<=0:
                        raise RuntimeError('Physics fixture protection expired')
                    if conditional_path is not None and env.snapshot.player is not None:
                        if np.max(np.abs(conditional_path[0]-[env.snapshot.player.x,env.snapshot.player.y]))>.005:
                            raise RuntimeError('Conditional aim validation player path changed')
                    info={'nb_nm_reason':'physics-probe-original-lives-not-NBNM-evaluation'}
                    if env.snapshot.screen!=7 or env.snapshot.player is None:break
                    if a.greedy_dodge and env.snapshot.player.deathbomb_timer>0:break
                else:
                    obs,_,done,truncated,info=env.step(0)
                    if done or truncated:break
            task_rows.append(dict(task=name,decisions=decision+1,samples=task_samples,reason=info.get('nb_nm_reason')))
            env.close();env=None
        write_json(a.output/'report.json',dict(status='complete',revision=REVISION,training_updates=0,
            tasks=task_rows,samples=sample_count,counts=dict(counts),reasons=dict(reasons),
            per_bullet96_ms={str(q):float(np.percentile(timings,q)*1000) for q in (50,95,99)} if timings else {},
            contradictions=bad,laser_checks=dict(laser_checks),laser_contradictions=laser_bad,
            benchmarks=benchmarks,seconds=time.monotonic()-start))
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc(),counts=dict(counts),
            benchmarks=benchmarks,laser_checks=dict(laser_checks),laser_contradictions=laser_bad));raise
    finally:
        if env is not None:env.close()
        cleanup_native_temp(temp)


if __name__=='__main__':main()
