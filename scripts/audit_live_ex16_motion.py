"""Observe EX16 endpoints on an owned live worker without changing its inputs."""
import os
from pathlib import Path


def snapshot(output,expected_pid):
    import gc
    from collections import Counter
    import fly_touhou.hazard_reader as hazard
    from fly_touhou.native_bullet_turn import read_turn_metadata,predict_turn_prefix
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned native worker')
    reader=next(x for x in gc.get_objects() if isinstance(x,hazard.VersionedHazardReader))
    observed=reader.read();rows=observed.extra['hazard_metadata']['bullets'];examples=[]
    for raw in rows:
        if raw.get('native_ex_active_flags')!=16:continue
        detailed=dict(raw);detailed.update(read_turn_metadata(reader.mem,raw['id']))
        p=predict_turn_prefix(detailed)
        examples.append(dict(raw=detailed,mode=p['mode'],reason=p['reason'],known=p['known_by_frame']))
        if len(examples)==4:break
    function=hazard.VersionedHazardReader.read
    closure={name:cell.cell_contents for name,cell in zip(function.__code__.co_freevars,function.__closure__ or ())}
    write_json(Path(output),dict(PID=os.getpid(),frame=observed.frame,stage=observed.stage,
        flags=dict(Counter(str(x.get('native_ex_active_flags')) for x in rows)),examples=examples,
        audit_calls=closure.get('calls'),audit_pairs=closure.get('pairs'),
        coherent_snapshot=reader.global_frame()==observed.frame,extra_native_steps=0))


def arm(output,expected_pid,pair_limit=96,include_polar_growth=False):
    import math
    import fly_touhou.hazard_reader as hazard
    from fly_touhou.native_bullet_turn import read_turn_metadata,predict_turn_prefix
    from fly_touhou.native_bullet_release import predict_bullet_release
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Expected owned native worker')
    original=hazard.VersionedHazardReader.read
    anchors={};counts={};failures=[];pairs=0;calls=0;last_frame=None;max_error=0.;examples=[]
    modes=('EX16','EX8_growth') if include_polar_growth else ('EX16',)
    by_mode={mode:0 for mode in modes};by_mode_dt={mode:{} for mode in modes}
    mode_limit=max(1,pair_limit//len(modes));turns=[];saved_pairs=0;saved_turns=0
    call_limit=4096 if include_polar_growth else 1024
    def save(status,error=None):
        nonlocal saved_pairs,saved_turns
        write_json(Path(output),dict(status=status,PID=os.getpid(),pairs=pairs,calls=calls,
            by_dt=counts,outside_envelope_count=len(failures),failures=failures,
            maximum_center_error_px=max_error,examples=examples,error=error,
            by_mode=by_mode,by_mode_dt=by_mode_dt,EX16_turn_intervals=turns,
            turn_intervals_are_not_forecast_validation=True,
            reader_restored=status not in ('armed','recording'),
            policy_observations_changed=False,extra_native_steps=0,
            scope='Same-ID live-phase motion endpoints with unchanged shape; EX16 turns captured separately, not certified. No collision or counterfactual proof.'))
        saved_pairs=pairs;saved_turns=len(turns)
    def read(self,*args,**kwargs):
        nonlocal pairs,calls,last_frame,max_error
        snapshot=original(self,*args,**kwargs)
        try:
            if snapshot.frame==last_frame:return snapshot
            last_frame=snapshot.frame;calls+=1
            rows={r['id']:r for r in snapshot.extra['hazard_metadata']['bullets']}
            for identity,anchor in list(anchors.items()):
                dt=snapshot.frame-anchor['frame'];raw=rows.get(identity);old=anchor['raw'];p=anchor['prediction']
                paired=(dt in (4,8,12,24) and raw is not None and snapshot.stage==anchor['stage']
                        and raw['native_timer']==old['native_timer']+dt
                        and raw['native_hitbox_value']==old['native_hitbox_value']
                        and raw['native_scale']==old['native_scale'] and raw.get('native_state') in (1,2))
                mode=anchor['mode']
                if (paired and include_polar_growth and mode=='EX16' and len(turns)<12
                        and not anchor.get('turn_recorded')
                        and dt>max(0,old['native_ex16_duration']-old['native_ex16_tick'])):
                    after=dict(raw);after.update(read_turn_metadata(self.mem,identity))
                    if after['native_ex16_completed']>old['native_ex16_completed']:
                        turns.append(dict(frame=snapshot.frame,dt=dt,before=old,after=after))
                        anchor['turn_recorded']=True
                if (paired and p['known_by_frame'][dt] and by_mode[mode]<mode_limit
                        and raw.get('native_ex_active_flags')==old.get('native_ex_active_flags')):
                    expected=p['positions'][dt];actual=raw['native_position']
                    error=math.hypot(actual[0]-expected[0],actual[1]-expected[1])
                    max_error=max(max_error,error);pairs+=1;key=str(dt);counts[key]=counts.get(key,0)+1
                    by_mode[mode]+=1;by_mode_dt[mode][key]=by_mode_dt[mode].get(key,0)+1
                    a,b=p['bounds'][dt];dx=b[0]-a[0];dy=b[1]-a[1];length=dx*dx+dy*dy
                    fraction=max(0.,min(1.,((actual[0]-a[0])*dx+(actual[1]-a[1])*dy)/length)) if length else 0.
                    distance=math.hypot(actual[0]-a[0]-fraction*dx,actual[1]-a[1]-fraction*dy)
                    entry=dict(frame=snapshot.frame,dt=dt,predicted=expected,observed=actual,
                        mode=mode,error_px=error,envelope_distance_px=distance,envelope_px=p['extra_radii'][dt],raw=old)
                    if (len(examples)<8 or not any(x['mode']==mode and x['dt']==dt for x in examples)):
                        examples.append(entry)
                    if distance>p['extra_radii'][dt]+1e-3:failures.append(entry)
                if dt>=24 or dt<0 or raw is None:anchors.pop(identity,None)
            sufficient=all(v>=mode_limit for v in by_mode.values()) and (not include_polar_growth or bool(turns))
            if failures or sufficient or calls>=call_limit:
                hazard.VersionedHazardReader.read=original
                save('motion_mismatch' if failures else 'completed' if sufficient else 'observation_limit_reached' if pairs else 'no_samples');return snapshot
            for raw in rows.values():
                if len(anchors)>=8:break
                if raw['id'] in anchors:continue
                detailed=dict(raw)
                if raw.get('native_ex_active_flags')==16 and raw.get('native_state')==1:
                    mode='EX16';detailed.update(read_turn_metadata(self.mem,raw['id']))
                    prediction=predict_turn_prefix(detailed,24)
                elif include_polar_growth and raw.get('native_ex_active_flags')==8 and raw.get('native_state')==2:
                    mode='EX8_growth';prediction=predict_bullet_release(detailed,24)
                else:continue
                near_turn=(include_polar_growth and mode=='EX16' and len(turns)<12
                    and 0<=detailed.get('native_ex16_duration',-1)-detailed.get('native_ex16_tick',-2)<=8)
                if (by_mode[mode]<mode_limit and prediction['known_by_frame'][4]) or near_turn:
                    anchors[raw['id']]=dict(frame=snapshot.frame,stage=snapshot.stage,
                        mode=mode,raw=detailed,prediction=prediction)
            if (pairs!=saved_pairs and pairs%8==0) or len(turns)!=saved_turns:save('recording')
        except Exception as error:
            hazard.VersionedHazardReader.read=original
            try:save('diagnostic_failed',str(error))
            except Exception:pass
        return snapshot
    hazard.VersionedHazardReader.read=read
    save('armed')
