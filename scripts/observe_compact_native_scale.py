"""Bounded metadata observation in the already-running frozen worker.

Adds no model input and does not step/reset the game or run another episode.
"""
import copy
import os
from pathlib import Path
from collections import deque


def install(output,*,pid=29084,created=1793587568.0975215):
    import psutil
    from apply_compact_live_wall import frames
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=pid or abs(psutil.Process().create_time()-created)>.01:
        raise ValueError('Observation belongs to the current frozen seed3 worker')
    worker=next(f for f in frames() if f.f_code.co_name=='_worker' and Path(f.f_code.co_filename).name=='thrl_vector.py')
    env=worker.f_locals['env'];reader=env.controller.reader
    forecast_enabled=bool((reader.profile.hooks or {}).get('native_ex_scale_forecast'))
    owner=next(c for c in type(reader).__mro__ if '_bullet_metadata' in c.__dict__)
    if owner._bullet_metadata.__name__=='observed':
        closure=dict(zip(owner._bullet_metadata.__code__.co_freevars,owner._bullet_metadata.__closure__))
        owner._bullet_metadata=closure['original'].cell_contents
    original=owner._bullet_metadata
    history={};pairs=[];last_saved=0;counts={4:0,12:0,24:0}
    target=Path(output)
    def save(status):
        write_json(target,dict(status=status,worker_PID=os.getpid(),game_PID=env.controller.pid,
            forecast_enabled=forecast_enabled,model_observations_unchanged=True,counts=counts,pairs=pairs))
    def observed(self,bullet):
        nonlocal last_saved
        result=original(self,bullet)
        if result.get('native_state')!=1 or result.get('native_ex_active_flags')!=0x600000:
            history.pop(bullet.id,None);return result
        current=copy.deepcopy(result);current['kind']=bullet.kind
        current['observed_native_frame']=self.global_frame()
        previous=history.setdefault(bullet.id,deque(maxlen=32))
        # A coherent game snapshot may be read repeatedly without advancing.
        # Keep distinct native frames so duplicate reads cannot evict24f history.
        if previous and previous[-1]['observed_native_frame']==current['observed_native_frame']:return result
        if previous and previous[-1]['native_timer']>current['native_timer']:previous.clear()
        for earlier in previous:
            frames_apart=current['observed_native_frame']-earlier['observed_native_frame']
            if frames_apart in counts and counts[frames_apart]<32 and earlier['kind']==current['kind']:
                pairs.append(dict(frames=frames_apart,before=earlier,after=current))
                counts[frames_apart]+=1
        previous.append(current)
        finished=min(counts.values())>=32
        if len(pairs)-last_saved>=32 or finished:
            save('captured' if finished else 'observing');last_saved=len(pairs)
        if finished:owner._bullet_metadata=original
        return result
    owner._bullet_metadata=observed
    save('observing')
    return dict(status='observing',worker_PID=os.getpid(),output=str(target),forecast_enabled=forecast_enabled)
