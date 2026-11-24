"""Long route advice from complete original-game NBNM demonstrations.

This changes teacher targets on student states, not the brain's inputs or its
action execution. A recorded success does not certify a deviating trajectory.
"""
from pathlib import Path
import json
import numpy as np

SCHEMA='nbnm-complete-demonstration-route-guide-v1'
HORIZONS=(24,96,240,600)


def scene_signature(snapshot):
    """Small scene-matching descriptor, never a collision-safety label.

    Preserve coarse world occupancy, motion, object sizes and laser geometry.
    The live native geometry/teacher retains responsibility for all dangers.
    """
    grid=np.zeros((4,8,8),np.float32)
    points=[(b.x,b.y,b.vx,b.vy,b.radius) for b in snapshot.bullets if b.active]
    points += [(e.x,e.y,0.,0.,max(e.radius_x,e.radius_y)) for e in snapshot.enemies]
    for laser in snapshot.lasers:
        if laser.nodes:xy=laser.nodes[::max(1,len(laser.nodes)//12)]
        else:
            t=np.linspace(0,max(0,laser.length),12)
            xy=np.column_stack((laser.x+np.cos(laser.angle)*t,laser.y+np.sin(laser.angle)*t))
        points.extend((x,y,laser.speed*np.cos(laser.angle),laser.speed*np.sin(laser.angle),laser.width/2) for x,y in xy)
    if points:
        values=np.asarray(points,np.float32)
        ix=np.floor((values[:,0]+192)/48).astype(int);iy=np.floor(values[:,1]/56).astype(int)
        valid=(ix>=0)&(ix<8)&(iy>=0)&(iy<8);ix=ix[valid];iy=iy[valid];values=values[valid]
        np.add.at(grid[0],(iy,ix),1)
        for channel,column,scale in ((1,2,12.),(2,3,12.),(3,4,64.)):
            np.add.at(grid[channel],(iy,ix),np.clip(values[:,column]/scale,-1,1))
            grid[channel]/=np.maximum(grid[0],1)
        grid[0]=np.log1p(grid[0])/np.log(65.)
    return grid.ravel()


def build_bank(capture, output):
    capture=Path(capture);report=json.loads((capture/'report.json').read_text(encoding='utf-8'))
    if (not report.get('full_game_NBNM') or report.get('Misses') or report.get('Bombs')
            or report.get('feature_kind')!='route-matching-only'):
        raise ValueError('Require a complete native NBNM route-only capture')
    from .thrl_training_mode import seed_split
    source=json.loads((capture/'source.json').read_text(encoding='utf-8'))
    seed=source['metadata']['stages'][0]['native_seed']
    if seed_split(seed)!='train':raise ValueError('Use a training-split initial replay seed')
    phases=report['phase_names'];parts=[]
    for stage in report['stages']:
        with np.load(capture/stage['file'],allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        size=len(d['frame']);ends=np.full(size,size-1,np.int32)
        changes=np.r_[0,np.flatnonzero(np.diff(d['phase'])!=0)+1,size]
        for left,right in zip(changes[:-1],changes[1:]):ends[left:right]=right-1
        goals=[]
        for horizon in HORIZONS:
            ids=np.minimum(np.searchsorted(d['frame'],d['frame']+horizon),ends)
            goals.append(d['position'][ids])
        parts.append(dict(phase=d['phase'],clock=d['phase_clock'],position=d['position'],
            power=d['features'][:,2],signature=d['scene_signature'],
            waypoints=np.stack(goals,axis=1),phase_remaining=d['frame'][ends]-d['frame']))
    arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    valid=arrays['clock']>=0
    arrays={k:v[valid] for k,v in arrays.items()}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'routes.npz',**arrays)
    coverage={phase:int(np.count_nonzero(arrays['phase']==i)) for i,phase in enumerate(phases)}
    manifest=dict(schema=SCHEMA,data='routes.npz',phases=phases,rows=int(valid.sum()),
        coverage=coverage,source_report=str((capture/'report.json').resolve()),
        source_replay=report['source'],native_seed=seed,training_split_only=True,
        all_source_paths_reach_full_game_NBNM=True,waypoint_horizons=list(HORIZONS),
        teacher_only=True,added_neural_features=0,PPO_rows=0,
        scope='Recorded route continuations to task completion; advice at new states is rechecked by the live teacher')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


class DemonstrationRouteGuide:
    def __init__(self, manifest, *, source_switch_penalty=0., reference_max_gap=32,
                 source_switch_stages=None):
        self.source_switch_penalty=float(source_switch_penalty)
        self.reference_max_gap=int(reference_max_gap)
        self.source_switch_stages=None if source_switch_stages is None else tuple(source_switch_stages)
        self.last_reference=None
        if not 0<=self.source_switch_penalty<=2:raise ValueError('Bounded route-source continuity cost required')
        path=Path(manifest);self.path=path;self.queries=0;self.modified=path.stat().st_mtime_ns
        from .thrl_study_io import read_shared_json
        self.manifest=read_shared_json(path)
        if (self.manifest.get('schema')!=SCHEMA or not (self.manifest.get('all_source_paths_reach_full_game_NBNM')
                or self.manifest.get('all_source_paths_reach_task_NBNM'))
                or not self.manifest.get('training_split_only')):
            raise ValueError('A complete, training-only NBNM demonstration bank is required')
        with np.load(path.parent/self.manifest['data'],allow_pickle=False) as z:
            self.data={k:z[k].copy() for k in z.files}
        self.groups={phase:np.flatnonzero(self.data['phase']==i) for i,phase in enumerate(self.manifest['phases'])}

    def export_reference(self):
        if not self.source_switch_penalty or self.last_reference is None:return None
        return dict(manifest=str(self.path),data=self.manifest['data'],reference=self.last_reference)

    def restore_reference(self, state):
        self.last_reference=None
        if (state and self.source_switch_penalty and Path(state['manifest'])==self.path
                and state['data']==self.manifest['data']):
            self.last_reference=tuple(state['reference'])
        return self.last_reference is not None

    def query(self,snapshot,encoding):
        self.queries+=1
        if self.queries%64==0 and self.path.stat().st_mtime_ns!=self.modified:
            self.__init__(self.path,source_switch_penalty=self.source_switch_penalty,
                reference_max_gap=self.reference_max_gap,source_switch_stages=self.source_switch_stages)
        from .nbnm_route_atlas import phase_progress
        from .thrl_movement_teacher import action_velocities
        phase,clock=phase_progress(snapshot)
        info=dict(available=False,phase=phase,clock=clock,teacher_only=True,
            source_reaches_full_game_NBNM=bool(self.manifest.get('all_source_paths_reach_full_game_NBNM')),current_path_verified=False)
        rows=self.groups.get(phase)
        if rows is None or clock is None:return info
        d=self.data;rows=rows[np.abs(d['clock'][rows]-clock)<=24]
        if not len(rows):return info
        p=snapshot.player;position=np.asarray([p.x,p.y])
        distance=np.linalg.norm(d['position'][rows]-position,axis=1)
        power=np.abs(d['power'][rows]-snapshot.power/400)
        mismatch=np.sqrt(np.mean((d['signature'][rows]-scene_signature(snapshot))**2,axis=1))
        supported=(distance<=96)&(power<=.30)&(mismatch<=.25)
        if not supported.any():return info
        cost=distance/96+power/.30+mismatch/.25+np.abs(d['clock'][rows]-clock)/48
        cost[~supported]=np.inf
        ranking=cost.copy();previous=None
        penalty=(self.source_switch_penalty if self.source_switch_stages is None
                 or snapshot.stage in self.source_switch_stages else 0.)
        if penalty and self.last_reference is not None and 'source_id' in d:
            old_phase,old_clock,old_source=self.last_reference
            if phase==old_phase and 0<=clock-old_clock<=self.reference_max_gap:
                previous=old_source
                ranking+=penalty*(d['source_id'][rows]!=old_source)
        k=int(np.argmin(ranking));row=int(rows[k])
        # Continuity affects reference selection, never confidence, live
        # clearance, source eligibility or the neural policy's action mask.
        confidence=float(np.exp(-cost[k]))
        if confidence<.12:return info
        source=int(d['source_id'][row]) if 'source_id' in d else None
        self.last_reference=(phase,clock,source)
        velocity=action_velocities()
        if encoding['source_features25'][18]<0:velocity[9:]=velocity[:9]
        endpoints=np.clip(position+4*velocity,[-184,32],[184,432])
        goals=d['waypoints'][row]
        score=np.zeros(18,float)
        # Follow the next stretch of the demonstrated corridor; farther
        # waypoints add a weaker preference and never authorize a shortcut.
        for goal,weight in zip(goals,(1.,.35,.15,.05)):
            score+=weight*(np.linalg.norm(goal-position)-np.linalg.norm(goal-endpoints,axis=1))/18
        info['source_reaches_full_game_NBNM']=bool(d.get('source_full_game',np.ones(len(d['clock']),bool))[row])
        return dict(info,available=True,confidence=confidence,scores=score.tolist(),
            matched_row=row,position_error=float(distance[k]),scene_error=float(mismatch[k]),
            power_error=float(power[k]),waypoints=goals.tolist(),
            source_id=source,source_switch_penalty=penalty,
            prior_source_id=previous,source_changed=bool(previous is not None and source!=previous),
            recorded_phase_remaining=int(d['phase_remaining'][row]))


def apply_advice(probabilities,safe,advice,routes=None):
    p=np.asarray(probabilities,float)
    if not advice or not advice.get('available'):return p
    valid=np.asarray(safe,bool).copy()
    if routes is not None and np.any(valid&(np.asarray(routes['exits'])>0)):
        valid &= np.asarray(routes['exits'])>0
    if not valid.any():return p
    scores=np.asarray(advice['scores'],float);scores-=scores[valid].max()
    target=np.where(valid,np.exp(np.clip(4*scores,-60,0)),0.);target/=target.sum()
    mix=.65*float(advice['confidence'])
    return (1-mix)*p+mix*target


def guide_for_environment(env):
    settings=env.nbnm.get('demonstration_route_teacher') or {}
    if not settings.get('enabled'):return None
    path=settings['manifest']
    stages=settings.get('source_switch_stages')
    key=(str(path),float(settings.get('source_switch_penalty',0.)),
         int(settings.get('reference_max_gap',32)),None if stages is None else tuple(stages))
    guide=getattr(env,'_demonstration_guide',None)
    if guide is None or getattr(env,'_demonstration_guide_key',None)!=key:
        guide=env._demonstration_guide=DemonstrationRouteGuide(path,
            source_switch_penalty=settings.get('source_switch_penalty',0.),
            reference_max_gap=settings.get('reference_max_gap',32),
            source_switch_stages=settings.get('source_switch_stages'))
        env._demonstration_guide_path=path
        env._demonstration_guide_key=key
    return guide


def advice_for_environment(env):
    guide=guide_for_environment(env)
    return None if guide is None else guide.query(env.snapshot,env.encoder.last_result)


def merge_verified_sources(manifests, teacher_manifests, output, *, versioned=False):
    """Merge native successes only; reject failed or evaluation trajectories."""
    from .thrl_training_mode import seed_split
    from .thrl_study_io import read_shared_json
    parts=[];phases=[];sources=[]
    def ids(names):
        for name in names:
            if name not in phases:phases.append(name)
        return np.asarray([phases.index(n) for n in names],np.int16)
    for path in map(Path,manifests):
        guide=DemonstrationRouteGuide(path);d=guide.data;m=guide.manifest
        mapped=ids(m['phases']);part={k:d[k].copy() for k in ('clock','position','power','signature','waypoints','phase_remaining')}
        part['phase']=mapped[d['phase']]
        part['source_full_game']=d.get('source_full_game',np.full(len(d['clock']),bool(m.get('all_source_paths_reach_full_game_NBNM'))))
        part['source_id']=np.full(len(part['clock']),len(sources),np.int32)
        parts.append(part);sources.append(dict(kind='verified_demonstration',manifest=str(path)))
    for path in map(Path,teacher_manifests):
        m=read_shared_json(path)
        for record in m['histories']:
            if not record.get('nbnm_success') or record.get('split')!='train' or seed_split(record['native_seed'])!='train':continue
            file=path.parent/record['file']
            with np.load(file,allow_pickle=False) as z:
                if not all(k in z.files for k in ('phase','clock','position','frame','power','signature')):continue
                d={k:z[k].copy() for k in ('phase','clock','position','frame','power','signature')}
            if not len(d['frame']):continue
            ends=np.empty(len(d['frame']),int);waypoints=np.empty((len(ends),len(HORIZONS),2),np.float32)
            cuts=np.r_[0,np.flatnonzero(d['phase'][1:]!=d['phase'][:-1])+1,len(ends)]
            for left,right in zip(cuts[:-1],cuts[1:]):
                ends[left:right]=right-1
                for j,horizon in enumerate(HORIZONS):
                    target=np.minimum(np.searchsorted(d['frame'][left:right],d['frame'][left:right]+horizon)+left,right-1)
                    waypoints[left:right,j]=d['position'][target]
            valid=d['clock']>=0
            if record['task']['mode']=='attack':
                # A native card entry/exit has no active Boss for a few ticks.
                # Its stage clock is a practice clock, NOT the real road's
                # timeline. Reusing it as road advice mixed unrelated waves.
                valid &= np.array([not str(p).endswith('/road') for p in d['phase']])
            if not valid.any():continue
            part={k:d[k][valid] for k in ('clock','position','power','signature')}
            part.update(phase=ids(d['phase'][valid].tolist()),waypoints=waypoints[valid],
                phase_remaining=(d['frame'][ends]-d['frame'])[valid],
                source_full_game=np.full(int(valid.sum()),bool(record.get('full_game_NBNM'))),
                source_id=np.full(int(valid.sum()),len(sources),np.int32))
            kind=('verified_student' if record.get('source_kind') in
                  ('native_verified_student_continuation','successful_student_behavior') else 'verified_rule_teacher')
            parts.append(part);sources.append(dict(kind=kind,file=str(file),task=record['task'],seed=record['native_seed']))
    if not parts:raise ValueError('No verified routes')
    arrays={k:np.concatenate([d[k] for d in parts]) for k in parts[0]}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    import time
    data_name=('routes-'+str(time.time_ns())+'.npz') if versioned else 'routes.npz'
    np.savez_compressed(output/data_name,**arrays)
    result=dict(schema=SCHEMA,data=data_name,phases=phases,rows=len(arrays['clock']),
        coverage={name:int((arrays['phase']==i).sum()) for i,name in enumerate(phases)},
        sources=sources,training_split_only=True,all_source_paths_reach_task_NBNM=True,
        road_source_revision='exclude-card-practice-transitions-v1',
        all_source_paths_reach_full_game_NBNM=bool(arrays['source_full_game'].all()),
        waypoint_horizons=list(HORIZONS),teacher_only=True,added_neural_features=0,PPO_rows=0,
        scope='Native-verified complete source tasks. Advice at deviating student states is not a verified continuation.')
    from .thrl_durable_records import write_json
    write_json(output/'manifest.json',result)
    return result


class RouteBankRefresh:
    """Publish new native successes at a learner boundary, without blocking reads."""
    def __init__(self,settings):
        self.settings=settings;self.previous=None

    def refresh(self):
        settings=self.settings
        sources=list(map(Path,settings['demonstrations']))
        teachers=[Path(p) for p in settings['teacher_manifests'] if Path(p).exists()]
        current=tuple((str(p),p.stat().st_mtime_ns) for p in sources+teachers)
        if current==self.previous:return None
        result=merge_verified_sources(sources,teachers,settings['output'],versioned=True)
        self.previous=current
        return dict(rows=result['rows'],sources=len(result['sources']),phases=len(result['phases']),
            manifest=str(Path(settings['output'])/'manifest.json'),native_verified_only=True)
