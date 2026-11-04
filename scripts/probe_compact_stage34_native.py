"""Frozen rule-teacher probe with exact native one-tick evidence in stages3/4.

No neural network or optimizer is loaded. All controls use the existing owned
hidden original-game backend. Raw game input is retained for reproduction.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import pickle
import struct
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def snapshot_context(s):
    return dict(frame=s.frame,stage=s.stage,stage_frame=s.stage_frame,
        player=asdict(s.player) if s.player else None,lives=s.lives,bombs=s.bombs,
        power=s.power,difficulty=s.difficulty,dialogue=s.dialogue,
        resources=s.extra.get('resources'),native_rng_state=s.extra.get('native_rng_state'),
        thrl_events=s.extra.get('thrl_events'),native_collision_player=(s.extra.get('native_collision_geometry') or {}).get('player'),
        lasers=[asdict(l) for l in s.lasers])


class Capture:
    def __init__(self,env,output):
        self.env=env;self.output=output;self.buffer=[];self.chunks=[]
        self.programs={};self.program_files=[];self.rows=0;self.transitions=0
        self.motion_errors=[];self.contact_errors=[];self.laser_bursts=[]
        self.laser_window=0;self.laser_ids=set();self.last_capture_frame=None
        self.tape_record_count=0

    def read(self,address,size):
        try:return self.env.controller.reader.mem.read(int(address),int(size))
        except Exception:return None

    def pages(self,start,size):
        result={}
        for address in range(start & ~4095,(start+size+4095)&~4095,4096):
            data=self.read(address,4096)
            if data is not None:result[address]=data
        return result

    def program(self,s,row):
        motion=row['native_motion'];owner=motion['owner_address']
        raw=self.read(owner+0x11cc,4)
        if raw is None:return None
        manager=struct.unpack('<I',raw)[0]
        raw=self.read(manager,0x100)
        if raw is None:return None
        count=struct.unpack_from('<i',raw,8)[0];table=struct.unpack_from('<I',raw,0x8c)[0]
        key=(s.stage,manager,table)
        if key in self.programs:return self.programs[key]
        if not 0<count<=20000:return None
        entries_raw=self.read(table,count*8)
        if entries_raw is None:return None
        entries=[];blocks={manager:raw,table:entries_raw};pages={}
        for name,sub in struct.iter_unpack('<II',entries_raw):
            name_raw=self.read(name,128)
            if name_raw is not None:blocks[name]=name_raw
            entries.append(dict(name_ptr=name,sub_ptr=sub,name=(name_raw or b'').split(b'\0')[0].decode('ascii','replace')))
        ordered=sorted({r['sub_ptr'] for r in entries if r['sub_ptr']})
        for i,address in enumerate(ordered):
            size=min(65536,ordered[i+1]-address) if i+1<len(ordered) else 16384
            pages.update(self.pages(address,max(16,size)))
        name=f'ecl-program-stage{s.stage}-{len(self.program_files):03}.pkl.gz'
        with gzip.open(self.output/name,'wb',compresslevel=2) as stream:
            pickle.dump(dict(stage=s.stage,frame=s.frame,manager=manager,count=count,table=table,
                entries=entries,blocks=blocks,pages=pages),stream,protocol=5)
        self.programs[key]=name;self.program_files.append(name)
        return name

    def capture_frame(self,s,*,laser=False):
        if self.last_capture_frame==s.frame:return
        bosses=[r for r in s.extra.get('resources',{}).get('enemies',()) if r.get('boss') and 'native_motion' in r]
        if s.stage not in (3,4) or not bosses and not laser:return
        self.last_capture_frame=s.frame;blocks={};programs=[]
        for row in bosses:
            name=self.program(s,row)
            if name:programs.append(name)
            motion=row['native_motion']
            raw=self.read(row['id'],0x5400)
            if raw is not None:blocks[row['id']]=raw
            for context in motion['ecl_contexts']:
                raw=self.read(context['address'],0x11c4)
                if raw is not None:blocks[context['address']]=raw
                if context['instruction']:blocks.update(self.pages(context['instruction'],16384))
        if laser:
            for item in s.lasers:
                raw=self.read(item.id,0x1000)
                if raw is not None:blocks[item.id]=raw
        mem=self.env.controller.reader.mem
        for address,size in ((mem.base+0xdb510,0x180),(mem.base+0xf5800,0x100),(mem.base+0xd4cec,256)):
            data=self.read(address,size)
            if data is not None:blocks[address]=data
        globals_block=blocks.get(mem.base+0xdb510)
        if globals_block is not None:
            manager=struct.unpack_from('<I',globals_block,0x34)[0]
            data=self.read(manager,0x100)
            if data is not None:blocks[manager]=data
        pointer=self.read(mem.base+898684,4)
        if pointer:
            player=struct.unpack('<I',pointer)[0]
            data=self.read(player,0x700)
            if data is not None:blocks[player]=data
        raw=self.read(mem.base+0xD8F58,4)
        if raw is not None:blocks[mem.base+0xD8F58]=raw
        row=dict(snapshot=snapshot_context(s),bosses=bosses,blocks=blocks,
            programs=sorted(set(programs)),native_game_speed=struct.unpack('<f',raw)[0] if raw is not None else None,
            laser_burst=laser)
        encoding=getattr(self.env.encoder,'last_result',None)
        if encoding and (encoding['frame'],encoding['stage'])==(s.frame,s.stage):row['compact_encoding']=encoding
        self.buffer.append(row);self.rows+=1
        if len(self.buffer)>=128:self.flush()

    def flush(self):
        from fly_touhou.thrl_durable_records import write_json
        if self.buffer:
            name=f'native-frames-{len(self.chunks):04}.pkl.gz'
            with gzip.open(self.output/name,'wb',compresslevel=2) as stream:pickle.dump(self.buffer,stream,protocol=5)
            self.chunks.append(dict(file=name,rows=len(self.buffer),first_frame=self.buffer[0]['snapshot']['frame'],last_frame=self.buffer[-1]['snapshot']['frame']))
            self.buffer=[]
        write_json(self.output/'capture-index.json',dict(schema='paused-stage34-native-frame-evidence-v1',
            rows=self.rows,chunks=self.chunks,programs=self.program_files,transitions=self.transitions,
            motion_errors=self.motion_errors,contact_errors=self.contact_errors,laser_bursts=self.laser_bursts,
            training=False,neural_model_loaded=False,raw_memory_is_observed_current_state=True))

    def inspect(self,before,after):
        self.transitions+=1
        if after.stage not in (3,4):return
        old={r['id']:r for r in before.extra.get('resources',{}).get('enemies',()) if r.get('boss')}
        new={r['id']:r for r in after.extra.get('resources',{}).get('enemies',()) if r.get('boss')}
        for identity in old.keys() & new.keys():
            a,b=old[identity],new[identity]
            prediction=a.get('native_motion_forecast') or {};positions=prediction.get('positions',())
            if len(positions)>1 and after.frame-before.frame==1:
                gap=((positions[1][0]-b['x'])**2+(positions[1][1]-b['y'])**2)**.5
                if gap>1 and len(self.motion_errors)<200:
                    self.motion_errors.append(dict(before=before.frame,after=after.frame,stage=after.stage,
                        identity=identity,predicted=positions[1][:2],actual=[b['x'],b['y']],error=gap,
                        unknown_from=prediction.get('unknown_from_frame'),reason=prediction.get('unknown_reason')))
            ticks=(a.get('native_contact') or {}).get('forecast',{}).get('ticks',())
            if ticks and ticks[0].get('contactable') is not None and ticks[0]['contactable']!=b['contactable'] and len(self.contact_errors)<200:
                self.contact_errors.append(dict(before=before.frame,after=after.frame,stage=after.stage,
                    identity=identity,predicted=ticks[0]['contactable'],actual=b['contactable'],known=ticks[0].get('known')))
        present={l.id for l in after.lasers if l.width>0}
        newborn=present-self.laser_ids
        if after.stage==4 and newborn and len(self.laser_bursts)<8:
            self.laser_bursts.append(dict(frame=after.frame,stage_frame=after.stage_frame,ids=sorted(newborn)))
            self.laser_window=32
        self.laser_ids=present
        self.capture_frame(after,laser=self.laser_window>0)
        if self.laser_window:self.laser_window-=1


def replay_archived_prefix(env,source,output,*,through_stage=2):
    """Re-execute recorded teacher actions with native physics, no replanning.

    This diagnostic prefix is never a student rollout. Rewards still observe
    every real transition; expensive sensory/teacher recomputation is skipped
    only while reusing the recorded action list. Native scene reads continue.
    """
    import numpy as np
    from fly_touhou.thrl_durable_records import write_json
    record=json.loads((source/'index.json').read_text(encoding='utf-8'))
    expected_seed=record['episode_summary']['full_story_native_seed']
    if env.native_seed!=expected_seed:raise ValueError('Archived action prefix needs its recorded native seed')
    teacher=env._teacher;encode=env._encode;original_observation=env._last_measurable_observation
    env._teacher=None
    env._encode=lambda snapshot:original_observation
    count=0;begun=time.monotonic();maximum_position_error=0.
    try:
        stop=False
        for part in record['chunks']:
            with np.load(source/part['file'],allow_pickle=False) as tape:
                for i,action in enumerate(tape['actions']):
                    expected_stage=int(tape['stage'][i])
                    if expected_stage>through_stage:stop=True;break
                    s=env.snapshot;f=tape['features'][i]
                    x,y=float(f[0]*368-184),float(f[1]*400+32)
                    observed=np.array([s.player.x,s.player.y])
                    # The archived compact encoder clips the offscreen sprite
                    # during non-controllable respawn/death animation only.
                    if f[14]<.5 and f[17]<.5:observed=np.clip(observed,[-184,32],[184,432])
                    error=float(np.linalg.norm(observed-[x,y]))
                    maximum_position_error=max(maximum_position_error,error)
                    if s.stage!=expected_stage or error>.02 or s.bombs!=round(float(f[4])*8) or s.lives!=round(float(f[3])*8):
                        raise ValueError(f'Archived prefix diverged at row{count}: stage{s.stage}/{expected_stage}, position error{error}, bombs{s.bombs}/{round(float(f[4])*8)}, lives{s.lives}/{round(float(f[3])*8)}')
                    _,_,done,truncated,info=env.step(int(action));count+=1
                    if count%256==0:
                        write_json(output/'progress.json',dict(pid=os.getpid(),game_pid=env.pid,phase='replaying_recorded_teacher_prefix',
                            decisions=count,stage=info['stage'],stage_frame=info['stage_frame'],native_frame=env.snapshot.frame,
                            misses=info['misses'],Bombs=info['bombs_used'],elapsed_seconds=time.monotonic()-begun,
                            maximum_position_error=maximum_position_error,optimizer_updates=0))
                    if done or truncated:raise RuntimeError('Archived opening prefix unexpectedly ended natively')
            if stop:break
    finally:
        env._teacher=teacher;env._encode=encode
    env.encoder.reset();teacher.reset()
    obs=env._encode(env.snapshot);info=env._label_current(env._info())
    write_json(output/'prefix-replay.json',dict(source=str(source),native_seed=env.native_seed,
        recorded_teacher_decisions_reexecuted=count,maximum_position_error=maximum_position_error,
        landing_stage=env.snapshot.stage,landing_stage_frame=env.snapshot.stage_frame,
        seconds=time.monotonic()-begun,teacher_replanned_decisions=0,PPO_samples=0,resources_injected=False))
    return obs,info,count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=202611070);p.add_argument('--max-decisions',type=int,default=45000)
    p.add_argument('--max-stage',type=int,default=4)
    p.add_argument('--prefix-archive',type=Path,help='Completed same-seed rule-teacher native episode;replay its stages1/2 without replanning')
    p.add_argument('--prefix-through-stage',type=int,default=2)
    p.add_argument('--max-new-decisions',type=int,default=None,
        help='Bound the new frozen evidence window after a recorded prefix')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-paused-probe-'+str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_job_paths import configure_native_temp,cleanup_native_temp
    from fly_touhou.compact_full_story import factory,story_kwargs
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    temp=configure_native_temp(a.output);env=None;capture=None;begun=time.monotonic();decisions=0
    try:
        cfg=read_shared_json(a.environment_config);cfg['seed']=a.seed;cfg['compact_full_native_teacher_support']=True
        env=factory(story_kwargs(cfg,a.output,1,'train',teacher=True)[0])
        obs,info=env.reset();capture=Capture(env,a.output)
        write_json(a.output/'base.json',dict(seed=env.native_seed,policy_seed=a.seed,snapshot=snapshot_context(env.snapshot),
            teacher_config=env._teacher.config(),options=cfg,scope='frozen rule teacher original-game diagnostic;not PPO or success evaluation'))
        env.controller.begin_input_tape();native_advance=env._advance_native
        if a.prefix_archive:
            obs,info,decisions=replay_archived_prefix(env,a.prefix_archive,a.output,through_stage=a.prefix_through_stage)
        start_decisions=decisions
        def advance(mask,ticks):
            if env.snapshot.stage not in (3,4):return native_advance(mask,ticks)
            for _ in range(ticks):
                before=env.snapshot;capture.capture_frame(before,laser=capture.laser_window>0)
                after=native_advance(mask,1);capture.inspect(before,after)
            return after
        env._advance_native=advance
        while decisions<a.max_decisions:
            if (a.output/'STOP').exists():break
            if a.max_new_decisions is not None and decisions-start_decisions>=a.max_new_decisions:break
            label=info['teacher_label'];before=info;action=int(label['action'])
            original=env._teacher.diagnostics()
            import numpy as np
            from fly_touhou.thrl_teacher import native36_to_source19
            from fly_touhou.compact_partial_risk import veto_selected_clear
            source_probabilities=native36_to_source19(env._teacher.probabilities(env.snapshot))
            before_features=obs['features'].copy();target=before['compact_targets']
            obs,_,done,truncated,info=env.step(action);decisions+=1
            elapsed=int(info['episode_frames']-before['episode_frames'])
            hit=bool(info['misses']>before['misses'] or
                (obs['features'][17]>.5 and before_features[17]<.5))
            outcome=int(hit) if target['normal'] and action<18 else -1
            partial=np.asarray(target.get('partial_point_danger',np.zeros((3,18),bool)))
            revised_partial,veto=veto_selected_clear(partial,np.asarray(action),np.asarray(outcome),
                np.asarray(target['normal']),np.asarray(elapsed))
            append_jsonl(a.output/'actions.jsonl',dict(decision=decisions,action=action,
                before_frame=before['compact_observation_frame'],after_frame=env.snapshot.frame,
                stage=before['stage'],stage_frame=before['stage_frame'],episode_frames=before['episode_frames'],
                bombs=before['bombs'],misses=before['misses'],Bombs_used=before['bombs_used'],
                source_teacher_Bomb_probability=original.get('bomb_probability'),
                revised_Bomb_probability=label['probabilities'][18],
                source_teacher_probabilities=source_probabilities.tolist(),
                revised_teacher_probabilities=label['probabilities'],
                guidance_weight=label.get('guidance_weight'),
                resource_correction=label.get('resource_reference_correction'),
                geometry_trusted=target['trusted'],normal=target['normal'],elapsed=elapsed,
                contact_outcome=outcome,contact_evidence=info.get('compact_contact_evidence'),
                partial_point_danger=partial.tolist(),partial_point_danger_after_veto=revised_partial.tolist(),
                partial_point_selected=bool(partial[0,action]) if action<18 else None,
                partial_point_veto=bool(veto),features=before_features.tolist(),margins=target['margins']))
            if decisions%128==0 or done or info['stage']>a.max_stage:
                capture.flush()
                write_json(a.output/'input-tape.json',dict(seed=env.native_seed,segments=env.controller._input_tape,
                    decisions=decisions,frame=env.snapshot.frame,stage=env.snapshot.stage))
                write_json(a.output/'progress.json',dict(pid=os.getpid(),game_pid=env.pid,phase='frozen_rule_teacher_native_probe',
                    decisions=decisions,stage=info['stage'],stage_frame=info['stage_frame'],native_frame=env.snapshot.frame,
                    misses=info['misses'],Bombs=info['bombs_used'],rows=capture.rows,
                    motion_error_count=len(capture.motion_errors),laser_bursts=len(capture.laser_bursts),
                    elapsed_seconds=time.monotonic()-begun,optimizer_updates=0))
            if done or truncated or info['stage']>a.max_stage:break
        write_json(a.output/'result.json',dict(status='completed_probe',decisions=decisions,stage=info['stage'],
            native_terminal=done,episode_summary=info.get('episode_summary'),max_stage_probe=a.max_stage,
            optimizer_updates=0,neural_model_loaded=False,seconds=time.monotonic()-begun))
    except BaseException as exc:
        write_json(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),decisions=decisions));raise
    finally:
        if capture is not None:capture.flush()
        if env is not None:
            if env.controller is not None:
                write_json(a.output/'input-tape.json',dict(seed=env.native_seed,segments=env.controller.end_input_tape(),decisions=decisions))
            env.close()
        write_json(a.output/'native-temp-cleanup.json',cleanup_native_temp(temp))


if __name__=='__main__':main()
