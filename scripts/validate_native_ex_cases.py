"""Exercise EX instructions in a separately owned original-game process.

Only this script's private-desktop process is modified. These are physics
fixtures, never training samples or evidence of NBNM game completion.
"""
from pathlib import Path
import argparse,json,sys,struct,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='*')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.nbnm_run import runtime,read,environment,task_settings
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.native import ProcessMemory
    from fly_touhou.native_bullet_program import REVISION,predict_program
    temp=runtime(a.output);env=None;mem=None;results=[]
    def command(op,ints=(0,0,0,0),fs=(0.,0.,0.,0.),concurrent=0):
        return struct.pack('<4f4iIIi',*fs,*ints,op,concurrent,0)
    cases=[]
    for side,x,y,vx,vy in [(1,0,-35,0,-3),(2,0,480,0,3),(4,-220,200,-3,0),(8,220,200,3,0)]:
        cases.append((f'wrap_{side}',(x,y,vx,vy),[command(4096,(1,side,0,0)),command(8,(8,0,0,0),(.2,.03,0,0))]))
    for mode in (0,1):cases.append((f'retention_{mode}',(0,200,1,0),[command(256,(3,mode,0,0)),command(8,(8,0,0,0),(.2,.03,0,0))]))
    for label,x,y,vx,vy in [('away',220,200,1,0),('toward',220,200,-1,0),('corner',220,-30,-1,1)]:
        cases.append(('retention_'+label,(x,y,vx,vy),[command(256,(24,1,0,0)),command(8,(8,0,0,0),(.2,.03,0,0))]))
    for mode in (0,1,2,3,4):cases.append((f'turn_{mode}',(0,200,1,0),[command(16,(3,2,mode,0),(.3,2.,0.,0.))]))
    cases.append(('aim_acceleration',(0,200,1,0),[command(4,(6,0,0,0),(.2,999.,.1,0.))]))
    for mode in (5,6):cases.append((f'random_turn_{mode}',(0,200,1,0),[command(16,(3,2,mode,0),(.6,2.,0.,0.))]))
    for animation in (0,1,2):
        cases.append((f'animation_{animation}',(0,200,1,0),[command(2,(animation,0,0,0)),command(8,(8,0,0,0),(.2,.03,0,0))]))
    cases.append(('child_fixed',(0,200,1,0),[command(8192,(1,2,3,1),(0.,.3,2.,0.)),
        command(16384,(7,2,0,0)),command(8,(8,0,0,0),(.2,.03,0,0))]))
    for mode in (0,6,7,8):
        cases.append((f'child_mode{mode}',(0,200,1,0),[command(8192,(mode,2,4,1),(.2,.3,2.,1.5)),
            command(16384,(5,2,1,0)),command(2,(1,0,0,0)),command(2097152,(20,0,0,0),(1.,-999.,0.,0.))]))
    cases.append(('child_after_animation',(0,200,1,0),[command(2,(1,0,0,0)),
        command(8192,(8,3,3,1),(.2,.3,2.,1.5)),command(16384,(5,2,1,0)),command(2,(1,0,0,0))]))
    cases.append(('animation_wrap',(190,200,3,0),[command(4096,(1,15,0,0)),command(2,(1,0,0,0),concurrent=1)]))
    cases.append(('animation_bounce',(190,200,3,0),[command(64,(1,8,0,0),(-999.,0.,0.,0.)),command(2,(1,0,0,0),concurrent=1)]))
    cases.append(('moving_turn',(0,200,1,0),[command(16,(3,2,1,0),(.3,2.,0.,0.))]))
    cases.append(('moving_aim_acceleration',(0,200,1,0),[command(4,(6,0,0,0),(.2,999.,.1,0.))]))
    if a.cases:cases=[c for c in cases if c[0] in a.cases]
    try:
        spec=read(a.config);cfg=environment(spec,evaluation=True);cfg['nbnm']=task_settings(spec,spec['tasks']['s1_BossCard1'])
        profile=read(cfg['profile']);profile.setdefault('hooks',{})['native_ex_program']=REVISION
        write_json(a.output/'profile.json',profile);cfg['profile']=str(a.output/'profile.json')
        cfg['nbnm'].update(demonstration_route_teacher={'enabled':False},live_route_planner={'enabled':False})
        options=story_kwargs(cfg,a.output/'native',1,'development',teacher=False)[0];options.pop('online_teacher_revision',None)
        env=factory(options);env.reset(options={'native_seed':60006})
        mem=ProcessMemory(env.controller.reader.mem.pid,writable=True)
        player_address=mem.integer(mem.base+0xdb67c)
        mem.write(player_address+99012,struct.pack('<if',100000,100000.))
        for name,xyv,commands in cases:
            for _ in range(300):
                snap=env.controller.reader.read();rows=snap.extra['hazard_metadata']['bullets']
                live=[r for r in rows if r['native_state']==1]
                if live:break
                env.controller.advance(0,1)
            if not live:raise RuntimeError('No original live bullet available for fixture')
            addr=live[0]['id'];x,y,vx,vy=xyv;speed=float(np.hypot(vx,vy));angle=float(np.arctan2(vy,vx))
            mem.write(addr+0xbc0,struct.pack('<8f',x,y,0,vx,vy,0,speed,angle))
            mem.write(addr+0xc04,struct.pack('<I',0));mem.write(addr+0xc20,struct.pack('<i',0))
            mem.write(addr+0xbe8,struct.pack('<i',0));mem.write(addr+0xc0e,struct.pack('<H',1))
            mem.write(addr+0x20,struct.pack('<I',live[0]['native_flags']&~0x200))
            mem.write(addr+0xc28,b''.join(commands).ljust(18*44,b'\0'))
            mem.write(addr+0x12bc,struct.pack('<3f',x,y,0))
            if hasattr(env.controller.reader.mem,'invalidate_prefetch'):env.controller.reader.mem.invalidate_prefetch()
            snap=env.controller.reader.read();raw=next(r for r in snap.extra['hazard_metadata']['bullets'] if r['id']==addr)
            path=np.repeat([[snap.player.x,snap.player.y]],33,axis=0)
            mask=0
            if name.startswith('moving_'):
                sign=1 if snap.player.x<=0 else -1;mask=128 if sign==1 else 64
                path[:,0]=np.clip(path[:,0]+sign*4.5*np.arange(33),-184,184)
            rng_seed=struct.unpack('<H',mem.read(mem.base+0xdb510,2))[0]
            predicted=predict_program(raw,32,player_path=path,rng_seed=rng_seed if name.startswith('random') else None)
            before_ids={r['id'] for r in snap.extra['hazard_metadata']['bullets']}
            row=dict(case=name,raw=raw,forecast=predicted,steps=[],max_position_error=0.,child_samples=0,child_outside=0,max_child_error=0.)
            def distance(forecast,tick,point):
                bounds=np.asarray(forecast['bounds'][tick]);delta=bounds[1]-bounds[0];point=np.asarray(point)
                ratio=np.clip((point-bounds[0])@delta/max(delta@delta,1e-30),0,1)
                return max(0.,float(np.linalg.norm(point-bounds[0]-ratio*delta))-forecast['extra_radii'][tick])
            previous_frame=snap.frame
            for tick in range(1,33):
                snap=env.controller.advance(mask,1)
                if mask and np.linalg.norm(np.array([snap.player.x,snap.player.y])-path[tick])>.001:
                    raise RuntimeError('Moving-player fixture differs from supplied candidate path')
                if snap.frame!=previous_frame+1:raise RuntimeError('Original frame did not advance one tick')
                previous_frame=snap.frame
                actual=next((r for r in snap.extra['hazard_metadata']['bullets'] if r['id']==addr),None)
                if actual is None:row.setdefault('removed_at',tick)
                error=distance(predicted,tick,actual['native_position']) if actual else 0.
                if actual and actual['native_timer']!=raw['native_timer']+tick:
                    row['timer_change']=dict(tick=tick,before=raw['native_timer'],after=actual['native_timer'],
                        same_program=actual.get('native_ex_program_hex')==raw['native_ex_program_hex'],actual=actual)
                    if actual.get('native_ex_program_hex')!=raw['native_ex_program_hex']:
                        row['slot_reused_at']=tick;break
                row['max_position_error']=max(row['max_position_error'],error)
                row['steps'].append(dict(tick=tick,actual=actual,error=error))
                if name.startswith('child'):
                    children=[b for b in snap.extra['hazard_metadata']['bullets'] if b['id'] not in before_ids and b.get('native_ex_program_hex')==raw['native_ex_program_hex']]
                    for b in children:
                        value=min((distance(p,tick,b['native_position']) for p in predicted.get('virtual_children',())),default=float('inf'))
                        row['child_samples']+=1;row['child_outside']+=value>.002;row['max_child_error']=max(row['max_child_error'],value)
                    row['steps'][-1]['children']=children
            append_jsonl(a.output/'cases.jsonl',row)
            results.append({k:v for k,v in row.items() if k not in ('raw','forecast','steps')})
            write_json(a.output/'progress.json',results)
        write_json(a.output/'report.json',dict(status='complete',scope='synthetic EX fixtures in original engine',cases=results,training_updates=0))
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc(),cases=results));raise
    finally:
        if mem:mem.close()
        if env:env.close()
        cleanup_native_temp(temp)


if __name__=='__main__':main()
