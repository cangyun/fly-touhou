"""Original-engine line EX fixtures on a separate private desktop, no policy."""
from pathlib import Path
import argparse,json,sys,struct,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from fly_touhou.nbnm_run import runtime,read,environment,task_settings
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.native import ProcessMemory
    from fly_touhou.native_line_laser_paths import predict_native_line
    temp=runtime(a.output);env=None;mem=None;results=[]
    def command(op,ii=(0,0,0,0),fs=(0.,0.,0.,0.)):return struct.pack('<4f4iIIi',*fs,*ii,op,0,0)
    cases=[('reflection',170.,200.,0.,[command(64,(2,13,0,0),(-999.,0,0,0))]),
           ('polar_native_noop',0.,200.,0.,[command(8,(8,0,0,0),(.2,.03,0,0))]),
           ('acceleration',0.,200.,0.,[command(4,(6,0,0,0),(.2,.3,0,0))]),
           ('turn',0.,200.,0.,[command(16,(3,2,0,0),(.3,2.,0,0))])]
    try:
        spec=read(a.config);cfg=environment(spec,evaluation=True);cfg['nbnm']=task_settings(spec,spec['tasks']['s2_BossCard1'])
        profile=read(cfg['profile']);profile.setdefault('hooks',{})['native_ex_program']='th14-live-ex-program-v1'
        write_json(a.output/'profile.json',profile);cfg['profile']=str(a.output/'profile.json')
        options=story_kwargs(cfg,a.output/'native',1,'development',teacher=False)[0];options.pop('online_teacher_revision',None)
        env=factory(options)
        placeholder={k:np.zeros(s.shape,s.dtype) for k,s in env.observation_space.spaces.items()}
        env._encode=lambda snapshot:placeholder
        env.reset(options={'native_seed':60006})
        mem=ProcessMemory(env.controller.reader.mem.pid,writable=True)
        mem.write(mem.integer(mem.base+0xdb67c)+99012,struct.pack('<if',100000,100000.))
        for name,x,y,angle,commands in cases:
            if results:
                mem.close();mem=None;env.reset(options={'native_seed':60006})
                mem=ProcessMemory(env.controller.reader.mem.pid,writable=True)
                mem.write(mem.integer(mem.base+0xdb67c)+99012,struct.pack('<if',100000,100000.))
            for _ in range(2000):
                snap=env.controller.reader.read();lines=snap.extra.get('native_line_lasers',{}).get('lines',[])
                if lines:break
                env.controller.advance(0,1)
            if not lines:raise RuntimeError('No original line available')
            addr=lines[0]['id']
            mem.write(addr+0x54,struct.pack('<11f',x,y,0,3.,0.,0.,angle,40.,16.,3.,0.))
            mem.write(addr+0x594,struct.pack('<II',0,0));mem.write(addr+0x5b4,struct.pack('<i',0))
            mem.write(addr+0x5d0,struct.pack('<3f',60.,0.,600.))
            mem.write(addr+0x5f4,b''.join(commands).ljust(18*44,b'\0'))
            env.controller.reader.mem.invalidate_prefetch();snap=env.controller.reader.read()
            raw=next(r for r in snap.extra['native_line_lasers']['lines'] if r['id']==addr)
            f=predict_native_line(raw,horizon=24)
            row=dict(case=name,raw=raw,forecast=f,steps=[],max_error=0.,samples=0)
            before={r['id'] for r in snap.extra['native_line_lasers']['lines']}
            for t in range(1,25):
                snap=env.controller.advance(0,1);lines=snap.extra['native_line_lasers']['lines']
                actual=next((r for r in lines if r['id']==addr),None)
                if actual:
                    if actual['age']!=raw['age']+t:raise RuntimeError('Native line clock did not advance')
                    error=max(np.max(abs(np.array(f['origin'][t])-actual['origin'])),abs(f['length'][t]-actual['length']),
                        abs(np.arctan2(np.sin(f['angle_by_frame'][t]-actual['angle']),np.cos(f['angle_by_frame'][t]-actual['angle']))))
                    row['max_error']=max(float(error),row['max_error']);row['samples']+=1
                row['steps'].append(dict(tick=t,frame=snap.frame,actual=actual,children=[r for r in lines if r['id'] not in before]))
            append_jsonl(a.output/'cases.jsonl',row)
            results.append({k:v for k,v in row.items() if k not in ('raw','forecast','steps')});write_json(a.output/'progress.json',results)
        write_json(a.output/'report.json',dict(status='complete',cases=results,training_updates=0))
    except BaseException:
        write_json(a.output/'failure.json',dict(error=traceback.format_exc(),cases=results));raise
    finally:
        if mem:mem.close()
        if env:env.close()
        cleanup_native_temp(temp)


if __name__=='__main__':main()
