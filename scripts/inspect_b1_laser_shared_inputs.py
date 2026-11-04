"""Read-only inspection of existing V5 shared observations; no native calls.

The repeated read is a transport-stability check, not an atomic game-frame
proof. Never unlink borrowed mappings, inject a controller or alter actions.
"""
from pathlib import Path
import argparse,sys
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_laser_observation import unbounded


def inspect(path):
    ownership=read(path);identity=ownership['owner']
    if ownership['status']!='active' or abs(psutil.Process(identity['pid']).create_time()-identity['create_time'])>.01:
        raise ValueError('Pinned active vector owner required')
    descriptors=ownership['shared_memory'];handles={};views={}
    try:
        for key in ('features','objects','maps'):
            row=descriptors[key];shape=tuple(row['shape'])
            if row['dtype']!='float32' or not 1<=shape[0]<=30:raise ValueError('Bounded managed FP32 mappings required')
            handles[key]=SharedMemory(name=row['name'],create=False)
            views[key]=np.ndarray(shape,dtype=np.float32,buffer=handles[key].buf)
            views[key].setflags(write=False)
        n=len(views['features'])
        if (views['features'].shape!=(n,25) or views['objects'].shape!=(n,96,29)
                or views['maps'].shape!=(n,16,92,96)):raise ValueError('Explicit V5 input shapes required')
        f=views['features'].copy();o=views['objects'].copy();m=views['maps'][:,10:13].copy()
        stable=np.array_equal(f,views['features']) and np.array_equal(o,views['objects'])
        rows=[]
        for world in range(n):
            selected=(o[world,:,0]>.5)&((o[world,:,3]>.5)|(o[world,:,4]>.5));lasers=o[world,selected]
            details=[]
            for token in lasers[:3]:
                v=unbounded(token[8:10],12);dv=unbounded(token[23:25],12)
                details.append(dict(line=bool(token[3]>.5),motion_observed=bool(token[26]>.5),
                    origin_velocity=(v-dv).tolist(),endpoint_velocity=(v+dv).tolist(),
                    width=float(unbounded(token[10],128)*2),native_state_bounded=float(token[27]),
                    native_iframes_bounded=float(token[28])))
            rows.append(dict(worker=world,stage=int(round(float(f[world,7])*6)),
                laser_segment_tokens=len(lasers),known_motion_segments=int((lasers[:,26]>.5).sum()),
                nonzero_map_velocity_cells=int(np.any(m[world,1:]!=0,axis=0).sum()),examples=details))
        return dict(schema='B1-V5-passive-shared-input-inspection-v1',utc=utc_now(),ownership=str(path),owner=identity,
            workers=n,repeated_read_stable=stable,atomic_native_frame_claimed=False,rows=rows,
            total_selected_laser_segments=sum(row['laser_segment_tokens'] for row in rows),
            total_known_motion_segments=sum(row['known_motion_segments'] for row in rows),
            native_instances_started=0,native_calls=0,policy_updates=0,desktop_input=False,mappings_modified=False)
    finally:
        views.clear()
        for handle in handles.values():handle.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--ownership',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous inspections')
    result=inspect(a.ownership);write(a.output,result)
    print({k:result[k] for k in ['workers','repeated_read_stable','total_selected_laser_segments','total_known_motion_segments','native_instances_started']})
