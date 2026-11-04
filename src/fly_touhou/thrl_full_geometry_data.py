"""Compact, lossless geometry from complete train-partition original games.

The source remains immutable. Test failures are not an input to this builder.
Maps are inspected for laser presence, then discarded; numeric values stay FP32.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import io
import json
import zlib

import numpy as np

from .thrl_durable_records import write_json as write
from .thrl_study_io import read_shared_json as read,utc_now


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_parts(count):
    if type(count) is not int or count<1:raise ValueError('A nonempty complete trajectory is required')
    # Broad coverage plus denser late-game coverage. Selection uses only row
    # positions, never actions, outcomes of held games, or independent tests.
    return sorted(set(np.rint(np.linspace(0,count-1,min(24,count))).astype(int).tolist())
        |set(range(max(0,count-96),count,4)))


def partition_seeds(all_seeds,held_seeds):
    all_seeds=set(map(int,all_seeds));held=set(map(int,held_seeds))
    if (len(all_seeds)!=30 or len(held)!=6 or not held<all_seeds
            or any(not 0<=seed<=65535 or seed%10>=6 for seed in all_seeds)):
        raise ValueError('Require30 original training seeds and6 trajectory-held seeds, never development/test seeds')
    return held


def calibrated_state(parent,point):
    import torch
    expected={key.removeprefix('risk.') for key in parent if key.startswith('risk.')}
    if set(point)!=expected:raise ValueError('The fitted point-risk state has different keys')
    for name,value in point.items():
        old=parent['risk.'+name]
        if value.shape!=old.shape or value.dtype!=old.dtype:
            raise ValueError('The fitted point-risk state has different shape or dtype')
    updated={key:value.clone() for key,value in parent.items()}
    for name,value in point.items():updated['risk.'+name]=value.clone()
    changed=[key for key in updated if not torch.equal(updated[key],parent[key])]
    if not changed:raise ValueError('The fitted candidate did not change any weights')
    if any(not key.startswith('risk.') for key in changed):raise ValueError('A non-point neural branch changed')
    return updated,changed


def _read_part(job):
    folder,index,part,held=job;spec=index['shards'][part]
    if Path(spec['name']).name!=spec['name']:raise ValueError('Unsafe shard path')
    payload=(folder/spec['name']).read_bytes()
    if len(payload)!=spec['bytes'] or hashlib.sha256(payload).hexdigest()!=spec['sha256']:
        raise ValueError('Native archive shard identity differs')
    raw=zlib.decompress(payload)
    if len(raw)!=spec['uncompressed_bytes'] or hashlib.sha256(raw).hexdigest()!=spec['uncompressed_sha256']:
        raise ValueError('Native archive decompression identity differs')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        n=spec['rows'];indices=np.arange(0,max(0,n-1),4)
        if not len(indices):return None
        f=z['features'];o=z['objects']
        if f.shape!=(n,25) or o.shape!=(n,96,29) or f.dtype!=np.float32 or o.dtype!=np.float32:
            raise ValueError('Require the original FP32 V5 feature/object fields')
        if not np.isfinite(f).all() or not np.isfinite(o).all():raise ValueError('Nonfinite original geometry')
        seed=index['metadata']['native_seed']
        if np.any(z['native_seeds']!=seed):raise ValueError('Mixed original seeds inside a trajectory')
        # Source channel10 is full-map projectile/laser occupancy. This gives a
        # stricter laser exclusion for native point-contact diagnostics than
        # merely checking the selected96 object tokens.
        lasers=np.any(z['maps'][indices,10]>0,axis=(1,2))
        elapsed=z['after_observation_frames'][indices]-z['before_frames'][indices]
        adjacent=z['before_frames'][indices+1]==z['after_observation_frames'][indices]
        miss=z['native_misses'][indices+1]-z['native_misses'][indices]
        result=dict(features=f[indices].copy(),objects=o[indices].copy(),
            next_features=f[indices+1].copy(),actions=z['actions'][indices].copy(),
            stage=z['stage'][indices].copy(),stage_frame=z['stage_frame'][indices].copy(),
            native_seed=np.full(len(indices),seed,np.int64),held=np.full(len(indices),held,bool),
            source_row=spec['start']+indices,has_laser_map=lasers,actual_elapsed=elapsed,
            adjacent_original_observations=adjacent,native_miss_delta=miss)
    return result


def build_archive(source,held_seeds,output,progress,check_stop,*,threads=4):
    source=Path(source);output=Path(output)
    if not 1<=threads<=4:raise ValueError('Bounded extraction threads required')
    root=read(source/'result.json')
    if root['status']!='completed' or root['complete_native_episodes']!=30 or root['native_seed_split']!='train':
        raise ValueError('Thirty complete original training trajectories are required')
    indices=[]
    for path in sorted((source/'trajectories').glob('world*/index.json')):
        index=read(path)
        if (index['status']!='completed' or index['natural_episode_complete'] is not True
                or index['metadata']['input_schema']!='semantic-separated-observed-laser-segments-v5'
                or index['metadata']['native_seed_split']!='train'):
            raise ValueError('Incomplete or incompatible native trajectory')
        indices.append((path,index))
    held=partition_seeds([x['metadata']['native_seed'] for _,x in indices],held_seeds)
    jobs=[];pins=[]
    for path,index in indices:
        parts=selected_parts(len(index['shards']));seed=index['metadata']['native_seed']
        jobs.extend((path.parent,index,part,seed in held) for part in parts)
        pins.append(dict(index=str(path),index_sha256=digest(path),seed=seed,held=seed in held,
            selected_shards=[index['shards'][i] for i in parts]))
    values=[]
    count=0
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for begin in range(0,len(jobs),threads*2):
            check_stop()
            for item in executor.map(_read_part,jobs[begin:begin+threads*2]):
                count+=1
                if item is not None:values.append(item)
            if count%24==0 or count==len(jobs):progress('extracting_full_native_geometry',
                source_shards_completed=count,source_shards_target=len(jobs))
    data={key:np.concatenate([row[key] for row in values]) for key in values[0]}
    if not np.array_equal(np.rint(data['features'][:,7]*6).astype(int),data['stage']):
        raise ValueError('Native stage labels and encoded stage differ')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'geometry.npz',**data)
    stats={}
    for name,mask in [('train',~data['held']),('held',data['held'])]:
        f=data['features'][mask];radius=f[:,13]*64
        stats[name]=dict(rows=int(mask.sum()),native_seeds=sorted(set(data['native_seed'][mask].tolist())),
            stage_counts={str(i):int((data['stage'][mask]==i).sum()) for i in range(1,7)},
            player_circle_min=float(radius.min()),player_circle_max=float(radius.max()),
            enlarged_player_states=int((radius>3).sum()))
    manifest=dict(schema='B1-full-original-training-geometry-v1',utc=utc_now(),source=str(source),
        source_result_sha256=digest(source/'result.json'),rows=len(data['features']),partitions=stats,
        immutable_source_pins=pins,data_sha256=digest(output/'geometry.npz'),original_FP32=True,
        independent_test_states_used=False,development_states_used=False,maps_kept=False,
        source_curves_not_reinterpreted_as_known_velocity=True)
    write(output/'manifest.json',manifest)
    return data,manifest
