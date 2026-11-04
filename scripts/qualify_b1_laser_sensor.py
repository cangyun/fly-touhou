"""Read-only V5 transport verification on recorded native laser geometry.

No native process, fitting or policy efficacy claim. A frozen V4 interpreter
subprocess supplies the historical reference; it does not run the original EXE.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,sys,subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_snapshots(folder):
    from fly_touhou.state import Snapshot
    from fly_touhou.thrl_study_io import read_shared_json
    for path in sorted(folder.rglob('*.json.gz')):
        receipt=read_shared_json(path.with_suffix('.receipt.json'));compressed=path.read_bytes()
        raw=gzip.decompress(compressed)
        if sha(path)!=receipt['sha256'] or hashlib.sha256(raw).hexdigest()!=receipt['uncompressed_sha256']:
            raise ValueError('Captured native snapshot identity changed')
        row=json.loads(raw)
        yield path,Snapshot.from_dict(row['snapshot'])


def legacy(a):
    from fly_touhou.thrl_inputs import make_encoder,SEPARATED_CONTACT_INPUT
    enc=make_encoder(SEPARATED_CONTACT_INPUT);arrays={}
    for i,(path,snapshot) in enumerate(raw_snapshots(a.raw_snapshots)):
        for key,value in enc.encode(snapshot).items():arrays[f'{i}_{key}']=value
    np.savez_compressed(a.output,**arrays)


def run(a):
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_study_io import utc_now
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_line_laser_motion import ObservedLineLaserTracker,line_endpoint
    from fly_touhou.thrl_laser_observation import segment_candidates,unbounded
    from fly_touhou.state import Snapshot,Player
    from fly_touhou.thrl_inputs import make_encoder,SEPARATED_CONTACT_INPUT as OLD,SEPARATED_LASER_INPUT as NEW
    from audit_b1_line_laser_motion import native_line
    if a.output.exists():raise ValueError('Preserve previous qualification evidence')
    a.output.mkdir(parents=True)
    reference=a.output/'frozen_V4_reference.npz'
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source',str(a.source),
        '--raw-snapshots',str(a.raw_snapshots),'--legacy-runtime',str(a.legacy_runtime),
        '--motion-qualification',str(a.motion_qualification),
        '--output',str(reference),'--legacy-only'],check=True,capture_output=True,text=True)
    encoder=make_encoder(OLD);new=make_encoder(NEW);snapshot_rows=[]
    with np.load(reference) as gold:
        for i,(path,snapshot) in enumerate(raw_snapshots(a.raw_snapshots)):
            old=encoder.encode(snapshot);fresh=new.encode(snapshot)
            exact=all(np.array_equal(v,gold[f'{i}_{k}']) for k,v in old.items())
            if not exact:raise ValueError('Historical V4 source behavior changed: '+str(path))
            if not np.array_equal(old['features'],fresh['features']):raise ValueError('Global feature drift')
            if not np.array_equal(old['maps'][np.r_[0:10,15]],fresh['maps'][np.r_[0:10,15]]):
                raise ValueError('Non-laser map channels changed')
            no_lasers=not snapshot.lasers
            if no_lasers and (not np.array_equal(old['maps'],fresh['maps']) or
                    not np.array_equal(old['objects'],fresh['objects'][:,:21])):
                raise ValueError('A laser-free original snapshot changed its old modalities')
            snapshot_rows.append(dict(path=str(path),sha256=sha(path),historical_V4_exact=exact,
                no_lasers=no_lasers,globals_exact=True,nonlaser_map_channels_exact=True))
    records=continuous=unknown=0;max_position=max_velocity=0.;patterns={};sources=[]
    from fly_touhou.thrl_study_io import read_shared_json
    pinned=read_shared_json(a.motion_qualification)
    if pinned.get('schema')!='B1-native-line-endpoint-measurement-qualification-v1':
        raise ValueError('Pinned earlier native motion qualification required')
    prefix={Path(row['path']).name:row for row in pinned['sources']}
    for path in sorted(a.source.glob('worker*.jsonl')):
        with _shared_binary(path) as f:raw=f.read()
        receipt=prefix[path.name];read_length=len(raw);raw=raw[:receipt['complete_prefix_bytes']]
        if hashlib.sha256(raw).hexdigest()!=receipt['complete_prefix_sha256']:
            raise ValueError('Pinned native motion prefix changed: '+str(path))
        sources.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            pinned_complete_bytes=len(raw),subsequent_bytes_not_in_qualification=read_length-len(raw)))
        for line in raw.splitlines():
            r=json.loads(line)
            if r['before']['node_count'] or r['after']['node_count']:continue
            old,after=native_line(r['before']),native_line(r['after'])
            tracker=ObservedLineLaserTracker();episode=r['native_episode']
            first=Snapshot(r['before_frame'],r['before_frame'],Player(0.,368.),stage=r['stage'],lasers=(old,))
            last=Snapshot(r['after_frame'],r['after_frame'],Player(0.,368.),stage=r['stage'],lasers=(after,))
            tracker.observe(first,native_episode=episode);motion=tracker.observe(last,native_episode=episode)
            row=motion.lines[0];candidates=segment_candidates(last,motion)
            if not candidates:continue
            token=np.asarray(candidates[0][2]);center=np.array([0.,368.])+unbounded(token[6:8],[384,448])
            half=unbounded(token[21:23],[384,448]);va=unbounded(token[8:10],12)-unbounded(token[23:25],12)
            vb=unbounded(token[8:10],12)+unbounded(token[23:25],12)
            position_error=max(np.linalg.norm(center-half-row.origin),np.linalg.norm(center+half-line_endpoint(after)))
            max_position=max(max_position,float(position_error));records+=1
            if bool(token[26])!=r['age_continuous']:raise ValueError('Unknown/stationary sensor distinction lost')
            if row.motion_observed:
                continuous+=1;patterns[r['pattern']]=patterns.get(r['pattern'],0)+1
                error=max(np.linalg.norm(va-r['measured_origin_velocity']),np.linalg.norm(vb-r['measured_endpoint_velocity']))
                max_velocity=max(max_velocity,float(error))
            else:
                unknown+=1
                if token[26]!=0:raise ValueError('Invalid motion was accepted')
    if (records!=1100 or continuous!=890 or unknown!=210 or len(snapshot_rows)!=20
            or max_position>2e-4 or max_velocity>1e-5):
        failure=dict(status='failed',records=records,continuous=continuous,unknown=unknown,
            snapshots=len(snapshot_rows),max_position=max_position,max_velocity=max_velocity)
        write(a.output/'failure.json',failure)
        raise ValueError('Expected bounded native qualification cohort/tolerance differs: '+json.dumps(failure))
    result=dict(schema='B1-observed-laser-V5-sensor-qualification-v1',status='qualified',utc=utc_now(),
        old_input_schema=OLD,new_input_schema=NEW,old_V4_all_snapshots_exact=True,
        full_native_snapshots=len(snapshot_rows),snapshot_results=snapshot_rows,
        native_laser_pairs=records,continuous_motion_pairs=continuous,unknown_motion_pairs=unknown,
        patterns=patterns,max_endpoint_roundtrip_error=max_position,max_velocity_roundtrip_error=max_velocity,
        frozen_reference=str(reference),frozen_reference_sha256=sha(reference),
        frozen_runtime=str(a.legacy_runtime),frozen_manifest_sha256=sha(a.legacy_runtime/'source_manifest.json'),
        motion_qualification=str(a.motion_qualification),motion_qualification_sha256=sha(a.motion_qualification),
        source=sources,laser_transport_player_position_is_synthetic=True,
        old_snapshot_diagnostics_not_used_for_fitting=True,new_original_games=0,optimizer_updates=0,
        native_collision_state_verified=False,curve_motion_verified=False,future_motion_guaranteed=False,
        native_efficacy_demonstrated=False,live_training_input_changed=False)
    write(a.output/'report.json',result)
    print(json.dumps({k:result[k] for k in ('status','full_native_snapshots','native_laser_pairs','continuous_motion_pairs',
        'unknown_motion_pairs','old_V4_all_snapshots_exact','max_endpoint_roundtrip_error','max_velocity_roundtrip_error')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['source','raw-snapshots','legacy-runtime','motion-qualification','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--legacy-only',action='store_true');a=p.parse_args()
    sys.path.insert(0,str((a.legacy_runtime if a.legacy_only else ROOT)/'src'))
    legacy(a) if a.legacy_only else run(a)
