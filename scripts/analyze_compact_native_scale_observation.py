"""Compare the new prediction with passive original-game scale/motion pairs."""
import argparse
from pathlib import Path
import numpy as np
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.native_bullet_release import predict_bullet_release


def analyze(source,output):
    data=read_shared_json(Path(source));groups={}
    for pair in data.get('pairs',[]):
        horizon=pair['frames'];before=dict(pair['before'],native_ex_scale_forecast_enabled=True)
        prediction=predict_bullet_release(before,horizon)
        row=groups.setdefault(str(horizon),dict(pairs=0,covered=0,max_scale_error=0.,max_position_envelope_error=0.))
        row['pairs']+=1
        if not all(prediction['known_by_frame'][1:]):continue
        row['covered']+=1
        row['max_scale_error']=max(row['max_scale_error'],abs(
            prediction['collision_scale_by_frame'][horizon]-pair['after']['native_scale']))
        a,b=np.asarray(prediction['bounds'][horizon]);actual=np.asarray(pair['after']['native_position'])
        segment=b-a;denominator=float(segment@segment)
        fraction=np.clip(float((actual-a)@segment)/denominator,0,1) if denominator else 0.
        extra=prediction.get('extra_radii',[0.]*(horizon+1))[horizon]
        error=max(0.,float(np.linalg.norm(actual-a-fraction*segment))-extra)
        row['max_position_envelope_error']=max(row['max_position_envelope_error'],error)
    result=dict(source=str(source),observation_status=data['status'],by_frames=groups,
        scope='Paired observations within one frozen original episode; not independent games or intervention outcomes',
        ready_to_review=bool(groups) and all(groups.get(str(h),{}).get('covered',0)>=32 for h in (4,12,24)),
        all_observed_covered_pairs_agree=bool(groups) and all(
            r['covered']>0 and r['max_scale_error']<=1e-5 and r['max_position_envelope_error']<=.01 for r in groups.values()))
    write_json(Path(output),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('output')
    args=parser.parse_args();print(analyze(args.source,args.output))
