"""Compare old/new geometry on the same saved original Wall scenes."""
import argparse
import gzip
import json
import pickle
import time
from pathlib import Path
import numpy as np
from fly_touhou.thrl_dodge_diagnostics import _shared_binary
from fly_touhou.thrl_study_io import read_shared_json
from fly_touhou.compact_story_sensation import CompactStoryEncoder


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with _shared_binary(args.capture/'wall-native.jsonl') as stream:
        frames={row['native_frame']:row for row in map(json.loads,stream)}
    programs=[r['program_hex'] for r in read_shared_json(args.capture/'wall-programs.json')['records']]
    results=[]
    for file in sorted(args.capture.glob('native-scene-*.pickle.gz')):
        with gzip.open(file,'rb') as stream:snapshot=pickle.load(stream)
        row=frames[snapshot.frame];outputs=[]
        for enabled in (False,True):
            encoder=CompactStoryEncoder();encoder.primary_attack_target=True
            for name in ('anticipate_arming','growth_motion_envelope','native_motion_forecast',
                         'native_focus_gate','native_curve_laser_geometry','unknown_laser_player_motion'):
                setattr(encoder.base,name,True)
            if enabled:snapshot.extra['native_reactive_orbs']=dict(row,programs=programs,motion_time_rate=1.)
            started=time.perf_counter();encoder.encode(snapshot);e=encoder.last_result;d=e['details']
            forecasts=e['source_native_forecasts']
            outputs.append(dict(enabled=enabled,seconds=time.perf_counter()-started,
                trusted=d['trusted'].tolist(),uncertainty=d['uncertainty'],margins=d['margins'].tolist(),
                reactive_candidates=sum(x['supported'] for x in forecasts.get('reactive_bullets',())),
                untriggered_prefixes=forecasts.get('reactive_untriggered_prefixes',0),
                unsupported=sum(not x['supported'] for x in forecasts.get('reactive_bullets',()))))
        change=np.asarray(outputs[1]['margins'])-outputs[0]['margins']
        results.append(dict(stage_frame=snapshot.stage_frame,phase=row['phase_names'],
            before=outputs[0],after=outputs[1],max_margin_change=float(np.abs(change).max())))
    report=dict(scope='Same native scenes; no student games or parameter updates',results=results)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(scenes=len(results),before_trusted=[r['before']['trusted'] for r in results],
        after_trusted=[r['after']['trusted'] for r in results],maximum_margin_change=max(r['max_margin_change'] for r in results))))


if __name__=='__main__':main()
