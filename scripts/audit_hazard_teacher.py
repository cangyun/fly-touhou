"""Read-only native v3 teacher failure and movement calibration audit."""
from collections import Counter
from pathlib import Path
import argparse
import gzip
import json
import math
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.connectome import file_sha256
from fly_touhou.state import decode_action


def candidates(snapshot, count=3):
    player = snapshot.get('player')
    if player is None:
        return []
    raw = {b['id']: b for b in snapshot['extra']['hazard_metadata']['bullets']}
    result = []
    for bullet in snapshot['bullets']:
        row = raw[bullet['id']]
        distance = math.hypot(bullet['x']-player['x'], bullet['y']-player['y'])
        result.append({'id':bullet['id'], 'center_distance':distance,
            'current_circle_clearance':distance-bullet['radius']-player['radius'],
            'native_state':row['native_state'], 'native_flags':row['native_flags'],
            'native_timer':row['native_timer'], 'native_animation_gate':row['native_animation_gate'],
            'position':[bullet['x'],bullet['y']], 'velocity':[bullet['vx'],bullet['vy']],
            'radius':bullet['radius'], 'legacy_active':bullet['active']})
    return sorted(result, key=lambda r:r['current_circle_clearance'])[:count]


def audit(path):
    speeds, raw_states, modes, focus_counts = {}, Counter(), Counter(), Counter()
    hits, summaries, commands, frames = [], [], Counter(), 0
    minimum = [math.inf,math.inf]
    maximum = [-math.inf,-math.inf]
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            record=json.loads(line)
            if record['record_type']=='metadata':
                header=record
                continue
            if record['record_type']=='episode_summary':
                summaries.append({k:record[k] for k in ('episode_id','steps','native_frames','max_power',
                    'edge_decisions','poc_entries','teacher_modes')})
                continue
            if record['record_type']!='transition':
                continue
            before,after=record['snapshot'],record['next_snapshot']
            p,q=before.get('player'),after.get('player')
            diagnostic=record['teacher_diagnostic']
            modes[diagnostic['mode']]+=1
            raw_states.update(str(b['native_state']) for b in before['extra']['hazard_metadata']['bullets'])
            frames+=record['elapsed_frames']
            if p is None or q is None:
                continue
            action=record['action']
            dx,dy,focused=decode_action(action%18)
            focus_counts[str(focused)]+=1
            commands['total']+=1
            for i,key in enumerate(('x','y')):
                minimum[i]=min(minimum[i],p[key],q[key])
                maximum[i]=max(maximum[i],p[key],q[key])
            if action>=18:
                commands['bomb_requested']+=1
                commands['bomb_requested_while_invulnerable']+=p['invulnerable_frames']>0
                commands['bomb_requested_in_state4']+=p['state']==4
            if p['state']!=4 and q['state']==4:
                hits.append({'episode_id':record['episode_id'],'step':record['step'],'raw_line_1based':number,
                    'frame':before['frame'],'stage':before['stage'],'stage_frame':before['stage_frame'],
                    'action':action,'player_before':p,'player_after':q,
                    'lives_before':before['lives'],'bombs_before':before['bombs'],
                    'mode':diagnostic['mode'],'risk':diagnostic['risk'],
                    'raw_hazard_counts':diagnostic['dynamic_hazard_prediction']['hazards'],
                    'nearest_bullet_candidates_before':candidates(before),
                    'nearest_bullet_candidates_after':candidates(after),
                    'laser_count_before':len(before['lasers']),'laser_count_after':len(after['lasers'])})
            # Measured displacement only. These exclusions avoid menu, death,
            # respawn and wall-clamped samples; do not infer unobserved bounds.
            if (p['state']==q['state']==1 and before['stage']==after['stage']
                    and before['lives']==after['lives'] and action<18
                    and not before['dialogue'] and not after['dialogue']
                    and abs(p['x'])<170 and abs(q['x'])<170
                    and 30<p['y']<418 and 30<q['y']<418 and (dx or dy)):
                displacement=math.hypot(q['x']-p['x'],q['y']-p['y'])
                speed=displacement/record['elapsed_frames']
                if speed>0:
                    group=f'{"focused" if focused else "normal"}:{dx},{dy}'
                    speeds.setdefault(group,[]).append(speed)
    calibration={key:{'samples':len(values),'median_pixels_per_native_frame':statistics.median(values),
        'minimum':min(values),'maximum':max(values)} for key,values in sorted(speeds.items())}
    return {'schema':'native-hazard-teacher-audit-v1','source':str(path),'source_sha256':file_sha256(path),
        'teacher_kind':header['teacher_kind'],'encoder_schema':header['encoder']['schema'],
        'native_frames':frames,'episodes':summaries,'commands':dict(commands),'modes':dict(modes),
        'raw_bullet_state_occurrences':dict(raw_states),'movement_calibration':calibration,
        'observed_player_extrema':{'minimum':minimum,'maximum':maximum},'hit_entries':hits,
        'limits':['Nearest geometry is a candidate explanation, not proof of the native collision source.',
            'Recorded teacher forecasts do not prove alternate action survival.',
            'Two random native campaigns do not isolate a causal teacher-version effect.'],
        'metadata_erratum':{'field':'teacher_diagnostic.student_observation_modified',
            'recorded':False,'correct_interpretation':'The collector explicitly used the v3 hazard encoder; only the complete old v2 prefix was unchanged. The inherited diagnostic flag is stale; original records are preserved.'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    output=options.output.resolve()
    if output.exists():
        raise FileExistsError('Audit evidence is never overwritten')
    result=audit(options.trace.resolve(strict=True))
    result['audit_source_sha256']=file_sha256(__file__)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'output':str(output),'sha256':file_sha256(output),
        'hits':len(result['hit_entries']),'native_frames':result['native_frames'],
        'commands':result['commands'],'calibration':result['movement_calibration']},ensure_ascii=False))


if __name__=='__main__':
    main()
