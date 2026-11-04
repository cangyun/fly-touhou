"""Load only complete, source-linked original-game motion measurements."""
from dataclasses import replace
import json
import math
from pathlib import Path

from .connectome import file_sha256


def calibrated_planner(config, path, *, exe_sha256):
    path=Path(path).resolve(strict=True)
    report=json.loads(path.read_text(encoding='utf-8-sig'))
    if (report.get('schema')!='original-th14-motion-calibration-v1'
            or report.get('all_measurements_complete') is not True
            or report.get('desktop_input') is not False
            or report.get('gameplay_resource_overrides') is not False
            or report.get('exe_sha256')!=exe_sha256):
        raise ValueError('Incomplete or different original-game motion calibration')
    cases=report['cases']
    if len(cases)!=5 or {r['name'] for r in cases}!={'left','right','top','bottom','focused'}:
        raise ValueError('All four native bounds and focused speed must be measured')
    identities=set()
    for row in cases:
        context=row['native_context']
        identities.add((context['exe_sha256'],context['character'],context['subshot'],context['difficulty']))
        if row['status']!='measured' or file_sha256(row['trace'])!=row['trace_sha256']:
            raise ValueError('Motion calibration trace changed or failed to finish')
    if identities!={(exe_sha256,0,0,3)}:
        raise ValueError('Motion calibration must identify selected TH14 character0/shot0/Lunatic')
    values=report['planner_overrides']
    if set(values)!={'left','right','top','bottom','normal_speed','focused_speed'} or any(
            isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values.values()):
        raise ValueError('Invalid measured motion settings')
    if not (values['left']<values['right'] and values['top']<values['bottom']
            and 0<values['focused_speed']<=values['normal_speed']<20):
        raise ValueError('Invalid measured motion bounds or speeds')
    return replace(config,**values), {'path':str(path),'sha256':file_sha256(path),
        'identity':{'character':0,'subshot':0,'difficulty':3,'exe_sha256':exe_sha256},
        'planner_overrides':values,'student_observation_prefix_changed':False}
