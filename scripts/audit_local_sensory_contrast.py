"""CPU-only input-code contrast; no neural recurrence, game or learning.

Mirroring a single nearby bullet is a constructed stimulus, not a measured
alternate native world. CPU sigmoid/linear need not equal CUDA bitwise; input
event comparisons are estimates and say nothing by themselves about behavior.
"""
from copy import deepcopy
from dataclasses import replace
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from fly_touhou.experiment import file_sha256  # noqa: E402
from fly_touhou.hazard_observation import HazardObservationEncoder  # noqa: E402
from fly_touhou.state import Snapshot  # noqa: E402
from fly_touhou.whole_brain import WholeBrainStudent  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--traces', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases-per-trace', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior sensory diagnoses')
    torch.set_num_threads(2)
    record = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = record['model_config']
    sensor = HazardObservationEncoder.from_specification(record['encoder'])
    weight, bias = (record['trainable_parameters'][name] for name in ('input_encoder.weight','input_encoder.bias'))
    owner = SimpleNamespace(_input_channels=torch.arange(weight.shape[0], dtype=torch.int64))
    cases = []
    for path in args.traces:
        count, last_step = 0, -100
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                if item.get('record_type') != 'transition' or item['step'] - last_step < 12:
                    continue
                snapshot = Snapshot.from_dict(item['snapshot'])
                player = snapshot.player
                if player is None or player.state != 1 or player.invulnerable_frames > 0:
                    continue
                raw = {row['id']: row for row in snapshot.extra['hazard_metadata']['bullets']}
                candidates = [b for b in snapshot.bullets if not b.intangible and raw[b.id]['native_flags'] & 2
                              and raw[b.id]['native_state'] in (1,2) and abs(b.x-player.x)>2.]
                if not candidates:
                    continue
                bullet = min(candidates, key=lambda b:(b.x-player.x)**2+(b.y-player.y)**2)
                if math.hypot(bullet.x-player.x,bullet.y-player.y)>48.:
                    continue
                alternate = deepcopy(snapshot)
                mirrored = replace(bullet, x=2*player.x-bullet.x, vx=-bullet.vx)
                alternate = replace(alternate, bullets=tuple(mirrored if b.id==bullet.id else b for b in alternate.bullets))
                row = next(r for r in alternate.extra['hazard_metadata']['bullets'] if r['id']==bullet.id)
                row.update(native_position=[mirrored.x,mirrored.y], native_velocity=[mirrored.vx,mirrored.vy],
                           native_angle=math.atan2(math.sin(row['native_angle']),-math.cos(row['native_angle'])))
                observations = torch.from_numpy(np.stack([sensor.encode(snapshot), sensor.encode(alternate)]))
                with torch.no_grad():
                    p = torch.sigmoid(F.linear(observations, weight, bias))*config['max_input_rate_hz']*config['dt_ms']/1000.
                    evidence = item['action_evidence']
                    key = int(evidence['noise_key'])
                    first = int(evidence['noise_counter'])-config['substeps'] if 'noise_counter' in evidence else (
                        item['neural_noise_counter']-config['substeps'])
                    counters = torch.arange(first,first+config['substeps'],dtype=torch.int64)
                    uniforms = WholeBrainStudent._uniform(owner, torch.full_like(counters,key), counters)
                    changed = int(((uniforms<p[0]) != (uniforms<p[1])).sum())
                cases.append({'trace':str(path.resolve()),'step':item['step'],'bullet_id':bullet.id,
                    'relative_position':[bullet.x-player.x,bullet.y-player.y],
                    'changed_raw_features':int((observations[0]!=observations[1]).sum()),
                    'mean_input_event_probability_change':float((p[0]-p[1]).abs().mean()),
                    'expected_changed_input_events_under_shared_uniform_noise':float((p[0]-p[1]).abs().sum()*config['substeps']),
                    'CPU_same_counter_threshold_input_event_changes':changed,
                    'CPU_original_input_events':int((uniforms<p[0]).sum()),
                    'original_chosen_action':item['action']})
                count, last_step = count+1, item['step']
                if count>=args.cases_per_trace:
                    break
    if not cases:
        raise ValueError('No eligible local constructed contrast cases')
    changes = [row['CPU_same_counter_threshold_input_event_changes'] for row in cases]
    report = {'schema':'local-raw-sensory-input-contrast-v1','checkpoint_sha256':file_sha256(args.checkpoint),
        'script_sha256':file_sha256(__file__),'traces':[{ 'path':str(p.resolve()),'sha256':file_sha256(p)} for p in args.traces],
        'scope':'constructed single-bullet input contrast on CPU; no brain inference, actual alternate game, or training',
        'limitations':['CPU input-event estimates do not prove bitwise CUDA events or descending action differences.',
                       'A single current observation does not replace sensory information accumulated in the recurrent state.'],
        'substeps':config['substeps'],'input_ports':weight.shape[0], 'cases':cases,
        'summary':{'cases':len(cases),'zero_changed_event_cases':changes.count(0),
                   'changed_event_median':float(np.median(changes)),'changed_event_maximum':max(changes)}}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary']))


if __name__=='__main__':
    main()
