"""Replay frozen failures with the actual captured recurrent memory, on CPU.

Separates the learned body from the fixed neural risk readouts. Geometry and
instantaneous sensor ablations are diagnostic hypotheses, not native trials.
Right-censored no-hit games are retained as censored, never invented failures.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write
from fly_touhou.thrl_failure_trace_audit import classify
from evaluate_thrl_avoidance_probe import without_bullet_information


def failure_cases(report):
    selected = [row for row in report['episode_results']
                if row['policy'] in ('trained_sampled', 'trained_greedy')]
    if {row['policy'] for row in selected} != {'trained_sampled', 'trained_greedy'}:
        raise ValueError('Require both trained conditions')
    failed = []
    for row in selected:
        if type(row.get('first_collision_observed')) is not bool or type(row.get('right_censored_at_cap')) is not bool:
            raise ValueError('Explicit observed collision and censoring fields required')
        if row['first_collision_observed'] == row['right_censored_at_cap']:
            raise ValueError('Observed collision and right censoring must be exclusive and complete')
        if row['first_collision_observed']:
            if not row.get('failure_trace'):
                raise ValueError('A genuinely observed trained failure lost its trace')
            failed.append(row)
        elif not row['right_censored_at_cap']:
            raise ValueError('Unobserved hit must not silently become a completed failure')
    return selected, failed


def summarize(rows):
    modeled = [row for row in rows if 'adjusted_proxy_collision_mass' in row]
    def mean(key, group):
        return float(np.mean([row[key] for row in group])) if group else None
    return dict(cases=len(rows), categories=dict(Counter(row['category'] for row in rows)),
        geometry_covered_cases=len(modeled),
        mean_base_proxy_collision_mass=mean('base_proxy_collision_mass', modeled),
        mean_adjusted_proxy_collision_mass=mean('adjusted_proxy_collision_mass', modeled),
        mean_base_proxy_buffer_unsafe_mass=mean('base_proxy_buffer_unsafe_mass', modeled),
        mean_adjusted_proxy_buffer_unsafe_mass=mean('adjusted_proxy_buffer_unsafe_mass', modeled),
        mean_base_instantaneous_sensor_TV=mean('base_instantaneous_sensor_TV', rows),
        mean_adjusted_instantaneous_sensor_TV=mean('adjusted_instantaneous_sensor_TV', rows),
        max_replay_probability_error=max((row['replay_probability_error'] for row in rows), default=None),
        max_Bomb_probability_difference=max((row['Bomb_probability_difference'] for row in rows), default=None))


def run(args):
    if args.output.exists():
        raise ValueError('Preserve previous attribution reports')
    report = read(args.report)
    if (report.get('status') != 'completed'
            or read(args.report.parent/'cleanup.json') != dict(closed=True, close_errors=[])):
        raise ValueError('Require a completed diagnostic with its native pool closed')
    selected, failed = failure_cases(report)
    pin = report['pins']['trained']
    checkpoint = Path(pin['copy'])
    payload_bytes = checkpoint.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != pin['sha256']:
        raise ValueError('Frozen model provenance changed')
    import io
    import torch
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_policy_actions import policy_logits
    from fly_touhou.thrl_inputs import PRIORITY_INPUT_SCHEMAS, object_dimension
    torch.set_num_threads(2)
    payload = torch.load(io.BytesIO(payload_bytes), map_location='cpu', weights_only=False)
    model = model_from_config(**payload['model_config']).cpu().eval().requires_grad_(False)
    if model.input_schema not in PRIORITY_INPUT_SCHEMAS or not hasattr(model, 'risk') or not hasattr(model, 'base'):
        raise ValueError('Use the declared separated B1 policy with neural entity readout')
    model.load_state_dict(payload['model'], strict=True)
    cases = []
    for case in failed:
        pointer = case['failure_trace']
        arrays_path = Path(pointer['arrays']).resolve()
        if not arrays_path.is_relative_to((args.report.parent/'failure_traces').resolve()):
            raise ValueError('Trace is outside the source diagnostic')
        receipt = read(pointer['receipt'])
        if (hashlib.sha256(arrays_path.read_bytes()).hexdigest() != pointer['sha256']
                or receipt['sha256'] != pointer['sha256'] or receipt['checkpoint_sha256'] != pin['sha256']
                or (receipt['policy'], receipt['game']) != (case['policy'], case['game'])):
            raise ValueError('Actual failure provenance differs')
        with np.load(arrays_path, allow_pickle=False) as arrays:
            geometry = classify(arrays, receipt)
            item = dict(case=case, geometry=geometry,
                features=arrays['features'][-1].copy(), objects=arrays['objects'][-1].copy(),
                maps=arrays['last_preaction_maps'].copy(), hidden=arrays['hidden'][-1].copy().reshape(-1),
                logged_probabilities=arrays['probabilities'][-1].copy(), action=int(arrays['action'][-1]))
        for key in ['features', 'objects', 'maps', 'hidden', 'logged_probabilities']:
            if item[key].dtype != np.float32 or not np.isfinite(item[key]).all():
                raise ValueError('Preserve actual finite FP32 observations and recurrent context')
        if (item['features'].shape != (25,)
                or item['objects'].shape != (96, object_dimension(model.input_schema))
                or item['maps'].shape != (16, 92, 96)
                or item['hidden'].shape != (model.gru_hidden_size,)):
            raise ValueError('Captured modalities do not belong to the frozen model')
        cases.append(item)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(cases), 8):
            group = cases[start:start+8]
            source = {key: np.stack([item[key] for item in group]) for key in ['features', 'maps', 'objects']}
            batch = {key: torch.from_numpy(value) for key,value in source.items()}
            hidden = torch.from_numpy(np.stack([item['hidden'] for item in group]))[None].contiguous()
            adjusted_logits, _, _ = model.forward_step(batch['features'], batch['maps'], hidden, objects=batch['objects'])
            base_logits, _, _ = model.base.forward_step(batch['features'], batch['maps'], hidden, objects=batch['objects'])
            blind = {key: torch.from_numpy(value) for key,value in without_bullet_information(source).items()}
            blind_adjusted, _, _ = model.forward_step(blind['features'], blind['maps'], hidden, objects=blind['objects'])
            blind_base, _, _ = model.base.forward_step(blind['features'], blind['maps'], hidden, objects=blind['objects'])
            movement = lambda logits: policy_logits(logits)[:,:18].softmax(-1).numpy()
            adjusted, base = movement(adjusted_logits), movement(base_logits)
            ablated_adjusted, ablated_base = movement(blind_adjusted), movement(blind_base)
            full_adjusted = policy_logits(adjusted_logits).softmax(-1).numpy()
            full_base = policy_logits(base_logits).softmax(-1).numpy()
            for i, item in enumerate(group):
                error = float(np.abs(adjusted[i]-item['logged_probabilities']).max())
                if error > 3e-5:
                    raise ValueError('Actual recurrent policy probabilities failed replay: '+str(error))
                case, geometry, action = item['case'], item['geometry'], item['action']
                row = dict(policy=case['policy'], game=case['game'], native_seed=case['native_seed'],
                    observed_collision=True, native_frames=case['native_frames'], stage=case['stage'],
                    category=geometry['category'], uncertainty_by_horizon=geometry.get('uncertainty_by_horizon'),
                    action=action, base_argmax=int(base[i].argmax()), adjusted_argmax=int(adjusted[i].argmax()),
                    base_executed_action_probability=float(base[i,action]), adjusted_executed_action_probability=float(adjusted[i,action]),
                    base_Bomb_probability=float(full_base[i,18]), adjusted_Bomb_probability=float(full_adjusted[i,18]),
                    Bomb_probability_difference=float(abs(full_adjusted[i,18]-full_base[i,18])),
                    base_instantaneous_sensor_TV=float(.5*np.abs(base[i]-ablated_base[i]).sum()),
                    adjusted_instantaneous_sensor_TV=float(.5*np.abs(adjusted[i]-ablated_adjusted[i]).sum()),
                    replay_probability_error=error, model_probabilities_are_native_counterfactuals=False)
                if 'modeled_clearances_clipped64' in geometry:
                    margin = np.asarray(geometry['modeled_clearances_clipped64'])
                    row.update(base_proxy_collision_mass=float(base[i,margin<=0].sum()),
                        adjusted_proxy_collision_mass=float(adjusted[i,margin<=0].sum()),
                        base_proxy_buffer_unsafe_mass=float(base[i,margin<=1].sum()),
                        adjusted_proxy_buffer_unsafe_mass=float(adjusted[i,margin<=1].sum()),
                        base_argmax_proxy_margin=float(margin[base[i].argmax()]),
                        adjusted_argmax_proxy_margin=float(margin[adjusted[i].argmax()]),
                        native_executed_action_proxy_margin=float(margin[action]))
                rows.append(row)
    result = dict(schema='B1-actual-memory-failure-neural-risk-attribution-v1', status='completed', utc=utc_now(),
        source_report=str(args.report), source_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256=pin['sha256'], input_schema=model.input_schema,
        selected_trained_games=len(selected), observed_collision_games=len(failed),
        right_censored_games=sum(case['right_censored_at_cap'] for case in selected),
        conditions={key: summarize([row for row in rows if row['policy']==key])
                    for key in ['trained_sampled', 'trained_greedy']},
        summary=summarize(rows), rows=rows, device='cpu', native_instances_started=0,
        optimizer_updates=0, fitting_on_these_failures=False, source_data_modified=False,
        actual_recurrent_memory_used=True, instantaneous_ablation_holds_memory_fixed=True,
        instantaneous_ablation_is_distribution_shift=True, ablation_measures_native_performance=False,
        scope='Observed captured failures only;right-censored games are not failures;geometry is current-motion proxy')
    args.output.mkdir(parents=True)
    write(args.output/'report.json', result)
    print(json.dumps(dict(input_schema=model.input_schema, selected_games=len(selected),
        observed_failures=len(failed), right_censored=result['right_censored_games'], summary=result['summary'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
