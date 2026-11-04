"""Read-only, explicitly partial native failure diagnostics for a frozen policy."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys


def finite(value):
    import math
    return float(value) if math.isfinite(float(value)) else None


def run(args):
    import numpy as np
    import psutil
    import torch
    sys.path.insert(0, str(args.runtime/'src'))
    from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
    from fly_touhou.thrl_durable_records import write_json as write
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_pair_risk import action_clearances
    from fly_touhou.thrl_policy_actions import policy_logits
    plan = read(args.source/'plan.json')
    if args.output.exists() or plan['native_seed_split'] != 'development':
        raise ValueError('Use a fresh diagnostic output and explicitly frozen development policy')
    checkpoint = plan['policies'][args.role]
    path = Path(checkpoint['checkpoint'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint['sha256']:
        raise ValueError('Frozen policy changed')
    torch.set_num_threads(2)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = model_from_config(**payload['model_config'])
    model.load_state_dict(payload['model'], strict=True); model.eval()
    name = 'fly_touhou._prepared_player_aware_diagnostic_teacher'
    spec = importlib.util.spec_from_file_location(name, args.diagnostic_teacher)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    teacher = module.MovementSafetyTeacher()
    with _shared_binary(args.source/args.role/'games/episodes.jsonl') as stream:
        rows = [json.loads(line) for line in stream.read().splitlines() if line.strip()]
    results = []
    with torch.inference_mode():
        for row in rows:
            if row['outcome'] != 'gameover':
                continue
            pointer = [p for p in row['diagnostic_traces']['events'] if p['event']=='natural_terminal_tail'][0]
            trace = read(pointer['receipt'])
            if (trace['checkpoint_sha256'] != checkpoint['sha256']
                    or trace['game']+1 != row['evaluation_game'] or trace['native_seed'] != row['native_seed']
                    or hashlib.sha256(Path(trace['arrays']).read_bytes()).hexdigest() != trace['sha256']):
                raise ValueError('Actual native failure identity differs')
            with np.load(trace['arrays'], allow_pickle=False) as data:
                fs, objects, actions, native = [data[key] for key in ('features','objects','action','native_frame')]
                ids = np.flatnonzero(module.normal_unprotected(fs))
                if not len(ids):
                    results.append(dict(game=row['evaluation_game'], reason='no_saved_normal_unprotected_state'))
                    continue
                index = int(ids[-1])
                f, o, maps = fs[index], objects[index], data['maps'][index]
                hidden = data['hidden'][index].reshape(1,1,-1)
                ft, ot, mt, ht = [torch.as_tensor(x) for x in (f[None], o[None], maps[None], hidden)]
                base, _, _ = model.base.forward_step(ft, mt, ht, objects=ot)
                logits = model._adjust(base, ft, ot)
                probabilities = torch.softmax(policy_logits(logits), -1)[0].numpy()
                recorded = data['probabilities'][index]
                delta = float(abs(probabilities-recorded).max())
                chosen = int(actions[index])
                if delta >= 2e-4 or int(probabilities.argmax()) != chosen:
                    raise ValueError('Exact-policy recorded-context inference did not reproduce its decision')
                prediction = action_clearances(model.risk, ft, ot[...,:21])[0].numpy()
                origin = teacher._scene(f, o)[0]
                points = o.copy()
                points[(points[:,3]>.5)|(points[:,4]>.5)] = 0
                margins = teacher.clearances(f, points, np.repeat(origin[None],18,axis=0), teacher.velocities, duration=4)
                after = fs[index+1] if index+1<len(fs) else data['after_features']
                ticks = (int(native[index+1]-native[index]) if index+1<len(fs)
                    else trace['after_facts']['observation_native_frame']-int(native[index]))
                eligible = (fs[:,24]>.5)&(fs[:,4]>0)&(fs[:,19]>=.25)&(fs[:,19]<=1)
                leaks = [dict(row=int(k), episode_frame=int(data['episode_frame'][k]),
                    remaining_frames=float(fs[k,19]*8), stock=float(fs[k,4]*8), action=int(actions[k]),
                    Bomb_probability=float(data['probabilities'][k,18])) for k in np.flatnonzero(eligible&(actions!=18))]
                results.append(dict(game=row['evaluation_game'], native_seed=row['native_seed'],
                    end_Bomb=row['bombs_remaining'], Miss=row['misses'], Bomb=row['bombs_used'],
                    final_normal_row=index, final_normal_episode_frame=int(data['episode_frame'][index]),
                    position=origin.tolist(), player_circle=float(f[13]*64), action=chosen,
                    policy_probability_reproduction_error=delta, actual_next_native_tick_delta=ticks,
                    actual_next_state4=bool(after[24]>.5), next_player_circle=float(after[13]*64),
                    last_normal_selected_point_count=int(((o[:,0]>.5)&((o[:,1]>.5)|(o[:,2]>.5))&(o[:,13]>.5)).sum()),
                    point_geometric_chosen4_margin=finite(margins[chosen]) if chosen<18 else None,
                    point_neural_chosen4_margin=finite(prediction[chosen]) if chosen<18 else None,
                    point_geometric_best4_margin=finite(margins.max()),
                    point_geometric_buffered4_actions=np.flatnonzero(margins>1).tolist(),
                    all_point_geometric4_margins=[finite(x) for x in margins],
                    all_neural_point4_margins=[finite(x) for x in prediction],
                    teacher_uncertainty4=teacher._uncertainty(f,o,maps,horizon=4),
                    teacher_uncertainty24=teacher._uncertainty(f,o,maps,horizon=24),
                    terminal_saved_tail_missed_usable_deathbomb_decisions=leaks,
                    whole_game_usable_deathbomb_decisions=row['reward_diagnostics']['deathbomb_usable_decision_count'],
                    whole_game_usable_Bomb_requests=row['reward_diagnostics']['deathbomb_usable_bomb_requests'],
                    source_trace=pointer, unexecuted_alternatives_are_geometric_proxies=True))
    if any(not torch.equal(value,payload['model'][key]) for key,value in model.state_dict().items()):
        raise ValueError('Read-only diagnostic changed weights')
    result = dict(schema='B1-frozen-development-terminal-geometry-diagnostic-v1', utc=utc_now(), role=args.role,
        checkpoint_sha256=checkpoint['sha256'], completed_games_at_snapshot=len(rows),
        failed_games_analyzed=len(results), rows=results, partial_diagnostic_not_full_game_score=True,
        checkpoint_unchanged=True, native_alternatives_executed=False, new_original_worlds=0,
        new_optimizer_updates=0, current_evaluation_unchanged=True,
        diagnostic_teacher_sha256=hashlib.sha256(args.diagnostic_teacher.read_bytes()).hexdigest(),
        infinite_unbounded_point_margin_encoded_as_null=True)
    write(args.output, result)
    print(json.dumps({**result, 'rows': [{k:v for k,v in row.items() if k not in (
        'source_trace','all_point_geometric4_margins','all_neural_point4_margins')} for row in results]},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--diagnostic-teacher', type=Path, required=True)
    parser.add_argument('--role', choices=('baseline','candidate'), default='candidate')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
