"""CPU audit of sealed GRU training, sampling, losses and optimizer continuity."""
from pathlib import Path
import argparse
import gzip
import json
import sys

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'src'))
from fly_touhou.dense_geometry_ppo import forecast_regret_batch
from fly_touhou.diagnostic_gru import DiagnosticGRU, replay_history
from fly_touhou.experiment import file_sha256, verify_events
from fly_touhou.first_hit_policy_gradient import first_hit_reward, finite_lesson_returns
from fly_touhou.geometry_observation import GeometryObservationEncoder
from fly_touhou.native_diagnostic import checked_snapshot, write_json
from fly_touhou.right_aligned_credit import right_aligned_segments
from fly_touhou.state import Snapshot
from fly_touhou.training_surrogate import recorded_policy_objective


def equal(a, b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior audits')
    torch.set_num_threads(4)
    manifest = json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    events = verify_events(args.run/'events.jsonl')
    assert events[0]['payload']['manifest_sha256'] == file_sha256(args.run/'manifest.json')
    protocol = json.loads(Path(manifest['files']['protocol']['path']).read_text(encoding='utf-8'))
    encoder = GeometryObservationEncoder.from_specification(protocol['encoder'])
    parent = torch.load(args.run/'checkpoint_0000.pt', map_location='cpu', weights_only=True)
    sampler = torch.Generator(device='cpu')
    results = []
    for update in range(1, args.through+1):
        path = args.run/f'checkpoint_{update:04d}.pt'
        metric_path = args.run/f'update_{update:04d}.json'
        seal = next(e['payload'] for e in events if e['event'] == 'diagnostic_update_finished'
                    and e['payload']['counters']['updates'] == update)
        assert file_sha256(path) == seal['checkpoint']['sha256']
        assert file_sha256(metric_path) == seal['metric']['sha256']
        metric = json.loads(metric_path.read_text(encoding='utf-8'))
        current = torch.load(path, map_location='cpu', weights_only=True)
        behavior_path = args.run/'training'/f'behavior_{update:04d}.pt.gz'
        binding = next(e['payload'] for e in events if e['event'] == 'sealed_complete_behavior_before_optimization'
                       and e['payload']['path'] == str(behavior_path))
        assert file_sha256(behavior_path) == binding['sha256']
        with gzip.open(behavior_path, 'rb') as f: data = torch.load(f, map_location='cpu', weights_only=True)
        assert file_sha256(data['parent_checkpoint']['path']) == data['parent_checkpoint']['sha256']
        assert equal(torch.load(data['parent_checkpoint']['path'], map_location='cpu', weights_only=True), parent)
        trace = Path(data['native_trajectory']['path'])
        assert file_sha256(trace) == data['native_trajectory']['sha256']
        with gzip.open(trace, 'rt', encoding='utf-8') as f: rows = [json.loads(line) for line in f]
        assert len(rows) == len(data['actions']) == metric['samples']
        sampler.set_state(parent['sampler'])
        for i, row in enumerate(rows):
            snapshot = Snapshot.from_dict(row['snapshot'])
            checked_snapshot(snapshot, protocol)
            assert row['role'] == 'training' and row['teacher_action_calls'] == row['safety_takeovers'] == 0
            assert row['episode_id'] == data['episode_ids'][i]
            assert torch.equal(torch.from_numpy(encoder.encode(snapshot)), data['observations'][i])
            assert row['conditional_logp18'] == data['old_logp18'][i].tolist()
            assert int(torch.multinomial(data['old_logp18'][i].exp(), 1, generator=sampler)) == row['action'] == int(data['actions'][i])
        assert torch.equal(sampler.get_state(), data['sampler_after_collection'])
        assert torch.equal(sampler.get_state(), current['sampler'])
        rewards = np.asarray([first_hit_reward(row)[0] for row in rows])
        ended = np.asarray([row['terminated'] or row['truncated'] for row in rows])
        assert torch.equal(torch.from_numpy(rewards), data['rewards'])
        assert torch.equal(torch.from_numpy(finite_lesson_returns(rewards, ended)), data['finite_returns'])
        costs, weights, active = forecast_regret_batch(rows)
        assert torch.equal(torch.from_numpy(costs), data['forecast_costs'])
        assert torch.equal(torch.from_numpy(weights), data['danger_weights'])
        assert torch.equal(torch.from_numpy(active), data['discriminative'])
        advantage = data['finite_returns']-data['baseline_values']
        advantage = ((advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)).clamp(-5,5)
        assert torch.equal(advantage, data['advantages'])
        assert metric['segments'] == [list(v) for v in right_aligned_segments(data['episode_ids'], 8)]
        model = DiagnosticGRU(); model.load_state_dict(parent['actor'])
        logp, states = replay_history(model, data['observations'], data['episode_ids'])
        lp_error = float((logp-data['old_logp18']).abs().max())
        state_error = float((states-data['states_before']).abs().max())
        assert lp_error < 2e-6 and state_error < 2e-6
        arguments = [data[k] for k in ('actions','old_selected_logp','advantages','forecast_costs','danger_weights')]
        actual_before = recorded_policy_objective(data['old_logp18'], *arguments)
        assert actual_before == metric['before']
        proposal = torch.load(metric['proposals']['path'], map_location='cpu', weights_only=True)
        assert file_sha256(metric['proposals']['path']) == metric['proposals']['sha256']
        assert len(proposal['proposal_logp18']) == len(metric['trials'])
        for lp, trial in zip(proposal['proposal_logp18'], metric['trials']):
            objective = recorded_policy_objective(lp, *arguments)
            assert objective == trial['objective']
            divergence = (data['old_logp18'].exp()*(data['old_logp18']-lp)).sum(-1)
            assert float(divergence.mean()) == trial['mean_KL']
            assert float(divergence.max()) == trial['maximum_KL']
            valid = -1e-9 <= trial['mean_KL'] <= .01 and objective['total'] < actual_before['total']-1e-8*max(1,abs(actual_before['total']))
            assert trial['qualifies'] == valid
        assert recorded_policy_objective(proposal['accepted_logp18'], *arguments) == metric['after']
        changed = {k:int((v != parent['actor'][k]).sum()) for k,v in current['actor'].items()}
        assert changed == metric['changed_parameters']
        if not metric['accepted']:
            assert equal(current['actor'], parent['actor'])
            assert equal(current['actor_optimizer'], parent['actor_optimizer'])
        c = current['counters']; p = parent['counters']
        assert c['updates'] == p['updates']+1 == update
        assert c['training_episodes'] == p['training_episodes']+4
        assert c['training_decisions'] == p['training_decisions']+len(rows)
        assert c['training_native_frames'] == p['training_native_frames']+4*len(rows)
        assert c['accepted_actor_updates'] == p['accepted_actor_updates']+int(metric['accepted'])
        assert c['critic_updates'] == p['critic_updates']+32
        assert all(int(v['step']) == c['accepted_actor_updates'] for v in current['actor_optimizer']['state'].values())
        assert all(int(v['step']) == c['critic_updates'] for v in current['critic_optimizer']['state'].values())
        results.append({'update': update, 'samples': len(rows), 'accepted': metric['accepted'],
            'maximum_GPU_CPU_logp_error': lp_error, 'maximum_GPU_CPU_state_error': state_error,
            'checkpoint_sha256': file_sha256(path), 'counters': c,
            'source_observations_sampler_rewards_advantages_and_all_proposals_verified': True})
        parent = current
    value = {'schema':'independent-GRU-diagnostic-learner-readback-v1','status':'passed',
        'auditor_path':str(Path(__file__)),'auditor_sha256':file_sha256(Path(__file__)),
        'updates':results,'G2_or_G3_achieved':False,'final_acceptance':False}
    write_json(args.output,value,exclusive=True)
    print(json.dumps(value,ensure_ascii=False))


if __name__ == '__main__':
    main()
