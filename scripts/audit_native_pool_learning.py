"""CPU audit of retained real full-brain native pool updates and checkpoints."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402
from fly_touhou.experiment import file_sha256, _file_lock  # noqa: E402
from fly_touhou.whole_brain_ppo import load_rollout, WholeBrainRollout, generalized_advantages  # noqa: E402
from fly_touhou.whole_brain_rollout_pool import SAMPLER_SCHEMA  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-updates', type=int,
                        help='Audit only this sealed prefix, preserving an exact copied event prefix')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior native pool evidence')
    torch.set_num_threads(2)
    parent_hash = file_sha256(args.parent)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    with _file_lock(args.run / '.events.lock'):
        event_text = (args.run / 'events.jsonl').read_text(encoding='utf-8')
    lines = event_text.splitlines()
    events = [json.loads(line) for line in lines]
    if args.first_updates is not None:
        if args.first_updates < 1:
            raise ValueError('A positive sealed-update prefix is required')
        checkpoints_in_order = [i for i, event in enumerate(events) if event['event'] == 'checkpoint'
                                and event['payload']['tag'] == 'ppo-latest']
        if len(checkpoints_in_order) < args.first_updates:
            raise ValueError('Requested completed checkpoint prefix is not yet available')
        end = checkpoints_in_order[args.first_updates - 1] + 1
        events, lines = events[:end], lines[:end]
    prefix_path = args.output.with_suffix('.events_prefix.jsonl')
    with prefix_path.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    epochs = [event['payload'] for event in events if event['event'] == 'epoch']
    checkpoints = {event['payload']['epoch']: event['payload'] for event in events
                   if event['event'] == 'checkpoint' and event['payload']['tag'] == 'ppo-latest'}
    sealed = {event['payload']['path']: event['payload'] for event in events
              if event['event'] == 'ppo_behavior_rollout_sealed'}
    manifest = json.loads((args.run / 'manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    assert config['initial_checkpoint_sha256'] == parent_hash
    assert config['campaign_runtime']['parallel_collection']['worlds'] == 2
    assert len(epochs) >= 1
    report = {'schema': 'real-complete-brain-native-pool-learning-audit-v1',
              'parent_sha256': parent_hash, 'run': str(args.run.resolve()),
              'native_game_started_by_auditor': False, 'new_learning_by_auditor': False,
              'source': {'path': str(Path(__file__).resolve()), 'sha256': file_sha256(__file__)},
              'event_prefix': {'path': str(prefix_path.resolve()), 'sha256': file_sha256(prefix_path)},
              'scope': 'only the inspected sealed update prefix; no entire-run completion or playing ability claim',
              'updates': []}
    previous = parent
    first = True
    for item in epochs:
        epoch, metrics = item['epoch'], item['metrics']
        assert metrics['behavior_replay_max_logp_error'] <= 1e-9
        assert metrics['teacher_action_calls'] == 0 and metrics['safety_takeovers'] == 0
        replay_path = args.run / 'rollouts' / f'{epoch:06d}.states.pt.gz'
        evidence = sealed[str(replay_path.resolve())]
        assert file_sha256(replay_path) == evidence['sha256']
        saved = load_rollout(replay_path)
        assert saved['graph_fingerprint'] == parent['graph_fingerprint']
        for name, value in saved['behavior_parameters'].items():
            assert torch.equal(value, previous['trainable_parameters'][name])
        rollout = WholeBrainRollout(**{key: (saved[key].numpy() if torch.is_tensor(saved[key]) else saved[key])
                                      for key in WholeBrainRollout.__dataclass_fields__})
        target_verification=None
        if config.get('action_feedback'):
            from fly_touhou.geometry_action_feedback import GeometryActionFeedback,annotate_action_targets
            feedback=GeometryActionFeedback(frame_skip=config['frame_skip'])
            assert feedback.specification()==config['action_feedback']
            mask,weights,metadata=rollout.action_set_mask.copy(),rollout.action_set_weights.copy(),rollout.action_set_metadata
            annotate_action_targets(rollout,evidence['native_trajectory']['path'],feedback)
            import numpy as np
            np.testing.assert_array_equal(mask,rollout.action_set_mask)
            np.testing.assert_array_equal(weights,rollout.action_set_weights)
            assert metadata==rollout.action_set_metadata
            assert metrics['action_set_coefficient']==feedback.specification()['coefficient']
            assert metrics['training_action_target_queries']==len(rollout)
            assert metrics['action_set_loss']>=0
            target_verification={'before_state_queries':len(rollout),'all_masks_weights_and_details_recomputed':True,
                'action_set_loss':metrics['action_set_loss'],'maximum_weight':metadata['maximum_weight'],
                'reasons':metadata['reasons'],'inference_action_override':False}
        teacher_verification=None
        if config.get('teacher_feedback'):
            from fly_touhou.native_teacher_feedback import NativeV6TrainingFeedback,annotate_teacher_targets
            expert=NativeV6TrainingFeedback();metadata=rollout.teacher_target_metadata
            assert expert.specification()==config['teacher_feedback']
            assert metadata['before_teacher_state']==previous['rl_state']['teacher_feedback_state']['worlds']
            expert.load_state_dict(metadata['before_teacher_state'])
            expected,weights=rollout.teacher_targets.copy(),rollout.teacher_target_weights.copy()
            actual=annotate_teacher_targets(rollout,evidence['native_trajectory']['path'],expert)
            np.testing.assert_array_equal(expected,rollout.teacher_targets)
            np.testing.assert_array_equal(weights,rollout.teacher_target_weights)
            assert metadata==actual and metrics['teacher_coefficient']==config['teacher_feedback']['coefficient']
            assert metrics['training_teacher_queries']==len(rollout) and metrics['teacher_loss']>=0
            teacher_verification={'before_state_queries':len(rollout),'all_probabilities_weights_modes_recomputed':True,
                'teacher_loss':metrics['teacher_loss'],'active_targets':metadata['active_targets'],
                'inference_action_override':False}
        assert len(rollout) == 64 and rollout.independent_boundaries == (32,)
        assert set(rollout.episode_ids[:32]).isdisjoint(rollout.episode_ids[32:])
        assert all(end <= 32 or start >= 32 for start, end in rollout.segments())
        # Exact numerical GAE equivalence to two genuinely separate trajectories.
        import numpy as np
        whole = generalized_advantages(rollout.rewards, rollout.values, rollout.next_values,
            rollout.terminated, rollout.truncated, gamma=config['gamma'], gae_lambda=config['gae_lambda'],
            independent_boundaries=rollout.independent_boundaries)
        parts = [generalized_advantages(rollout.rewards[a:b], rollout.values[a:b], rollout.next_values[a:b],
            rollout.terminated[a:b], rollout.truncated[a:b], gamma=config['gamma'], gae_lambda=config['gae_lambda'])
            for a, b in ((0, 32), (32, 64))]
        for column in range(2):
            np.testing.assert_array_equal(whole[column], np.concatenate([part[column] for part in parts]))
        reward_verification = None
        if config.get('reward_profile', {}).get('schema') in ('native-life-avoidance-v1', 'native-life-emergency-bomb-v1','native-life-emergency-geometry-v2'):
            from fly_touhou.campaign_rewards import NativeLifeAvoidanceReward, NativeEmergencyBombReward, NativeGeometryEmergencyReward
            from fly_touhou.planner import PlannerConfig
            from fly_touhou.state import Snapshot
            reward_class = {'native-life-avoidance-v1':NativeLifeAvoidanceReward,
                'native-life-emergency-bomb-v1':NativeEmergencyBombReward,
                'native-life-emergency-geometry-v2':NativeGeometryEmergencyReward}[config['reward_profile']['schema']]
            planner_config = PlannerConfig(**config['reward_profile']['prediction_source']['planner'])
            evaluators = [reward_class(gamma=config['gamma'], planner_config=planner_config) for _ in range(2)]
            assert all(e.specification() == config['reward_profile'] for e in evaluators)
            native_evidence = evidence['native_trajectory']
            assert file_sha256(native_evidence['path']) == native_evidence['sha256']
            verified, maximum_error, dense = set(), 0., 0
            with gzip.open(native_evidence['path'], 'rt', encoding='utf-8') as stream:
                for line in stream:
                    transition = json.loads(line)
                    if transition.get('record_type') != 'transition':
                        continue
                    world = int(re.search(r':world([0-9]+):', transition['episode_id']).group(1))
                    index = world*32 + transition['rollout_step']
                    assert index not in verified and rollout.episode_ids[index] == transition['episode_id']
                    before, after = Snapshot.from_dict(transition['snapshot']), Snapshot.from_dict(transition['next_snapshot'])
                    reward, components, details = evaluators[world].evaluate(before, after,
                        transition['elapsed_frames'], terminated=transition['terminated'], action=transition['action'])
                    if after.stage > before.stage > 0:
                        reward += 5.
                        components['stage_clear'] = 5.
                    error = max(abs(reward - transition['reward']), abs(reward - rollout.rewards[index]))
                    assert error < 1e-10
                    assert components == transition['reward_components']
                    assert details == transition['training_reward_diagnostics']
                    maximum_error = max(maximum_error, error)
                    dense += abs(components['avoidance_potential_difference']) > 1e-5
                    verified.add(index)
            assert len(verified) == 64
            reward_verification = {'actual_native_transitions_verified': len(verified),
                'maximum_recorded_reward_error': maximum_error, 'dense_feedback_decisions': dense,
                'all_diagnostic_components_match_measured_states': True}
        archive = checkpoints[epoch]
        assert file_sha256(archive['path']) == archive['sha256']
        current = torch.load(archive['path'], map_location='cpu', weights_only=True)
        state = current['rl_state']
        if teacher_verification is not None:
            assert state['teacher_feedback_state']['worlds']==rollout.teacher_target_metadata['after_teacher_state']
        assert current['neurons'] == 138639 and current['edge_records'] == 15091983
        assert current['graph_fingerprint'] == parent['graph_fingerprint']
        assert state['completed_timesteps'] == previous['rl_state']['completed_timesteps'] + 64
        assert state['native_frames'] == previous['rl_state']['native_frames'] + 256
        assert state['updates'] == previous['rl_state']['updates'] + 1
        parallel = state['parallel_sampler_state']
        assert parallel['schema'] == SAMPLER_SCHEMA and parallel['worlds'] == 2
        assert sum(parallel['episodes_started']) == state['episodes_started']
        assert torch.equal(state['action_sampler_state'], parallel['sampler_states'][0])
        before_adam = previous['rl_state']['optimizer_state_dict']['state']
        after_adam = state['optimizer_state_dict']['state']
        assert set(before_adam) == set(after_adam)
        increments = {str(key): float(after_adam[key]['step'] - before_adam[key]['step']) for key in before_adam}
        assert all(delta == metrics['optimizer_steps'] for delta in increments.values())
        assert metrics['actor_update_l2'] > 0 and metrics['internal_gain_logit_update_l2'] > 0
        geometry_update=None
        if current['model_config'].get('geometry_prefix_dim'):
            before_geometry=previous['trainable_parameters']['geometry_input_encoder.weight']
            after_geometry=current['trainable_parameters']['geometry_input_encoder.weight']
            geometry_update=float((after_geometry.double()-before_geometry.double()).norm())
            assert geometry_update>0 and metrics['geometry_encoder_gradient_norm']>0
        report['updates'].append({'epoch': epoch, 'native_decisions': len(rollout),
            'independent_boundary': 32, 'brain_source_records': 15091983, 'neurons': 138639,
            'behavior_parameters_match_preserved_preceding_checkpoint': True,
            'first_actor_equals_pinned_parent': first,
            'actual_logged_behavior_replay_max_logp_error': metrics['behavior_replay_max_logp_error'],
            'GAE_equal_to_independent_worlds': True, 'all_Adam_step_increments': increments,
            'real_optimizer_steps': metrics['optimizer_steps'],
            'actor_update_l2': metrics['actor_update_l2'],
            'internal_gain_update_l2': metrics['internal_gain_logit_update_l2'],
            'geometry_encoder_update_l2':geometry_update,
            'collect_seconds': metrics['collect_seconds'], 'epoch_seconds': item['elapsed_seconds'],
            'reward_verification': reward_verification,'action_target_verification':target_verification,
            'teacher_target_verification':teacher_verification,
            'checkpoint_sha256': archive['sha256']})
        previous, first = current, False
    runtime = [event['payload'] for event in events if event['event'] == 'ppo_owned_campaign_runtime_audit']
    pids = sorted({item['pid'] for item in runtime})
    assert len(pids) >= 2
    for item in runtime:
        assert item['configuration']['window_visibility'].startswith('hidden owned original')
        assert item['configuration']['desktop_input'] is False
    report.update(status='passed', original_native_processes=pids,
                  final_checkpoint_sha256=checkpoints[epochs[-1]['epoch']]['sha256'])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'native_processes': pids,
                      'updates': len(epochs), 'decisions': 64*len(epochs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
