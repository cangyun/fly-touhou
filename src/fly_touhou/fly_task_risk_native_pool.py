"""One frozen DN-risk candidate with16 real workers and optional read-only data."""
from pathlib import Path
import hashlib
import json

import torch

from .fly_visual_native_pool import PoolWideVisualNativePolicy, PoolReadyVector
from .fly_task_risk_native_evaluation import (TaskRiskWideVisualNativePolicy,
    development_seed_plan, verify_seed_rows)


class PoolTaskRiskVisualNativePolicy(PoolWideVisualNativePolicy, TaskRiskWideVisualNativePolicy):
    def __init__(self, *args, recorder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder, self._encoded_for_recording = recorder, None
        self._recording_hook = (self.student.sensory_adapter.register_forward_hook(self._remember_encoded)
                                if recorder is not None else None)

    def _remember_encoded(self, module, args, output):
        self._encoded_for_recording = output.detach()

    @torch.no_grad()
    def forward_step(self, features, maps, cursor, *, objects):
        context = self._context
        result = super().forward_step(features, maps, cursor, objects=objects)
        if self.recorder is not None:
            self.recorder.record(self, context, features, maps, objects, result[0], self._encoded_for_recording)
        self._encoded_for_recording = None
        return result

    def close_recording(self):
        if self._recording_hook is not None:
            self._recording_hook.remove()
            self._recording_hook = None
        return None if self.recorder is None else self.recorder.close()


def evaluate_task_risk_native_pool(student, risk_head, config, output, *, capacity_probe,
        parameter_checkpoint_sha256, reference_raw_report, diagnostic_output=None,
        seed=2027431601, brain_seed=613711, check_stop=lambda: None, progress=None):
    from .thrl_actor_bc_evaluation import capacity_gate
    from .thrl_training_mode import validate_training_mode, episode_protocol
    from .thrl_inputs import SEPARATED_LASER_INPUT
    from .thrl_full_story_evaluation import evaluate
    from .thrl_durable_records import write_json
    from .thrl_dodge_diagnostics import _shared_binary
    import train_thrl_native as native_entry
    if (not validate_training_mode(config) or config.get('input_schema') != SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True):
        raise ValueError('Existing natural headless V5/source19 full-story configuration required')
    if (student.training or risk_head.training or any(p.requires_grad for p in student.parameters())
            or any(p.requires_grad for p in risk_head.parameters())):
        raise ValueError('Independent frozen complete-brain base and trained DN head required')
    plan = development_seed_plan(20, seed, brain_seed)
    with _shared_binary(reference_raw_report) as stream:
        reference_bytes = stream.read()
    reference = json.loads(reference_bytes.decode('utf-8-sig'))
    if (reference.get('policy_frozen') is not True or reference.get('PPO_updates') != 0
            or reference.get('action_selection') != 'greedy' or reference.get('native_seed_split') != 'development'):
        raise ValueError('Complete frozen raw20 reference required')
    verify_seed_rows(reference['episode_results'], plan)
    def guard(startup=False):
        check_stop()
        capacity = capacity_probe()
        capacity_gate(capacity['native_worlds'], 16, startup=startup,
            main_identity_alive=capacity['main_identity_alive'])
    guard(startup=True)
    recorder = None
    if diagnostic_output is not None:
        from .fly_dn_diagnostic_capture import DNDiagnosticCapture
        recorder = DNDiagnosticCapture(diagnostic_output)
    policy = PoolTaskRiskVisualNativePolicy(student, risk_head, beta=4., neural_batch_size=16,
        episodes=20, brain_seed=brain_seed, recorder=recorder).eval()
    kwargs = native_entry.make_env_kwargs(config, Path(output), 16, 'development')
    if (len(kwargs) != 16 or any(not k['headless'] or not k['hide_window']
            or k['policy_action_count'] != 19 or k['frame_skip'] != 4
            or k['wall_time_limit_seconds'] is not None or 'online_teacher_revision' in k for k in kwargs)):
        raise ValueError('Sixteen private natural worlds; no teacher execution or truncation')
    protocol = dict(episode_protocol(config), neural_reset='new actual game only;no stage/Miss reset',
        brain_batch_size=16, real_native_workers=16, perception_batch='actual ready K<=16',
        ready_wait_ms=2, risk_beta=4., risk_input='same following DN only',
        Bomb_marginal='preserved at T1; greedy Bomb count can change',
        geometry_recording_used_for_actions=False,
        cross_perception_batch_bitwise_equivalence_claimed=False)
    result = None
    try:
        result = evaluate(torch, policy, kwargs, output, episodes=20, seed=seed,
            check_stop=guard, progress=progress, native_split='development',
            episode_protocol=protocol, summary=native_entry.development_summary,
            action_selection='greedy', action_temperature=1., vector_class=PoolReadyVector)
    finally:
        diagnostics = policy.close_recording()
    verify_seed_rows(result['episode_results'], plan)
    result['task_risk_policy'] = dict(schema='fly-task-DN-risk-pool16-frozen-development-v1',
        parameter_checkpoint_sha256=parameter_checkpoint_sha256,
        beta=4., brain_batch_size=16, real_native_workers=16,
        teacher_actor_used=False, own_DN_risk_used=policy.risk_forward_rows > 0,
        analytic_or_raw_object_execution_filter=False, optimizer_updates=0,
        brain_batch_calls=policy.brain_batch_calls,
        real_data_neural_forwards=policy.data_forwards,
        dummy_lane_forwards=policy.dummy_lane_forwards,
        episode_records=policy.episode_records,
        reference_raw_report_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        same_reference_native_seeds_verified=True,
        reference_and_candidate_perception_batch_counts_equal_claimed=False,
        old_T025_probabilities_claimed_equal=False, greedy_Bomb_count_preserved_claimed=False,
        risk_source_training_role='auxiliary only; this frozen candidate changes the readout protocol',
        automatic_deployment=False, DN_diagnostics=diagnostics)
    write_json(Path(output)/'report.json', result)
    return result
