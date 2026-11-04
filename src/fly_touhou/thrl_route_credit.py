"""One fixed B1 correction recipe; preserve the chosen policy and PPO Adam."""
from copy import deepcopy
from dataclasses import asdict
import math

REVISION='B1-soft-route-long-credit-v1'
DECISIONS=983040
ROLLOUT=7680


def configuration(base,parent,*,checkpoint,checkpoint_sha256,runtime,manifest_sha256,
                  native_read_report,job_id,control_root):
    from .thrl_inputs import SEPARATED_LASER_INPUT
    from .resource_teacher_v10 import REVISION as TEACHER
    from .thrl_movement_learning import LASER_ONLINE_REVISION,contract
    from .thrl_ppo import PPOConfig
    from .thrl_branching import validate_parallel_config
    if (parent.get('schema')!='th14-thrl-training-checkpoint-v1'
            or parent.get('optimizer_transaction_complete') is False
            or parent.get('model_config',{}).get('input_schema')!=SEPARATED_LASER_INPUT):
        raise ValueError('A complete original V5 PPO parent is required')
    c=deepcopy(base);retired={}
    for key in list(c):
        if (key.startswith('candidate_') or key in ('baseline_resume','sealed_avoidance_baseline',
                'pre_recovery_sealed_avoidance_baseline','initial_development_reused','initial_baseline_completed',
                'initial_development_not_repeated_during_recovery','learning_migration','infrastructure_retry_of',
                'infrastructure_retry_reason','geometry_repair_qualification','initial_baseline_semantics')):
            retired[key]=c.pop(key)
    parameters=deepcopy(c['parameters']);parameters.update(gae_lambda=.995,update_index=0)
    # Retain the actual128-step history. There is no claimed history expansion.
    c.update(experiment=REVISION,display_name='B1 · GRU · 路线信用传播与软指导',study_name=REVISION,
        seed=20261025,workers=30,rollout_total_decisions=ROLLOUT,trial_decisions=DECISIONS,
        development_decisions=[DECISIONS],development_episodes=20,
        full_story_burn_in=128,full_story_history_storage='raw-v1',
        source_runtime=str(runtime),source_manifest_sha256=manifest_sha256,
        profile=str(runtime/'profiles/th14_1.00b_original.json'),
        policy_model_config=deepcopy(parent['model_config']),model_config=deepcopy(parent['model_config']),
        parent_PPO_policy=dict(checkpoint=str(checkpoint),sha256=checkpoint_sha256,
            optimized_decisions=parent['candidate_optimized_decisions'],source_objective='fixed-method correction;preserve all tensors/Adam/RNG'),
        warm_start_checkpoint=str(checkpoint),warm_start_sha256=checkpoint_sha256,
        parameters=asdict(PPOConfig(**parameters)),teacher_pretraining=False,reset_Adam_after_BC=False,
        teacher_revision=TEACHER,movement_guidance_revision=LASER_ONLINE_REVISION,
        initial_avoidance_development=True,development_objective='avoidance-first-v1',
        avoidance_development_workers=20,avoidance_development_max_frames=16384,
        avoidance_development_seed=382191,checkpoint_every_updates=2,manual_start_only=False,
        native_frame_horizon=None,first_hit_terminal=False,continues_allowed=False,
        policy_action_count=19,finite_native_lives=True,autonomous_bomb=True,
        native_item_read_qualification=str(native_read_report),
        goal_panel_registration=dict(job_id=job_id,control_root=str(control_root)),
        initialization_semantics='New correction objective,exact chosen policy and inherited PPO Adam;fresh original worlds',
        evaluation_after_training_shutdown=True,formal_acceptance=False,
        retired_recovery_metadata=retired)
    c['ppo_baseline']={**c.get('ppo_baseline',{}),'gae_lambda':.995}
    c['ppo_search_space']={**c.get('ppo_search_space',{}),'gae_lambda':[.995]}
    c['lifetime_optimized_decisions_before_phase']=int(parent['lifetime_decisions'])
    c['learning_parameter_change']=dict(revision=REVISION,
        source_gae_lambda=parent['config']['parameters']['gae_lambda'],target_gae_lambda=.995,
        source_rollout_per_world=parent['config']['rollout_total_decisions']//parent['config']['workers'],target_rollout_per_world=256,
        actual_source_history=parent['config'].get('full_story_burn_in'),target_history=128,
        source_teacher=parent['config'].get('teacher_revision'),target_teacher=TEACHER,
        rewards_actions_observations_and_parameter_count_unchanged=True,
        actual_actor_output_gradient_diagnostics='first minibatch every8 completed rollouts',
        hypotheses_are_not_native_skill_evidence=True)
    c['capacity']=validate_parallel_config(c);contract(c)
    if c['parameters']['gamma']!=.999:raise ValueError('This correction keeps the existing .999 reward discount')
    return c


def trace_weights(gamma,lam,frames):
    """GAE TD-residual coefficient, not a bound on eventual credit learning."""
    return {str(f):(gamma*lam)**(f/4) for f in frames}
