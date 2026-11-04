"""Prepared frozen DN-risk readout comparison; definitions only, no native launch.

One existing complete-brain forward; same following DN drives actor and risk.
beta4 is an experimental frozen readout, beta0 is the exact raw-policy control.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

from .fly_visual_native_wide import WideVisualNativePolicy, validate_width
from .fly_dn_task_risk import FlyDNTaskRiskReadout
from .fly_centered_native_evaluation import native_brain_key
from .thrl_action_risk_readout import adjust_movement_logits
from .thrl_policy_actions import policy_logits
from .thrl_inputs import SEPARATED_LASER_INPUT

SCHEMA = 'fly-task-DN-risk-frozen-native-development-v1'
PARAMETER_SCHEMA = 'fly-visual-task-actor-critic-DN-risk-parameters-v1'


def _beta(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value not in (0,4):
        raise ValueError('Only the declared beta4 candidate or beta0 raw control is prepared')
    return float(value)


def _adjust(logits, risk, eligible, beta):
    if beta == 0 or not bool(eligible.any()):
        return logits
    source = policy_logits(logits[eligible])
    adjusted = adjust_movement_logits(source,risk,beta=beta)
    # The evaluator clamps once more. Reject, rather than silently change the
    # T1 normalization/Bomb marginal or replace a movement with a hard mask.
    if not bool(torch.isfinite(adjusted[:,:18]).all()) or bool((adjusted[:,:18].abs()>80).any()):
        raise RuntimeError('DN-risk candidate unavailable: adjusted movement outside backend clamp domain')
    output = logits.clone()
    output[eligible,:18] = adjusted[:,:18]
    return output  # original Bomb logit, including unavailable -inf, is untouched


class TaskRiskWideVisualNativePolicy(WideVisualNativePolicy):
    """Own-state/cursor/reset behavior is inherited unchanged from the raw policy."""
    def __init__(self,student,risk_head,*,beta=4.,neural_batch_size=16,episodes=20,
                 brain_seed=613711,synthetic_fixture=False):
        if not isinstance(risk_head,FlyDNTaskRiskReadout):
            raise ValueError('Both the original student and trained typed DN-risk head are required')
        risk_head.validate_brain(student.brain)
        super().__init__(student,neural_batch_size=neural_batch_size,episodes=episodes,
            brain_seed=brain_seed,synthetic_fixture=synthetic_fixture)
        self.risk_head,self.beta=risk_head,_beta(beta)
        self.risk_forward_batches=self.risk_forward_rows=self.risk_reweighted_rows=0

    @torch.no_grad()
    def forward_step(self,features,maps,cursor,*,objects):
        context=self._context
        logits,value,following_cursor=super().forward_step(features,maps,cursor,objects=objects)
        if self.beta==0:
            return logits,value,following_cursor  # no risk-head call; exact raw logits/state/cursor
        # Exact normal_unprotected V5 predicate; raw hazards are never read here.
        eligible=(features[:,21]>.5)&(features[:,20]<=0)&(features[:,5]<=0)
        if not bool(eligible.any()):
            return logits,value,following_cursor
        selected=torch.nonzero(eligible,as_tuple=False).reshape(-1).cpu().tolist()
        states=[self.workers[context[i][0]].state for i in selected]
        # Use each actual following state, avoiding another full-neuron state
        # pack merely to read a tiny DN vector. No second brain forward.
        risk=torch.cat([self.risk_head(state) for state in states],dim=0)
        self.risk_forward_batches+=len(states);self.risk_forward_rows+=len(selected)
        adjusted=_adjust(logits,risk,eligible,self.beta)
        self.risk_reweighted_rows+=len(selected)
        return adjusted,value,following_cursor


def development_seed_plan(episodes=20,seed=2027431601,brain_seed=613711):
    """Same per-game SeedSequence and development partition as existing evaluate."""
    from .thrl_training_mode import draw_native_seed
    if (type(episodes) is not int or not 1<=episodes<=20 or type(seed) is not int or not 0<=seed<2**32
            or type(brain_seed) is not int or not 0<=brain_seed<2**31):
        raise ValueError('Finite1..20 games and explicit valid master/brain seeds required')
    plan=[]
    for game in range(episodes):
        e,a=np.random.SeedSequence([seed,game]).generate_state(2)
        plan.append(dict(evaluation_game=game+1,evaluation_seed=int(e),action_seed=int(a),
            native_seed=draw_native_seed(np.random.default_rng(int(e)),'development'),
            brain_rng_key=native_brain_key(game,brain_seed)))
    return plan


def verify_seed_rows(rows,plan):
    if len(rows)!=len(plan) or len({r.get('evaluation_game') for r in rows})!=len(plan):
        raise ValueError('Complete unique actual evaluation games required for seed verification')
    by_game={r['evaluation_game']:r for r in rows}
    for expected in plan:
        actual=by_game.get(expected['evaluation_game'],{})
        if any(actual.get(k)!=expected[k] for k in ('evaluation_seed','action_seed','native_seed')):
            raise ValueError('Applied native/environment/action seed differs from the planned game')
    return True


def evaluate_task_risk_native_wide(student,risk_head,config,output,*,capacity_probe,
        parameter_checkpoint_sha256,beta=4.,neural_batch_size=16,episodes=20,seed=2027431601,
        brain_seed=613711,reference_raw_report=None,check_stop=lambda:None,progress=None):
    """DEFINITION ONLY. Caller must explicitly load/freeze BOTH parameter components.

    capacity_probe() supplies existing owner-verified native_worlds and
    main_identity_alive (PID+ctime), using the prior development capacity gate.
    No current training process is stopped or manipulated here.
    """
    from .thrl_actor_bc_evaluation import capacity_gate
    from .thrl_training_mode import validate_training_mode,episode_protocol
    from .thrl_full_story_evaluation import evaluate
    from .thrl_durable_records import write_json
    from .thrl_dodge_diagnostics import _shared_binary
    import train_thrl_native as native_entry
    beta=_beta(beta);width=validate_width(neural_batch_size)
    plan=development_seed_plan(episodes,seed,brain_seed)
    if (not isinstance(parameter_checkpoint_sha256,str) or len(parameter_checkpoint_sha256)!=64
            or any(c not in '0123456789abcdef' for c in parameter_checkpoint_sha256)
            or not callable(capacity_probe)):
        raise ValueError('Explicit combined base+head checkpoint SHA and verified capacity probe required')
    if (not validate_training_mode(config) or config.get('input_schema')!=SEPARATED_LASER_INPUT
            or config.get('headless') is not True or config.get('autonomous_bomb') is not True
            or config.get('development_objective')=='avoidance-first-v1'):
        raise ValueError('Existing natural headless V5/source19 autonomous-Bomb configuration required')
    if student.training or risk_head.training or any(p.requires_grad for p in student.parameters()) or any(p.requires_grad for p in risk_head.parameters()):
        raise ValueError('Load an independent frozen eval base+head; never adopt a live training object')
    parameters=tuple(student.parameters())+tuple(risk_head.parameters())
    versions=tuple(p._version for p in parameters)
    workers=min(4,episodes)
    def guard(startup=False):
        check_stop();capacity=capacity_probe()
        if type(capacity.get('main_identity_alive')) is not bool:
            raise ValueError('Unknown main PID/ctime identity is not an exited main')
        if tuple(p._version for p in parameters)!=versions:
            raise RuntimeError('Frozen base/head parameters changed during development')
        capacity_gate(capacity['native_worlds'],workers,startup=startup,
            main_identity_alive=capacity['main_identity_alive'])
    guard(startup=True)
    reference=None;reference_sha=None
    if reference_raw_report is not None:
        with _shared_binary(reference_raw_report) as stream:data=stream.read()
        reference=json.loads(data.decode('utf-8-sig'));reference_sha=hashlib.sha256(data).hexdigest()
        if (reference.get('policy_frozen') is not True or reference.get('PPO_updates')!=0
                or reference.get('action_selection')!='greedy' or reference.get('native_seed_split')!='development'):
            raise ValueError('Reference must be the complete frozen raw development report')
        verify_seed_rows(reference['episode_results'],plan)
    policy=TaskRiskWideVisualNativePolicy(student,risk_head,beta=beta,neural_batch_size=width,
        episodes=episodes,brain_seed=brain_seed).eval()
    kwargs=native_entry.make_env_kwargs(config,Path(output),workers,'development')
    if (len(kwargs)!=workers or any(not k['headless'] or not k['hide_window'] or k['policy_action_count']!=19
            or k['frame_skip']!=4 or k['wall_time_limit_seconds'] is not None or 'online_teacher_revision' in k for k in kwargs)):
        raise ValueError('At most4 existing private natural workers; no teacher or truncation')
    protocol=dict(episode_protocol(config),neural_reset='new actual game only;no stage/Miss reset',
        neural_state='complete unchanged own brain state',brain_batch_size=width,real_native_workers=workers,
        risk_scope='following DN only;normal_unprotected f21/f20/f5',risk_beta=beta,
        Bomb_marginal='preserved at final T1;greedy Bomb frequency can change',
        paired_seed_claim='planned until actual complete rows are verified')
    result=evaluate(torch,policy,kwargs,output,episodes=episodes,seed=seed,check_stop=guard,
        progress=progress,native_split='development',episode_protocol=protocol,
        summary=native_entry.development_summary,action_selection='greedy',action_temperature=1.)
    guard()
    verify_seed_rows(result['episode_results'],plan)
    result['task_risk_policy']=dict(schema=SCHEMA,parameter_schema_required=PARAMETER_SCHEMA,
        combined_parameter_checkpoint_sha256=parameter_checkpoint_sha256,beta=beta,
        checkpoint_file_binding='declared by caller;load BOTH through existing TaskLearningBundle before this API',
        frozen_parameter_versions_unchanged=True,
        brain_batch_size=width,real_native_workers=workers,teacher_actor_used=False,
        teacher_hidden_used=False,own_DN_risk_used=policy.risk_forward_rows>0,
        own_DN_risk_configured=beta>0,raw_objects_or_analytic_risk_for_reweight=False,
        optimizer_updates=0,stage_or_Miss_resets=0,brain_batch_calls=policy.brain_batch_calls,
        total_neural_lane_forwards=width*policy.brain_batch_calls,
        risk_forward_batches=policy.risk_forward_batches,risk_forward_rows=policy.risk_forward_rows,
        risk_reweighted_rows=policy.risk_reweighted_rows,episode_records=policy.episode_records,
        actual_seed_plan_verified=True,seed_plan=plan,reference_raw_report_sha256=reference_sha,
        same_reference_initial_native_seeds_verified=reference is not None,
        reference_parameter_identity_verified_here=False,
        action_temperature=1.,old_T025_probabilities_claimed_equal=False,
        beta0_logits_state_cursor_exact=True,greedy_Bomb_frequency_preservation_claimed=False,
        target_risk_is_geometry_proxy_not_native_counterfactual_truth=True,
        proxy_direction_prior_may_explain_head_performance=True,automatic_deployment=False)
    write_json(Path(output)/'report.json',result)
    return result
