"""Reproduce one recorded optimizer step and attribute its policy change.

Works on saved data and a separate model. Does not access or control the game,
or mutate the live learner, checkpoint, optimizer, or training output.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import traceback

import psutil

from fly_touhou.thrl_study_io import read_shared_json


def run(args, status):
    os.environ.update(FLY_TOUHOU_DETERMINISTIC_SPMM='1', FLY_TOUHOU_FUSED_EDGE_GRAD='1',
        FLY_TOUHOU_SPMM_CACHE=str(args.cache/'spmm-kernel-cache'),
        FLY_TOUHOU_EDGE_GRAD_CACHE=str(args.cache/'kernel-cache'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    import numpy as np
    import torch
    from fly_touhou import connectome as graph_module, compact_story_sequence_groups as groups_module
    from fly_touhou.compact_story_behavior import story_behavior_logits
    from fly_touhou.compact_story_critical_replay import CriticalTeacherReplay
    from fly_touhou.compact_story_learning_rate import PolicyLearningRate
    from fly_touhou.compact_story_model import from_story_checkpoint, sequence
    from fly_touhou.compact_story_ppo import optimizer_for, restore_optimizer, update
    from fly_touhou.spiking_engineered_control import SpikingState

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=Path(args.source)
    launch=read_shared_json(source/'launch.json')
    training_path=read_shared_json(source/'initial-training-path.json')['path']
    training=torch.load(training_path,map_location='cpu',weights_only=False)
    tape=torch.load(source/'rollout-030.pt',map_location='cpu',weights_only=False)
    status('loading_separate_model')
    original_load=graph_module.load_connectome
    def load_graph(path,*,max_neurons=None,verify=True):
        return original_load(path,max_neurons=max_neurons,verify=False)
    graph_module.load_connectome=load_graph
    try:
        model=from_story_checkpoint(args.connectome,training['model_payload'],device='cuda:0')
    finally:
        graph_module.load_connectome=original_load
    optimizer=optimizer_for(model)
    restore_optimizer(optimizer,training['optimizer'])
    PolicyLearningRate(optimizer,scale=launch['policy_lr_scale'],
        min_scale=launch['minimum_policy_lr_scale'],adaptive=True)
    del training
    gc.collect()

    status('loading_recorded_critical_teacher_windows')
    bank=CriticalTeacherReplay(balanced=True,protected_exit_forecast=model.protected_exit_forecast,
        native_motion_forecast=model.native_motion_forecast,native_focus_gate=model.native_focus_gate,
        unknown_laser_player_motion=model.unknown_laser_player_motion,
        native_curve_laser_geometry=model.native_curve_laser_geometry)
    sources=read_shared_json(source/'critical-teacher-replay.json')['source_tapes']
    target=(source/'rollout-030.pt').resolve()
    found=False
    for name in sources:
        path=Path(name)
        if path.resolve()==target:
            old=tape;found=True
        else:
            old=torch.load(path,map_location='cpu',weights_only=False)
        bank.add(old,source=str(path.resolve()))
        if found:break
        del old
    if not found:raise ValueError('Recorded first-cycle critical replay source missing')

    seed=args.seed+30
    T,N=tape['actions'].shape
    original_groups=groups_module.world_aligned_groups
    selected=original_groups(T,N,16,np.random.default_rng(seed))
    second=selected[1]
    times=torch.tensor([t for t,w in second])[None]+torch.arange(16)[:,None]
    worlds=torch.tensor([w for t,w in second])[None].expand(16,-1)
    features=tape['features'][times,worlds].to(model.device)
    actions=tape['actions'][times,worlds].to(model.device)
    starts=tape['episode_starts'][times,worlds].to(model.device)
    old_logp=tape['old_log_probs'][times,worlds].to(model.device)
    before_state=SpikingState.pack([tape['block_states'][t].row(w) for t,w in second]).to(model.device)
    def measured():
        with torch.no_grad():
            pred=sequence(model,features,starts,before_state,actions,checkpointed=False)
            logits=story_behavior_logits(pred['logits'],
                tape['metadata']['movement_sampling_temperature'],tape['metadata']['bomb_sampling_temperature'])
            logp=logits.log_softmax(-1).gather(-1,actions[...,None])[...,0]
            delta=logp-old_logp
            return dict(approximate_KL=float((delta.exp()-1-delta).mean()),
                mean_abs_logp_change=float(delta.abs().mean()),max_abs_logp_change=float(delta.abs().max()))
    model.eval()
    pre=measured()
    old_parameters={id(p):p.detach().cpu().clone() for group in optimizer.param_groups for p in group['params']}
    # Same permutation and RNG consumption as the actual learner; only stop
    # the isolated copy after its first optimizer update.
    def first_only(*values):return original_groups(*values)[:1]
    groups_module.world_aligned_groups=first_only
    status('reproducing_one_update')
    try:
        report=update(model,optimizer,tape,epochs=1,micro_scenes=16,seed=seed,
            critical_replay=bank,world_aligned=True,checkpoint_chunks=True,
            boss_attack_coefficient=launch['boss_attack_guidance_coefficient'],
            conditional_wall_coefficient=launch['conditional_wall_guidance_coefficient'])
    finally:
        groups_module.world_aligned_groups=original_groups
    after_parameters={id(p):p.detach().cpu().clone() for group in optimizer.param_groups for p in group['params']}
    model.eval()
    full=measured()
    ablations={}
    if not args.sweep_only:
        status('measuring_individual_parameter_groups')
        for group in optimizer.param_groups:
            if group['name']=='three_critics':continue
            with torch.no_grad():
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.copy_((after_parameters if g is group else old_parameters)[id(p)])
            ablations[group['name']]=measured()
    scales={}
    status('measuring_context_step_scales')
    for factor in (0.,.1,.25,.5,1.):
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group['params']:
                    before=old_parameters[id(p)];after=after_parameters[id(p)]
                    p.copy_(before+(after-before)*factor if group['name']=='new_context_ports' else after)
        scales[str(factor)]=measured()
    return dict(scope=('One saved cycle30 update reconstructed on a separate full-brain model; '
        'individual parameter-group changes are nonlinear interventions, not additive causal shares.'),
        pre_update=pre,after_full_update=full,only_changed_parameter_group=ablations,
        context_step_multiplier_with_other_groups_unchanged=scales,
        reproduced_first_update=report['logs'][0],rollout=str(source/'rollout-030.pt'),
        interpretation=('Compare the reproduced first-update loss and next-batch KL against learning.jsonl '
            'before attributing changes. Context interpolation describes this single step, not a future run. '
            'The two minibatches contain correlated training states.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','connectome','cache','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,default=202610335)
    parser.add_argument('--sweep-only',action='store_true')
    args=parser.parse_args()
    process=psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    status_file=args.output.with_suffix('.status.json')
    def status(phase,**fields):
        status_file.write_text(json.dumps(dict(phase=phase,PID=process.pid,
            create_time=process.create_time(),**fields)),encoding='utf-8')
    try:
        status('starting')
        result=run(args,status)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        status('completed',report=str(args.output))
    except Exception:
        status('failed',error=traceback.format_exc())
        raise
