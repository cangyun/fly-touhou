"""Small CUDA teacher-supervision branch from an existing full-brain checkpoint.

No native games are started.  The PPO learner continues independently.  This
tests whether recorded ResourceV10 stage3/4 movement distributions can train
the actual LIF brain; its held-out imitation metric is not a game score.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def restore_ppo_lr_groups(updated_optimizer, source_optimizer):
    """Keep teacher-updated Adam moments but restore the source PPO LR schedule."""
    updated = updated_optimizer['param_groups']
    source = source_optimizer['param_groups']
    if [g.get('name') for g in updated] != [g.get('name') for g in source]:
        raise ValueError('Teacher and source optimizer groups differ')
    keys=('lr','policy_base_lr','policy_group_multiplier','policy_reference_base_lr','policy_min_lr_scale')
    groups=[]
    for current, original in zip(updated, source):
        group=dict(current)
        for key in keys:
            if key in original:group[key]=original[key]
            else:group.pop(key,None)
        groups.append(group)
    return dict(updated_optimizer,param_groups=groups)


def configure_teacher_context(rates,override=None):
    """Keep the saved context LR unless this pilot explicitly changes it."""
    group=next(g for g in rates.optimizer.param_groups if g['name']=='new_context_ports')
    value=group['policy_group_multiplier'] if override is None else override
    rates.set_group_multiplier('new_context_ports',value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-checkpoint', 'connectome', 'manifest', 'output', 'kernel-cache'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--updates', type=int, default=4)
    p.add_argument('--eval-per-stage', type=int, default=1)
    p.add_argument('--data-type', choices=('legacy','native'), default='legacy')
    p.add_argument('--stages', nargs='+', type=int, default=[3,4])
    p.add_argument('--risk-lr-multiplier', type=float,
        help='Optional relative learning rate for the existing risk head;Adam moments are preserved')
    p.add_argument('--movement-lr-multiplier', type=float,
        help='Optional relative learning rate for the existing movement head;Adam moments are preserved')
    p.add_argument('--teacher-risk-loss-scale', type=float, default=1.,
        help='Scale only the teacher risk loss during updates;held-out metrics use the original objective')
    p.add_argument('--context-lr-multiplier',type=float,
        help='Optional context LR multiplier; otherwise preserve the saved optimizer value')
    p.add_argument('--wait-for-collection-progress',type=Path,
        help='Start the pilot at the beginning of a live student collection')
    p.add_argument('--trainer-pid',type=int)
    p.add_argument('--trainer-created',type=float)
    p.add_argument('--min-cycle',type=int,default=1)
    a = p.parse_args()
    if a.updates < 1 or a.eval_per_stage < 1:
        raise ValueError('Positive training and evaluation counts required')
    waiting=(a.wait_for_collection_progress is not None,a.trainer_pid is not None,a.trainer_created is not None)
    if any(waiting) and not all(waiting):
        raise ValueError('Collection wait needs progress path, trainer PID and creation time')
    a.output.mkdir(parents=True, exist_ok=False)
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    configure_generated_caches()
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_FUSED_EDGE_GRAD='1', FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_EDGE_GRAD_CACHE=str(a.kernel_cache/'kernel-cache'),
        FLY_TOUHOU_SPMM_CACHE=str(a.kernel_cache/'spmm-kernel-cache'))
    from fly_touhou.thrl_durable_records import write_json
    start = time.monotonic()
    def progress(phase, **values):
        write_json(a.output/'progress.json', dict(phase=phase, seconds=time.monotonic()-start, **values))

    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if a.wait_for_collection_progress is not None:
            from audit_compact_replay_first_groups import wait_for_collection
            progress('waiting_for_student_collection',min_cycle=a.min_cycle)
            if not wait_for_collection(a.wait_for_collection_progress,
                    dict(PID=a.trainer_pid,create_time=a.trainer_created),a.min_cycle,7200,max_decisions=1024):
                raise RuntimeError('Trainer exited or collection wait expired')
        import torch
        from fly_touhou import connectome as graph_module
        from fly_touhou.compact_rule_teacher_replay import RuleTeacherArchiveReplay
        from fly_touhou.compact_story_model import from_story_checkpoint
        from fly_touhou.compact_story_ppo import optimizer_for, restore_optimizer
        from fly_touhou.compact_story_learning_rate import PolicyLearningRate
        from fly_touhou.whole_brain_ppo import clip_gradients_fp64
        torch.set_num_threads(2)
        progress('loading_existing_training_checkpoint')
        saved = torch.load(a.training_checkpoint, map_location='cpu', weights_only=False)
        payload = saved['model_payload']
        with (a.output/'parent-parameters.pt').open('wb') as stream:
            torch.save(payload, stream)
        original_load = graph_module.load_connectome
        def load_graph(path, *, max_neurons=None, verify=True):
            return original_load(path, max_neurons=max_neurons, verify=False)
        graph_module.load_connectome = load_graph
        try:
            model = from_story_checkpoint(a.connectome, payload, device='cuda:0')
        finally:
            graph_module.load_connectome = original_load
        model.pad_candidate_batch = False
        optimizer = optimizer_for(model)
        restore_optimizer(optimizer, saved['optimizer'])
        rates = PolicyLearningRate(optimizer)
        context_multiplier=configure_teacher_context(rates,a.context_lr_multiplier)
        if a.risk_lr_multiplier is not None:
            rates.set_group_multiplier('risk',a.risk_lr_multiplier)
        if a.movement_lr_multiplier is not None:
            rates.set_group_multiplier('movement',a.movement_lr_multiplier)
        if a.data_type == 'native':
            from fly_touhou.compact_native_rule_replay import NativeRuleTeacherReplay
            replay_type = NativeRuleTeacherReplay
        else:
            replay_type = RuleTeacherArchiveReplay
        train = replay_type(a.manifest, split='train')
        held = replay_type(a.manifest, split='heldout')
        def choose(replay, stage, n):
            choices = [i for i, row in enumerate(replay.windows) if row['stage']==stage]
            if not choices:
                raise ValueError('Missing teacher windows for stage '+str(stage))
            if n == 1:
                return [choices[len(choices)//2]]
            # Spread a short pilot across the whole stage and both teacher
            # games instead of repeatedly sampling the first game's opening.
            return [choices[round(i*(len(choices)-1)/(n-1))] for i in range(n)]
        train_by_stage = {stage: choose(train, stage, max(1, (a.updates+len(a.stages)-1)//len(a.stages))) for stage in a.stages}
        held_indices = sum((choose(held, stage, a.eval_per_stage) for stage in a.stages), [])
        def evaluate(label):
            model.eval(); rows = []
            for index in held_indices:
                with torch.no_grad():
                    value, info = held.loss(model, index=index)
                rows.append(dict(info, loss=float(value)))
                progress('heldout_teacher_'+label, completed=len(rows), target=len(held_indices), rows=rows)
            model.train()
            def count(r):
                return r.get('rule_archive_teacher_rows',r.get('native_rule_movement_rows',0))
            weight = sum(count(r) for r in rows)
            if a.data_type == 'native':
                movement=(sum(r['native_rule_movement_loss']*count(r) for r in rows)/weight if weight else None)
                weak_rows=sum(r.get('native_rule_weak_behavior_rows',0) for r in rows)
                weak_loss=(sum(r.get('native_rule_weak_behavior_loss',0)*r.get('native_rule_weak_behavior_rows',0)
                               for r in rows)/weak_rows if weak_rows else None)
                pairs=sum(r.get('risk_proxy_ordered_pairs',0) for r in rows)
                mixed=sum(r.get('risk_proxy_mixed_scenes',0) for r in rows)
                return dict(rows=rows, eligible_rows=weight,
                    mean_loss=sum(r['loss'] for r in rows)/len(rows), movement_loss=movement,
                    weak_behavior_rows=weak_rows,weak_behavior_loss=weak_loss,
                    risk_loss=sum(r['native_rule_risk_loss'] for r in rows)/len(rows),
                    risk_proxy_ordered_pairs=pairs,
                    risk_proxy_order_accuracy=(sum(r.get('risk_proxy_correct_pairs',0) for r in rows)/pairs if pairs else None),
                    risk_proxy_mixed_scenes=mixed,
                    policy_proxy_danger_choices=sum(r.get('policy_proxy_danger_choices',0) for r in rows),
                    risk_proxy_scope='Trusted 4-frame modeled geometry;not observed outcomes of unexecuted movements')
            if not weight:raise ValueError('No held-out teacher movement labels in chosen windows')
            return dict(rows=rows, eligible_rows=weight,
                mean_loss=sum(r['loss']*count(r) for r in rows)/weight)

        before = evaluate('before')
        updates = []
        for step in range(a.updates):
            stage = a.stages[step % len(a.stages)]
            index = train_by_stage[stage][(step//len(a.stages)) % len(train_by_stage[stage])]
            optimizer.zero_grad(set_to_none=True)
            loss, info = (train.loss(model,index=index,risk_loss_scale=a.teacher_risk_loss_scale)
                          if a.data_type=='native' else train.loss(model,index=index))
            loss.backward()
            gradient_by_group = {}
            for group in optimizer.param_groups:
                parts = [p.grad.detach().float().square().sum() for p in group['params'] if p.grad is not None]
                gradient_by_group[group['name']] = float(torch.sqrt(torch.stack(parts).sum())) if parts else 0.
            norm = clip_gradients_fp64([p for group in optimizer.param_groups for p in group['params']], 1.)
            optimizer.step()
            updates.append(dict(info, loss=float(loss.detach()), gradient_norm=float(norm),
                                gradient_norm_by_group=gradient_by_group))
            progress('teacher_supervised_update', completed=step+1, target=a.updates, latest=updates[-1])
        after = evaluate('after')
        # Persist the measured result before writing large checkpoints; if a
        # disk fills up the actual experiment outcome must not be lost.
        write_json(a.output/'measurements.json',dict(status='evaluated',updates=updates,
            heldout_before=before,heldout_after=after,supervised_updates=len(updates),
            teacher='ResourceV10',PPO_samples=0))
        costs = dict(payload.get('costs', {}))
        costs['offline_rule_teacher_updates'] = int(costs.get('offline_rule_teacher_updates', 0))+len(updates)
        candidate = model.parameter_payload(costs)
        parameters_temporary=a.output/'candidate-parameters.pt.tmp'
        try:
            with parameters_temporary.open('wb') as stream:
                torch.save(candidate, stream)
            os.replace(parameters_temporary,a.output/'candidate-parameters.pt')
        except BaseException:
            parameters_temporary.unlink(missing_ok=True)
            raise
        teacher_optimizer=optimizer.state_dict()
        teacher_group_rates={g['name']:g['lr'] for g in teacher_optimizer['param_groups']}
        ppo_optimizer=restore_ppo_lr_groups(teacher_optimizer,saved['optimizer'])
        saved.update(model_payload=candidate, optimizer=ppo_optimizer,
                     torch_RNG=torch.get_rng_state(), CUDA_RNG=torch.cuda.get_rng_state_all(),
                     offline_rule_teacher_updates=len(updates),
                     teacher_pilot_lr_rates=teacher_group_rates,
                     PPO_lr_groups_restored_from_source=True)
        temporary=a.output/'candidate-training.pt.tmp'
        try:
            with temporary.open('wb') as stream:
                torch.save(saved, stream)
            os.replace(temporary,a.output/'candidate-training.pt')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        report = dict(status='completed', teacher='ResourceV10', source_checkpoint=str(a.training_checkpoint),
            CUDA=torch.cuda.get_device_name(), original_games_started=0, new_PPO_samples=0,
            supervised_updates=len(updates), updates=updates, heldout_before=before, heldout_after=after,
            candidate_parameters=str(a.output/'candidate-parameters.pt'),
            candidate_training=str(a.output/'candidate-training.pt'),
            context_port_lr_multiplier=context_multiplier, risk_lr_multiplier=a.risk_lr_multiplier,
            movement_lr_multiplier=a.movement_lr_multiplier, policy_lr_scale=rates.scale,
            policy_group_learning_rates=teacher_group_rates,
            PPO_resume_learning_rates={g['name']:g['lr'] for g in ppo_optimizer['param_groups']},
            PPO_lr_groups_restored_from_source=True,
            teacher_risk_loss_scale=a.teacher_risk_loss_scale,
            inherited_Adam_moments=True,
            heldout_metric=('Combined movement and risk proxy loss on heldout teacher windows;see manifest for the game split;not original-game performance'
                if a.data_type=='native' else 'Cross entropy to rule-teacher movement on partial old observations;not original-game performance'),
            heldout_not_used_for_training=True, brain_topology_changed=False,
            seconds=time.monotonic()-start)
        write_json(a.output/'report.json', report)
        progress('completed', report=str(a.output/'report.json'))
        print(json.dumps(dict(status='completed', heldout_before=before['mean_loss'],
                              heldout_after=after['mean_loss'])))
    except BaseException as exc:
        write_json(a.output/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        progress('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
