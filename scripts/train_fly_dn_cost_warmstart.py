"""Explicit brain-only D95-cost warmstart;default is a NoTorch plan.

One completed 22-state aligned source is retained. Its frozen policy collects
its own trajectories with a zero-effect DN head. Only that head is then fitted
to D95 continuous costs, followed by frozen original-game development20.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
SCHEMA = 'fly-DN-cost-head-only-warmstart-original30-fixed-DNstats-v2'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def head_calibration_step(model, optimizer, normalized_DN_features, teacher_costs):
    """One cached-DN-only Adam step; no brain/critic forward or parameter detach.

    Kept separate so the same production step can be checked on a tiny CPU
    model. Source22 have no gradient and Adam therefore preserves their steps
    and moments; all26 Parameter objects remain owned by the same optimizer.
    """
    import torch
    from fly_touhou.fly_dn_continuous_risk_policy import continuous_cost_calibration_loss
    optimizer.zero_grad(set_to_none=True)
    predicted = model.risk_head.costs_from_features(normalized_DN_features.detach())
    loss = continuous_cost_calibration_loss(predicted, teacher_costs,
        torch.ones(len(predicted),dtype=torch.bool,device=predicted.device))['loss']
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite continuous cost calibration')
    loss.backward()
    if any(p.grad is not None for p in list(model.brain.parameters())+list(model.critic.parameters())):
        raise RuntimeError('Head-only calibration must never update a source brain/critic parameter')
    norm = torch.nn.utils.clip_grad_norm_(model.risk_head.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=loss.detach(),gradient_norm=norm.detach())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('environment-config', 'training-source', 'predecessor-output', 'output', 'd95-checkpoint'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--training-source-sha256', required=True)
    p.add_argument('--connectome', type=Path, default=ROOT / 'data/connectome')
    p.add_argument('--collect-rollouts', type=int, default=8)
    p.add_argument('--head-updates', type=int, default=2000)
    p.add_argument('--minibatch-size', type=int, default=1024)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--brain-seed', type=int, default=613711)
    p.add_argument('--action-seed', type=int, default=971137)
    p.add_argument('--head-seed', type=int, default=71921)
    p.add_argument('--execute-cuda', action='store_true')
    a = p.parse_args(argv)
    if (not 1 <= a.collect_rollouts <= 32 or not 1 <= a.head_updates <= 20000
            or not 1 <= a.minibatch_size <= 8192 or a.max_seconds < 1
            or len(a.training_source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in a.training_source_sha256)
            or any(not 0 <= s < 2**31 for s in (a.brain_seed, a.action_seed, a.head_seed))):
        p.error('Explicit finite collection/calibration budgets and actual lowercase source SHA required')
    return a


def plan(a):
    return dict(schema=SCHEMA, status='prepared_not_executed', Torch_imported='torch' in sys.modules,
        source=str(a.training_source), source_sha256=a.training_source_sha256,
        source_kind='aligned22', source_brain_critics_actor_Adam_preserved=True,
        predecessor_must_be_completed_exited_cleaned=True, actual_native_worlds=30,
        collection_decisions=a.collect_rollouts * 7680, collection_rollouts=a.collect_rollouts,
        collection_policy='existing complete brain;zero-effect following-DN risk head;student actions only',
        teacher_actions=0, safety_takeovers=0, PPO_updates=0, base22_optimizer_updates=0,
        head_calibration_Adam=a.head_updates, head_minibatch=a.minibatch_size,
        head_architecture='DN1299 rate/reference150Hz ->64SiLU->18softplus', head_beta=8.,
        head_fixed_statistics='train-only mean/std;active SD>=1e-4 median floor;inactive scale1;fit once;no clipping',
        head_learning_rate=3e-4, base_learning_rates=3e-5,
        calibration_loss='within-scene min-centered continuous D95 costs;SmoothL1 scale8 beta.1',
        cost_label_domain='all normal-unprotected states;independent of Resource confidence',
        raw_objects_used_only_for_training_labels=True, risk_input='same following DN rate only',
        final_frozen_development_games=20, final_development_native_workers=20,
        final_action_selection='argmax T1',
        final_development_seed=2027431601, formal_stability_test=False,
        no_best_checkpoint_selection=True, head_stopping='fixed final update;only physical/error/STOP guard',
        maximum_seconds=a.max_seconds, neural_batch_width=16, real_neural_groups=[16,14],
        native_frame_cap=None, Miss_cap=None, continues=False,
        RAM_limit=None, VRAM_limit=None, artificial_memory_caps=False,
        source_code_running_v4_modified=False, GPU_or_native_started=False)


def execute(a):
    from train_fly_task_supervision import require_completed_predecessor
    predecessor = require_completed_predecessor(a.predecessor_output)
    if a.output.exists():
        raise FileExistsError('New output required;no automatic overwrite/retry')
    if sha(a.training_source) != a.training_source_sha256:
        raise ValueError('Actual full trained source SHA differs')
    config = json.loads(a.environment_config.read_text('utf-8-sig'))
    from fly_touhou.thrl_training_mode import validate_training_mode
    if (not validate_training_mode(config) or config.get('reward_schema') != 'th14-completion-reward-v3'
            or config.get('training_mode') != 'full-story-v1' or config.get('headless') is not True):
        raise ValueError('Existing private natural full-story completion-v3 environment required')
    if 'torch' in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0'):
        raise ValueError('Fresh explicit GPU0 process required')
    import psutil
    current = psutil.Process()
    name = Path(config['game_exe']).name.lower()
    if [p.info for p in psutil.process_iter(['pid', 'name']) if (p.info['name'] or '').lower() == name]:
        raise RuntimeError('Existing original game;refusing overlap or touching user games')
    from fly_touhou.owned_worker_affinity import worker_cpu_pool, spread_owned_workers
    cpus = current.cpu_affinity(); worker_cpus = worker_cpu_pool(cpus)
    current.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); current.cpu_affinity(cpus[-2:])
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-DN-cost-warmstart-' + str(os.getpid()))
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_job_paths import configure_native_temp
    a.output.mkdir(parents=True, exist_ok=False); (a.output / 'traces').mkdir()
    configure_native_temp(a.output)
    launch = dict(plan(a), status='initializing_explicit_execution', PID=os.getpid(),
                  create_time=current.create_time(), predecessor_verified=predecessor)
    sources = [Path(__file__)] + [ROOT/'src/fly_touhou'/n for n in (
        'fly_dn_cost_model.py', 'fly_dn_cost_checkpoint.py', 'fly_dn_cost_collector.py',
        'fly_dn_cost_labels.py', 'fly_dn_cost_native_policy.py', 'fly_dn_continuous_risk_policy.py',
        'fly_d95_reference_refinement.py', 'fly_gru_aligned_model.py', 'whole_brain.py',
        'fly_representation_trace.py', 'fly_input_drive.py', 'fly_visual_online_checkpoint.py')]
    launch['source_sha256s'] = {str(p): sha(p) for p in sources}
    launch['source_snapshot_scope'] = 'explicit listed sources;not a full runtime snapshot'
    snapshot = a.output/'source-snapshot'; snapshot.mkdir()
    for p in sources:
        shutil.copyfile(p, snapshot/p.name)
    write_json(a.output/'launch.json', launch)
    started = time.monotonic(); last_heartbeat = [started]
    costs = dict(completed_collection_rollouts=0, student_actions_executed=0,
                 head_calibration_Adam=0, PPO_updates=0, base22_optimizer_updates=0,
                 D95_risk_label_query_rows=0, D95_additional_normal_cost_queries=0,
                 teacher_actions=0, safety_takeovers=0, extra_brain_forwards_for_labels=0)
    phase = {'phase': 'initializing'}
    torch = model = optimizer = vector = collector = None
    source = parent = algorithm = None
    writer = None
    training_closed = evaluation_started = evaluation_completed = False
    close_errors = []
    latest_checkpoint = None

    def progress(row):
        nonlocal writer
        phase.clear(); phase.update(row)
        write_json(a.output/'progress.json', dict(row, schema=SCHEMA, costs=copy.deepcopy(costs),
            elapsed_seconds=time.monotonic()-started, PID=os.getpid()))
        if writer is not None:
            prefix='fly_iterations/DN_cost_warmstart_v1/'
            try:
                # Risk pretraining has its own optimizer-step axis. Native
                # decisions never get relabeled as extra training decisions.
                if row.get('phase')=='head_calibration' and 'head_update' in row:
                    for key,value in row.items():
                        if key!='head_update' and type(value) in (int,float):
                            writer.add_scalar(prefix+'head_pretraining/'+key,value,
                                              global_step=row['head_update'])
                elif row.get('phase')=='final_development_only' and row.get('evaluation'):
                    ev=row['evaluation']
                    for key in ('completed','target','policy_decisions','wall_seconds'):
                        if type(ev.get(key)) in (int,float):
                            writer.add_scalar(prefix+'frozen_development_progress/'+key,ev[key],
                                              global_step=int(ev.get('policy_decisions',0)))
                else:
                    for key in ('student_actions_executed','completed_collection_rollouts',
                                'D95_risk_label_query_rows','head_calibration_Adam','PPO_updates'):
                        writer.add_scalar(prefix+'counts/'+key,costs[key],
                                          global_step=costs['student_actions_executed'])
                writer.flush()
            except (OSError,RuntimeError) as error:
                write_json(a.output/'tensorboard-error.json',dict(error=repr(error),
                    training_continues_from_primary_JSON_records=True))
                writer=None

    def guard():
        now = time.monotonic()
        if (a.output/'STOP').exists():
            raise RuntimeError('OperatorSTOP')
        if now-started > a.max_seconds:
            raise RuntimeError('Finite wall budget reached')
        if shutil.disk_usage(a.output).free < 4*1024**3:
            raise RuntimeError('Output disk below4GiB')
        if now-last_heartbeat[0] > 30:
            last_heartbeat[0] = now
            progress(dict(phase))

    def save(tag):
        nonlocal latest_checkpoint
        from fly_touhou.fly_dn_cost_checkpoint import training_checkpoint_payload
        counters = dict(copy.deepcopy(costs), source_aligned_counters=copy.deepcopy(inherited),
            lifetime_completed_rollouts=int(inherited.get('lifetime_completed_rollouts',0)),
            lifetime_Adam_steps=int(inherited.get('lifetime_Adam_steps',0))+costs['head_calibration_Adam'],
            lifetime_base22_Adam_steps=int(inherited.get('lifetime_Adam_steps',0)),
            collection_is_not_PPO_optimized_decisions=True)
        payload = training_checkpoint_payload(model, optimizer, counters=counters,
                                             algorithm=algorithm, parent_provenance=parent)
        payload['input_drive_protocol'] = copy.deepcopy(source['input_drive_protocol'])
        path = a.output/(tag+'-training.pt'); tmp = path.with_suffix('.pt.tmp')
        with tmp.open('wb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path); latest_checkpoint = str(path)
        parameters=a.output/(tag+'-parameters.pt'); temporary=parameters.with_suffix('.pt.tmp')
        with temporary.open('wb') as stream:
            torch.save(model.parameter_checkpoint_payload(),stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,parameters)
        return path

    def close_training():
        nonlocal training_closed
        if not training_closed:
            training_closed = True
            if vector is not None:
                try: vector.close()
                except BaseException as e: close_errors.append(repr(e))
                close_errors.extend(str(x) for x in vector.close_errors)
        all_errors = list(close_errors)
        cleanup = a.output/'development/native_cleanup.json'
        if evaluation_started:
            if cleanup.exists():
                recorded = json.loads(cleanup.read_text('utf-8-sig')).get('close_errors')
                if isinstance(recorded,list): all_errors.extend(str(x) for x in recorded)
                else: all_errors.append('Development cleanup error list unavailable/unverified')
            else:
                all_errors.append('Development cleanup receipt unavailable')
        write_json(a.output/'training-cleanup.json', dict(close_errors=all_errors,
            training_closed=training_closed, worlds=30, development_started=evaluation_started,
            development_completed=evaluation_completed))
        return all_errors

    try:
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True); torch.cuda.init(); torch.manual_seed(a.head_seed)
        from fly_touhou.fly_dn_cost_checkpoint import read_training_source, restore_training_source
        from fly_touhou.fly_gru_aligned_model import FlyGRUAlignedActorCritic
        from fly_touhou.fly_dn_cost_collector import DNCostOnlinePool
        from fly_touhou.fly_dn_continuous_risk_policy import (
            continuous_cost_calibration_loss,training_constant_direction_cost)
        from fly_touhou.fly_d95_reference_refinement import D95RiskPenalty
        from fly_touhou.fly_task_reference_pool import make_task_reference_vector
        from fly_touhou.fly_visual_online_collector_continuation import MovementDiagnosticsVector
        from fly_touhou.thrl_vector import THRLNativeVector
        from train_fly_visual_long_continuous import staggered_reset_vector
        from fly_touhou.fly_representation_trace import record_rollout_representations
        from fly_touhou.fly_input_drive import input_drive_protocol
        source = read_training_source(a.training_source, expected_sha256=a.training_source_sha256,
                                      source_kind='aligned22')
        source_input_drive = source.get('input_drive_protocol')
        if source_input_drive != input_drive_protocol('poisson'):
            raise ValueError('Explicit unchanged source Poisson input-drive protocol required')
        parameter_path = a.output/'source-base-parameters.pt'
        torch.save(source['recovery']['model_payload'], parameter_path)
        base = FlyGRUAlignedActorCritic.from_actor_critic_checkpoint(parameter_path, a.connectome,
            expected_sha256=sha(parameter_path), device='cpu').to('cuda:0')
        model, optimizer, inherited, transition = restore_training_source(base, source,
            source_kind='aligned22', expected_algorithm=source['recovery']['algorithm'],
            risk_learning_rate=3e-4, beta=8., seed=a.head_seed)
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb=json.loads((ROOT/'artifacts/engineering/tensorboard-live-v3/configuration.json').read_text('utf-8-sig'))
            writer=SummaryWriter(log_dir=str(Path(tb['log_directory'])/'fly-migration-20261026'),
                filename_suffix='.DN-cost-head-warmstart',flush_secs=5)
            write_json(a.output/'tensorboard.json',dict(existing_server='http://127.0.0.1:6006/',
                new_server_started=False,tag_prefix='fly_iterations/DN_cost_warmstart_v1/',
                head_training_axis='actual risk-head Adam updates;not native decisions'))
        except (ImportError,OSError,KeyError) as error:
            write_json(a.output/'tensorboard-error.json',dict(error=repr(error),
                training_continues_from_primary_JSON_records=True))
        algorithm = dict(schema=SCHEMA, phase='head-only-supervised-calibration', head_updates=a.head_updates,
            head_minibatch=a.minibatch_size, risk_learning_rate=3e-4, base_learning_rates=3e-5,
            cost_scale=8., Huber_beta=.1, beta=8., collection_rollouts=a.collect_rollouts,
            train_split='own game_id%5!=4', held_split='own game_id%5==4',
            base22_updates=0, PPO_updates=0, loss='continuous within-scene relative D95 cost',
            input_drive_protocol=copy.deepcopy(source_input_drive),
            head_feature_statistics_protocol=model.risk_head.configuration()['feature_statistics_protocol'],
            no_best_selection=True, source_algorithm=copy.deepcopy(source['recovery']['algorithm']))
        parent = dict(source=str(a.training_source), source_sha256=a.training_source_sha256,
            predecessor=predecessor, actual_Adam_transition=transition, source_paid_history_preserved=True,
            input_drive_protocol=copy.deepcopy(source_input_drive),
            base22_remain_fixed=True, D95_checkpoint=str(a.d95_checkpoint), D95_sha256=sha(a.d95_checkpoint))
        write_json(a.output/'initialization.json', parent)
        save('initial')
        launch.update(status='online_cost_collection', CUDA_device_name=torch.cuda.get_device_name(0),
                      input_drive_protocol=copy.deepcopy(source_input_drive),
                      actual_Adam_states_restored=22, risk_head_new_parameter_tensors=4)
        write_json(a.output/'launch.json', launch)
        base_parameters = list(base.brain.parameters())+list(base.critic.parameters())
        base_values = [p.detach().cpu().clone() for p in base_parameters]
        base_Adam = {p:copy.deepcopy(optimizer.state[p]) for p in base_parameters}
        penalty = D95RiskPenalty.from_checkpoint(a.d95_checkpoint, device=model.device)
        vector = MovementDiagnosticsVector(make_task_reference_vector(config, a.output/'training',
            vector_class=staggered_reset_vector(THRLNativeVector)))
        write_json(a.output/'worker-affinity-before-reset.json',
                   spread_owned_workers(os.getpid(),current.create_time(),worker_cpus))
        collector = DNCostOnlinePool(model,vector,reference_refiner=penalty,geometry_label_budget=2048,
                                    brain_seed=a.brain_seed,action_seed=a.action_seed)
        write_json(a.output/'sampling-capture.json',collector.enable_sampling_capture())
        collector.start(); model.eval()
        vectors=[]; labels=[]; games=[]; valid=[]
        for number in range(a.collect_rollouts):
            progress(dict(phase='collecting_cost_labels',collection_rollout=number)); guard()
            rollout = collector.collect(steps=256,
                policy_version=int(inherited.get('lifetime_completed_rollouts',0)),check_stop=guard)
            # Source brain and zero-initialized head are unchanged across the full dataset.
            arrays={name:getattr(rollout,name).cpu().numpy() for name in (
                'DN_features','encoded_visual','features','raw_logits','composed_logits','predicted_costs',
                'actions','action_probs','old_log_probs','legal_actions','game_ids','native_episodes',
                'brain_rng_keys','noise_counters','d95_risk_costs','d95_risk_queried_mask','d95_risk_valid_mask')}
            arrays.update({'task_'+name:getattr(rollout.task_targets,name) for name in
                ('margins','trusted','outcomes','unprotected_mask')})
            path=a.output/'traces'/f'rollout-{number:03}.npz'; temporary=path.with_suffix('.npz.tmp')
            import numpy as np
            with temporary.open('wb') as stream:
                np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary,path)
            write_json(path.with_suffix('.json'),dict(metadata=rollout.metadata,
                step_info=rollout.step_info,completed_episodes=rollout.completed_episodes,
                trace_sha256=sha(path),PPO_updates=0,source_brain_parameters_fixed=True))
            rd=a.output/'neural-diagnostics'/f'rollout-{number:06}'/'representations'
            record_rollout_representations(model,rollout,rd,max_scenes=512)
            vectors.append(rollout.DN_features.reshape(-1,rollout.DN_features.shape[-1]))
            labels.append(rollout.d95_risk_costs.reshape(-1,18))
            games.append(rollout.game_ids.reshape(-1))
            valid.append((rollout.d95_risk_queried_mask&rollout.d95_risk_valid_mask).reshape(-1))
            costs['student_actions_executed']+=7680; costs['completed_collection_rollouts']=number+1
            for key in ('D95_risk_label_query_rows','D95_additional_normal_cost_queries'):
                costs[key]+=rollout.metadata[key]
            del arrays,rollout; gc.collect()
        if close_training(): raise RuntimeError('Cost collection cleanup incomplete')
        collector=vector=None; gc.collect()
        x=torch.cat(vectors); y=torch.cat(labels); game=torch.cat(games); usable=torch.cat(valid)
        train=usable&(game%5!=4); held=usable&(game%5==4)
        if not bool(train.any()) or not bool(held.any()):
            raise RuntimeError('Collected trajectories need nonempty disjoint train/held games')
        xt=x[train].to(model.device); yt=y[train].to(model.device)
        xh=x[held].to(model.device); yh=y[held].to(model.device)
        train_games=sorted(torch.unique(game[train]).tolist())
        held_games=sorted(torch.unique(game[held]).tolist())
        del vectors,labels,games,valid,x,y,game,usable
        constant=training_constant_direction_cost(yt,torch.ones(len(yt),dtype=torch.bool,device=model.device))
        prior=constant.expand(len(yh),-1)
        with torch.no_grad():
            prior_metrics=continuous_cost_calibration_loss(prior,yh,
                torch.ones(len(yh),dtype=torch.bool,device=model.device))
            prior_regret=(yh[:,int(constant.argmin())]-yh.amin(-1)).mean()
        write_json(a.output/'cost-training-split.json',dict(train_games=train_games,held_games=held_games,
            train_rows=len(xt),held_rows=len(xh),constant_direction=int(constant.argmin()),
            constant_held_loss=float(prior_metrics['loss']),constant_held_cost_regret=float(prior_regret),
            teacher_costs_are_not_native_counterfactual_truth=True))
        order_rng=torch.Generator(device='cpu').manual_seed(a.head_seed)
        head=model.risk_head
        feature_statistics = head.fit_feature_statistics(xt)
        feature_statistics.update(train_games=train_games,held_games_used=False,
            feature_mean_raw_sha256=hashlib.sha256(head.feature_mean.detach().cpu().numpy().tobytes()).hexdigest(),
            feature_scale_raw_sha256=hashlib.sha256(head.feature_scale.detach().cpu().numpy().tobytes()).hexdigest())
        write_json(a.output/'DN-feature-statistics.json',feature_statistics)
        for update in range(a.head_updates):
            guard()
            index=torch.randint(len(xt),(min(a.minibatch_size,len(xt)),),generator=order_rng).to(model.device)
            step = head_calibration_step(model,optimizer,xt[index],yt[index])
            loss,norm = step['loss'],step['gradient_norm']; costs['head_calibration_Adam']+=1
            if (update+1)%100==0 or update+1==a.head_updates:
                with torch.no_grad():
                    hp=head.costs_from_features(xh)
                    validation=continuous_cost_calibration_loss(hp,yh,
                        torch.ones(len(yh),dtype=torch.bool,device=model.device))
                    regret=(yh.gather(-1,hp.argmin(-1,keepdim=True)).squeeze(-1)-yh.amin(-1)).mean()
                progress(dict(phase='head_calibration',head_update=update+1,
                    train_loss=float(loss),held_loss=float(validation['loss']),held_cost_regret=float(regret),
                    constant_held_loss=float(prior_metrics['loss']),constant_held_cost_regret=float(prior_regret),
                    head_grad_norm=float(norm),train_rows=len(xt),held_rows=len(xh)))
        write_json(a.output/'risk-head-calibration-result.json',dict(head_updates=costs['head_calibration_Adam'],
            train_rows=len(xt),held_rows=len(xh),held_loss=float(validation['loss']),held_cost_regret=float(regret),
            constant_held_loss=float(prior_metrics['loss']),constant_held_cost_regret=float(prior_regret),
            policy_selected=f'fixed final{a.head_updates} head;no best-of selection',
            feature_statistics=feature_statistics,native_evaluation_still_required=True))
        for p,value in zip(base_parameters,base_values):
            if not torch.equal(p.detach().cpu(),value): raise RuntimeError('Head calibration changed source weights')
            if set(optimizer.state[p])!=set(base_Adam[p]) or any(
                    not torch.equal(optimizer.state[p][k],base_Adam[p][k]) for k in base_Adam[p]):
                raise RuntimeError('Head calibration changed a source Adam moment or step')
        write_json(a.output/'base22-preservation.json',dict(all22_parameters_exact=True,
            all22_Adam_steps_and_moments_exact=True,source_sha256=a.training_source_sha256,
            new_head_Adam=costs['head_calibration_Adam'],PPO_updates=0))
        save('final')
        del xt,yt,xh,yh,base_values,base_Adam; gc.collect(); torch.cuda.empty_cache()
        model.eval().requires_grad_(False)
        frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        from fly_touhou.fly_dn_cost_native_policy import evaluate_dn_cost_native_pool
        from fly_touhou.fly_dn_cost_model import PARAMETER_SCHEMA
        evaluation_started=True; progress(dict(phase='final_development_only'))
        evaluation=evaluate_dn_cost_native_pool(model,config,a.output/'development',episodes=20,
            native_workers=20,seed=2027431601,brain_seed=a.brain_seed,check_stop=guard,
            worker_cpus=worker_cpus,
            source_schema=PARAMETER_SCHEMA,checkpoint_sha256=sha(a.output/'final-parameters.pt'),
            progress=lambda row:progress(dict(phase='final_development_only',evaluation=row)))
        evaluation_completed=True
        if any(not torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters()):
            raise RuntimeError('Frozen original evaluation changed a model parameter')
        if close_training(): raise RuntimeError('Final development cleanup incomplete')
        if writer is not None:
            try:
                for key,value in evaluation.items():
                    if type(value) in (int,float):
                        writer.add_scalar('fly_iterations/DN_cost_warmstart_v1/frozen_development/'+key,
                                          value,global_step=20)
                writer.flush()
            except (OSError,RuntimeError) as error:
                write_json(a.output/'tensorboard-error.json',dict(error=repr(error),
                    development_complete_primary_report_preserved=True))
        write_json(a.output/'result.json',dict(schema=SCHEMA,status='completed',costs=costs,
            development=evaluation,source=parent,final_training_checkpoint=latest_checkpoint,
            source_base22_preserved=True,formal_stability_test=False,
            elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        # The source is immutable;paid head updates get a new full checkpoint.
        checkpoint_error=None
        if model is not None and optimizer is not None and algorithm is not None:
            try: save('stopped')
            except BaseException as second: checkpoint_error=repr(second)
        write_json(a.output/'failure.json',dict(schema=SCHEMA,status='failed_or_explicit_stop',
            error=repr(error),costs=costs,last_progress=phase,latest_recoverable_checkpoint=latest_checkpoint,
            checkpoint_error=checkpoint_error,no_automatic_retry=True,source_checkpoint_unmodified=True))
        raise
    finally:
        try:
            if writer is not None:
                try: writer.close()
                except (OSError,RuntimeError) as error:
                    write_json(a.output/'tensorboard-close-error.json',dict(error=repr(error)))
        finally:
            close_training()
            if model is not None:
                model.brain.clear_cuda_capture(); model.brain.clear_cached_propagation()


def main(argv=None):
    a=parse_args(argv)
    if not a.execute_cuda:
        print(json.dumps(plan(a),ensure_ascii=False)); return
    execute(a)


if __name__=='__main__':
    main()
