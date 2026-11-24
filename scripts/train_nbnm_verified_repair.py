"""Separate verified-route learning from PPO, preserving the original run.

The default repair trains the full signed brain, sensory interfaces and
readouts. A readout-only repair keeps the original brain computation fixed.
Successful native histories are supervised data, never student PPO samples.
This creates a separate candidate requiring frozen native evaluation before
promotion.  It cannot silently overwrite or resume the original long run.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def replay_index(seed, update, pool):
    # The correction pool cycles task groups with index % group_count.
    # Multiplying the update by2 would starve one of two stage groups.
    return int(seed)+int(update)+(0 if pool=='correction' else 1000003)


def rehearsal_requests(seed, update, correction_sequences, has_early_retention, *, reference_every_update=False):
    if has_early_retention and (reference_every_update or update%2==0):
        name, index = 'early_retention', update//2
    else:
        name, index = 'retention', update
    return (('correction',replay_index(seed,update,'correction'),correction_sequences),
            (name,replay_index(seed,index,name),1))


def freeze_shared_representation(model, *, train_witness=True):
    """Keep the sensory and recurrent computation fixed during a readout repair."""
    model.requires_grad_(False)
    model.route.requires_grad_(True)
    witness = getattr(model, 'corridor_witness_head', None)
    if witness is not None and train_witness:
        witness.requires_grad_(True)


def scheduled_prefixes(args, pools, completed):
    requests={}
    every_update=getattr(args,'reference_every_update',False)
    for update in range(completed,args.updates):
        for name,index,count in rehearsal_requests(args.seed,update,args.correction_sequences,
                pools['early_retention'] is not None, reference_every_update=every_update):
            requests.setdefault(name,[]).append((index,count))
    histories={}
    for name,schedule in requests.items():
        histories.update(pools[name].fixed_replay_histories(schedule))
    return histories


def prepare_fixed_prefixes(args, model, spec, pools, proc, started):
    from fly_touhou.nbnm_fixed_readout_cache import identity, read_states
    from fly_touhou.nbnm_prefix import FixedPrefixCache
    from fly_touhou.nbnm_run import atomic_torch_save
    from fly_touhou.thrl_durable_records import write_json

    histories=scheduled_prefixes(args,pools,0)
    signature=identity(model,spec,args.source,histories)
    output=args.output/'fixed-prefix-states.pt'
    progress=args.output/'fixed-prefix-preparation.json'
    prepared=read_states(output,signature,model)
    previous={}
    if prepared is None and args.reuse_prefix_archive is not None:
        previous=read_states(args.reuse_prefix_archive,signature,model,allow_subset=True) or {}
    snapshots={key:tuple(value.detach().cpu() for value in state.tensors())
               for key,state in previous.items()}
    del previous
    reused=len(snapshots)
    cache=FixedPrefixCache(model,{key:history for key,history in histories.items() if key not in snapshots})

    def report(phase):
        write_json(progress,dict(phase=phase,PID=proc.pid,create_time=proc.create_time(),
            planned_steps=cache.planned_steps,replayed_steps=cache.replayed_steps,
            prepared_windows=len(snapshots),target_windows=len(histories),archive=str(output),reused_windows=reused,
            optimizer_updates=0,elapsed_seconds=time.monotonic()-started))

    report('preparing')
    if prepared is not None:
        snapshots=dict.fromkeys(histories)
        report('already-prepared')
        return
    stop=args.output/'STOP'
    for key in histories:
        if key in snapshots:
            continue
        if stop.exists():
            report('stopped')
            return
        previous_steps=cache.replayed_steps
        snapshots[key]=tuple(value.detach().cpu() for value in cache[key].tensors())
        if cache.replayed_steps!=previous_steps:
            report('preparing')
    if identity(model,spec,args.source,histories)!=signature:
        raise RuntimeError('A source changed while preparing fixed recurrent states')
    staged=output.with_suffix(f'.{proc.pid}.pt')
    atomic_torch_save(dict(identity=signature,states=snapshots),staged)
    staged.replace(output)
    report('prepared')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Original or supervised repair checkpoint including Adam')
    parser.add_argument('--correction',type=Path,required=True,help='Native-verified successful route manifest')
    retention=parser.add_mutually_exclusive_group()
    retention.add_argument('--early-retention',type=Path,
                           help='Verified opening-stage histories for every other retention sequence')
    retention.add_argument('--reference-retention',type=Path,
                           help='Match movement score differences on recorded successful histories every other update')
    parser.add_argument('--reference-every-update',action='store_true',
                        help='Keep the same reference windows and reuse each on its adjacent update')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=32)
    parser.add_argument('--correction-sequences',type=int,default=1,
                        help='Number of verified windows sampled from one route per update')
    parser.add_argument('--seed',type=int,default=14271)
    parser.add_argument('--native-exit-weight',type=float,default=0.,
                        help='Fit missing 48/96-frame exit witnesses using measured continuations')
    parser.add_argument('--native-short-risk-weight',type=float,default=.5,
                        help='Weight of short-risk calibration from successful trajectories')
    parser.add_argument('--readout-only',action='store_true',
                        help='Update the movement and corridor-witness heads using the original full brain')
    parser.add_argument('--movement-only',action='store_true',
                        help='Restrict readout repair to the original movement head')
    parser.add_argument('--prefer-observed-branches',action='store_true',
                        help='Rank two actually executed branches at their shared original-game state')
    parser.add_argument('--balance-correction-seeds',action='store_true',
                        help='Rotate original seeds within each corrected stage')
    prefix_replay=parser.add_mutually_exclusive_group()
    prefix_replay.add_argument('--precompute-reference-prefixes',action='store_true',
                               help='Reuse identical original histories for frozen-body reference retention')
    prefix_replay.add_argument('--reuse-fixed-prefixes',action='store_true',
                               help='Reuse identical committed prefixes across scheduled readout-only windows')
    parser.add_argument('--reuse-movement-activity',action='store_true',
                        help='Reuse fixed descending activity across movement-only rehearsal windows')
    operation=parser.add_mutually_exclusive_group()
    operation.add_argument('--execute',action='store_true')
    operation.add_argument('--prepare-fixed-prefixes',action='store_true',
                           help='Compute fixed recurrent history states without an optimizer update')
    parser.add_argument('--reuse-prefix-archive',type=Path,
                        help='Reuse prepared windows of the same recurrent model without copying their archive')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.movement_only:
        args.readout_only=True
    if args.reference_every_update and args.reference_retention is None:
        parser.error('Reference retention on every update requires --reference-retention')
    if args.prepare_fixed_prefixes and (not args.reuse_fixed_prefixes or args.resume):
        parser.error('Fixed prefix preparation requires --reuse-fixed-prefixes and the original source')
    if args.reuse_prefix_archive is not None and not args.reuse_fixed_prefixes:
        parser.error('Reusing a separate prefix archive requires --reuse-fixed-prefixes')
    if args.reuse_movement_activity and not (args.movement_only and args.reuse_fixed_prefixes):
        parser.error('Cached movement activity requires --movement-only and --reuse-fixed-prefixes')
    if not (args.execute or args.prepare_fixed_prefixes):
        return run(args)
    from fly_touhou.nbnm_job_phase import gpu_phase
    with gpu_phase('evaluation' if args.prepare_fixed_prefixes else 'training'):
        return run(args)


def run(args):
    from fly_touhou.nbnm_rehearsal import SuccessRehearsal,verified_correction
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    if not args.source.exists():raise FileNotFoundError(args.source)
    if not 1<=args.updates<=128:raise ValueError('Use a bounded diagnostic repair phase of 1..128 updates')
    if args.correction_sequences<1:raise ValueError('Sample at least one verified sequence per update')
    if args.readout_only and args.native_short_risk_weight:
        raise ValueError('Readout-only repair requires zero short-risk weight')
    if args.movement_only and args.native_exit_weight:
        raise ValueError('Movement-only repair requires zero exit-readout weight')
    if args.precompute_reference_prefixes and not (args.readout_only and args.reference_retention):
        raise ValueError('Reference prefix reuse requires fixed recurrent weights and reference retention')
    if args.reuse_fixed_prefixes and not args.readout_only:
        raise ValueError('Prefix reuse requires fixed recurrent weights')
    correction=SuccessRehearsal(args.correction,native_short_risk_coefficient=args.native_short_risk_weight,
                                native_exit_coefficient=args.native_exit_weight,
                                prefer_observed_branches=args.prefer_observed_branches,
                                balance_native_seeds=args.balance_correction_seeds)
    if not all(verified_correction(row) for row in correction.rows):
        raise ValueError('Every correction requires a native-completed stage or declared road goal')
    retention_path=args.reference_retention or args.early_retention
    match_reference=args.reference_retention is not None
    early=SuccessRehearsal(retention_path,match_reference_scores=match_reference) if retention_path else None
    early_path=str(retention_path) if early is not None else None
    if early is not None and not all(verified_correction(row) for row in early.rows):
        raise ValueError('The separate retention set requires native-verified histories')
    args.output.mkdir(parents=True,exist_ok=True)
    plan=dict(source=str(args.source),correction=str(args.correction),
        corrective_histories=len(correction.rows),supervised_updates=args.updates,
        native_exit_weight=args.native_exit_weight,native_short_risk_weight=args.native_short_risk_weight,
        early_retention=early_path, early_retention_windows=(
            0 if early is None else args.updates if args.reference_every_update else (args.updates+1)//2),
        retention_matches_reference_scores=match_reference,
        reference_every_update=args.reference_every_update,
        precompute_reference_prefixes=args.precompute_reference_prefixes,
        reuse_fixed_prefixes=args.reuse_fixed_prefixes,
        reuse_prefix_archive=str(args.reuse_prefix_archive) if args.reuse_prefix_archive else None,
        reuse_movement_activity=args.reuse_movement_activity,
        verified_windows_per_update=args.correction_sequences,retention_windows_per_update=1,
        sequence_length=16,full_current_weight_prefix_burn_in=True,
        PPO_updates=0,gameplay_samples=0,topology_changed=False,
        readout_only=args.readout_only,movement_only=args.movement_only,
        prefer_observed_branches=args.prefer_observed_branches,
        balance_correction_seeds=args.balance_correction_seeds,
        trainable=('movement head' if args.movement_only else
                   'movement and corridor-witness heads' if args.readout_only else
                   'existing sensory parameters, internal brain parameters, movement and risk readouts'),
        unchanged='original checkpoint, native worlds, reward, action selection, sampling and PPO',
        original_Adam_moments_preserved=True,teacher_loss_competes_with_PPO=False,
        selection='separate candidate; native frozen comparison required',
        executed=args.execute)
    if not args.prepare_fixed_prefixes:
        write_json(args.output/'plan.json',plan)
    if not (args.execute or args.prepare_fixed_prefixes):
        print(json.dumps(plan));return

    import numpy as np
    import torch
    import psutil
    from fly_touhou.nbnm_run import runtime,load_cpu_checkpoint,load_model,atomic_torch_save
    from fly_touhou.compact_story_ppo import optimizer_for,restore_optimizer
    from fly_touhou.whole_brain_ppo import clip_gradients_fp64
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    temporary=runtime(args.output)
    proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    started=time.monotonic();completed=0
    checkpoint=args.output/'candidate-training.pt'
    if checkpoint.exists() and not (args.resume or args.prepare_fixed_prefixes):
        raise ValueError('Use --resume for an existing candidate')
    if args.resume and not checkpoint.exists():raise FileNotFoundError(checkpoint)
    try:
        source=(load_cpu_checkpoint(args.source) if args.prepare_fixed_prefixes
                else torch.load(args.source,map_location='cpu',weights_only=False))
        spec=copy.deepcopy(source['spec'])
        if source.get('in_progress') and len(source['in_progress']['logs'])<int(spec.get('updates_per_cycle',32)):
            raise ValueError('Finish the source batch before a separate repair phase')
        if source.get('schema')=='nbnm-verified-supervised-repair-v1':
            source_cursor=dict(source['source_cursor'])
            parent_updates=int(source.get('parent_supervised_updates',0))+int(source['supervised_updates'])
        else:
            source_cursor=dict(decisions=source['completed_decisions'],PPO_updates=source['optimizer_updates'])
            parent_updates=0
        current=torch.load(checkpoint,map_location='cpu',weights_only=False) if args.resume else source
        if args.resume:
            if current.get('schema')!='nbnm-verified-supervised-repair-v1':raise ValueError('Wrong candidate checkpoint')
            if current['source_cursor']!=source_cursor or current['correction']!=str(args.correction):
                raise ValueError('Repair source or verified teaching data changed')
            if current.get('native_exit_weight',0.)!=args.native_exit_weight:
                raise ValueError('Native exit supervision weight changed during candidate resume')
            if current.get('native_short_risk_weight',.5)!=args.native_short_risk_weight:
                raise ValueError('Short-risk supervision weight changed during candidate resume')
            if current.get('early_retention')!=early_path:
                raise ValueError('Opening-stage retention manifest changed during candidate resume')
            if current.get('retention_matches_reference_scores',False)!=match_reference:
                raise ValueError('Retention objective changed during candidate resume')
            if current.get('reference_every_update',False)!=args.reference_every_update:
                raise ValueError('Reference retention frequency changed during candidate resume')
            if current.get('readout_only',False)!=args.readout_only:
                raise ValueError('Trainable parameter groups changed during candidate resume')
            if current.get('movement_only',False)!=args.movement_only or (
                    current.get('prefer_observed_branches',False)!=args.prefer_observed_branches):
                raise ValueError('Movement repair or observed branch objective changed during resume')
            if current.get('balance_correction_seeds',False)!=args.balance_correction_seeds:
                raise ValueError('Correction seed sampling changed during resume')
            if (current.get('parent_supervised_updates',0)!=parent_updates
                    or Path(current['source']).resolve()!=args.source.resolve()
                    or current.get('correction_sequences',1)!=args.correction_sequences):
                raise ValueError('Repair parent or per-update sequence count changed during resume')
            completed=current['supervised_updates']
        model,_=load_model(spec,checkpoint if args.resume else args.source,
                           payload=source if args.prepare_fixed_prefixes else None)
        model.train()
        if args.readout_only:
            freeze_shared_representation(model,train_witness=not args.movement_only)
        retention_settings=spec.get('rehearsal',{})
        retention=None
        if not args.reference_every_update:
            retention=SuccessRehearsal(retention_settings['manifest'],
                additional_manifests=retention_settings.get('additional_manifests',()),
                scope_weights=retention_settings.get('scope_weights'),
                attack_spell_fraction=retention_settings.get('attack_spell_fraction'),
                challenging_fraction=retention_settings.get('challenging_fraction',0.),
                native_short_risk_coefficient=args.native_short_risk_weight)
        pools=dict(correction=correction,early_retention=early,retention=retention)
        if args.prepare_fixed_prefixes:
            del current,source
            prepare_fixed_prefixes(args,model,spec,pools,proc,started)
            return
        optimizer=optimizer_for(model)
        restore_optimizer(optimizer,current['optimizer'])
        del current,source

        def save():
            payload=model.parameter_payload(dict(NBNM_decisions=source_cursor['decisions'],
                PPO_updates=source_cursor['PPO_updates'],verified_supervised_updates=parent_updates+completed))
            atomic_torch_save(dict(schema='nbnm-verified-supervised-repair-v1',model_payload=payload,
                optimizer=optimizer.state_dict(),source=str(args.source),source_cursor=source_cursor,
                correction=str(args.correction),supervised_updates=completed,spec=spec,
                parent_supervised_updates=parent_updates,correction_sequences=args.correction_sequences,
                native_exit_weight=args.native_exit_weight,early_retention=early_path,
                native_short_risk_weight=args.native_short_risk_weight,readout_only=args.readout_only,
                retention_matches_reference_scores=match_reference,
                reference_every_update=args.reference_every_update,
                movement_only=args.movement_only,prefer_observed_branches=args.prefer_observed_branches,
                balance_correction_seeds=args.balance_correction_seeds,
                native_worlds='unchanged in original source checkpoint; candidate is not auto-promoted'),checkpoint)
            atomic_torch_save(payload,args.output/'candidate-parameters.pt')

        def status(phase,**fields):
            write_json(args.output/'progress.json',dict(phase=phase,PID=proc.pid,create_time=proc.create_time(),
                supervised_updates=completed,target=args.updates,PPO_updates=0,source_cursor=source_cursor,
                parent_supervised_updates=parent_updates,
                elapsed_seconds=time.monotonic()-started,**fields))

        save();status('verified_supervised_repair')
        reference_prefixes=None
        shared_prefixes=None
        if args.precompute_reference_prefixes and completed<args.updates and not (args.output/'STOP').exists():
            indices=[index for update in range(completed,args.updates)
                     for name,index,count in rehearsal_requests(args.seed,update,args.correction_sequences,True,
                         reference_every_update=args.reference_every_update)
                     if name=='early_retention']
            status('precomputing_reference_prefixes')
            start_cache=time.monotonic()
            reference_prefixes,replayed=early.fixed_reference_prefixes(model,indices)
            write_json(args.output/'reference-prefixes.json',dict(windows=len(indices),
                distinct_prefixes=len(reference_prefixes),
                unique_independent_steps=sum(offset for _,offset in reference_prefixes),
                replayed_steps=replayed,elapsed_seconds=time.monotonic()-start_cache))
            status('verified_supervised_repair')
        if args.reuse_fixed_prefixes and completed<args.updates and not (args.output/'STOP').exists():
            from fly_touhou.nbnm_prefix import FixedPrefixCache
            from fly_touhou.nbnm_fixed_readout_cache import identity, read_states
            archive=args.output/'fixed-prefix-states.pt'
            use_prepared=archive.is_file()
            histories=scheduled_prefixes(args,pools,0 if use_prepared else completed)
            prepared_source=None
            if use_prepared:
                shared_prefixes=read_states(archive,identity(model,spec,args.source,histories),model)
                if shared_prefixes is not None:
                    prepared_source=archive
            if shared_prefixes is None:
                if use_prepared and completed:
                    histories=scheduled_prefixes(args,pools,completed)
                if args.reuse_prefix_archive is not None:
                    matching=read_states(args.reuse_prefix_archive,identity(model,spec,args.source,histories),
                                         model,allow_subset=True)
                    if matching is not None and set(matching)==set(histories):
                        shared_prefixes=matching
                        prepared_source=args.reuse_prefix_archive
            loaded=shared_prefixes is not None
            if not loaded:
                shared_prefixes=FixedPrefixCache(model,histories)
            prefix_stats=dict(distinct_windows=len(histories),loaded_prepared_states=loaded,
                              planned_steps=0 if loaded else shared_prefixes.planned_steps,
                              prepared_archive=str(prepared_source) if prepared_source else None)
            write_json(args.output/'fixed-prefixes.json',dict(prefix_stats,replayed_steps=0))
            del histories
        movement_activity={} if args.reuse_movement_activity else None
        while completed<args.updates and not (args.output/'STOP').exists():
            begun=time.monotonic();optimizer.zero_grad(set_to_none=True)
            losses={};metrics={}
            for name,index,count in rehearsal_requests(args.seed,completed,args.correction_sequences,
                    early is not None,reference_every_update=args.reference_every_update):
                records=[]
                losses[name]=0.
                precomputed=reference_prefixes if name=='early_retention' else None
                if shared_prefixes is not None:
                    precomputed=shared_prefixes
                for loss,info in pools[name].iter_losses(model,index=index,windows=count,
                        fixed_prefixes=precomputed,movement_activity=movement_activity):
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite verified supervision')
                    divisor=info.get('rehearsal_windows_in_update',1)
                    (.5/divisor*loss).backward()
                    losses[name]+=float(loss.detach())/divisor
                    records.append(info)
                    del loss
                metrics[name]=records[0] if len(records)==1 else dict(windows=records,
                    rehearsal_rows=sum(row['rehearsal_rows'] for row in records),
                    rehearsal_shared_prefix_steps=records[0]['rehearsal_shared_prefix_steps'])
            gradient_norms={group['name']:sum(float(p.grad.detach().double().square().sum())
                for p in group['params'] if p.grad is not None)**.5 for group in optimizer.param_groups}
            norm=clip_gradients_fp64([p for group in optimizer.param_groups for p in group['params']],.5)
            if not np.isfinite(norm):raise FloatingPointError('Non-finite repair gradient')
            optimizer.step();completed+=1
            append_jsonl(args.output/'updates.jsonl',dict(update=completed,losses=losses,
                gradient_norm=norm,gradient_norms=gradient_norms,metrics=metrics,
                update_seconds=time.monotonic()-begun,PPO_rows=0))
            save()
            if shared_prefixes is not None and not prefix_stats['loaded_prepared_states']:
                write_json(args.output/'fixed-prefixes.json',dict(prefix_stats,replayed_steps=shared_prefixes.replayed_steps))
            status('verified_supervised_repair',losses=losses,gradient_norms=gradient_norms)
        save();status('paused' if completed<args.updates else 'complete')
        write_json(args.output/'result.json',dict(supervised_updates=completed,target=args.updates,
            parent_supervised_updates=parent_updates,readout_only=args.readout_only,
            retention_matches_reference_scores=match_reference,
            reference_every_update=args.reference_every_update,
            movement_only=args.movement_only,prefer_observed_branches=args.prefer_observed_branches,
            original_run_unchanged=True,source_cursor=source_cursor,PPO_updates=0,
            candidate=str(args.output/'candidate-parameters.pt'),native_improvement_not_yet_measured=True))
    except BaseException:
        failure='fixed-prefix-preparation-failure.json' if args.prepare_fixed_prefixes else 'failure.json'
        write_json(args.output/failure,dict(error=traceback.format_exc(),
            committed_supervised_updates=completed,original_run_unchanged=True))
        raise
    finally:
        cleanup_native_temp(temporary)


if __name__=='__main__':main()
