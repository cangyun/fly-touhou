"""Recompute training-state DN with a frozen SNN, then fit only its Bomb head.

No original game is launched. No development-game labels are fitted. The
ordinary/critical split is by world identity across tapes, not adjacent rows.
"""
import argparse, copy, gc, json, os, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','resume-training','connectome','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rollout',type=Path,action='append',required=True)
    p.add_argument('--updates',type=int,default=128)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:raise SystemExit('Explicit --execute required for CUDA training-state replay')
    if not 1<=a.updates<=512:raise ValueError('Bounded readout calibration budget1..512 required')
    import psutil
    from fly_touhou.thrl_durable_records import write_json
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic();proc=psutil.Process()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    launch=dict(PID=proc.pid,create_time=proc.create_time(),argv=sys.argv,CUDA_started=False,
        native_games_launched=0,new_PPO_samples=0,development_data_used=False,planned_readout_updates=a.updates)
    write_json(a.output/'launch.json',launch)
    try:
        import torch
        from fly_touhou.compact_story_model import from_story_checkpoint
        from fly_touhou.compact_story_critical_replay import CriticalTeacherReplay
        from fly_touhou.compact_story_bomb_calibration import log_odds,loss,metrics,reset_bomb_moments
        from fly_touhou.spiking_engineered_control import SpikingState
        torch.set_num_threads(2);torch.manual_seed(20261030)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        point=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        resume=torch.load(a.resume_training,map_location='cpu',weights_only=False)
        if set(resume['model_payload']['state_dict'])!=set(point['state_dict']) or any(
                not torch.equal(v,resume['model_payload']['state_dict'][k]) for k,v in point['state_dict'].items()):
            raise ValueError('Parameter and Adam checkpoint weights differ')
        bank=CriticalTeacherReplay(balanced=True)
        for path in a.rollout:
            tape=torch.load(path,map_location='cpu',weights_only=False)
            if not tape.get('metadata',{}).get('full_story') or tape['metadata'].get('teacher_actions')!=0:
                raise ValueError('Original student full-story TRAIN rollouts required')
            bank.add(tape,source=str(path.resolve()));del tape;gc.collect()
        write_json(a.output/'replay-bank.json',bank.summary())
        blocks=bank.sequences+bank.normal_sequences
        if not bank.sequences or not bank.normal_sequences:raise ValueError('Both replay classes are required')
        model=from_story_checkpoint(a.connectome,point).eval()
        launch.update(CUDA_started=True,CUDA_device=torch.cuda.get_device_name(),source_costs=point.get('costs'),
            training_sequences=len(blocks),heldout_world_rule='world %4 ==2 across all source tapes')
        write_json(a.output/'launch.json',launch)
        collected=[];reconstruction_error=0.
        # Actual recorded brain histories and actions; labels never execute.
        with torch.inference_mode():
            for first in range(0,len(blocks),4):
                if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
                chosen=blocks[first:first+4]
                state=SpikingState.pack([b['before'] for b in chosen]).to(model.device)
                for t in range(16):
                    features=torch.cat([b['data']['features'][t] for b in chosen]).to(model.device)
                    actions=torch.cat([b['data']['actions'][t] for b in chosen]).to(model.device)
                    starts=torch.cat([b['data']['episode_starts'][t] for b in chosen]).to(model.device)
                    state=state.reset(starts)
                    out,state,_=model.step(features,state,actions=actions)
                    risk=(8*out['horizon_scores'][...,0]-1).sigmoid().min(-1).values
                    urgency=((risk-.2)/.05).sigmoid()
                    if model.phase_valid_control:
                        vulnerable=(features[:,14]>.5)&(features[:,5]<=0)&(features[:,13]<=0)
                        urgency*=vulnerable|(features[:,17]>.5)
                    legal=out['legal_bomb'].cpu()
                    weights=torch.cat([b['data']['rule_weights'][t] for b in chosen])
                    critical=torch.cat([b['data']['deathbomb_mask'][t] for b in chosen])
                    active=legal&((weights>0)|critical)
                    if not bool(active.any()):continue
                    logmass=out['logits'][:,:18].logsumexp(-1)
                    reconstructed=(model.bomb(out['DN_mean'])[:,0]+model.bomb_urgency_gain*urgency).clamp(-80,80)
                    reconstruction_error=max(reconstruction_error,float((reconstructed[active.to(model.device)]-out['logits'][active.to(model.device),18]).abs().max()))
                    if reconstruction_error>2e-5:raise ValueError('Bomb logit decomposition differs from the current deployed policy')
                    collected.append(dict(DN=out['DN_mean'].cpu()[active],urgency=urgency.cpu()[active],
                        movement_logmass=logmass.cpu()[active],
                        teacher=torch.cat([b['data']['rule_probabilities'][t,:,18] for b in chosen])[active],
                        weights=weights[active],deathbomb=critical[active],
                        world=torch.tensor([b['source_world'] for b in chosen])[active],
                        block=torch.arange(first,first+len(chosen))[active]))
                write_json(a.output/'progress.json',dict(phase='replaying_original_training_states',
                    completed_blocks=min(first+4,len(blocks)),target_blocks=len(blocks),elapsed_seconds=time.monotonic()-begun))
        cache={k:torch.cat([r[k] for r in collected]) for k in collected[0]}
        cache['metadata']=dict(checkpoint=str(a.checkpoint.resolve()),training_only=True,
            development_data_used=False,reconstruction_max_error=reconstruction_error,
            history='Genuine stored pre-block state; current frozen brain; original student actions for16 recurrent steps',
            blocks=[{k:b[k] for k in ('source','source_time','source_world')} for b in blocks])
        torch.save(cache,a.output/'training-Bomb-DN.pt')
        # Convert outside inference mode so normal autograd may save these tensors.
        data={k:v.clone() for k,v in cache.items() if k!='metadata'}
        head=copy.deepcopy(model.bomb).cpu();gain=torch.nn.Parameter(model.bomb_urgency_gain.detach().cpu().clone())
        del model,bank,blocks,cache;gc.collect();torch.cuda.empty_cache()
        held=data['world']%4==2;fit=~held
        train={k:v[fit] for k,v in data.items()};validation={k:v[held] for k,v in data.items()}
        def measure(d):return metrics(log_odds(head,gain,d),d['teacher'],d['weights'],d['deathbomb'])
        report=dict(source_checkpoint=str(a.checkpoint),source_training_checkpoint=str(a.resume_training),
            baseline_training=measure(train),baseline_validation=measure(validation),nodes=[],
            objective='Existing rule0.1 and deathbomb0.05 CE, Bomb-dependent terms only',
            altered_parameters='Existing Bomb readout and neural urgency gain only',
            topology_changed=False,native_games_launched=0,new_PPO_samples=0,
            development_games_used=False,heldout_rule='world %4 ==2 across all source tapes',
            readout_native_benefit_unverified=True,reconstruction_max_error=reconstruction_error)
        best=None;best_loss=report['baseline_validation']['balanced_BCE']
        optimizer=torch.optim.Adam([*head.parameters(),gain],lr=3e-4)
        for update in range(1,a.updates+1):
            if (a.output/'STOP').exists():raise RuntimeError('Operator STOP')
            value=loss(log_odds(head,gain,train),train['teacher'],train['weights'],train['deathbomb'])
            optimizer.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_([*head.parameters(),gain],5.)
            optimizer.step()
            if update%16==0 or update==a.updates:
                node=dict(update=update,loss=float(value.detach()),training=measure(train),validation=measure(validation))
                report['nodes'].append(node)
                v=node['validation'];b=report['baseline_validation']
                # Never select a blanket Bomb suppression that loses the held-out critical decisions.
                if (v['balanced_BCE']<best_loss and v['deathbomb_family_choices']==v['deathbomb_rows']
                        and v['normal_family_Bomb_choices']==0
                        and v['normal_mean_Bomb_probability']<b['normal_mean_Bomb_probability']):
                    best_loss=v['balanced_BCE'];best=(copy.deepcopy(head.state_dict()),gain.detach().clone())
                    report['selected_update']=update;report['selected_validation']=v
                write_json(a.output/'progress.json',dict(phase='Bomb_readout_calibration',**node))
                print(json.dumps(node),flush=True)
        report.update(paid_readout_updates=a.updates,elapsed_seconds=time.monotonic()-begun,complete=True,
            candidate_selected=best is not None)
        if best is not None:
            original=point['state_dict'];point=copy.deepcopy(point)
            point['state_dict'].update({'bomb.'+k:v for k,v in best[0].items()})
            point['state_dict']['bomb_urgency_gain']=best[1]
            untouched=[k for k in original if not k.startswith('bomb.') and k!='bomb_urgency_gain']
            assert all(torch.equal(original[k],point['state_dict'][k]) for k in untouched)
            point['costs']=dict(point.get('costs',{}),bomb_readout_calibration_updates=report['selected_update'])
            point['bomb_readout_calibration']=dict(training_only=True,source=str(a.output/'training-Bomb-DN.pt'),
                selected_update=report['selected_update'],paid_readout_updates=a.updates,native_benefit_unverified=True)
            torch.save(point,a.output/'model-parameters.pt')
            resume['model_payload']=point;resume['optimizer']=reset_bomb_moments(resume['optimizer'])
            resume['Bomb_optimizer_reset_reason']='Bomb readout recalibrated with class contrasts;other Adam groups preserved'
            torch.save(resume,a.output/'resumable-training.pt')
            report.update(candidate=str(a.output/'model-parameters.pt'),resumable_training=str(a.output/'resumable-training.pt'),
                other_model_tensors_bitwise_unchanged=len(untouched),other_Adam_groups_preserved=True,
                new_Bomb_Adam_moments_reset=True)
        write_json(a.output/'report.json',report)
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise


if __name__=='__main__':main()
