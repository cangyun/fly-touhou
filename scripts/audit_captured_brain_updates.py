"""Full-source CUDA capture parity across real state inputs and parameter changes."""
from pathlib import Path
import argparse
import json
import traceback

import torch

from fly_touhou.connectome import file_sha256, load_connectome
from fly_touhou.whole_brain_learning import load_campaign_demonstrations, load_whole_brain_checkpoint, _prime_state
from benchmark_cached_cusparse import clone_state, require_same_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior capture update audits')
    torch.set_num_threads(4)
    digest=file_sha256(args.checkpoint)
    report={'schema':'complete-brain-captured-parameter-update-audit-v1',
        'checkpoint_sha256':digest,'script_sha256':file_sha256(Path(__file__)),
        'trace_sha256':file_sha256(args.trace),'native_game_actions':0,
        'checkpoint_modified':False,'actual_PPO_training_claimed':False}
    try:
        graph=load_connectome(args.source_dir,max_neurons=None)
        policy=load_whole_brain_checkpoint(args.checkpoint,graph,'cuda')
        model=policy.student.eval()
        model.verify_propagation_buffers()
        model.enable_cuda_capture()
        report['inference_execution']=model.inference_execution_specification()
        dataset=load_campaign_demonstrations([args.trace])
        if dataset.encoder.specification()!=policy.encoder.specification():
            raise ValueError('Observation specification differs')
        observations=torch.as_tensor(dataset.episodes[0].observations[[0,54,55,56]],device=model.device)
        original={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
        with torch.no_grad():
            state=_prime_state(model,observations[0],policy.warmup_ms)
            retained=None
            for i,observation in enumerate(observations):
                expected,end=model._forward_validated(observation.unsqueeze(0),clone_state(state))
                actual,after=model(observation.unsqueeze(0),clone_state(state))
                if not torch.equal(expected,actual):
                    raise AssertionError('Integrated captured forward differs')
                require_same_state(end,after)
                if retained is None:
                    retained=after
                    retained_copy=clone_state(after)
                state=after
            require_same_state(retained,retained_copy)
            report['retained_previous_states_not_rewritten']=True
            before=clone_state(state)
            observation=observations[-1].unsqueeze(0)
            unchanged_logits,_=model._forward_validated(observation,before)
            active=int(before.rate[0].argmax())
            model.action_readout.weight[0,active].add_(.05)
            model.input_encoder.bias.add_(.01)
            model.input_encoder.weight[:,0].add_(.001)
            if model.recurrent_gain_logits is not None:
                model.recurrent_gain_logits.add_(.01)
            expected,end=model._forward_validated(observation,clone_state(before))
            actual,after=model(observation,clone_state(before))
            if not torch.equal(expected,actual) or torch.equal(unchanged_logits,actual):
                raise AssertionError('Captured computation did not read the changed learned tensors exactly')
            require_same_state(end,after)
            report['parameter_change_read_from_original_storage']=True
            report['maximum_logit_change_after_parameter_change']=float((unchanged_logits-actual).abs().max())
            rejected_nan=False
            invalid=observation.clone()
            invalid[0,0]=float('nan')
            try:
                model(invalid,clone_state(before))
            except ValueError as error:
                rejected_nan='Non-finite' in str(error)
            if not rejected_nan:
                raise AssertionError('Public captured path accepted nonfinite game data')
            report['nonfinite_observation_rejected_before_replay']=True
            original_parameter=model.action_readout.weight
            model.action_readout.weight=torch.nn.Parameter(original_parameter.detach().clone())
            rejected_storage=False
            try:
                model(observation,clone_state(before))
            except RuntimeError as error:
                rejected_storage='configuration/storage/source buffers changed' in str(error)
            finally:
                model.action_readout.weight=original_parameter
            if not rejected_storage:
                raise AssertionError('Captured kernels accepted replaced parameter storage')
            report['replaced_parameter_storage_rejected']=True
            for name,p in model.named_parameters():
                p.copy_(original[name].to(model.device))
            capture=model._captured_inference
            incoming,outgoing=model._cached_recurrent,model._cached_recurrent_transpose
            model.to('cpu')
            if (not capture.closed or not incoming._closed or not outgoing._closed
                    or model._captured_inference is not None or model._cached_recurrent is not None):
                raise AssertionError('Moving model did not close graph/native descriptor ownership')
            model.verify_propagation_buffers()
            if any(not torch.equal(p,original[name]) for name,p in model.named_parameters()):
                raise AssertionError('Probe failed to restore exact learned parameters')
            report['model_move_closes_capture_and_native_owners']=True
            report['all_original_parameters_restored_exactly']=True
        if file_sha256(args.checkpoint)!=digest:
            raise AssertionError('Immutable parent changed')
        report.update(status='passed',neurons=graph.n_neurons,edge_records=graph.n_edges)
        print(json.dumps(report),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
