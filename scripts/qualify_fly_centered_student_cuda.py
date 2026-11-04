"""Explicit complete-graph centered32 training and checkpoint replay qualification."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from fly_touhou.fly_cached_distillation import (
    CachedFitConfig,FitGuard,atomic_json,file_sha,import_reader_preserving_CVD,process_identity,episode_key,
)
from fly_touhou.fly_centered_distillation import load_sensory_statistics,validate_statistics_against_reader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--fit-output',type=Path,required=True)
    parser.add_argument('--initial-parameters',type=Path,required=True)
    args=parser.parse_args()
    if args.audit.exists() or args.fit_output.exists():raise ValueError('Use two fresh distinct outputs')
    args.audit.mkdir(parents=True)
    config=CachedFitConfig(max_rows=32,validation_rows=32,max_updates=8,max_model_decisions=96,
        max_seconds=180.,tbptt=4,checkpoint_every_updates=4,gpu_fraction=.2,minimum_ram_GiB=10.)
    guard=FitGuard(args.audit,config)
    report=dict(schema='centered-student-full-graph-CUDA32-qualification-v1',script_sha256=file_sha(__file__),
        data_forward_budget=128,capture_setup_forward_budget=4,total_forward_budget=132,
        optimizer_step_budget=8,trained_data_rows=32,reloaded_data_rows=32,native_worlds=0,
        status='inspecting',fit_output=str(args.fit_output))
    try:
        module,sequence=import_reader_preserving_CVD()
        reader=module.FullEpisodeSensoryReader(
            './outputs/distillation/B1-d95-v5-sensory9721-resumable-cache-v1',
            './outputs/distillation/B1-d95-final19-labels-v1',guard)
        statistics=load_sensory_statistics(ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024/fly-migration-20261026/sensory-drive-diagnostic-v1/calibration-input-artifact-v1/train-world000-prefix2048-sensory-statistics.json')
        report.update(CVD_import_sequence=sequence,statistics_binding=validate_statistics_against_reader(statistics,reader),
            initial_SHA256=file_sha(args.initial_parameters),process=process_identity())
        if report['initial_SHA256']!='6c42592ae4bd43f567f3eb3149f9d35190f12cc251870d266395172a30ab7617':raise ValueError('Initial I/O source differs')
        if not args.execute_cuda:
            report.update(status='NoTorch_inspect_only',Torch_imported='torch' in sys.modules)
            atomic_json(args.audit/'inspect.json',report);return 0
        if os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'','-1'):raise ValueError('Explicit caller CUDA required')
        import torch,psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        torch.set_num_threads(2);torch.manual_seed(613711)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(.2,0)
        if torch.cuda.mem_get_info(0)[0]<6*1024**3:raise ValueError('Need6GiB free VRAM')
        from fly_touhou.fly_centered_distillation import CenteredFlyDistillationStudent
        from fly_touhou.fly_cached_distillation import fit_cached_student,LabelRows
        from fly_touhou.cached_whole_brain_bridge import CachedWholeBrainBridge
        from fly_touhou.fly_distillation_student import legal_action_distillation_kl
        model=CenteredFlyDistillationStudent.from_initial_parameter_checkpoint(args.initial_parameters,ROOT/'data/connectome',
            statistics=statistics,input_gain=452.74201651411596,propagation_backend='coo-cusparse-cached',device='cpu')
        model.to('cuda:0');guard.check('before_centered_capture')
        model.brain.enable_cuda_capture(1)
        report['capture_setup_forward_completed']=4
        fit=fit_cached_student(model,reader,['world000'],[],args.fit_output,config,guard=guard)
        if fit['status']!='completed_finite_cached_distillation' or fit['optimizer_updates']!=8:
            raise RuntimeError('Centered finite CUDA fit did not finish8updates: '+fit['status'])
        reference=fit['scopes']['frozen_after_train']['metrics']
        report['fit']=dict(result_path=str(args.fit_output/'result.json'),result_sha256=file_sha(args.fit_output/'result.json'),
            total_forward=fit['total_forward_passes'],updates=fit['optimizer_updates'],backward=fit['backward_passes'],CUDA=fit['CUDA'])
        model.brain.clear_cuda_capture();model.brain.clear_cached_propagation();del model
        torch.cuda.empty_cache()
        restored=CenteredFlyDistillationStudent.from_centered_parameter_checkpoint(args.fit_output/'last-completed-parameters.pt',ROOT/'data/connectome',device='cpu')
        restored.to('cuda:0');bridge=CachedWholeBrainBridge(restored)
        labels=LabelRows(guard);losses=[];counts=[0]*19;spikes=DN=0
        with torch.no_grad(),closing(reader.iter_episode('world000',max_rows=32,audit_dir=args.audit/'reloaded-eager-reader')) as rows:
            for row in rows:
                guard.check('reloaded_frozen_eager')
                target,legal=labels.read(row)
                logits,state=bridge.step(row,episode_rng_key=episode_key('world000',613711) if row.metadata.source_row==0 else None)
                loss=legal_action_distillation_kl(logits,torch.tensor(target,device='cuda:0')[None],torch.tensor(legal,device='cuda:0')[None])
                losses.append(float(loss));counts[int(logits.masked_fill(~torch.tensor(legal,device='cuda:0')[None],-float('inf')).argmax(-1))]+=1
                spikes+=int(state.window_spike_counts.sum());DN+=int(state.window_spike_counts[:,restored.brain.output_indices].sum())
        expected=reference['groups']['all']['mean_legal_KL'];mean=sum(losses)/len(losses)
        replay=dict(rows=len(losses),mean_legal_KL=mean,reference_mean_legal_KL=expected,
            KL_error=abs(mean-expected),greedy_counts=counts,greedy_counts_equal=counts==reference['student_greedy_counts'],
            hard_whole_spikes_equal=spikes==reference['hard_window_counts']['whole_brain_spikes'],
            hard_DN_spikes_equal=DN==reference['hard_window_counts']['descending_spikes'])
        passed=len(losses)==32 and replay['KL_error']<=1e-6 and all(replay[k] for k in ('greedy_counts_equal','hard_whole_spikes_equal','hard_DN_spikes_equal'))
        report.update(status='passed' if passed else 'checkpoint_replay_difference',replay=replay,
            actual_data_forward=fit['total_forward_passes']+len(losses),actual_total_forward=4+fit['total_forward_passes']+len(losses),
            actual_optimizer_steps=fit['optimizer_updates'],source_code_SHA256=file_sha(ROOT/'src/fly_touhou/fly_centered_distillation.py'),
            resources=guard.report(),avoidance_or_full_episode_learning_demonstrated=False)
        restored.brain.clear_cached_propagation()
        atomic_json(args.audit/'result.json',report)
        return 0 if passed else 2
    except BaseException as error:
        report.update(status='failed',reason=str(error),resources=guard.report())
        atomic_json(args.audit/'failure.json',report)
        raise


if __name__=='__main__':raise SystemExit(main())
