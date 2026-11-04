"""Frozen CUDA check of packed teacher histories; no optimizer or native game."""
from pathlib import Path
import argparse
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parameters','manifest','connectome','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    os.environ.update(FLY_TOUHOU_DETERMINISTIC_SPMM='1',
        FLY_TOUHOU_SPMM_CACHE=str(a.cache/'spmm-kernel-cache'),
        FLY_TOUHOU_EDGE_GRAD_CACHE=str(a.cache/'kernel-cache'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    from fly_touhou.runtime_cache_paths import configure_generated_caches
    configure_generated_caches()
    import numpy as np
    import torch
    from fly_touhou import connectome as graph_module
    from fly_touhou.compact_story_model import from_story_checkpoint
    from fly_touhou.compact_native_rule_replay import NativeRuleTeacherReplay
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    payload=torch.load(a.parameters,map_location='cpu',weights_only=False)
    load=graph_module.load_connectome
    graph_module.load_connectome=lambda path,**kw:load(path,**dict(kw,verify=False))
    try:model=from_story_checkpoint(a.connectome,payload,device='cuda:0').eval()
    finally:graph_module.load_connectome=load
    model.requires_grad_(False)
    replay=NativeRuleTeacherReplay(a.manifest,stage_weights={1:.05,2:.1,3:.3,4:.3,5:.125,6:.125})
    index=None
    for i,row in enumerate(replay.windows):
        if row['stage']!=3:continue
        with np.load(replay.path.parent/row['file'],allow_pickle=False) as z:
            start=replay.burn_in
            if (z['teacher_guidance_weight'][start:].sum()>=1
                    and z['unprotected_mask'][start:].all() and (z['actions'][start:]<18).all()):
                index=i;break
    if index is None:raise ValueError('A fully controllable stage3 reference window is required')
    with torch.no_grad():
        replay.loss(model,index=index)  # warm this process's cached kernels
        def measured(indices):
            torch.cuda.synchronize();begun=time.perf_counter()
            loss,metrics=replay.loss_batch(model,indices=indices)
            torch.cuda.synchronize()
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite teacher loss')
            return dict(loss=float(loss),seconds=time.perf_counter()-begun,metrics=metrics)
        single=measured([index]);repeated=measured([index]*7)
        torch.testing.assert_close(torch.tensor(single['loss']),torch.tensor(repeated['loss']),rtol=2e-5,atol=2e-6)
        indices=replay.sample_indices(np.random.default_rng(11),7)
        mixed=measured(indices)
    report=dict(status='passed',single=single,repeated_same_window=repeated,mixed_seven_windows=mixed,
        repeated_window_loss_absolute_difference=abs(single['loss']-repeated['loss']),
        mixed_forward_seconds_over_single=mixed['seconds']/single['seconds'],
        selected_windows=[replay.windows[i] for i in indices],parameters=str(a.parameters),
        full_connectome_model=True,optimizer_updates=0,native_games=0,live_training_modified=False,
        scope='Same-window duplication tests normalization on the real whole SNN. Seven distinct histories test the packed path. Timing covers teacher loading,burn-in and forward loss only,not full PPO or backward throughput;the live trainer shares this GPU.')
    write_json(a.output,report)
    print(report)


if __name__=='__main__':main()
