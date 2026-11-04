"""Read-only demonstration timing at a saved real state; no game or training."""
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from fly_touhou.runtime_cache_paths import configure_generated_caches, generated_kernel_cache_base
    configure_generated_caches()
    cache = generated_kernel_cache_base(Path('./outputs/distillation/compact-SNN-batch16-recovered-20261101-v2'))
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        FLY_TOUHOU_DETERMINISTIC_SPMM='1', FLY_TOUHOU_SPMM_CACHE=str(cache/'spmm-kernel-cache'))
    import numpy as np
    import torch
    from fly_touhou import connectome as graph_module
    from fly_touhou.compact_story_model import from_story_checkpoint
    from fly_touhou.compact_full_story import apply_sensor_patches
    from fly_touhou.compact_frozen_decoding import configure
    from fly_touhou.compact_frozen_prefix_replay import restore_brain
    from fly_touhou.compact_story_action_selection import StoryActionSelector
    from fly_touhou.thrl_study_io import read_shared_json
    from fly_touhou.thrl_durable_records import write_json
    bundle = Path('./outputs/demonstration/cycle087-v6-first-clear-20261102')
    capsule = Path('./outputs/distillation/compact-SNN-batch16-recovered-20261102-v5/development-cycle087-shield-route-v6-seed3/development/stage6-prefix/brain-world-0.pt')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = read_shared_json(bundle/'original-environment-config.json')
    point = torch.load(bundle/'cycle087-parameters.pt', map_location='cpu', weights_only=False)
    original = graph_module.load_connectome
    graph_module.load_connectome = lambda path, *, max_neurons=None, verify=True: original(path, max_neurons=max_neurons, verify=False)
    try:
        model = from_story_checkpoint(ROOT/'data/connectome', point, device='cuda:0').eval()
    finally:
        graph_module.load_connectome = original
    model.pad_candidate_batch = False
    configure(model, cfg)
    apply_sensor_patches(model, cfg)
    data = torch.load(capsule, map_location='cpu', weights_only=False)
    features = data['features']
    state = restore_brain(model, capsule, features, 3574003864)
    selector = StoryActionSelector('family_mode', 1, seed=2027432927)
    choose = lambda logits: selector(logits, [0])
    timings = []
    with torch.inference_mode():
        # Repeat the same saved state to isolate latency. These repetitions are
        # not new gameplay and must never be presented as a live trajectory.
        for i in range(18):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out, following, dn = model.infer_candidates(features[None], [state], action_selector=choose)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            if i >= 3:
                timings.append(elapsed)
            if i == 0:
                np.savez_compressed(args.output/'real-stage6-inference-sample.npz',
                    features=features, logits=out['logits'][0].cpu().numpy(),
                    selected_action=out['actions'][0].cpu().numpy(),
                    membrane=following[0].membrane[0].cpu().numpy(),
                    spikes=following[0].spikes[0].cpu().numpy().astype(np.uint8),
                    rate=following[0].rate[0].cpu().numpy(),
                    input_indices=model.core.input_indices.cpu().numpy(),
                    output_indices=model.core.output_indices.cpu().numpy(),
                    dn=dn[0].cpu().numpy())
    result=dict(scope='Repeated inference at an actual saved state;not a game,trajectory or end-to-end frame-rate test',
        checkpoint_cycle=87,stage=6,stage_frame=4,source=str(capsule),
        parameters_updated=0,game_started=False,mode='family_mode',
        nodes=model.core.n_neurons,neural_ticks=model.core.K,
        median_ms=1000*float(np.median(timings)),p95_ms=1000*float(np.quantile(timings,.95)),
        samples=len(timings),decision_budget_ms_at_60fps=1000*4/60,
        inference_only_game_fps=4/float(np.median(timings)),
        selected_action=int(out['actions'][0]),gpu=torch.cuda.get_device_name(),
        sample_is='One real next-state inference from stage6 entry;not continuous neural recording')
    write_json(args.output/'timing.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
