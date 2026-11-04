"""Measure a frozen post-PPO policy on the original six BC heldout trajectories.

Recompute this policy's own encoders and entire recurrent contexts. Reusing the
BC latent cache after shared-encoder PPO would give the wrong diagnostic.
These are imitation measurements on old training-partition trajectories, not
new native tests or proof of clear rate. No parameters or optimizers are changed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    sys.path.insert(0, str(a.runtime / 'src'))
    import numpy as np
    import psutil
    import torch
    from fly_touhou.thrl_study_io import read_shared_json as read, utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    from fly_touhou.thrl_full_teacher_data import FullTeacherDataset
    from fly_touhou.thrl_teacher_latents import TeacherLatentCache
    from fly_touhou.thrl_cached_actor_fit import begin_cached_actor_phase, forward_cached_actor
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_policy_actions import policy_logits

    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.runtime / 'source_manifest.json') != '3b8c9d6a506e9cc2cae282a095f627882c9beedbb0b0f379e22bdf5d9ce887c0':
        raise ValueError('Qualified model/data runtime changed')
    if not torch.cuda.is_available() or psutil.virtual_memory().available < 3 * 2**30:
        raise RuntimeError('CUDA and3GiB host working reserve required')
    reference = read(a.reference)
    if (reference.get('status') != 'completed' or reference.get('heldout_scored_labels') != 150224
            or reference.get('original_heldout_indices') != [8, 16, 17, 22, 24, 28]):
        raise ValueError('Original complete six-trajectory BC reference required')
    if sha(a.checkpoint) != '180aa51d0e06b3c7ff5a9caa8462acb1322d59fa32c7bcf4a42f87cbab32855a':
        raise ValueError('Only the already finished, fixed full-PPO candidate is qualified here')
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if sha(a.BC_checkpoint) != reference['actual_BC_sha256']:
        raise ValueError('Original BC identity differs')
    bc = torch.load(a.BC_checkpoint, map_location='cpu', weights_only=False)
    if payload['model_config'] != bc['model_config']:
        raise ValueError('Matched model configuration required')
    risk_keys = [k for k in payload['model'] if k.startswith(('risk.', 'laser_risk.'))]
    if len(risk_keys) != 18 or any(not torch.equal(payload['model'][k], bc['model'][k]) for k in risk_keys):
        raise ValueError('This comparison requires the same fixed physical readouts')
    changed_encoder = [k for k in payload['model'] if k.startswith(
        ('base.map_encoder.', 'base.feature_proj.', 'base.object_attention.'))
        and not torch.equal(payload['model'][k], bc['model'][k])]
    if not changed_encoder:
        raise ValueError('Expected the completed shared-encoder PPO candidate')
    dataset = FullTeacherDataset(a.dataset)
    if dataset.manifest_sha256 != reference['teacher_manifest_sha256']:
        raise ValueError('Teacher archive differs')
    split = dataset.split(seed=31)
    if (split['heldout'] != reference['original_heldout_indices']
            or split['heldout_native_seeds'] != reference['original_heldout_native_seeds']):
        raise ValueError('The original whole-trajectory split must be retained')

    a.output.mkdir(parents=True, exist_ok=False)
    a.created = True
    write(a.output / 'launch.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        utc=utc_now(), native_worlds_started=0, optimizer_updates=0, CUDA=True,
        checkpoint_sha256=sha(a.checkpoint), reference_sha256=sha(a.reference)))
    (a.output / 'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.08)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.monotonic()

    def stop():
        if (a.output / 'STOP').exists():
            raise RuntimeError('Operator stopped this read-only diagnostic')
        if time.monotonic() - started > 1800:
            raise TimeoutError('Finite1800second read-only diagnostic budget exceeded')
        time.sleep(.003)

    def progress(row):
        write(a.output / 'progress.json', dict(row, utc=utc_now(),
            optimizer_updates=0, native_worlds_started=0))

    class HeldoutView:
        def __init__(self):
            self.records = []
            offset = 0
            for original in split['heldout']:
                r = dataset.records[original]
                self.records.append({**r, 'index': len(self.records), 'offset': offset,
                                     'original_index': original})
                offset += r['rows']
            self.rows = offset
            names = ('all', 'normal_unprotected', 'protected_or_non_normal',
                     'teacher_moves', 'teacher_stays', 'teacher_Bomb')
            self.masks = {k: np.zeros(offset, bool) for k in names}
            self.masks.update({'stage' + str(s): np.zeros(offset, bool) for s in range(1, 7)})

        def iter_chunks(self, i, *, chunk_rows):
            record = self.records[i]
            for first, last, data, legal in dataset.iter_chunks(record['original_index'], chunk_rows=chunk_rows):
                lo, hi = record['offset'] + first, record['offset'] + last
                f, actions = data['features'], data['actions']
                normal = (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0)
                values = dict(all=np.ones(last-first, bool), normal_unprotected=normal,
                    protected_or_non_normal=~normal,
                    teacher_moves=(actions != 0) & (actions != 9) & (actions != 18),
                    teacher_stays=(actions == 0) | (actions == 9), teacher_Bomb=actions == 18)
                values.update({'stage' + str(s): data['stage'] == s for s in range(1, 7)})
                for k, value in values.items():
                    self.masks[k][lo:hi] = value
                yield first, last, data, legal

        def close_caches(self):
            dataset.close_caches()

        def descriptors(self, indices, *, length):
            return [(i, t) for i in indices for t in range(0, self.records[i]['rows']-length+1, length)]

    view = HeldoutView()
    model = model_from_config(**payload['model_config']).cuda()
    model.load_state_dict(payload['model'], strict=True)
    if any(not torch.equal(v.detach().cpu(), payload['model'][k]) for k, v in model.state_dict().items()):
        raise ValueError('Actual diagnostic weights do not match the fixed PPO checkpoint')
    begin_cached_actor_phase(model)
    model.requires_grad_(False)
    model.eval()
    # Never reuse the older BC encodings: the CNN and attention weights changed.
    cache = TeacherLatentCache(torch, model, view, on_progress=progress, check_stop=stop)
    costs = model.beta * torch.nn.functional.softplus(cache.risk)
    view.masks['frozen_risk_critical_proxy'] = (view.masks['normal_unprotected'] &
        ((costs.max(-1).values > 4) & (costs.min(-1).values < .1)).cpu().numpy())
    del costs
    descriptors, excluded = cache.valid_descriptors(range(6), length=16)
    if excluded or len(descriptors) * 16 != reference['heldout_scored_labels']:
        raise ValueError('Original heldout target coverage differs')
    cache.refresh_contexts()
    groups = {k: dict(n=0, correct=0, CE_sum=0., probability_sum=0.,
        label_counts=np.zeros(19, np.int64), correct_by_label=np.zeros(19, np.int64)) for k in view.masks}
    for first in range(0, len(descriptors), 64):
        stop()
        selected = descriptors[first:first+64]
        batch = cache.batch(selected, length=16)
        with torch.no_grad():
            logits = forward_cached_actor(model, **{k: batch[k] for k in (
                'encoded', 'features', 'risk', 'initial_hidden', 'dones')})
            clean = policy_logits(logits)
            target = batch['actions']
            losses = torch.nn.functional.cross_entropy(clean, target, reduction='none').cpu().numpy()
            probabilities = clean.softmax(-1).gather(1, target[:, None]).squeeze(1).cpu().numpy()
            prediction = clean.argmax(-1).cpu().numpy()
            truth, keep = target.cpu().numpy(), batch['legal'].cpu().numpy()
        positions = np.concatenate([np.arange(view.records[i]['offset']+t,
            view.records[i]['offset']+t+16) for i, t in selected])
        for name, mask in view.masks.items():
            picked = mask[positions] & keep
            n = int(picked.sum())
            if not n:
                continue
            g = groups[name]
            correct = prediction[picked] == truth[picked]
            g['n'] += n
            g['correct'] += int(correct.sum())
            g['CE_sum'] += float(losses[picked].sum(dtype=np.float64))
            g['probability_sum'] += float(probabilities[picked].sum(dtype=np.float64))
            g['label_counts'] += np.bincount(truth[picked], minlength=19)
            g['correct_by_label'] += np.bincount(truth[picked][correct], minlength=19)
        progress(dict(phase='reading_final_PPO', completed=min(first+64, len(descriptors)),
            total=len(descriptors), wall_seconds=time.monotonic()-started))
    normalized = {}
    for name, g in groups.items():
        n, counts = g['n'], g['label_counts']
        old = reference['comparisons']['selected_BC'][name]
        if n != old['samples'] or counts.tolist() != old['label_counts']:
            raise ValueError('Same original labels/group definitions required:' + name)
        normalized[name] = dict(samples=n, action_agreement=g['correct']/n if n else None,
            hard_teacher_CE=g['CE_sum']/n if n else None,
            mean_teacher_action_probability=g['probability_sum']/n if n else None,
            label_counts=counts.tolist(), correct_by_label=g['correct_by_label'].tolist(),
            agreement_change_from_BC=g['correct']/n-old['action_agreement'] if n else None,
            CE_change_from_BC=g['CE_sum']/n-old['hard_teacher_CE'] if n else None)
    cache.assert_valid()
    if any(not torch.equal(v.detach().cpu(), payload['model'][k]) for k, v in model.state_dict().items()):
        raise ValueError('Read-only diagnostic unexpectedly changed weights')
    if sha(a.checkpoint) != '180aa51d0e06b3c7ff5a9caa8462acb1322d59fa32c7bcf4a42f87cbab32855a':
        raise ValueError('Production checkpoint changed during the diagnostic')
    report = dict(schema='B1-fixed-full-PPO-old-BC-heldout-retention-v1', status='completed', utc=utc_now(),
        checkpoint=str(a.checkpoint), checkpoint_sha256=sha(a.checkpoint),
        reference=str(a.reference), reference_sha256=sha(a.reference),
        BC_checkpoint_sha256=sha(a.BC_checkpoint), teacher_manifest_sha256=dataset.manifest_sha256,
        original_heldout_indices=split['heldout'], original_heldout_native_seeds=split['heldout_native_seeds'],
        heldout_raw_rows=view.rows, heldout_scored_labels=len(descriptors)*16,
        final_PPO=normalized, BC_reference={k: reference['comparisons']['selected_BC'][k] for k in normalized},
        parent_reference={k: reference['comparisons']['parent_before_BC'][k] for k in normalized},
        changed_encoder_tensor_count=len(changed_encoder), encodings_recomputed_for_current_encoder=True,
        full_original_prefix_context_recomputed_for_current_GRU=True, original_fixed_risk_tensors_equal=True,
        old_BC_heldout_is_not_independent_of_later_onpolicy_PPO=True,
        agreement_is_not_clear_rate_or_uniquely_optimal_action=True,
        interpretations_require_frozen_native_results=True,
        cache=cache.stats, optimizer_updates=0, native_worlds_started=0,
        current_frozen_evaluation_configuration_modified=False, CUDA=True,
        wall_seconds=time.monotonic()-started)
    write(a.output / 'report.json', report)
    progress(dict(phase='completed', completed=len(descriptors)*16, total=len(descriptors)*16))
    print(json.dumps({k: v for k, v in report.items() if k not in (
        'final_PPO', 'BC_reference', 'parent_reference', 'cache')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('runtime', 'dataset', 'reference', 'checkpoint', 'BC-checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        if getattr(args, 'created', False):
            from fly_touhou.atomic_json_io import atomic_write_json as write
            write(args.output / 'failure.json', dict(traceback=traceback.format_exc(),
                optimizer_updates=0, native_worlds_started=0, no_automatic_restart=True))
        raise
