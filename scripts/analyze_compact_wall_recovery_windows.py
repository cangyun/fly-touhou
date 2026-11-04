"""Locate teacher-supported space recovery before recorded native contacts.

Read-only analysis of saved student states. A forecast-safe move is not a
measured alternate trajectory, and overlapping windows are not independent.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=int, default=128)
    args = parser.parse_args()
    import torch
    import psutil
    from fly_touhou.compact_observation_io import load_observations
    from fly_touhou.compact_guided_wall_candidate import eligible_and_target
    from fly_touhou.thrl_durable_records import write_json

    torch.set_num_threads(2)
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity(process.cpu_affinity()[-2:])
    d, io = load_observations(args.rollout,
        skip_tensors=('recorded_selected_DN', 'recorded_pooled_DN'))
    valid = d.get('valid_mask', torch.ones_like(d['actions'], dtype=torch.bool))
    normal = d['unprotected_mask'] & valid
    known = d.get('recorded_pre_action_trusted', d['trusted'])
    eligible, recovery, target = eligible_and_target(d['features'], d['margins'],
        known, normal, d['rule_probabilities'], d['rule_weights'])
    probability = d['recorded_policy_logits'][..., :18].softmax(-1)
    recovery_mass = (probability * recovery).sum(-1)
    chosen_recovery = recovery.gather(-1, d['actions'].clamp_max(17)[..., None])[..., 0]
    chosen_recovery &= d['actions'] < 18
    stages = d['recorded_native_stage']
    stage_report = {}
    for stage in range(1, 7):
        mask = eligible & (stages == stage)
        if not bool(mask.any()):
            continue
        stage_report[str(stage)] = dict(rows=int(mask.sum()),
            teacher_recovery_mass=float(target[mask].mean()),
            student_recovery_mass=float(recovery_mass[mask].mean()),
            actually_selected_recovery=int((mask & chosen_recovery).sum()))
    contacts = []
    frames = d.get('recorded_episode_frames')
    for t, w in ((d['outcomes'] == 1) & normal & (d['actions'] < 18)).nonzero().tolist():
        first = max(0, t - args.history)
        starts = d['episode_starts'][first:t + 1, w].nonzero().flatten()
        if len(starts):
            first += int(starts[-1])
        indices = eligible[first:t, w].nonzero().flatten() + first
        missed = indices[~chosen_recovery[indices, w]]
        def detail(index):
            return dict(rollout_index=int(index), decisions_before_contact=t-int(index),
                episode_frame=int(frames[index, w]) if frames is not None else None,
                teacher_recovery_mass=float(target[index, w]),
                student_recovery_mass=float(recovery_mass[index, w]),
                action=int(d['actions'][index, w]),
                recovery_directions=recovery[index, w].nonzero().flatten().tolist(),
                chosen_margin_4_12_24=d['margins'][index, w, :, int(d['actions'][index, w])].tolist()
                    if d['actions'][index, w] < 18 else None)
        contacts.append(dict(time=t, world=w, stage=int(stages[t, w]),
            episode_frame=int(frames[t, w]) if frames is not None else None,
            bombs_feature=float(d['features'][t, w, 4]),
            left_censored=first == 0 and not bool(d['episode_starts'][0, w]),
            eligible_recovery_rows=len(indices), non_recovery_choices=len(missed),
            nearest_non_recovery=detail(int(missed[-1])) if len(missed) else None,
            earliest_non_recovery=detail(int(missed[0])) if len(missed) else None))
    report = dict(source=str(args.rollout), history_decisions=args.history,
        stages=stage_report, contacts=contacts,
        contacts_with_prior_recovery_opportunities=sum(c['eligible_recovery_rows'] > 0 for c in contacts),
        contacts_with_prior_non_recovery_choices=sum(c['non_recovery_choices'] > 0 for c in contacts),
        observation_io=io, live_changes=False, CUDA_initialized=torch.cuda.is_initialized(),
        scope='Actual contacts, prior positive-weight teacher guidance and recorded student decisions. Recovery directions have forecast margins, not native counterfactual outcomes. Correlated windows do not prove causal benefit.')
    write_json(args.output, report)
    print({k: v for k, v in report.items() if k not in ('contacts', 'observation_io')})


if __name__ == '__main__':
    main()
