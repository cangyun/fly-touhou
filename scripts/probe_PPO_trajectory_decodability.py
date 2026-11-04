"""Disposable CPU risk decoders on one actually collected, fixed-policy rollout.

Whole worlds, including all their games, are held out together. Geometry
labels are the original stored GRU-style four-frame hypotheses, not measured
counterfactual game outcomes. No fitted weights are exported or deployed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--source-parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--current-head-comparison', action='store_true',
                        help='Same existing LayerNorm head and minibatches; compare only head learning rates')
    parser.add_argument('--joint-head-comparison', action='store_true',
                        help='Same actual actor/risk heads and original guidance; compare their learning rates')
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        parser.error('Positive fixed fitting budget and a fresh report required')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import psutil
    current = psutil.Process()
    cpus = current.cpu_affinity()
    current.cpu_affinity(cpus[-4:-2] if len(cpus) >= 4 else cpus)
    current.nice(psutil.IDLE_PRIORITY_CLASS)
    import numpy as np
    import torch
    from torch import nn
    from fly_touhou.continuous_guided_dagger import LayerNormalizedRisk, PARAMETER_SCHEMA
    from fly_touhou.continuous_signed_clearance import signed_clearance4f_terms
    from fly_touhou.fly_dn_cost_native_policy import compose_from_active_mask
    from fly_touhou.fly_gru_aligned_guidance import gru_aligned_guidance_loss
    from fly_touhou.continuous_object_avoidance import GUIDANCE_CONFIG
    from fly_touhou.thrl_dodge_diagnostics import _shared_binary
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    begun = time.monotonic()
    launch = dict(PID=current.pid, create_time=current.create_time(), argv=sys.argv,
                  CUDA_used=False, native_started=False, production_training_modified=False)
    write_json(args.output.with_suffix('.launch.json'), launch)
    wanted = ('features', 'objects', 'encoded_visual', 'DN_features', 'game_ids',
              'first_observed_contact', 'composed_logits', 'predicted_costs',
              'task_margins', 'task_trusted', 'task_unprotected_mask', 'task_outcomes',
              'rule_probabilities', 'rule_weights', 'actions')
    with _shared_binary(args.rollout) as stream:
        with np.load(stream, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in wanted}
    time_steps, worlds = saved['features'].shape[:2]
    if worlds != 30 or args.rollout.name != 'rollout-000.npz':
        raise ValueError('This diagnostic requires the first fixed-policy 30-world rollout')
    flat = {key: value.reshape((-1,) + value.shape[2:]) for key, value in saved.items()}
    f = flat['features']
    normal = (f[:, 21] > .5) & (f[:, 20] <= 0) & (f[:, 5] <= 0)
    eligible = flat['task_trusted'][:, 0] & flat['task_unprotected_mask'] & normal
    world = np.tile(np.arange(worlds), time_steps)[eligible]
    game = flat['game_ids'][eligible]
    features = torch.from_numpy(f[eligible])
    target = flat['task_margins'][eligible, 0].clip(-64, 64)
    positive, negative = target <= 0, target > 1
    mixed = positive.any(-1) & negative.any(-1)
    contacts = flat['first_observed_contact'][eligible] & mixed
    if len(target) < 1000 or mixed.sum() < 100:
        raise ValueError('Insufficient trusted on-policy observations')
    margins = torch.from_numpy(flat['task_margins'][eligible])
    trust = torch.from_numpy(flat['task_trusted'][eligible])
    unprotected = torch.from_numpy(flat['task_unprotected_mask'][eligible])
    outcomes = torch.from_numpy(flat['task_outcomes'][eligible])
    rule_probabilities = torch.from_numpy(flat['rule_probabilities'][eligible])
    rule_weights = torch.from_numpy(flat['rule_weights'][eligible])
    actions = torch.from_numpy(flat['actions'][eligible])
    views = dict(
        raw=np.concatenate((f[eligible], flat['objects'][eligible].reshape(len(target), -1)), -1),
        encoded384=flat['encoded_visual'][eligible],
        DN1299=flat['DN_features'][eligible])

    def scores(risk, selected=None):
        selected = mixed if selected is None else mixed & selected
        z, pos, neg = np.asarray(risk)[selected], positive[selected], negative[selected]
        if not len(z):
            return dict(rows=0)
        pairs = pos[:, :, None] & neg[:, None, :]
        delta = z[:, :, None] - z[:, None, :]
        ranks = np.where(delta > 0, 1., np.where(delta == 0, .5, 0.))
        auc = (ranks * pairs).sum((1, 2)) / pairs.sum((1, 2))
        chosen = z.argmin(-1)
        return dict(rows=len(z), within_scene_ranking_AUC=float(auc.mean()),
                    selected_proxy_unsafe_count=int(pos[np.arange(len(z)), chosen].sum()),
                    selected_proxy_safe_count=int(neg[np.arange(len(z)), chosen].sum()),
                    selected_proxy_unsafe_fraction=float(pos[np.arange(len(z)), chosen].mean()))

    # The score before softplus cannot be reconstructed from underflowed costs.
    # Replay only the small original risk head on the actual recorded DN.
    with _shared_binary(args.source_parameters) as stream:
        source = torch.load(stream, map_location='cpu', weights_only=False)
    if source.get('schema') != PARAMETER_SCHEMA:
        raise ValueError('Actual calibrated guided parameter checkpoint required')
    source_schema = source['schema']
    risk_state = {key.removeprefix('risk_head.'): value
                  for key, value in source['state_dict'].items() if key.startswith('risk_head.')}
    warm_risk_state = {key: value.clone() for key, value in risk_state.items()}
    warm_actor_state = {key.removeprefix('actor.'): value.clone()
                       for key, value in source['state_dict'].items() if key.startswith('actor.')}
    head = LayerNormalizedRisk()
    head.load_state_dict(risk_state, strict=True)
    del source, risk_state
    with torch.no_grad():
        latent, costs = head.scores_and_costs(torch.from_numpy(views['DN1299']))
    replay_error = float((costs - torch.from_numpy(flat['predicted_costs'][eligible])).abs().max())
    baseline = dict(original_risk_CPU_replayed_latent=scores(latent.numpy()),
                    actual_composed_policy=scores(-flat['composed_logits'][eligible, :18].clip(-80, 80)),
                    risk_cost_CPU_replay_max_abs_error=replay_error)
    del head, costs, latent
    if args.current_head_comparison:
        views = {f'existing_DN_head_LR{rate:g}': views['DN1299']
                 for rate in (3e-5, 3e-4, 1e-3)}
    joint_rates = dict(same_LR=(3e-5, 3e-5), faster_risk=(3e-5, 3e-4),
                       faster_actor_and_risk=(3e-4, 3e-4))
    if args.joint_head_comparison:
        views = {key: views['DN1299'] for key in joint_rates}
    results, prior = {}, np.zeros_like(target)
    for name, values in views.items():
        x = torch.from_numpy(values.astype(np.float32))
        predictions = np.zeros_like(target)
        joint_risk_predictions = np.zeros_like(target)
        fold_results = []
        for fold in range(5):
            held = world % 5 == fold
            train = ~held
            if set(game[train]) & set(game[held]):
                raise ValueError('A whole native game appeared in both splits')
            torch.manual_seed(292910 + fold)
            if args.current_head_comparison or args.joint_head_comparison:
                z = x
                net = LayerNormalizedRisk()
                net.load_state_dict(warm_risk_state, strict=True)
                rate = joint_rates[name][1] if args.joint_head_comparison else float(name.rsplit('LR', 1)[1])
                def predict_margin(values):
                    return 1. - .5 * net.layers(values)
            else:
                center = x[train].mean(0)
                scale = x[train].std(0, unbiased=False)
                scale = scale.clamp_min(max(1e-5, float(scale.median()) * .05))
                z = ((x - center) / scale).clamp(-20, 20)
                net = nn.Sequential(nn.Linear(x.shape[-1], 128), nn.SiLU(),
                                    nn.Linear(128, 128), nn.SiLU(),
                                    nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 18))
                nn.init.normal_(net[-1].weight, std=.01)
                nn.init.zeros_(net[-1].bias)
                rate = .001
                def predict_margin(values):
                    return 8. * net(values)
            if args.joint_head_comparison:
                actor = nn.Sequential(nn.LayerNorm(1299), nn.Linear(1299, 256),
                                      nn.SiLU(), nn.Linear(256, 19))
                actor.load_state_dict(warm_actor_state, strict=True)
                optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=joint_rates[name][0]),
                                              dict(params=net.parameters(), lr=rate)])
            else:
                optimizer = torch.optim.Adam(net.parameters(), lr=rate)
            ids = torch.from_numpy(np.flatnonzero(train))
            generator = torch.Generator().manual_seed(292910 + fold)
            for _ in range(args.steps):
                batch = ids[torch.randint(len(ids), (256,), generator=generator)]
                optimizer.zero_grad(set_to_none=True)
                predicted = predict_margin(z[batch])
                task = dict(margins=margins[batch], trusted=trust[batch],
                            unprotected_mask=unprotected[batch])
                risk_terms = signed_clearance4f_terms(predicted, task, features[batch],
                                                     torch.ones(len(batch), dtype=torch.bool))
                loss = risk_terms['loss']
                if args.joint_head_comparison:
                    raw = actor(z[batch])
                    cost = torch.nn.functional.softplus(net.layers(z[batch]))
                    combined = compose_from_active_mask(raw, cost.detach(),
                        torch.ones(len(batch), dtype=torch.bool), beta=8., check_finite=False)
                    legal = torch.ones_like(combined, dtype=torch.bool)
                    legal[:, 18] = False
                    combined = combined.masked_fill(~legal, -torch.inf)
                    guide = gru_aligned_guidance_loss(combined[None], margins[batch][None],
                        trust[batch][None], actions[batch][None], outcomes[batch][None],
                        unprotected_mask=unprotected[batch][None], config=GUIDANCE_CONFIG,
                        rule_probabilities=rule_probabilities[batch][None], rule_weights=rule_weights[batch][None])
                    loss = risk_terms['weighted_loss'] + guide['total_loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite disposable fitting loss')
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                output = -predict_margin(z).numpy()
                if args.joint_head_comparison:
                    joint_risk_predictions[held] = output[held]
                    combined = compose_from_active_mask(actor(z), net(z),
                        torch.ones(len(z), dtype=torch.bool), beta=8., check_finite=False)
                    output = -combined[:, :18].clamp(-80., 80.).numpy()
            predictions[held] = output[held]
            known = positive[train] | negative[train]
            prior[held] = positive[train].sum(0) / known.sum(0).clip(1)
            fold_results.append(dict(fold=fold, held_worlds=np.unique(world[held]).tolist(),
                train_rows=int(train.sum()), held_rows=int(held.sum()),
                training_loss=float(loss.detach()), train=scores(output, train), held=scores(output, held)))
            write_json(args.output.with_suffix('.progress.json'), dict(view=name, fold=fold,
                fold_complete=True, wall_seconds=time.monotonic()-begun, CPU_only=True))
        results[name] = dict(input_dimensions=x.shape[-1], hidden=[128, 128, 128],
            parameters=sum(parameter.numel() for parameter in net.parameters()) +
                       (sum(parameter.numel() for parameter in actor.parameters()) if args.joint_head_comparison else 0),
            risk_parameters=sum(parameter.numel() for parameter in net.parameters()),
            actor_parameters=sum(parameter.numel() for parameter in actor.parameters()) if args.joint_head_comparison else 0,
            actor_hidden=[256] if args.joint_head_comparison else None,
            evaluated_output='composed final policy' if args.joint_head_comparison else 'risk scores',
            learning_rate=rate, original_head_initialization=args.current_head_comparison or args.joint_head_comparison,
            actor_learning_rate=joint_rates[name][0] if args.joint_head_comparison else None,
            Adam_steps_per_fold=args.steps, held_mixed=scores(predictions),
            held_actual_contact_rows=scores(predictions, contacts), folds=fold_results)
        if args.joint_head_comparison:
            results[name].update(risk_only_held_mixed=scores(joint_risk_predictions),
                                risk_only_held_contact=scores(joint_risk_predictions, contacts))
        del x, z, net, optimizer
    baseline['held_training_direction_prevalence'] = scores(prior)
    result = dict(schema='disposable-PPO-own-state-risk-decoding-v1', rollout=str(args.rollout),
        source_parameters=str(args.source_parameters), source_rows=time_steps*worlds,
        source_parameter_schema=source_schema,
        current_head_learning_rate_comparison=args.current_head_comparison,
        joint_actor_and_risk_comparison=args.joint_head_comparison,
        PPO_in_disposable_fitting=False, critic_in_disposable_fitting=False,
        disposable_Adam_starts_fresh=True,
        eligible_rows=len(target), known_mixed_rows=int(mixed.sum()),
        mixed_actual_contact_rows=int(contacts.sum()), unique_actual_games=len(np.unique(game)),
        whole_worlds_and_games_separated=True, training_standardization_fit_without_held_rows=True,
        targets='stored original trusted4f geometry;proxy unsafe<=0;safe>1',
        fitting=('original safe/future/hit/rule guidance + original weighted signed4f.05;supervised-only cost routing'
                 if args.joint_head_comparison else
                 'original signed4f Huber/8 + .2 known BCE, three original weighted query pools;no other losses'),
        labels_are_native_counterfactual_outcomes=False, baseline=baseline, results=results,
        no_new_game_collection=True, CUDA_used=False, brain_replayed=False,
        models_deployed=False, fitted_weights_exported=False, production_training_modified=False,
        wall_seconds=time.monotonic()-begun,
        scope='same-policy actual on-policy states, exploratory whole-world decoding;different dimensions and parameter counts prevent causal architecture claims;not a native gameplay evaluation')
    assert not torch.cuda.is_initialized()
    write_json(args.output, result)
    print(json.dumps(dict(eligible_rows=len(target), mixed_rows=int(mixed.sum()), baseline=baseline,
        results={key:value['held_mixed'] for key,value in results.items()}, wall_seconds=result['wall_seconds'])))


if __name__ == '__main__':
    main()
