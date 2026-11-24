"""Train long-risk readouts on supported continuations, not forced straight runs.

No action score is changed here. The existing DN-only readout learns these
labels through the existing risk loss. A positive path is a conditional native
geometry forecast, not a measured counterfactual or knowledge of future spawns.
Missing finite-search exits remain unknown; immediate contact is unavoidable
within a 4-frame command and remains a valid negative for longer horizons.
"""
import numpy as np

REVISION = 'supported-continuation-risk-v1'


def enabled(config):
    from .nbnm_policy_continuation_risk import REVISION as POLICY_REVISION
    revision = (config or {}).get('risk_target_revision')
    if revision not in (None, REVISION, POLICY_REVISION):
        raise ValueError('Unknown NBNM risk-target revision')
    return revision in (REVISION, POLICY_REVISION)


def targets(label, expert):
    margins = np.asarray(label['margins'], np.float32)
    trust = np.asarray(label['trusted'], bool)
    risk_trust = np.asarray(label.get('risk_training_trusted', trust), bool)
    if margins.shape != (3, 18) or trust.shape != (3,) or risk_trust.shape != (3,):
        raise ValueError('Three native horizons and 18 candidate directions required')
    result = margins.copy()
    valid = np.zeros((3, 18), bool)
    source = np.zeros((3, 18), np.uint8)
    # Exactly preserve the 4f risk-label contract, including exposed calibration.
    valid[0] = risk_trust[0]
    enclosure=bool(label.get('enclosing_ex_forecast_present'))
    if enclosure:
        # Being outside ALL possible random/release trajectories proves a
        # conditional safe path. Being inside their union does NOT prove a
        # collision. Exact discrete overlaps retain separate partial labels.
        valid[0] &= margins[0]>0
    source[0, valid[0]] = 4
    normal = bool(label['normal'])
    if not normal:
        return dict(risk_continuation_margins=result, risk_continuation_valid=valid,
                    risk_continuation_source=source)
    immediate_contact = trust[0] & (margins[0] <= 0) & (not enclosure)
    for h in (1, 2):
        # A safe straight continuation is itself a witnessed candidate.
        straight_safe = trust[h] & (margins[h] > 1)
        valid[h] = straight_safe | immediate_contact
        source[h, straight_safe] = 1
        # The first command cannot turn before its fourth native tick.
        result[h, immediate_contact] = margins[0, immediate_contact]
        source[h, immediate_contact] = 2
    detail = expert.get('nb_nm_route') or {}
    horizon = int(detail.get('route_horizon_frames', 0))
    if (float(expert.get('guidance_weight', 0)) > 0 and detail.get('route_search')
            and horizon in (12, 24) and trust[:(2 if horizon == 12 else 3)].all()):
        exits = np.asarray(detail.get('exits', []))
        best = np.asarray(detail.get('best_margin', []), np.float32)
        if exits.shape != (18,) or best.shape != (18,):
            raise ValueError('Route witnesses must describe all 18 first actions')
        supported = (margins[0] > 1) & (exits > 0) & np.isfinite(best) & (best > 1)
        for h in (1, 2):
            if (12, 24)[h-1] > horizon:
                continue
            use = supported & (~valid[h] | (best > result[h]))
            # A 24f path's minimum clearance is a conservative 12f lower bound.
            result[h, use] = best[use]
            valid[h, use] = True
            source[h, use] = 3
    return dict(risk_continuation_margins=result, risk_continuation_valid=valid,
                risk_continuation_source=source)


def conflict_metrics(batch, model):
    """Score decomposition on collected student states; no action intervention."""
    import torch
    logits = batch['recorded_policy_logits'][..., :18].float()
    risk = (8*batch['recorded_neural_horizon_scores'].float()-1).sigmoid()
    reference = batch['rule_probabilities'][..., :18]
    teacher = reference.argmax(-1)
    student = logits.argmax(-1)
    def take(values, index):
        return values.gather(-1, index[..., None]).squeeze(-1)
    first_valid = batch['risk_continuation_valid'][..., 0, :]
    normal = batch['unprotected_mask'] & (batch['rule_weights'] > 0)
    eligible = normal & take(first_valid, teacher) & (take(batch['margins'][..., 0, :], teacher) > 1)
    weights = getattr(model, 'movement_risk_horizon_weights', (8.,4.,2.))
    long_cost = model.neural_risk_scale*(weights[1]*risk[...,1]+weights[2]*risk[...,2])
    gap = take(logits, teacher)-take(logits, student)
    without_long = gap+take(long_cost, teacher)-take(long_cost, student)
    reversal = eligible & (gap < 0) & (without_long > 0)
    # The historical reversal is only a teacher-versus-chosen comparison.
    # A third candidate may still win. Report the full18-action ablation too
    # so that a pairwise rank change is not mistaken for causal action choice.
    no_long_student=(logits+long_cost).argmax(-1)
    global_reversal=eligible&(student!=teacher)&(no_long_student==teacher)
    long_helped_teacher=eligible&(student==teacher)&(no_long_student!=teacher)
    witness=batch.get('risk_route_witness_valid',
        (batch['risk_continuation_source']==3)&batch['risk_continuation_valid'])
    supported = (witness[..., 1:, :]
        & (batch['margins'][..., 1:, :] <= 0)).any(-2)
    probabilities = logits.softmax(-1)
    rows = eligible & supported.any(-1)
    # Long-target gradients also reach shared neural features. Keeping the 4f
    # labels unchanged does not itself guarantee preserved short-risk accuracy.
    measured = (batch['unprotected_mask'] & (batch['outcomes'] >= 0)
        & (batch['actions'] < 18) & (batch['native_action_frames'] > 0)
        & (batch['native_action_frames'] <= 4))
    actual_hit = measured & (batch['outcomes'] == 1)
    selected_risk = take(risk[...,0], batch['actions'].clamp(0,17))
    return dict(route_risk_guided_rows=int(eligible.sum()),
        route_risk_long_reversal_rows=int(reversal.sum()),
        route_risk_global_reversal_rows=int(global_reversal.sum()),
        route_risk_long_helped_teacher_rows=int(long_helped_teacher.sum()),
        route_risk_teacher_mode_rows=int((eligible&(student==teacher)).sum()),
        route_risk_teacher_mode_without_long_rows=int((eligible&(no_long_student==teacher)).sum()),
        route_risk_supported_reversal_rows=int((reversal & take(supported,teacher)).sum()),
        route_risk_supported_turn_rows=int(rows.sum()),
        route_risk_supported_turn_probability_sum=float(((probabilities*supported).sum(-1)*rows).sum()),
        route_risk_native_contact_rows=int(actual_hit.sum()),
        route_risk_native_contact_predicted_below_half=int((actual_hit & (selected_risk < .5)).sum()),
        route_risk_native_clear_rows=int((measured & ~actual_hit).sum()),
        route_risk_native_clear_predicted_above_half=int((measured & ~actual_hit & (selected_risk >= .5)).sum()))
