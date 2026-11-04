"""Recheck V10 route preferences using the same-frame full native geometry.

Enabled by compact_full_native_teacher_support. With native forecasts this
also rechecks previously nonzero labels. Unknown immediate geometry abstains
from distribution supervision; protected states retain the resource teacher.
Neither this support filter nor V10 is a complete native ECL predictor.
"""
from __future__ import annotations

import math
import numpy as np

REVISION = 'compact-full-native-partial-danger-and-constant-route-v3'


def refine_archived_preferences(data, *, support_mix=.1, margin=1.):
    """Apply the new preference rules to stored current-native supervision.

    Executed actions, outcomes, features and forecast trust remain unchanged.
    This cannot raise confidence or turn a forecast into a native outcome.
    """
    targets=np.asarray(data['targets']).copy()
    weights=np.asarray(data['teacher_guidance_weight']).copy()
    partial=np.asarray(data.get('partial_point_danger',np.zeros_like(data['margins'],bool)))
    counts=dict(partial_danger_preferences=0,constant12_preferences=0)
    for i in np.flatnonzero(data['unprotected_mask']):
        p=targets[i].astype(float);move_mass=p[:18].sum()
        if move_mass<=0:continue
        known=data['trusted'][i];m=data['margins'][i]
        if not known[0]:
            bad=partial[i,0];remaining=~bad
            if not (bad.any() and remaining.any() and p[:18][bad].sum()>0):continue
            preferred=np.where(remaining,p[:18],0.)
            if preferred.sum()==0:
                preferred=(remaining&(m[0]==np.max(m[0,remaining]))).astype(float)
            support=remaining;weights[i]=0
            counts['partial_danger_preferences']+=1
        else:
            safe=m[0]>margin;constant12=safe&(m[1]>margin)
            turning=bool(known[2]) and bool((safe&(data['features'][i,75:93]>0)).any())
            if not (weights[i]>0 and known[1] and constant12.any() and not turning):continue
            support=safe;preferred=np.where(constant12,p[:18],0.)
            if preferred.sum()==0:preferred=constant12.astype(float)
            counts['constant12_preferences']+=1
        preferred/=preferred.sum()
        targets[i,:18]=move_mass*((1-support_mix)*preferred+support_mix*support/support.sum())
    return targets,weights,dict(revision=REVISION,**counts,executed_actions_changed=False,
        factual_outcomes_changed=False,forecast_trust_changed=False,confidence_increased=False)


def correct_avoidable_bomb_reference(label,encoding):
    """Remove a teacher Bomb proposal contradicted by its reliable move route.

    The older resource planner requests Bomb if all *constant* paths fail,
    while its later movement filter may find a positive turning continuation.
    Preserve native deathbomb guidance and uncertain cases. This modifies a
    training reference, never a student's legal actions or executed action.
    """
    result=dict(label)
    detail=dict(revision='native-positive-route-resource-reference-v1',applied=False,
        student_action_mask_changed=False,deathbomb_reference_preserved=True)
    current=encoding['details']
    if not current.get('normal'):
        return result,dict(detail,reason='protected_or_deathbomb_reference_retained')
    if not (current.get('full_observed_hazards') and np.asarray(current['trusted'],bool).all()
            and float(label.get('guidance_weight',1.))>0):
        return result,dict(detail,reason='route_not_reliably_covered')
    x=np.asarray(encoding['features'])
    route=(np.asarray(current['margins'])[0]>1)&(x[75:93]>0)
    p=np.asarray(label['probabilities'],float)
    if not route.any() or p[18]<=0:
        return result,dict(detail,reason='no_positive_turning_route_or_no_Bomb_proposal')
    move=p[:18].copy()
    if move.sum()==0:move=route.astype(float)
    revised=np.r_[move/move.sum(),0.]
    detail.update(applied=True,reason='positive_4_plus_12_frame_route_replaces_constant_path_Bomb_proposal',
        previous_Bomb_probability=float(p[18]),positive_route_actions=np.flatnonzero(route).tolist(),
        future_survival_guaranteed=False)
    result.update(probabilities=revised.tolist(),action=int(revised.argmax()),resource_reference_correction=detail)
    return result,detail


def restore_full_native_support(label, encoding, *, support_mix=.1, margin=1.):
    """Return an independent source19 label and auditable fallback details.

    Positive immediate support is not a claim that a longer safe route exists.
    When known, the encoder's existing4+12-frame positive continuation search
    can concentrate the preference component on a route it actually found.
    Its missing routes are never negative labels:every immediate-safe action
    retains the V10 support floor.  No executed action is changed here.
    """
    p = np.asarray(label['probabilities'], dtype=np.float64)
    if (p.shape != (19,) or not np.isfinite(p).all() or np.any(p < 0)
            or abs(float(p.sum())-1.) > 2e-6):
        raise ValueError('Normalized finite source19 teacher probabilities required')
    if (not math.isfinite(support_mix) or not 0 < support_mix <= 1
            or not math.isfinite(margin) or margin < 0):
        raise ValueError('A convex support mixture and nonnegative clearance margin required')
    result = dict(label)
    detail = dict(revision=REVISION, applied=False,
        preserves_Bomb_mass=True, changes_policy_inputs=False,
        teacher_executes_action=False, native_counterfactuals_evaluated=False)
    native_repair=encoding.get('source_native_forecasts') is not None
    if float(label.get('guidance_weight', 1.)) != 0 and not native_repair:
        return result, dict(detail, reason='existing_nonzero_guidance_retained')
    if (label.get('frame'), label.get('stage')) != (encoding.get('frame'), encoding.get('stage')):
        raise ValueError('Full-native support requires the same current frame and stage')
    current = encoding['details']
    source=np.asarray(encoding.get('source_features25',()),dtype=float)
    if source.shape==(25,) and source[18]<0:
        # Native Shift is ignored in this state. Retain the teacher's fast
        # task preferences and give each identical-command pair equal mass;
        # never prefer a falsely predicted slower displacement.
        move_mass=float(p[:18].sum());fast=p[:9]
        fast=fast/fast.sum() if fast.sum()>0 else np.full(9,1/9)
        p=p.copy();p[:18]=np.tile(fast*move_mass*.5,2)
        detail['native_focus_unavailable']=True
    if not current.get('normal'):
        return result, dict(detail, reason='protected_or_non_normal_reference_retained')
    if not current.get('full_observed_hazards'):
        return result, dict(detail, reason='partial_geometry_cannot_restore_guidance')
    trusted = np.asarray(current['trusted'], dtype=bool)
    clearances = np.asarray(current['margins'], dtype=float)
    if trusted.shape != (3,) or clearances.shape != (3, 18) or not np.isfinite(clearances).all():
        raise ValueError('Finite current3x18 compact margins and three trust flags required')
    if not trusted[0]:
        if native_repair:
            result.update(guidance_weight=0.,guidance_weight_reason='native_motion_or_contact_program_unresolved')
            partial=np.asarray(current.get('partial_point_danger',np.zeros((3,18),bool)),dtype=bool)
            # A separate unknown object must not make the rule teacher ignore
            # a known point collision. Remaining moves are still uncertain:
            # retain zero distribution-guidance weight and the original Bomb
            # mass. Only its preference among movement candidates changes.
            bad=partial[0] if partial.shape==(3,18) else np.zeros(18,bool)
            remaining=~bad;move_mass=float(p[:18].sum())
            if bad.any() and remaining.any() and move_mass>0 and p[:18][bad].sum()>0:
                preferred=np.where(remaining,p[:18],0.)
                if preferred.sum()==0:
                    best=np.max(clearances[0,remaining])
                    preferred=(remaining&(clearances[0]==best)).astype(float)
                preferred/=preferred.sum()
                revised=p.copy()
                revised[:18]=move_mass*((1.-support_mix)*preferred+
                    support_mix*remaining.astype(float)/int(remaining.sum()))
                detail.update(applied=True,reason='known_point_danger_removed_from_uncertain_teacher_movement',
                    known_danger_actions=np.flatnonzero(bad).tolist(),
                    remaining_actions_not_certified_safe=True,guidance_weight_restored=False,
                    preference_only=True)
                result.update(probabilities=revised.tolist(),action=int(revised.argmax()),
                    full_native_support=detail)
                return result,detail
        return result, dict(detail, reason='immediate_geometry_still_unknown')
    safe = clearances[0] > margin
    if not safe.any():
        if native_repair:
            result.update(guidance_weight=0.,guidance_weight_reason='no_buffered_native_first_step')
        return result, dict(detail, reason='no_buffered_immediate_support')

    uniform = safe.astype(float) / int(safe.sum())
    preference_support = safe
    route_support = np.zeros(18, dtype=bool)
    x = np.asarray(encoding.get('features', ()), dtype=float)
    # The continuation interval ends at16 frames;use the enclosing24-frame
    # known flag rather than certifying it from a shorter12-frame interval.
    if trusted[2] and x.ndim == 1 and len(x) >= 93:
        route_support = safe & (x[75:93] > 0)
        if route_support.any():
            preference_support = route_support
    # The sampled16-frame continuation search is incomplete. If it finds
    # nothing, a verified12-frame constant route is stronger evidence than
    # immediate4-frame clearance alone. Keep the immediate-safe support
    # floor: absence of a sampled continuation is not proof of collision.
    constant12=safe&(clearances[1]>margin) if trusted[1] else np.zeros(18,bool)
    if not route_support.any() and constant12.any():
        preference_support=constant12
    preferred = np.where(preference_support, p[:18], 0.)
    if preferred.sum() > 0:
        preferred = preferred / preferred.sum()
    else:
        preferred = preference_support.astype(float) / int(preference_support.sum())
    revised = p.copy()
    revised[:18] = float(p[:18].sum()) * ((1.-support_mix)*preferred + support_mix*uniform)
    detail.update(applied=True, reason='full_native_current_frame_soft_support',
        safe_actions=np.flatnonzero(safe).tolist(), safe_first_actions=int(safe.sum()),
        support_mix=float(support_mix), clearance_margin=float(margin),
        original_guidance_weight=float(label.get('guidance_weight',1.)),
        original_guidance_weight_reason=label.get('guidance_weight_reason'),
        source_object_rows=current.get('entities'),
        positively_supported_route_actions=np.flatnonzero(route_support).tolist(),
        positive_constant12_actions=np.flatnonzero(constant12).tolist(),
        preference_horizon_frames=16 if route_support.any() else 12 if constant12.any() else 4,
        continuation_search='existing4+12-frame proxy;three nearby second actions;positive evidence only',
        missing_continuation_is_not_a_negative_label=True,
        does_not_certify_long_term_escape=True)
    result.update(probabilities=revised.tolist(), action=int(revised.argmax()),
        guidance_weight=1., guidance_weight_reason=detail['reason'],
        full_native_support=detail)
    return result, detail
