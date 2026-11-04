"""Optional native-outcome correction for the existing four-frame risk head.

Geometry describes conditional motion/flags; a measured student contact can
contradict it. This helper builds targets only for the movement actually
executed by a vulnerable student. Unknown geometry does not erase a measured
outcome. Unsampled actions never receive invented native labels.

An explicit training option reuses the existing risk BCE with these factual
targets. Its live value is recorded in the job launch. There is no additional
risk head, ranking objective, policy override, neuron, edge, or native process.
Development/test trajectories must never be supplied to training.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

REVISION = 'compact-selected-native-four-frame-contact-targets-v1'
EXPOSURE_REVISION = 'native-four-frame-protection-expiry-eligibility-v1'
RECOVERY_REVISION = 'recorded-native-expiry-outcome-recovery-v1'


def recover_recorded_native_outcomes(data):
    """Recover filtered TRAINING targets from the collector's paired facts.

    This never changes the archive, selected actions, rewards or PPO targets.
    No neighboring observation, forecast or unexecuted direction supplies a
    label: the collector already recorded this exact transition's contact.
    """
    outcomes=data['outcomes']
    facts=data.get('recorded_observed_contacts')
    report=dict(revision=RECOVERY_REVISION,recovered_rows=0,recovered_contacts=0,recovered_clears=0)
    if facts is None:return outcomes,report
    actions=data['actions'];dt=data['native_action_frames'];f=data['features']
    valid=data.get('valid_mask',torch.ones_like(actions,dtype=torch.bool))
    exposed=native_exposure_mask(f,data['unprotected_mask'])
    complete=((facts&(dt>0)&(dt<=4))|(~facts&(dt==4)))
    recover=valid&exposed&(f[...,14]>.5)&(actions>=0)&(actions<18)&(outcomes<0)&complete
    report.update(recovered_rows=int(recover.sum()),recovered_contacts=int((recover&facts).sum()),
                  recovered_clears=int((recover&~facts).sum()))
    return torch.where(recover,facts.to(outcomes.dtype),outcomes),report


def native_exposure_mask(features, unprotected):
    """Measured exposure during the action, including a shield ending in1..4f.

    Works on NumPy collection rows and Torch batches. Does not promote state4,
    a fully shielded move, or an unobserved outcome into a native risk label.
    """
    timer = features[...,13]
    remaining = 60 * timer / (1 - timer + 1e-12)
    return unprotected | ((features[...,14] > .5) & (timer > 0) & (remaining <= 4.0001))


def native_contact_weights(actions, outcomes, unprotected, elapsed_frames):
    """Balance measured positive/negative rows over one complete minibatch.

    Compute once before microbatch splitting, then slice these same weights.
    ``outcomes`` is the collector's native contact fact, not a geometric or
    teacher prediction. Bomb, protected, deathbomb and unknown rows are unused.
    A contact within1..4 native frames is positive; a clear needs all four.
    """
    if (actions.shape != outcomes.shape or actions.shape != unprotected.shape
            or actions.shape != elapsed_frames.shape
            or unprotected.dtype != torch.bool):
        raise ValueError('Paired executed actions, native outcomes and vulnerability required')
    observed = (((outcomes == 1) & (elapsed_frames > 0) & (elapsed_frames <= 4))
                | ((outcomes == 0) & (elapsed_frames == 4)))
    valid = unprotected & (actions >= 0) & (actions < 18) & observed
    positive = valid & (outcomes == 1)
    negative = valid & (outcomes == 0)
    counts = torch.stack((negative.sum(), positive.sum()))
    classes = (counts > 0).sum().clamp_min(1)
    weights = (negative.float() / counts[0].clamp_min(1)
               + positive.float() / counts[1].clamp_min(1)) / classes
    return weights.detach(), counts.detach()


def native_contact_bce(horizon_scores, actions, outcomes, unprotected, elapsed_frames, *, weights=None):
    """Calibrate only the selected action's existing4f score from native facts.

    The score-to-logit convention is the production risk head's ``8*s-1``.
    Actual counts remain diagnostic facts, not inferred collision identities.
    No margin regression target is fabricated from a binary native outcome.
    """
    if horizon_scores.shape != (*actions.shape, 18, 3):
        raise ValueError('Paired source18 by three-horizon risk scores required')
    valid_weights, counts = native_contact_weights(actions, outcomes, unprotected, elapsed_frames)
    if weights is None:
        weights = valid_weights
    elif weights.shape != actions.shape:
        raise ValueError('Slice the complete-minibatch native weights with the same rows')
    observed = (((outcomes == 1) & (elapsed_frames > 0) & (elapsed_frames <= 4))
                | ((outcomes == 0) & (elapsed_frames == 4)))
    valid = unprotected & (actions >= 0) & (actions < 18) & observed
    selected = horizon_scores[..., 0].gather(-1, actions.clamp(0, 17)[..., None])[..., 0]
    logits = torch.where(valid, 8 * selected - 1, 0.)
    target = (outcomes == 1).to(logits.dtype)
    weight = weights.to(device=logits.device, dtype=logits.dtype) * valid.to(logits.dtype)
    loss = (weight * F.binary_cross_entropy_with_logits(logits, target, reduction='none')).sum()
    return loss, dict(measured_clear_rows=counts[0], measured_contact_rows=counts[1],
                      native_target_directions=valid.sum(), unexecuted_native_targets=0)
