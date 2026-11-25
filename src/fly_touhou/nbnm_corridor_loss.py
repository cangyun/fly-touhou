"""Distil counterfactual corridor quality from DN and supervise the student's own policy."""
import math

import torch
from torch.nn import functional as F

from .nbnm_corridor import WIDTH, FEATURE_WIDTH


def risk_readout(head, candidates):
    # Pure DN readout. Normalization is per candidate, independent of ready-batch size.
    x=F.layer_norm(candidates,(candidates.shape[-1],))
    return head(x)


def compose(logits, risk, features, vulnerable, weight):
    if features.shape[-1]!=FEATURE_WIDTH:
        return logits
    cue=features[...,281:407].reshape(*features.shape[:-1],18,WIDTH)
    observed=cue[...,6].amax(-1)>0
    # The confidence bit gates the whole scene. Neither per-action corridor
    # scores nor the teacher choose an action outside the connectome.
    cost=risk.detach().sigmoid()
    cost=(.65*cost[...,0]+.35*cost[...,1])*(observed & vulnerable)[...,None]
    return torch.cat((logits[...,:18]-weight*(cost-cost.mean(-1,keepdim=True)),logits[...,18:]),-1)


def compose_witness(logits, witness, features, vulnerable, weight):
    if features.shape[-1]!=FEATURE_WIDTH:
        return logits
    present=features[...,281:407].reshape(*features.shape[:-1],18,WIDTH)[...,6].amax(-1)>0
    p=witness.sigmoid()
    # Relative support across first moves, with a bounded effect. This is an
    # independent DN estimate of a found exit, separate from predicted danger.
    utility=.4*p[...,0]+.6*p[...,1]
    delta=(utility-utility.mean(-1,keepdim=True))*(present & vulnerable)[...,None]
    return torch.cat((logits[...,:18]+weight*delta,logits[...,18:]),-1)


def prioritize_short_risk(logits, route_and_risk, short_risk, vulnerable, settings):
    """Apply a shared reduction to learned corridor scores in a strongly conflicting scene."""
    movements = logits[..., :18]
    first = movements.argmax(dim=-1, keepdim=True)
    selected_risk = short_risk.gather(-1, first).squeeze(-1)
    conflict = (vulnerable & (selected_risk >= settings["trigger_risk"])
                & (short_risk.amin(dim=-1) <= settings["alternative_risk"]))
    weight = settings["remaining_corridor_weight"]
    combined = weight * movements + (1. - weight) * route_and_risk
    return torch.cat((torch.where(conflict[..., None], combined, movements), logits[..., 18:]), dim=-1)


def endpoint_adjustment(prediction,features,settings):
    """Centered DN-only correction at selected stages during observed spells."""
    active=torch.zeros_like(features[...,0],dtype=torch.bool)
    stage=(features[...,111]*6).round()
    for value in settings['stages']:
        active=active|(stage==value)
    active=active&(features[...,274]>.5)
    if settings.get('paired_only',False):
        # This readout learned a relative endpoint preference only on scenes that
        # contained both types. The switch is common to every movement logit.
        cue=features[...,281:407].reshape(*features.shape[:-1],18,WIDTH)
        status=cue[...,3]
        credible=cue[...,6]>0
        limited=((status>0)&(status<=.25)&credible).any(-1)
        open_route=((status>=.75)&credible).any(-1)
        active=active & limited & open_route
    signed=prediction.squeeze(-1).tanh()
    return settings['weight']*(signed-signed.mean(-1,keepdim=True))*active[...,None]


def _matching_move(features,actions):
    chosen=F.one_hot(actions.clamp(0,18).long(),19)[...,:18].bool()
    idle=torch.arange(18,device=features.device)%9==0
    same_velocity=(features[...,11]<0)[...,None] | idle
    return chosen | (torch.cat((chosen[...,9:],chosen[...,:9]),-1)&same_velocity)


def lure_targets(batch):
    supported = batch['nbnm_lure_moves'] & (batch['margins'][..., 0, :] >= 6)
    immediate = _matching_move(batch['features'], batch['actions']) & (batch['outcomes'] == 1)[..., None]
    exact = F.one_hot(batch['actions'].clamp(0, 18).long(), 19)[..., :18].bool()
    followed_contact = exact & batch['nbnm_lure_course_hits'].any(-1)[..., None]
    return supported & ~immediate & ~followed_contact & batch['unprotected_mask'][..., None]


def exit_supervision(batch, road_stages=None, *, allow_48=False):
    """Teach the first-step safety and the reachable exits supported by this scene."""
    f = batch['features']
    active = batch['unprotected_mask'] & (f[..., 72] > .5)
    if road_stages is not None:
        stage = (f[..., 111] * 6).round()
        active_stage = torch.zeros_like(active)
        for number in road_stages:
            active_stage |= stage == number
        active &= active_stage & (f[..., 273] <= .5)
    if 'valid_mask' in batch:
        active &= batch['valid_mask']

    cue = f[..., 281:].reshape(*f.shape[:-1], 18, WIDTH)
    first_risk = f[..., 18:36]
    actions = batch['actions']
    observed_hit = _matching_move(f, actions) & (batch['outcomes'] == 1)[..., None]
    blocked = (first_risk >= math.tanh(1. / 8.)) | (cue[..., 2] == -1.) | observed_hit
    allowed = ~blocked
    clear = allowed & (first_risk < 0.)
    teach_short = active & blocked.any(-1) & clear.any(-1)

    # Low-confidence geometry gives a graded preference. Native collisions
    # supersede a forecast for the executed first step or the matched suffix.
    actual_move = F.one_hot(actions.clamp(0, 18).long(), 19)[..., :18].bool()
    contradicted = actual_move & batch['corridor_course_hits'].any(-1)[..., None]
    coverage = cue[..., 6]
    found48 = (clear & (coverage > 0) & (cue[..., 2] >= .75)
               & (cue[..., 0] >= 1. / 9.) & ~contradicted)
    found = found48 & (cue[..., 3] >= .75) & (cue[..., 1] >= 1. / 9.)
    recovered = batch.get('corridor_recovered_exits')
    if recovered is not None:
        native_exit = recovered[..., 1] & allowed
        found |= native_exit
        coverage = torch.where(native_exit, 1., coverage)
    if allow_48:
        # Use the shorter witness only while no 96-frame exit is supported.
        # Keep its preference weak since the unobserved continuation can close.
        use_shorter = ~found.any(-1, keepdim=True)
        found = torch.where(use_shorter, found48, found)
        coverage = torch.where(use_shorter, .5 * coverage, coverage)

    x, y = f[..., 0], f[..., 1]
    dx, dy = torch.minimum(x, 1. - x), torch.minimum(y, 1. - y)
    near_wall = (dx <= .08) | (dy <= .08)
    if road_stages is not None:
        near_wall = (dx * 368 <= 32) | near_wall
    scarce = found.sum(-1) <= 4
    strongest = (coverage * found).amax(-1)
    preferred = found & (coverage >= strongest[..., None])
    alternatives = (allowed & ~preferred).any(-1)
    rows = active & (near_wall | scarce) & found.any(-1) & alternatives
    return allowed, teach_short, preferred, strongest * rows


def recovered_exits(features,actions,native_survived):
    """A successful original continuation witnesses an exit from only the executed first move."""
    cues=features[...,281:407].reshape(*features.shape[:-1],18,WIDTH)
    chosen=F.one_hot(actions.clamp(0,18).long(),19)[...,:18].bool()
    return chosen[...,None] & native_survived.unsqueeze(-2) & (cues[...,2:4]<0) & (cues[...,6:7]>0)


def witness_targets(features,actions=None,outcomes=None,course_hits=None,*,recovered=None):
    x=features[...,281:407].reshape(*features.shape[:-1],18,WIDTH)
    status=x[...,2:4]
    found=status>0
    # When learning physical reachability, an exhausted beam (-.5) is
    # censored. Only a found path or a blocked first move supplies a label.
    evidence=(found | (status==-1)) if recovered is not None else (status!=0)
    searched=evidence & (x[...,6:7]>0)
    if actions is not None:
        # Immediate contact contradicts both encodings of an identical
        # native move. Later contact only contradicts the followed suffix.
        contradicted=_matching_move(features,actions).unsqueeze(-1)&(outcomes==1)[...,None,None]
        if course_hits is not None:
            exact=F.one_hot(actions.clamp(0,18).long(),19)[...,:18].bool()
            contradicted=contradicted | (exact[...,None]&course_hits.unsqueeze(-2))
        found=found & ~contradicted
        searched=searched | contradicted
    confidence=x[...,6:7].clamp_min(.25)
    if recovered is not None:
        found=found | recovered
        searched=searched | recovered
        confidence=torch.where(recovered,1.,confidence)
    positive=found.sum(-2,keepdim=True).clamp_min(1)
    negative=(searched & ~found).sum(-2,keepdim=True).clamp_min(1)
    # Equal mass to recovered/unrecovered directions even for a single exit.
    weights=torch.where(found,positive.reciprocal(),negative.reciprocal())*searched*confidence
    return found.to(x.dtype),weights


def course_targets(features,actions=None,outcomes=None,course_hits=None,*,allow_replans=False,recovered=None):
    """Prefer a still-safe previous suffix when its two margins compete with new exits."""
    if features.shape[-1]!=FEATURE_WIDTH:
        blank=torch.zeros((*features.shape[:-1],18),dtype=torch.bool,device=features.device)
        return blank,blank.any(-1)
    x=features[...,281:407].reshape(*features.shape[:-1],18,WIDTH)
    available=(x[...,3]>0)&(x[...,6]>0)
    alias=torch.arange(18,device=x.device)%9==0
    physical_alias=(features[...,11]<0)[...,None] | alias
    if actions is not None:
        immediate=_matching_move(features,actions) & (outcomes==1)[...,None]
        available=available & ~immediate
        if course_hits is not None:
            # Identical first displacement can still have a different suffix.
            chosen=F.one_hot(actions.clamp(0,18).long(),19)[...,:18].bool()
            available=available & ~(chosen & course_hits[...,1:2])
    retained=(x[...,3]==.75)&available
    margins=x[...,:2]
    best=(margins*available[...,None]).amax(-2,keepdim=True)
    competitive=(margins+.035>=.80*best).all(-1)
    retain=retained & competitive & (x[...,0]>.11) & (x[...,1]>.11)
    # A continuation can start with either encoding of a physically identical
    # direction. A recorded collision invalidates both of those encodings.
    same_focus=torch.cat((retain[...,9:],retain[...,:9]),-1)
    retain=retain | (same_focus & physical_alias & available & competitive)
    rows=retain.any(-1) & (available.sum(-1)>1)
    if not allow_replans:
        return retain,rows
    # A new witness with at least the same margins is a valid student
    # decision. In particular, a flat open field has no preferred course.
    reference=(margins*retain[...,None]).amax(-2,keepdim=True)
    no_worse=(margins+1e-6>=reference).all(-1)
    # Match the 96-frame risk ordering once both 48-frame approaches clear
    # eight native pixels. The extra pixel must be on the same witnessed path.
    gap=margins[...,1]
    old_gap=reference[...,1]
    native_gain=8*(gap-old_gap)/((1-gap)*(1-old_gap)).clamp_min(1e-4)
    wider=(margins[...,0]>=.5)&(reference[...,0]>=.5)&(native_gain>=1.)
    fresh=available & (x[...,2]>0) & (x[...,3]>=.75) & (no_worse | wider)
    acceptable=retain | (rows[...,None] & fresh)
    if recovered is not None:
        acceptable=acceptable | (rows[...,None] & recovered[...,1])
    return acceptable,rows & ~acceptable.all(-1)


def continuation_preference(batch):
    """Trust a checked suffix, and allow a meaningfully wider executable replacement."""
    features = batch['features']
    cue = features[..., 281:407].reshape(*features.shape[:-1], 18, WIDTH)
    clearance = 8 * cue[..., :2] / (1 - cue[..., :2]).clamp_min(1e-6)
    action = batch['actions']
    chosen = F.one_hot(action.clamp(0, 18).long(), 19)[..., :18].bool()
    course_hit = chosen & batch['corridor_course_hits'].any(-1)[..., None]
    short_margin = batch['margins'][..., 0, :]
    trusted_tail = cue[..., 6] >= .5
    # With an incomplete long forecast, the planner still rechecks the
    # first 24 frames with observed geometry. Give that prefix a weak
    # preference when a short alternative is already known to be blocked.
    trusted_prefix = batch['trusted'].all(-1) & (short_margin <= 1.).any(-1)
    usable = (short_margin > 1.) & (features[..., 18:36] < 0.) & ~course_hit
    usable &= trusted_tail | ((cue[..., 6] > 0.) & trusted_prefix[..., None])

    supported48 = usable & (cue[..., 2] >= .75) & (clearance[..., 0] > 1.)
    supported96 = supported48 & (cue[..., 3] >= .75) & (clearance[..., 1] > 1.)
    use96 = supported96.any(-1, keepdim=True)
    supported = torch.where(use96, supported96, supported48)
    retained = supported & (torch.where(use96, cue[..., 3], cue[..., 2]) == .75)

    previous = clearance.masked_fill(~retained[..., None], 0.).amax(-2, keepdim=True)
    gain = clearance - previous
    wider = torch.where(use96, (gain >= -1.).all(-1) & (gain.amax(-1) >= 2.),
                        gain[..., 0] >= 2.)
    preferred = retained | (supported & wider & trusted_tail & retained.any(-1, keepdim=True))

    # A verified pause is also the first command of a full corridor. The
    # environment already checked that there is no tactical reason to move.
    certified_wait = batch.get('nbnm_calm_moves', torch.zeros_like(supported96))
    wait = supported96 & trusted_tail & certified_wait & (clearance >= 8.).all(-1)
    can_pause = wait.any(-1)
    preferred = torch.where(can_pause[..., None], wait, preferred)
    active = batch['unprotected_mask'] & batch['trusted'][..., 0] & (batch['outcomes'] == 0)
    if 'valid_mask' in batch:
        active &= batch['valid_mask']
    if 'terminated' in batch:
        active &= ~batch['terminated']
    has_preference = preferred.any(-1) & ~preferred.all(-1)
    fully_supported = use96[..., 0] & (preferred & trusted_tail).any(-1)
    strength = torch.where(fully_supported, 1., .25) * (active & has_preference)
    if 'corridor_verified_exits' in batch:
        new, weight = timed_preference(batch, cue, clearance, active & ~can_pause, course_hit)
        preferred = torch.where((weight > 0)[..., None], new, preferred)
        strength = torch.where(weight > 0, weight, strength)
        if 'nbnm_lure_course_hits' in batch:
            lure = lure_targets(batch)
            agreement = lure & preferred
            use_lure = active & ~can_pause & lure.any(-1) & ((strength < 1.) | agreement.any(-1))
            shared = torch.where((strength == 1.)[..., None], agreement, lure)
            preferred = torch.where(use_lure[..., None], shared, preferred)
            strength = torch.where(use_lure, strength.clamp_min(.5), strength)
    return preferred, strength, can_pause & active & has_preference


def timed_preference(batch, cue, clearance, active, course_hit):
    """Accept a verified 48f exit and explicitly teach a checked turn deadline."""
    verified = batch['corridor_verified_exits']
    room = batch['nbnm_corridor_room']
    native = batch.get('corridor_native_exit_margin')
    if native is not None:
        # For the executed first move, use the measurements of the actual
        # continuation as a unit. Do not attach them to a different planned turn.
        chosen = F.one_hot(batch['actions'].clamp(0, 18).long(), 19)[..., :18].bool()
        better = (native[..., 0] > 0.)[..., None] & (~verified[..., 0]
                 | ((native[..., 1] > 0.)[..., None] & ~verified[..., 1]))
        actual = chosen & better
        verified = torch.where(actual[..., None], (native > 0.)[..., None, :], verified)
        clearance = torch.where(actual[..., None], native[..., None, :], clearance)
        room = torch.where(actual, batch['corridor_native_exit_room'][..., None], room)
    short = batch['margins'][..., 0, :]
    viable = (short > 1.) & (batch['features'][..., 18:36] < 0.) & ~course_hit
    from .nbnm_corridor_evidence import PREFERRED_CLEARANCE
    safe48 = verified[..., 0] & viable & (clearance[..., 0] >= PREFERRED_CLEARANCE)
    safe96 = verified[..., 1] & safe48 & (clearance[..., 1] >= PREFERRED_CLEARANCE)
    best48 = clearance[..., 0].masked_fill(~safe48, 0.).amax(-1, keepdim=True)
    eligible = safe48 & (clearance[..., 0] >= best48 - 2.)
    # At comparable 48f margins, an independently checked 96f exit has
    # priority. A longer uncertain prediction cannot displace a checked prefix.
    farther = eligible & safe96
    eligible &= farther | ~farther.any(-1, keepdim=True)
    has_room = (eligible & (room >= 16.)).any(-1, keepdim=True)
    eligible &= (room >= 8.) | ~has_room

    retained = eligible & (cue[..., 2] == .75)
    target = torch.where(retained.any(-1, keepdim=True), retained, eligible)
    heading = torch.arange(18, device=cue.device) % 9
    same = heading == batch['nbnm_previous_heading'][..., None]
    turn = batch['nbnm_due_turns'] & viable & ~same
    urgent = turn.any(-1) & ~(safe48 & same).any(-1)
    can_turn = eligible & ~same
    turn_target = torch.where(can_turn.any(-1, keepdim=True), can_turn, turn)
    target = torch.where(urgent[..., None], turn_target, target)

    # On an open field, an observed repeat of the student's last action is
    # not a motive to move. Preserve the quiet-scene supervision separately.
    exits = (viable & (cue[..., 2] >= .75)).sum(-1)
    pressure = (batch['margins'][..., 2, :] <= 1.).any(-1) | ((exits > 0) & (exits <= 4))
    applied = active & (pressure | urgent) & target.any(-1) & ~target.all(-1)
    strength = torch.where((target & safe48).any(-1), 1., .5) * applied
    return target, strength


def continuation_reference(probabilities, preferred, strength):
    """Preserve the teacher's relative preferences within the agreed action set."""
    allowed = probabilities[..., :18] * preferred
    mass = allowed.sum(-1, keepdim=True)
    fallback = preferred / preferred.sum(-1, keepdim=True).clamp_min(1)
    target = torch.where(mass > 1e-8, allowed / mass.clamp_min(1e-8), fallback)
    target = torch.cat((target, torch.zeros_like(probabilities[..., 18:])), -1)
    return torch.lerp(probabilities, target, strength[..., None])


def course_loss(logits,targets,rows,normalizer=None, *, row_weights=None):
    """Choose any equally viable continuation and allow a fresh plan after risk changes."""
    logits=torch.where(rows[...,None],logits,torch.zeros_like(logits))
    acceptable=targets | ~rows[...,None]
    delta=torch.logsumexp(logits,-1)-torch.logsumexp(logits.masked_fill(~acceptable,-1e8),-1)
    denom=rows.sum() if normalizer is None else normalizer
    return (delta*(rows if row_weights is None else row_weights)).sum()/denom.clamp_min(1)


def compose_course(logits,course,features,vulnerable,settings):
    if features.shape[-1]!=FEATURE_WIDTH:
        return logits
    present=features[...,281:407].reshape(*features.shape[:-1],18,WIDTH)[...,6].amax(-1)>0
    enabled=present & vulnerable
    if settings.get('revalidated_course_only'):
        # The DN may express a soft continuation preference only while a
        # previous path still competes from this actual state. When it
        # closes, the ordinary actor and risk heads choose a fresh route.
        _,competitive=course_targets(features,allow_replans=True)
        enabled=enabled & competitive
    stages=settings.get('keep_course_stages')
    if stages is not None:
        selected=torch.zeros_like(enabled)
        for stage in stages:selected=selected | ((features[...,111]*6).round()==stage)
        enabled=enabled & selected
    if 'keep_course_spell_numbers' in settings:
        number=(features[...,275]*32).round()
        selected=torch.zeros_like(enabled)
        for spell in settings['keep_course_spell_numbers']:
            selected=selected | (number==spell)
        enabled=enabled & (features[...,274]>.5) & selected
    preference=course.sigmoid()
    relative=(preference-preference.mean(-1,keepdim=True))*enabled[...,None]
    return torch.cat((logits[...,:18]+settings.get('keep_course_weight',2.)*relative,logits[...,18:]),-1)


def witnessed_contacts(rollout, *, course_key='recorded_corridor_course', return_clear=False):
    """Native outcome of this specific path, only up to its first deviation."""
    actions=rollout['actions']
    course=rollout[course_key]
    valid=(rollout.get('valid_mask',torch.ones_like(actions,dtype=torch.bool))
        & rollout['unprotected_mask'] & (actions>=0) & (actions<18) & (rollout['outcomes']>=0))
    frames=rollout['recorded_episode_frames']
    episodes=rollout['recorded_native_episode']
    stage=rollout['recorded_native_stage']
    duration=rollout['native_action_frames']
    hits=torch.zeros((*actions.shape,2),dtype=torch.bool,device=actions.device)
    clear = torch.zeros_like(hits) if return_clear else None
    alive=valid.clone()
    for k in range(min(24,len(actions))):
        size=len(actions)-k
        good=(alive[:size] & valid[k:] & (course[:size,:,k]==actions[k:])
            & (frames[k:]==frames[:size]+4*k) & (episodes[k:]==episodes[:size])
            & (stage[k:]==stage[:size]) & (duration[k:]>0) & (duration[k:]<=4))
        collision=good & (rollout['outcomes'][k:]==1)
        if k<12:hits[:size,:,0]|=collision
        hits[:size,:,1]|=collision
        if return_clear and k in (11,23):
            clear[:size, :, int(k == 23)] = good & ~collision & (duration[k:] == 4)
        alive[:size]=good & ~rollout['terminated'][k:] & ~collision & (duration[k:]==4)
    return (hits, clear) if return_clear else hits


def native_exit_measurements(rollout):
    """Recognize played routes that clear the interval and leave a measured exit."""
    from .nbnm_policy_continuation_risk import observed_futures
    from .nbnm_corridor_evidence import PREFERRED_CLEARANCE

    known, hits = observed_futures(rollout, (48, 96), same_stage=True)
    actions = rollout['actions']
    valid = rollout.get('valid_mask', torch.ones_like(actions, dtype=torch.bool))
    picked = rollout['margins'][..., 0, :].gather(-1, actions.clamp(0, 17)[..., None])[..., 0]
    trusted = rollout['trusted']
    picked = picked.masked_fill(~valid | ~trusted[..., 0], 0.)
    frames = rollout['recorded_episode_frames']
    episodes = rollout['recorded_native_episode']
    stages = rollout['recorded_native_stage']
    f = rollout['features']
    x, y = 368 * f[..., 0] - 184, 400 * f[..., 1] + 32
    space = torch.maximum(184 - x.abs(), torch.minimum(y - 32, 432 - y))
    has_exit = trusted[..., 1] & ((rollout['margins'][..., 1, 1:9] >= PREFERRED_CLEARANCE).sum(-1) >= 2)

    measured = torch.zeros_like(known, dtype=picked.dtype)
    room = torch.zeros_like(picked)
    for index, steps in enumerate((12, 24)):
        size = len(actions) - steps
        if size <= 0:
            continue
        same = ((frames[steps:] == frames[:size] + 4 * steps) & (episodes[steps:] == episodes[:size])
                & (stages[steps:] == stages[:size]))
        bottleneck = picked.unfold(0, steps, 1)[:size].amin(-1)
        qualified = (known[:size, :, index] & ~hits[:size, :, index] & same & valid[steps:]
                     & has_exit[steps:] & (bottleneck >= PREFERRED_CLEARANCE))
        measured[:size, :, index] = torch.where(qualified, bottleneck, 0.)
        if index == 0:
            room[:size] = space[steps:]
        else:
            # Both horizons must refer to the same played trajectory.
            measured[:size, :, 1] *= measured[:size, :, 0] > 0.
    return measured, room


def targets(features,actions=None,outcomes=None,course_hits=None,*,recovered=None):
    x=features[...,281:407].reshape(*features.shape[:-1],18,WIDTH)
    evidence=x[...,2:4]
    good=evidence>0
    blocked=evidence==-1
    confidence=x[...,6:7]
    # Only real contact DURING this chosen four-frame command contradicts
    # every possible continuation from it. A later error leaves alternatives open.
    if actions is not None:
        chosen=F.one_hot(actions.clamp(0,18).long(),19)[...,:18].bool().unsqueeze(-1)
        observed=(outcomes==1).unsqueeze(-1)
        hit=_matching_move(features,actions).unsqueeze(-1) & observed.unsqueeze(-2)
        blocked=blocked|hit
        good=good & ~hit
        if course_hits is not None:
            # The student followed this particular proposed course until
            # native contact. Train its risk with that observed result.
            # A different turn after the same first move is a different
            # course. Its failure cannot be inferred from this record.
            failed_course=chosen & course_hits.unsqueeze(-2)
            good=good & ~failed_course
            blocked=blocked | failed_course
    # A found 96-frame path can end in a two-wall corner or lack an onward
    # exit while another such path stays open. The planner marks these *relative*
    # bottlenecks with .25 (new) or .1875 (retained). Keep the witnessed exit
    # positive in witness_targets; teach the risk and action heads its limited
    # continuation value. A necessary narrow exit has no relative penalty.
    limited=(evidence[...,1]>0)&(evidence[...,1]<=.25)&good[...,1]
    wider=(good[...,1]&(evidence[...,1]>=.75)).any(-1,keepdim=True)
    limited=limited & wider
    y=torch.where(good,1-x[...,:2],torch.ones_like(evidence))
    y=torch.stack((y[...,0],torch.where(limited,y[...,1].clamp_min(.75),y[...,1])),-1)
    weights=torch.where(blocked,torch.ones_like(evidence),good.to(x.dtype)*confidence)
    supported=(good[...,0]|good[...,1])
    # Soft local route preferences weight wide openings. Planner failures are
    # censored for risk. A witnessed alternative can still guide a turn.
    # A narrow passage is actionable. Its existence matters even when the
    # bottleneck is small; confidence weights fitting, not which exits exist.
    quality=(.65*good[...,0]*(.5+.5*x[...,0])+.35*good[...,1]*(.5+.5*x[...,1])
        +.08*((evidence==.75)&good).all(-1))*confidence[...,0]
    quality=torch.where(limited,.5*quality,quality)
    if recovered is not None:
        # An actually played 96f deviation supplies a viable first move, not
        # a measured clearance. Preserve other exits and absolute risk labels.
        # A 48f continuation alone cannot displace a known 96f alternative.
        weights=weights.masked_fill(recovered,0.)
        alternate=recovered[...,1] | (recovered[...,0] & ~good[...,1].any(-1,keepdim=True))
        boundary=.70*quality.amax(-1,keepdim=True)
        quality=torch.where(alternate,torch.maximum(quality,boundary),quality)
    best=quality.amax(-1)
    prefer=supported.any(-1)&((best-quality.amin(-1))>.10*best)
    return y,weights,quality,prefer


def _competitive_exits(quality):
    return (quality>0)&(quality>=.70*quality.amax(-1,keepdim=True))


def full_horizon_preference(features, values, recovered):
    """Choose among complete turning routes when available. Reweight scarce exits in the actor loss."""
    risk, certainty, quality, _ = values
    cues = features[..., 281:407].reshape(*features.shape[:-1], 18, WIDTH)
    geometric = (cues[..., 3] > 0) & (certainty[..., 1] > 0) & (risk[..., 1] < 1.)
    witnessed = geometric | recovered[..., 1]
    # A positively observed native deviation remains an alternative even when
    # the beam missed it. If no 96f route was found, retain 48f guidance.
    active = geometric.any(-1, keepdim=True)
    quality = torch.where(active & ~witnessed, 0., quality)
    best = quality.amax(-1)
    prefer = (best > 0) & ((best - quality.amin(-1)) > .10 * best)
    # Count physical first velocities. Fast/slow idle coincide; all nine
    # velocities coincide with their focused aliases when focus is unavailable.
    focus = features[..., 11:12] >= 0
    fast, slow = witnessed[..., :9], witnessed[..., 9:]
    shared = ~focus | (torch.arange(9, device=features.device) == 0)
    count = (fast | (slow & shared)).sum(-1) + (slow & ~shared).sum(-1)
    legal = 9 + 8 * focus.squeeze(-1).to(quality.dtype)
    # A minibatch has the same total coefficient, whether the only surviving
    # opening is one first move or most of the field remains open.
    mass = torch.where(active.squeeze(-1), legal / count.clamp_min(1), 1.) * prefer
    return (risk, certainty, quality, prefer), mass


def supported_exit_order(features,values):
    """Compare two witnessed continuations after both clear the near bottleneck."""
    risk,certainty,_,_=values
    cues=features[...,281:407].reshape(*features.shape[:-1],18,WIDTH)
    # The planner stores gap/(8+gap). At least eight native pixels at 48
    # frames prevents a wider 96-frame endpoint from buying a risky approach.
    eligible=((cues[...,0]>=.5)&(cues[...,2]>0)&(cues[...,3]>0)
        &(certainty>0).all(-1)&(risk[...,1]<1))
    actions=torch.arange(18,device=features.device)
    eligible=eligible & (actions!=9) & ((features[...,11:12]>=0)|(actions<9))
    long_risk=risk[...,1]
    best=long_risk.masked_fill(~eligible,2.).argmin(-1,keepdim=True)
    gap=cues[...,1]
    native_gap=8*gap/(1-gap).clamp_min(.01)
    difference=(long_risk-long_risk.gather(-1,best)).clamp_min(0.)
    compare=eligible & (native_gap.gather(-1,best)-native_gap>=1.) & (difference>0)
    confidence=cues[...,6]
    weights=compare*torch.minimum(confidence,confidence.gather(-1,best))
    weights=weights/compare.sum(-1,keepdim=True).clamp_min(1)
    return best,weights,difference


def supported_exit_order_loss(long_logits,labels,normalizer=None):
    best,weight,gap=labels
    predicted=long_logits.sigmoid()
    relative=predicted-predicted.gather(-1,best)
    total=weight.sum() if normalizer is None else normalizer
    return ((gap-relative).clamp_min(0.)*weight).sum()/total.clamp_min(1.)


def opening_loss(logits, values, normalizer=None, *, row_mass=None):
    """Learn the relative evidence for exits without calling unsearched moves collisions."""
    _,_,quality,prefer=values
    support=_competitive_exits(quality)
    support=support|~prefer[...,None]
    total=torch.logsumexp(logits,-1)-torch.logsumexp(logits.masked_fill(~support,-1e8),-1)
    mass=prefer if row_mass is None else row_mass
    n=mass.sum() if normalizer is None else normalizer
    return (total*mass).sum()/n.clamp_min(1)


def restrict_reference(probabilities,weights,values):
    """Within the current viable set the teacher may express a tactical preference."""
    risk,certainty,quality,prefer=values
    blocked=((risk==1.) & (certainty>=1.)).all(-1)
    # Retain tactical preferences in unexplored areas. A confirmed contact
    # on this exact proposed course cannot receive a positive reference here.
    filter_row=prefer | blocked.any(-1)
    viable=torch.where(prefer[...,None],_competitive_exits(quality),~blocked)
    p=probabilities[...,:18]*viable
    mass=p.sum(-1,keepdim=True)
    p=p/mass.clamp_min(1e-12)
    reference=torch.cat((p,torch.zeros_like(p[...,:1])),-1)
    return (torch.where(filter_row[...,None],reference,probabilities),
        torch.where(filter_row & (mass[...,0]<=1e-8),torch.zeros_like(weights),weights))


def loss(risk,logits,values,normalizer=None,*,policy_preference=None):
    y,w,quality,prefer=values
    denom,decisions=(w.sum(),prefer.sum()) if normalizer is None else normalizer
    score=(F.binary_cross_entropy_with_logits(risk,y,reduction='none')*w).sum()/denom.clamp_min(1)
    # Support-set likelihood does not force the teacher's favourite action.
    logit=logits[...,:18].float()
    normal=~torch.isfinite(logit).any(-1)
    logit=torch.where(normal[...,None],torch.zeros_like(logit),logit)
    if policy_preference is None:
        route=opening_loss(logit,values,decisions)
    else:
        preferred,mass,policy_norm=policy_preference
        route=opening_loss(logit,preferred,policy_norm,row_mass=mass)
    # The risk head also learns which first actions lead to verified geometric
    # witnesses when the other branches are censored. This is a relative
    # ordering loss, separate from predicting geometric margin or contact.
    predicted=risk.sigmoid()
    order=opening_loss(-6*(.65*predicted[...,0]+.35*predicted[...,1]),values,decisions)
    return score+.5*order,route

