"""CPU summaries from a committed rollout, with no model or native execution."""


def behavior_summary(data):
    import torch
    from .thrl_movement_teacher import MovementSafetyTeacher

    f = data['features']
    actions = data['actions']
    normal = data['unprotected_mask']
    logits = data['recorded_policy_logits']
    probability = data.get('recorded_behavior_policy_logits', logits).softmax(-1)
    family = torch.where(logits[..., 18] > logits[..., :18].logsumexp(-1),
                         18, logits[..., :18].argmax(-1))
    legal_stock = (f[..., 4] > 0) & torch.isfinite(logits[..., 18])
    trusted = data['trusted'][..., 0]
    all_safe = (data['margins'][..., 0, :] > 1).all(-1)
    groups = {}
    for name, mask in dict(normal_stocked=normal & legal_stock,
                           normal_stocked_all_safe=normal & legal_stock & trusted & all_safe,
                           protected_stocked=~normal & legal_stock,
                           deathbomb_stocked=data['deathbomb_mask'] & legal_stock).items():
        n = int(mask.sum())
        teacher = mask & (data['rule_weights'] > 0)
        groups[name] = dict(rows=n, requests=int((mask & (actions == 18)).sum()),
            expected_requests=float(probability[..., 18][mask].sum()),
            mean_probability=float(probability[..., 18][mask].mean()) if n else None,
            family_Bomb_states=int((mask & (family == 18)).sum()),
            teacher_rows=int(teacher.sum()),
            teacher_mean_probability=float(data['rule_probabilities'][..., 18][teacher].mean()) if teacher.any() else None)

    xy = f[..., :2] * f.new_tensor([368, 400]) + f.new_tensor([-184, 32])
    lo, hi = f.new_tensor([-184, 32]), f.new_tensor([184, 432])
    velocity = f.new_tensor(MovementSafetyTeacher().velocities)
    # The native focus gate can make a requested Shift move at fast speed.
    # Match the sensory encoder when summarizing wall clamping and Boss-axis
    # approach; this diagnostic does not alter the sampled actions.
    fast_velocity = velocity[torch.arange(18, device=f.device) % 9]
    velocity = torch.where((f[..., 11] < 0)[..., None, None], fast_velocity, velocity)
    endpoint = (xy[..., None, :] + 4 * velocity).maximum(lo).minimum(hi)
    stationary = (endpoint - xy[..., None, :]).abs().amax(-1) < 1e-4
    sampled_still = stationary.gather(-1, actions.clamp_max(17)[..., None])[..., 0] & (actions < 18)
    family_still = stationary.gather(-1, family.clamp_max(17)[..., None])[..., 0] & (family < 18)
    mass = (probability[..., :18] * stationary).sum(-1)
    conditional_mass = mass / probability[..., :18].sum(-1).clamp_min(1e-12)
    edge = (xy - lo).minimum(hi - xy)
    corner = edge.amax(-1)
    wall = {}
    for name, mask in dict(normal=normal, exact_corner=normal & (corner <= 1e-4),
            near_corner16=normal & (corner <= 16),
            near_edge16_except_corner=normal & (edge.amin(-1) <= 16) & (corner > 16),
            interior=normal & (edge.amin(-1) > 16)).items():
        n = int(mask.sum())
        teacher = mask & (data['rule_weights'] > 0)
        wall[name] = dict(rows=n, fraction_of_unprotected_rows=n / max(1, int(normal.sum())),
            stationary_sampled=int((mask & sampled_still).sum()),
            stationary_sampled_fraction=float(sampled_still[mask].float().mean()) if n else None,
            stationary_family_states=int((mask & family_still).sum()),
            stationary_family_fraction=float(family_still[mask].float().mean()) if n else None,
            outward_stationary_commands=int((mask & sampled_still & (actions != 0) & (actions != 9)).sum()),
            mean_conditional_stationary_probability=float(conditional_mass[mask].mean()) if n else None,
            teacher_rows=int(teacher.sum()),
            mean_teacher_stationary_probability=float((data['rule_probabilities'][..., :18] * stationary)[teacher].sum(-1).mean()) if teacher.any() else None)
    result = dict(Bomb_inventory_conditioned=groups, wall_and_stationary=wall,
        scope='Training states only. Bomb requests differ from native consumption. Stationary means wall-clamped displacement over four frames; focus, shooting and carried brain state may differ. Regions except the broad normal group and nested exact/near corner groups are disjoint.')
    from .compact_guided_wall_candidate import eligible_and_target
    eligible,recovery,target=eligible_and_target(f,data['margins'],data['trusted'],normal,
        data['rule_probabilities'],data['rule_weights'])
    count=int(eligible.sum())
    movement_probability=logits[...,:18].softmax(-1)
    result['teacher_supported_wall_recovery']=dict(rows=count,
        mean_teacher_recovery_probability=float(target[eligible].mean()) if count else None,
        mean_student_recovery_probability=float((movement_probability*recovery).sum(-1)[eligible].mean()) if count else None,
        scope='Training states with teacher-supported conditional recovery;not independent games or native counterfactual outcomes.')
    if f.shape[-1] == 129:
        dx, dy = f[..., 114] * 368, f[..., 115] * 400
        boss = (f[..., 113] > .5) & (f[..., 117] > .5) & (dy < 0)
        toward = dx.abs()[..., None] > (dx[..., None] - (endpoint[..., 0] - xy[..., 0, None])).abs()
        sampled_toward = toward.gather(-1, actions.clamp_max(17)[..., None])[..., 0] & (actions < 18)
        family_toward = toward.gather(-1, family.clamp_max(17)[..., None])[..., 0] & (family < 18)
        attack = {}
        for name, mask in dict(normal_boss_above=normal & boss,
                safe_misaligned_boss=normal & boss & (dx.abs() > 24) & trusted & all_safe).items():
            n = int(mask.sum())
            teacher = mask & (data['rule_weights'] > 0)
            attack[name] = dict(rows=n,
                mean_abs_target_dx=float(dx[mask].abs().mean()) if n else None,
                sampled_toward_fraction=float(sampled_toward[mask].float().mean()) if n else None,
                family_toward_fraction=float(family_toward[mask].float().mean()) if n else None,
                family_stationary_fraction=float(family_still[mask].float().mean()) if n else None,
                mean_student_toward_probability=float((probability[..., :18] * toward).sum(-1)[mask].mean()) if n else None,
                teacher_rows=int(teacher.sum()),
                mean_teacher_toward_probability=float((data['rule_probabilities'][..., :18] * toward).sum(-1)[teacher].mean()) if teacher.any() else None)
        attack['mean_native_power'] = float((f[..., 2] * 400).mean())
        attack['scope'] = 'Boss-axis approach over four wall-clamped frames, not measured shots or damage. The safe subgroup requires all18 immediate forecasts above1px and known geometry; longer routes can still be unsafe. Overlapping training states, not independent games.'
        result['attack_alignment'] = attack
        from .compact_selective_attack_guidance import eligible_and_target as selective_attack_targets
        eligible,approach,target=selective_attack_targets(f,data['margins'],data['trusted'],normal,
            data['rule_probabilities'],data['rule_weights'])
        count=int(eligible.sum())
        behavior=probability[...,:18]/probability[...,:18].sum(-1,keepdim=True).clamp_min(1e-12)
        result['teacher_supported_selective_attack']=dict(rows=count,
            mean_teacher_approach_probability=float(target[eligible].mean()) if count else None,
            mean_student_approach_probability_T1=float((movement_probability*approach).sum(-1)[eligible].mean()) if count else None,
            mean_behavior_approach_probability=float((behavior*approach).sum(-1)[eligible].mean()) if count else None,
            scope='Teacher-supported conditional short-horizon approach on training states;not independent games or native counterfactual outcomes.')
    if all(key in data for key in ('outcomes', 'episode_starts', 'terminated')):
        result['continued_policy_outcomes'] = continued_policy_outcomes(data)
    return result


def continued_policy_outcomes(data):
    """First observed event in subsequent decisions, keeping all censoring.

    In particular, do not exclude contacts because their following rows are
    protected. Never use post-action consistency veto to choose starting rows.
    """
    import torch
    actions, outcomes = data['actions'], data['outcomes']
    features = data['features']
    length, worlds = actions.shape
    starts = torch.arange(length, device=actions.device)[:, None].expand(-1, worlds)
    columns = torch.arange(worlds, device=actions.device)[None, :].expand(length, -1)
    selected = data['margins'].gather(-1, actions.clamp_max(17)[..., None, None].expand(length, worlds, 3, 1))[..., 0]
    result = {}
    names = {0:'no_contact_in_observed_window', 1:'contact', 2:'censored_Bomb',
             3:'censored_protected_or_unobserved', 4:'censored_episode_reset',
             5:'censored_terminal', 6:'censored_tape_end'}
    for horizon, (frames, decisions) in enumerate(((4, 1), (12, 3), (24, 6))):
        eligible = data['unprotected_mask'] & (actions < 18) & (features[..., 72 + horizon] > .5)
        alive = eligible.clone()
        code = torch.zeros_like(actions)
        changed = torch.zeros_like(eligible)

        def finish(mask, value):
            nonlocal alive
            mask = alive & mask
            code[mask] = value
            alive = alive & ~mask

        for offset in range(decisions):
            index = starts + offset
            finish(index >= length, 6)
            index = index.clamp_max(length - 1)
            if 'valid_mask' in data:finish(~data['valid_mask'][index,columns],6)
            if offset:
                finish(data['episode_starts'][index, columns], 4)
            following_action = actions[index, columns]
            following_outcome = outcomes[index, columns]
            finish(following_action == 18, 2)
            finish(following_outcome < 0, 3)
            changed |= alive & (following_action != actions)
            finish(following_outcome == 1, 1)
            # A terminal after the last observed decision does not censor that
            # completed window; earlier terminals do censor longer windows.
            if offset + 1 < decisions:
                finish(data['terminated'][index, columns], 5)
        rows = {}
        for name, mask in dict(forecast_danger=eligible & (selected[..., horizon] <= 0),
                forecast_boundary=eligible & (selected[..., horizon] > 0) & (selected[..., horizon] <= 1),
                forecast_safe=eligible & (selected[..., horizon] > 1)).items():
            counts = {label:int((mask & (code == value)).sum()) for value, label in names.items()}
            completed = mask & ((code == 0) | (code == 1))
            rows[name] = dict(starts=int(mask.sum()), **counts,
                changed_action_then_no_contact=int((mask & changed & (code == 0)).sum()),
                unchanged_action_then_no_contact=int((mask & ~changed & (code == 0)).sum()),
                contact_after_action_change=int((mask & changed & (code == 1)).sum()),
                measured_contact_fraction_excluding_censored=float((mask & (code == 1)).sum()/completed.sum()) if completed.any() else None)
        result[str(frames)+'f_label_next'+str(decisions)+'_decisions'] = rows
    result['scope'] = ('Observed student continuation, not held-action counterfactual execution. '
        'Windows overlap and are not independent episodes. Time is counted in recorded decisions; '
        'native elapsed frames were not archived per row. Bomb/protection/reset/terminal/tape boundaries '
        'are censored, not successes. Different movement commands also change focus/carried brain state. '
        'Uses pre-action known flags, never post-contact supervision acceptance.')
    return result
