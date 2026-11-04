"""Declare paired frozen decoder variants and compare complete native games."""
from .compact_story_idle_prior import checked_idle_scale


def variants(modes, inherited_scale, contrast_scales=None):
    if not contrast_scales:
        return [dict(label=mode, mode=mode, idle_prior_scale=inherited_scale,
                     folder='development' if modes == ['joint_mode'] else 'evaluation-' + mode)
                for mode in modes]
    scales = [checked_idle_scale(x) for x in contrast_scales]
    if len(modes) != 1 or len(scales) != 2 or scales[0] == scales[1]:
        raise ValueError('Paired prior comparison requires one decoder and two distinct scales')
    if scales[0] != inherited_scale:
        raise ValueError('First paired scale must equal the checkpoint idle-prior scale')
    return [dict(label=modes[0] + '-idle' + format(scale, 'g').replace('.', 'p'),
                 mode=modes[0], idle_prior_scale=scale,
                 folder='evaluation-' + modes[0] + '-idle' + format(scale, 'g').replace('.', 'p'))
            for scale in scales]


def paired_report(baseline, candidate):
    from .compact_native_budget import compare
    if baseline['action_selection'] != candidate['action_selection']:
        raise ValueError('Paired prior comparison requires the same action decoder')
    if baseline.get('policy_seed') != candidate.get('policy_seed'):
        raise ValueError('Paired prior comparison requires the same policy seed')
    old = {g['evaluation_game']: g for g in baseline['episode_results']}
    new = {g['evaluation_game']: g for g in candidate['episode_results']}
    if len(old) != baseline['episodes'] or set(old) != set(new) or len(new) != candidate['episodes']:
        raise ValueError('Missing or duplicate complete paired episodes')
    rows = []
    for index in sorted(old):
        a, b = old[index], new[index]
        if a['evaluation_seed'] != b['evaluation_seed']:
            raise ValueError('Paired original opening seeds differ')
        def describe(g):
            return dict(outcome=g['outcome'],maximum_stage=max(g['stage_history']),
                        bombs_used=g['bombs_used'],misses=g.get('misses'),
                        boss_damage=g.get('reward_diagnostics',{}).get('boss_hp_fraction'),
                        episode_frames=g.get('episode_frames'))
        rows.append(dict(game=index,evaluation_seed=a['evaluation_seed'],
                         baseline=describe(a),candidate=describe(b)))
    return dict(**compare(candidate,baseline),pairs=rows,
                shared_weights=True,changed_option='idle_prior_scale',
                optimizer_updates=0,brain_topology_changed=False,
                selection_scope='Paired development evidence only; no statistical significance or stable-clear claim. An improved heuristic does not automatically promote the candidate.')
