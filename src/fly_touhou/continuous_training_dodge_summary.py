"""Observed training contacts, explicitly separate from frozen game cohorts."""
from __future__ import annotations
from statistics import median


def training_contact_summary(collection):
    mode=collection['collection'].get('action_selection')
    completed=collection.get('completed_training_games',[])
    groups={'all_completed':completed}
    if mode=='student-greedy24-explore6':
        groups.update(greedy24_completed=[r for r in completed if int(r['worker'])<24],
            explore6_completed=[r for r in completed if int(r['worker'])>=24])
    result={}
    for name,games in groups.items():
        rows=[r['summary'] for r in games]
        contacts=[int(r['native_frames']) for r in rows if r['first_collision_observed']]
        result[name]=dict(completed_games=len(rows),observed_contacts=len(contacts),
            natural_clear_censored=sum(bool(r.get('natural_clear_right_censored')) for r in rows),
            completed_games_reaching_stage2=sum(max(r.get('stage_history',[1]))>=2 for r in rows),
            completed_contact_median_frames=float(median(contacts)) if contacts else None,
            completed_contact_mean_frames=sum(contacts)/len(contacts) if contacts else None,
            completed_contact_max_frames=max(contacts) if contacts else None)
    return dict(groups=result,action_selection=mode,frozen_evaluation=False,
        excludes_unfinished_games=True,one_game_may_span_parameter_updates=True,
        completed_contact_median_is_not_unbiased_survival_median=True)
