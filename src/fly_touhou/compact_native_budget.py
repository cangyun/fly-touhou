"""Small-budget development gate using frozen complete original-game results.

This is an allocation heuristic on a development set,not a significance test
or a claim of stable clear rate. Episode length alone never grants more work.
"""
from statistics import mean,median


def metrics(report):
    n=report['episodes'];games=report['episode_results']
    if report.get('evaluation_split','development')!='development' or any(g.get('native_seed_split','development')!='development' for g in games):
        raise ValueError('Held-out test games cannot select training budgets or checkpoints')
    if (report.get('full_game') is not True or report.get('teacher_actions')!=0
            or report.get('optimizer_updates')!=0 or n<1 or len(games)!=n
            or any(g['outcome'] not in ('gameover','clear') for g in games)):
        raise ValueError('Budget evidence must be frozen,student-controlled full native games')
    first={};first_contact={};contacts=0
    for event in report.get('dodge_diagnostics',{}).get('event_summaries',[]):
        if 'life_lost' in event['kinds']:
            first.setdefault(event['world']+1,event['after'])
        if 'contact_window_entered' in event['kinds']:
            contacts+=1;first_contact.setdefault(event['world']+1,event['after'])
    complete_first_miss=all(g['evaluation_game'] in first for g in games)
    damage=[g.get('reward_diagnostics',{}).get('boss_hp_fraction') for g in games]
    return dict(episodes=n,completion=[report['clears']/n,*[report['stage_counts'][str(s)]/n for s in range(6,1,-1)]],
        mean_boss_damage_fraction=mean(damage) if all(x is not None for x in damage) else None,
        median_first_miss=median(first[g['evaluation_game']]['episode_frames'] for g in games) if complete_first_miss else None,
        median_first_observed_contact=median(first_contact[g['evaluation_game']]['episode_frames'] for g in games)
            if all(g['evaluation_game'] in first_contact for g in games) else None,
        mean_observed_contact_windows=contacts/n if 'dodge_diagnostics' in report else None,
        mean_Bomb_before_first_miss=mean(first[g['evaluation_game']]['bombs_used'] for g in games) if complete_first_miss else None,
        mean_Bomb_used=mean(g['bombs_used'] for g in games))


def compare(candidate,incumbent):
    new=metrics(candidate);old=metrics(incumbent)
    if new['episodes']!=old['episodes']:raise ValueError('Development budget comparison needs the same episode count')
    reason='no_verified_native_benefit';improved=False
    if new['completion']>old['completion']:reason='clear_or_stage_progress';improved=True
    elif new['completion']<old['completion']:reason='clear_or_stage_regression'
    else:
        fields=('mean_boss_damage_fraction','median_first_miss','mean_Bomb_before_first_miss')
        if all(new[k] is not None and old[k] is not None for k in fields):
            resources=new['mean_Bomb_before_first_miss']<=old['mean_Bomb_before_first_miss']+.25
            dodge=(new['median_first_miss']>=max(old['median_first_miss']*1.2,old['median_first_miss']+120)
                and new['mean_boss_damage_fraction']>=old['mean_boss_damage_fraction']*.95)
            combat=(new['mean_boss_damage_fraction']>=max(old['mean_boss_damage_fraction']*1.1,old['mean_boss_damage_fraction']+.05)
                and new['median_first_miss']>=old['median_first_miss']*.95)
            if resources and dodge:
                contact_fields=('median_first_observed_contact','mean_observed_contact_windows')
                contact_supported=(all(new[k] is not None and old[k] is not None for k in contact_fields)
                    and new['median_first_observed_contact']>=max(old['median_first_observed_contact']*1.2,old['median_first_observed_contact']+120)
                    and new['mean_observed_contact_windows']<=old['mean_observed_contact_windows'])
                reason=('dodging_gain_without_resource_or_attack_regression' if contact_supported
                    else 'survival_gain_without_resource_or_attack_regression');improved=True
            elif resources and combat:reason='attack_gain_without_resource_or_dodging_regression';improved=True
    return dict(improved=improved,reason=reason,candidate=new,incumbent=old,
        scope='Fixed development-budget heuristic;not formal-test evidence or statistical significance')


class NativeBudgetAllowance:
    """Keep worlds alive across an evidence-supported finite budget extension.

    The reference is external evidence, never assigned to the new initial
    weights. A longer episode alone cannot unlock additional collection.
    """
    def __init__(self,initial_decisions,reference):
        if initial_decisions<=0:raise ValueError('Positive initial decision allowance required')
        metrics(reference)
        self.initial_decisions=initial_decisions;self.reference=reference;self.unlocked=False
    def can_collect(self,spent,rows):return self.unlocked or spent+rows<=self.initial_decisions
    def observe(self,report):
        decision=compare(report,self.reference)
        self.unlocked=self.unlocked or decision['improved']
        return dict(decision,extension_unlocked=self.unlocked,initial_decisions=self.initial_decisions)
