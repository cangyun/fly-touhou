"""Batch-boundary schedule controls; never reset the live game or brain."""
from .thrl_study_io import read_shared_json


def read_collection_control(schedule,path):
    if not path.exists():return None
    data=read_shared_json(path)
    if set(data)-{'revision','segment_decisions','max_updates','evaluation_mode'}:
        raise ValueError('Unknown collection control setting')
    revision=data['revision']
    if type(revision) is not int or revision<1:raise ValueError('Positive collection-control revision required')
    if revision<=schedule.revision:return None
    mode=data.get('evaluation_mode')
    if mode not in (None,'final_only'):raise ValueError('Supported evaluation override: final_only')
    before=dict(segment_decisions=schedule.segment_decisions,max_updates=schedule.max_updates,
        next_evaluation_decisions=schedule.next_evaluation,evaluation_every_decisions=schedule.evaluation_every)
    schedule.configure(data.get('segment_decisions',schedule.segment_decisions),
        data.get('max_updates',schedule.max_updates))
    if mode=='final_only':
        schedule.next_evaluation=schedule.decision_budget
        schedule.evaluation_every=schedule.decision_budget
        schedule.evaluation_mode=mode
    schedule.revision=revision
    return dict(revision=revision,before=before,
        after=dict(segment_decisions=schedule.segment_decisions,max_updates=schedule.max_updates,
            next_evaluation_decisions=schedule.next_evaluation,evaluation_every_decisions=schedule.evaluation_every,
            evaluation_mode=getattr(schedule,'evaluation_mode','periodic')),
        remaining_decision_budget_changed=False,evaluation_spacing_changed=mode is not None,
        native_worlds_reset=False,carried_brain_state_reset=False)


def install_for_running_schedule():
    """Upgrade the parser of the already-running Nov3 job at its first batch.

    That process loaded the previous ReadySchedule class before this option
    existed. Its position-summary function is imported after collection and
    before the next schedule read. Replacing only this method there lets the
    existing instance accept the explicit control file without restarting its
    sixteen games. Newly started jobs already use this parser and do nothing.
    """
    from .compact_story_schedule import ReadySchedule
    if getattr(ReadySchedule,'supports_live_evaluation_control',False):return False
    ReadySchedule.read_control=read_collection_control
    ReadySchedule.supports_live_evaluation_control=True
    return True
