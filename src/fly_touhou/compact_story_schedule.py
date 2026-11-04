"""Decision-based collection, evaluation and compute budgets for live worlds."""
from .thrl_study_io import read_shared_json


class ReadySchedule:
    def __init__(self,*,workers,segment_decisions,max_updates,decision_budget,
                 evaluation_every,first_evaluation=0,checkpoint_every=0):
        self.workers=workers;self.decision_budget=decision_budget
        self.evaluation_every=evaluation_every
        self.next_evaluation=first_evaluation or evaluation_every
        self.checkpoint_every=checkpoint_every
        self.next_checkpoint=checkpoint_every or decision_budget
        self.revision=0
        if decision_budget<=0 or decision_budget%256 or evaluation_every<=0 or evaluation_every%256:
            raise ValueError('Positive whole256-decision budgets and evaluation intervals required')
        if self.next_evaluation<=0 or self.next_evaluation%256:
            raise ValueError('First frozen evaluation must be a whole256-decision boundary')
        self.configure(segment_decisions,max_updates)

    def configure(self,segment_decisions,max_updates):
        if type(segment_decisions) is not int or segment_decisions%256 or segment_decisions<self.workers*16:
            raise ValueError('Global segment needs whole256-row batches and at least16 decisions/world')
        if type(max_updates) is not int or max_updates<0:
            raise ValueError('Nonnegative Adam quota required;zero means no extra compute cap')
        self.segment_decisions=segment_decisions;self.max_updates=max_updates

    def read_control(self,path):
        if not path.exists():return None
        data=read_shared_json(path)
        if set(data)-{'revision','segment_decisions','max_updates'}:raise ValueError('Unknown collection control setting')
        revision=data['revision']
        if type(revision) is not int or revision<1:raise ValueError('Positive collection-control revision required')
        if revision<=self.revision:return None
        before=dict(segment_decisions=self.segment_decisions,max_updates=self.max_updates)
        self.configure(data.get('segment_decisions',self.segment_decisions),data.get('max_updates',self.max_updates))
        self.revision=revision
        return dict(revision=revision,before=before,
            after=dict(segment_decisions=self.segment_decisions,max_updates=self.max_updates),
            remaining_decision_budget_changed=False,evaluation_spacing_changed=False,
            native_worlds_reset=False,carried_brain_state_reset=False)

    def next_size(self,paid):
        remaining=min(self.decision_budget-paid,self.next_evaluation-paid)
        size=min(self.segment_decisions,remaining)
        if size<=0:return 0
        if size%256 or size<self.workers*16:
            raise ValueError('Remaining decision boundary cannot form complete real-world blocks')
        return size

    def evaluation_due(self,paid):
        return paid>=self.next_evaluation or paid>=self.decision_budget

    def evaluated(self,paid):
        while self.next_evaluation<=paid:self.next_evaluation+=self.evaluation_every

    def checkpoint_due(self,paid):
        return paid>=self.next_checkpoint

    def checkpointed(self,paid):
        if self.checkpoint_every:
            while self.next_checkpoint<=paid:self.next_checkpoint+=self.checkpoint_every

