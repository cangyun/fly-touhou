"""One declared movement intervention in an otherwise fixed native action tape.

Original prefix observations must match the captured source. Alternative
outcomes are measured by the same original no-Bomb probe, not by a teacher or
collision formula. The original continuation is fixed, not a new closed-loop
policy and not a training demonstration.
"""
import hashlib
import numpy as np

from .thrl_native_trace_replay import RecordedAvoidanceCase


class NativeActionIntervention(RecordedAvoidanceCase):
    def __init__(self,report_path,row,*,action,offset=1):
        super().__init__(report_path,row)
        if type(action) is not int or not 0<=action<18 or type(offset) is not int or not 1<=offset<=min(128,len(self.actions)):
            raise ValueError('Declared source18 action and last128-step intervention required')
        self.original_actions=self.actions
        self.intervention_index=len(self.actions)-offset
        self.source_action=self.actions[self.intervention_index]
        self.intervention_action=action
        self.offset=offset
        self.baseline=action==self.source_action
        self.actions=self.actions[:self.intervention_index]+bytes([action])+self.actions[self.intervention_index+1:]
        self.executed=bytearray()
        self.branch_info=None

    def compare_before(self,observation,info):
        if self.baseline or self.position<=self.intervention_index:
            super().compare_before(observation,info)
            if self.differences:
                raise ValueError('Native intervention prefix differs from its original captured state')
        if self.position==self.intervention_index:
            self.branch_info={k:info.get(k) for k in
                ('observation_native_frame','episode_frames','stage','stage_frame','native_episode')}

    def acknowledge(self):
        self.executed.append(self.next_action())
        super().acknowledge()

    def finish_intervention(self,observation,info,*,terminal):
        if self.branch_info is None or self.position<=self.intervention_index:
            raise ValueError('Original native prefix ended before the declared intervention')
        expected=(self.tail_count if self.baseline else
                  max(0,self.intervention_index-self.tail_begin+1))
        if self.differences or self.checked_tail_rows!=expected:
            raise ValueError('Native prefix checkpoint coverage is incomplete')
        if len(self.executed)!=self.position or bytes(self.executed)!=self.actions[:self.position]:
            raise ValueError('Acknowledged intervention actions differ from the declared tape')
        if info.get('native_episode')!=self.branch_info['native_episode']:
            raise ValueError('Native world changed during an action intervention')
        # Last source intervals in this diagnostic are ordinary four-tick
        # movement, not a dialogue/loading bridge. Keep that scope explicit.
        elapsed=info.get('observation_native_frame',-1)-self.branch_info['observation_native_frame']
        if elapsed!=4*(self.position-self.intervention_index):
            raise ValueError('Native action probe crossed an undeclared nonplay interval')
        hit=bool(info.get('probe_episode',{}).get('first_collision_observed'))
        if terminal:
            probe=info.get('probe_episode')
            if not probe or not hit or probe.get('actual_bombs_used')!=0:
                raise ValueError('Unexpected terminal in a pure movement intervention')
            endpoint_frames=probe['native_frames']
        else:
            if (self.position!=len(self.actions) or info.get('misses')!=0
                    or info.get('bombs_used')!=0 or observation['features'][21]<=.5):
                raise ValueError('Unverified clean native comparison endpoint')
            endpoint_frames=info['episode_frames']
        source_endpoint_verified=False
        if self.baseline:
            baseline_result=super().finish(observation,info,terminal=terminal)
            if not baseline_result['original_native_replay_verified']:
                raise ValueError('Control replay of the unchanged original action did not match')
            source_endpoint_verified=True
        return dict(policy=self.row['policy'],game=self.row['game'],offset=self.offset,
            intervention_index=self.intervention_index,source_action=self.source_action,
            intervention_action=self.intervention_action,unchanged_action_control=self.baseline,
            original_endpoint_replay_verified=source_endpoint_verified,
            source_evaluation_seed=self.row['evaluation_seed'],native_seed=self.row['native_seed'],
            source_first_collision_frames=self.row['native_frames'],
            observed_endpoint_frames=endpoint_frames,observed_collision=hit,
            prevented_source_hit_through_comparison_horizon=not hit and self.position==len(self.actions),
            prefix_original_inputs_verified=True,checked_prefix_tail_rows=self.checked_tail_rows,
            acknowledged_decisions=self.position,
            action_tape_sha256=hashlib.sha256(bytes(self.executed)).hexdigest(),
            endpoint_xy=[float(observation['features'][0]*368-184),
                         float(observation['features'][1]*400+32)],
            endpoint_state_onehot=np.asarray(observation['features'][21:25]).tolist(),
            actual_Bombs=0,native_counterfactual_executed=not self.baseline,
            continuation='same recorded source actions after one changed four-native-frame movement',
            new_policy_efficacy_claimed=False,training_label=False)
