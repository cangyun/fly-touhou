"""Exercise unequal native latency, live state continuity and real-row PPO."""
import numpy as np
import torch

from fly_touhou.compact_story_ready import ReadyCollector
from fly_touhou.compact_story_coverage import StoryCoverage


class DifferentSpeedWorlds:
    def __init__(self,worlds=4):
        self.worlds=worlds;self._inflight={};self.clock=0.
        self.position=np.zeros(worlds,int);self.episodes=np.zeros(worlds,int)
        self.resets=np.zeros(worlds,int);self.needs_reset=np.zeros(worlds,bool)
        self.last_action=np.zeros(worlds,int);self.executed=np.zeros(worlds,int)

    def snapshot(self,w):
        f=np.zeros(129,np.float32);f[:2]=[.25+w*.02,.75]
        f[2]=.4;f[4]=.375;f[14]=1;f[18:72]=-.5;f[72:93]=1;f[93:111]=.5
        stage=int(min(6,1+self.position[w]//24));f[111]=stage/6
        f[114]=self.position[w]/200
        label=dict(frame=int(self.position[w]*4),native_episode=int(self.episodes[w]),normal=True,
            margins=np.full((3,18),10.,np.float32),trusted=[True]*3)
        teacher=dict(frame=label['frame'],native_episode=label['native_episode'],
            probabilities=np.r_[np.full(18,1/18),0.].tolist(),guidance_weight=1.)
        info=dict(compact_targets=label,teacher_label=teacher,stage=stage,episode_frames=label['frame'],
            bombs_used=0,misses=0,bombs=3,lives=2,policy_action=int(self.last_action[w]),
            reward_mapping_complete=True,reward_vector=[.001,0.,0.],
            episode_summary=dict(stage=stage,game_cleared=False,frames=label['frame']))
        return dict(features=f),info

    def initial(self):
        rows=[self.snapshot(w) for w in range(self.worlds)]
        return {'features':np.stack([o['features'] for o,i in rows])},[i for o,i in rows]

    def submit_at(self,indices,operation,actions=None,options=None):
        for j,w in enumerate(indices):
            assert w not in self._inflight
            assert operation=='reset' or not self.needs_reset[w]
            delay=(1+5*w) if operation=='step' else (7+3*w)
            self._inflight[w]=(self.clock+delay,operation,None if actions is None else actions[j])

    def receive_ready(self,**kwargs):
        kwargs['check_stop']()
        w=min(self._inflight,key=lambda w:(self._inflight[w][0],w))
        due,operation,action=self._inflight.pop(w);self.clock=due
        done=False
        if operation=='reset':
            assert self.needs_reset[w]
            self.position[w]=0;self.episodes[w]+=1;self.resets[w]+=1;self.needs_reset[w]=False
        else:
            self.position[w]+=1;self.executed[w]+=1;self.last_action[w]=action
            done=self.position[w]==43;self.needs_reset[w]=done
        observation,info=self.snapshot(w)
        return [dict(index=w,operation=operation,observation=observation,info=info,
            reward=.001,terminated=done,truncated=False)]


def build(tmp_path):
    from test_candidate_spiking import model as base_fixture
    from fly_touhou.compact_story_model import CompactStoryPolicy
    torch.set_num_threads(2);torch.manual_seed(715)
    model=CompactStoryPolicy(base_fixture()).eval();model.pad_candidate_batch=False
    model.phase_valid_control=True;model.soft_risk_composition=True;model.native_contact_risk_calibration=True
    c=ReadyCollector.__new__(ReadyCollector)
    c.workers=4;c.vector=DifferentSpeedWorlds(4);c.output=tmp_path/'native';c.output.mkdir()
    c.obs,c.infos=c.vector.initial();c.state=model.initial_state(4);c.starts=np.ones(4,bool)
    c.actions=0;c.completed=0;c.coverage=StoryCoverage(4);c.policy_version=0
    c.native_position_hold_forecast=False;c.native_hidden_enemy_contacts=False;c.native_ex8_polar4_forecast=False
    c.ready_batch_size=7;c.ready_wait_ms=1.;c.movement_sampling_temperature=.5;c.bomb_sampling_temperature=1.
    return c,model


def test_ready_worlds_continue_through_updates_and_keep_real_ppo_rows(tmp_path):
    from fly_touhou.compact_story_ppo import update,optimizer_for
    from fly_touhou.compact_story_ready_data import ready_sequence_groups
    from fly_touhou.compact_story_storage import save_rollout
    c,model=build(tmp_path)
    first=c.rollout(model,steps=128)
    counts=first['metadata']['world_decisions']
    assert sum(counts)==512 and all(n%16==0 for n in counts)
    assert counts[0]>counts[3] and not c.vector._inflight
    assert int(first['valid_mask'].sum())==512<first['valid_mask'].numel()
    assert c.coverage.completed.tolist()==c.vector.resets.tolist()
    assert c.vector.resets[0]>c.vector.resets[3]
    groups=ready_sequence_groups(first['valid_mask'],16,np.random.default_rng(2))
    selected={(t+j,w) for g in groups for t,w in g for j in range(16)}
    assert selected==set(map(tuple,first['valid_mask'].nonzero().tolist()))
    assert len(groups)==2 and all(len(g)==16 for g in groups)
    # Lossless tapes retain only the real worlds' original recurrent states.
    path=tmp_path/'ready.pt';save_rollout(first,path,compressed=True)
    restored=torch.load(path,weights_only=False)
    for t,block in first['block_states'].items():
        assert restored['block_states'][t].worlds==block.worlds
        for w in block.worlds:
            for a,b in zip(restored['block_states'][t].row(w).tensors(),block.row(w).tensors()):
                assert torch.equal(a,b)
    position=c.vector.position.copy();resets=c.vector.resets.copy()
    carried=[v.clone() for v in c.state.tensors()]
    optimizer=optimizer_for(model)
    result=update(model,optimizer,restored,epochs=1,micro_scenes=16,world_aligned=True,seed=71)
    assert result['optimizer_updates']>0
    assert np.array_equal(c.vector.position,position) and np.array_equal(c.vector.resets,resets)
    for a,b in zip(carried,c.state.tensors()):assert torch.equal(a,b)
    second=c.rollout(model,steps=64)
    assert second['metadata']['behavior_policy_version']==2
    assert second['metadata']['student_actions']==256 and c.actions==768
    assert second['recorded_episode_frames'][0].tolist()==(4*position).tolist()
    for w in range(4):
        for a,b in zip(carried,second['block_states'][0].row(w).tensors()):assert torch.equal(a[w:w+1],b)
    assert not c.vector._inflight and c.coverage.completed.tolist()==c.vector.resets.tolist()


def test_gae_uses_full_world_histories_and_only_real_rows_for_normalization(tmp_path):
    from fly_touhou.compact_story_ready_data import prepare_ready_targets
    from fly_touhou.fly_gru_aligned_ppo import aligned_ppo_config
    from fly_touhou.thrl_ppo import compute_gae,combine_advantages
    c,model=build(tmp_path);data=c.rollout(model,steps=128)
    cfg=aligned_ppo_config();valid=data['valid_mask'];expected=torch.zeros_like(data['old_values'])
    targets=torch.zeros_like(expected)
    # Place a delayed consequence after the first BPTT boundary.
    data['reward_vectors'][25,0,0]=-3.
    for w,n in enumerate(valid.sum(0).tolist()):
        a,r=compute_gae(data['reward_vectors'][:n,w:w+1],data['old_values'][:n,w:w+1],
            data['bootstrap_values'][w:w+1],data['terminated'][:n,w:w+1],data['episode_starts'][:n,w:w+1],
            gamma=cfg.gamma,gae_lambda=cfg.gae_lambda)
        expected[:n,w]=a[:,0];targets[:n,w]=r[:,0]
    advantages,returns,combined=prepare_ready_targets(data,cfg,'cpu')
    torch.testing.assert_close(advantages[valid],expected[valid])
    torch.testing.assert_close(returns[valid],targets[valid])
    torch.testing.assert_close(combined[valid],combine_advantages(expected[valid],cfg.objective_weights))
    assert torch.count_nonzero(combined[~valid])==0
    old=advantages[0,0,0].clone();data['reward_vectors'][25,0,0]+=3.
    changed,_,_=prepare_ready_targets(data,cfg,'cpu')
    assert abs(float(old-changed[0,0,0]))>1.  # credit crosses16, not a per-block terminal


def test_reactive_wall_rollout_is_accepted_by_current_late_stage_replay(tmp_path):
    from fly_touhou.compact_native_forecast import native_repair_revisions
    from fly_touhou.compact_late_stage_replay import LateStageReplay,SENSOR_FLAGS
    from fly_touhou.compact_story_critical_replay import CriticalTeacherReplay
    c,model=build(tmp_path)
    c.native_wall_reactive_forecast='th14-distance-reactive-orbs-v1'
    c.vector.position[:]=48
    c.obs,c.infos=c.vector.initial()
    data=c.rollout(model,steps=64)
    revisions=native_repair_revisions(reactive_orbs=c.native_wall_reactive_forecast)
    assert data['metadata']['native_repair_revisions']==revisions
    assert 'reactive_orbs' not in native_repair_revisions()
    critical=CriticalTeacherReplay(native_repair_revisions=revisions)
    assert not critical._old_sensor_semantics(data)
    late=LateStageReplay(sensor_flags={key:data['metadata'][key] for key in SENSOR_FLAGS},
        native_repair_revisions=revisions)
    assert late.add(data,source='fresh-wall-rollout')>0
    assert late.incompatible_sources==[]


def test_lower_field_mode_updates_real_rollouts_and_carries_exception_metadata(tmp_path):
    from fly_touhou.compact_story_ppo import update,optimizer_for
    c,model=build(tmp_path)
    model.lower_field_default=True;model.combat_position_guidance_coefficient=.1
    c.infos[0]['compact_targets']['lower_field_exception']=True
    rollout=c.rollout(model,steps=64)
    assert rollout['lower_field_exception'][0,0]
    assert not rollout['lower_field_exception'][0,1]
    result=update(model,optimizer_for(model),rollout,epochs=1,micro_scenes=16,world_aligned=True,seed=72)
    assert result['optimizer_updates']==1
