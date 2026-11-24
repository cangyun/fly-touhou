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


def test_original_th08_capture_and_ppo_keep_native_contact_and_censor_forecasts(tmp_path):
    from types import SimpleNamespace

    from fly_touhou.compact_story_ppo import optimizer_for, update
    from fly_touhou.nbnm_corridor import SCHEMA
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.th08_training import TH08TrainingEnv

    settings = dict(risk_mode="route", rule_guidance_coefficient=0., future_guidance_coefficient=0.,
                    risk_target_revision="student-native-continuation-risk-v2",
                    objective=dict(progress_revision="native-events-v1", progress_weight=.2),
                    corridor_risk=dict(revision="student-reachable-corridors-v1", beam=1, loss_weight=.3,
                                       actor_weight=4., witness_weight=8.))
    adapter = TH08TrainingEnv(executable=None, profile=None, d3d8to9=None, private_appdata=tmp_path,
                              input_schema=SCHEMA, settings=settings)
    adapter.world = SimpleNamespace(encoder=SimpleNamespace(last_result=dict(
        margins=np.full((5, 18), 8., np.float32), known=np.zeros(3, bool))))

    class StoryWorlds(DifferentSpeedWorlds):
        def snapshot(self, index):
            step = self.position[index]
            contact = bool(step == 43)
            stage = 1 if step < 24 else 2
            frames = int(step * 4)
            features = np.zeros(407, np.float32)
            features[14] = 1.
            features[111] = stage / 6.
            features[281:].reshape(18, 7)[:, 2:4] = -1.
            features[281:].reshape(18, 7)[:, 6] = .25
            observation = dict(features=features)
            adapter.native_episode = int(self.episodes[index])
            info = adapter._policy_info(observation, dict(stage=stage, stage_frame=frames, native_frames=frames,
                outcome="nbnm-failure" if contact else None, contact=contact, full_game_NBNM=False,
                misses=0, Bombs=0), int(self.last_action[index]), previous_stage=1 if step == 24 else stage)
            return observation, info

    collector, model = build(tmp_path)
    configure_model(model, settings, reset_critics=False)
    collector.vector = StoryWorlds(collector.workers)
    collector.obs, collector.infos = collector.vector.initial()
    rollout = collector.rollout(model, total_decisions=256)
    assert int(rollout["recorded_observed_contacts"].sum()) > 0
    contact_rows = rollout["recorded_observed_contacts"]
    assert (rollout["outcomes"][contact_rows] == 1).all()
    assert (rollout["reward_vectors"][contact_rows][:, 0] == -1.).all()
    assert float(rollout["reward_vectors"][..., 2].sum()) > 0
    assert not rollout["trusted"].any() and not rollout["rule_weights"].any()

    result = update(model, optimizer_for(model), rollout, micro_scenes=16, native_only_corridors=True)
    assert result["optimizer_updates"] == 1
    metrics = result["logs"][0]
    assert metrics["corridor_native_recovered48"] > 0
    assert 0 < metrics["corridor_label_actions48"] <= 2 * int(contact_rows.sum())


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


def test_unavailable_reset_stays_local_and_rejoins_the_next_segment(tmp_path):
    class UnavailableReset(DifferentSpeedWorlds):
        def receive_ready(self, **options):
            result = super().receive_ready(**options)
            row, = result
            if row["index"] == 0 and row["operation"] == "reset" and self.resets[0] == 1:
                row["info"].update(native_reset_unavailable=True,
                                   native_start_retries=[{"attempt": i} for i in (1, 2, 3)])
                self.needs_reset[0] = True
            return result

    collector, model = build(tmp_path)
    collector.vector = UnavailableReset(4)
    collector.obs, collector.infos = collector.vector.initial()
    collector.worker_groups = ["spell_focus", "spell_focus", "spell", "spell"]
    collector.group_weights = {"spell_focus": .5, "spell": .5}

    first = collector.rollout(model, total_decisions=256)
    assert first["metadata"]["native_suspended_worlds"] == [0]
    assert first["metadata"]["native_reset_discarded_rows"] == 11
    assert first["metadata"]["world_decisions"][:2] == [32, 96]
    assert sum(first["metadata"]["world_decisions"][2:]) == 128
    assert int(first["valid_mask"].sum()) == 256
    assert collector.vector.executed[0] == 43

    next_segment = collector.rollout(model, total_decisions=256)
    assert next_segment["metadata"]["native_suspended_worlds"] == []
    assert next_segment["metadata"]["world_decisions"][0] > 0
    assert collector.actions == 512


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
