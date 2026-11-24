"""Matched starts and extending real command tapes across variable loading."""
from pathlib import Path
import copy
import importlib.util
import pytest


def test_safe_idle_tie_break_preserves_threats_collection_and_boss_alignment():
    import math
    import torch
    from fly_touhou.compact_story_idle_prior import prefer_settled_idle
    f=torch.zeros(9,407)
    f[:,1]=(310-32)/400
    f[:,14]=1
    f[:,18:72]=math.tanh((1-40)/8)
    f[:,72:75]=1
    f[1,54+17]=math.tanh((1-25)/8)  # Only one candidate lacks 24-frame room.
    f[2,74]=0
    f[3,118]=.3                    # Recoverable power at the player's position.
    f[4,123]=1                     # Nearby valuable item.
    f[5:8,113]=1
    f[6:8,117]=1
    f[6:8,115]=-.3                 # Boss above the player, aligned in row 6.
    f[7,114]=.2
    f[8,13]=.2                     # Invulnerability timer.
    logits=torch.zeros(9,19)
    logits[:,3]=.3
    logits[6,3]=.7
    logits[:,18]=-torch.inf
    out,gate=prefer_settled_idle(logits,f,.5)
    assert gate.tolist()==[True,False,False,False,False,False,True,False,False]
    torch.testing.assert_close(out[:,1:],logits[:,1:])
    torch.testing.assert_close(out[~gate],logits[~gate])
    assert out[0].argmax()==0 and out[6].argmax()==3
    assert (logits[:,0]==0).all()


def test_route_aware_wait_requires_an_exit_after_stopping_and_preserves_focus():
    import math
    import torch
    from fly_touhou.compact_story_idle_prior import prefer_route_aware_wait
    f=torch.zeros(10,407)
    f[:,1]=(310-32)/400
    f[:,14]=f[:,112]=1
    f[:,72:75]=1
    h=f[:,18:72].reshape(-1,3,18)
    h[:]=math.tanh((1.-45.)/8.)
    h[:,:,7]=.5                              # A separate direction is dangerous.
    routes=f[:,281:].reshape(-1,18,7)
    routes[:,:,:2]=.75
    routes[:,:,2:4]=1
    routes[:,:,6]=.75
    z=torch.zeros(len(f),19)
    z[:,2]=.3
    z[:,18]=-torch.inf
    z[1,11]=.35                              # Stop without releasing Shift.
    routes[2,0,3]=-.5                        # Four safe ticks would forfeit the exit.
    h[3,2,0]=math.tanh((1.-18.)/8.)
    f[4,123]=1                               # Collectible at the player's current position.
    routes[5,0,6]=.25
    h[5,2,0]=math.tanh((1.-36.)/8.)          # Conditional tail needs wider short clearance.
    f[6,113]=f[6,117]=1
    f[6,114]=.5                              # Boss has moved out of alignment.
    f[7,112]=0
    z[8,18]=2
    near=torch.full((len(f),18,3),-1.)
    far=torch.full((len(f),18,2),-3.)
    far[9,0,1]=1                             # Long-risk head advises leaving.
    out=dict(logits=z,horizon_scores=near,corridor_logits=far)
    revised,eligible,idle=prefer_route_aware_wait(out,f,.5)
    assert eligible.tolist()==[True,True,False,False,False,False,False,False,False,False]
    assert revised.argmax(-1).tolist()==[0,9,2,2,2,2,2,2,18,2]
    assert idle[:2].tolist()==[0,9]
    torch.testing.assert_close(revised[~eligible],z[~eligible])
    torch.testing.assert_close(revised[:,18],z[:,18])


def test_frozen_heading_can_yield_on_the_very_next_decision():
    import math
    import torch
    from fly_touhou.action_query_residual_objects import source18_action_descriptors
    from fly_touhou.compact_story_idle_prior import prefer_safe_heading
    n=17
    f=torch.zeros(n,407)
    f[:,14]=1
    f[:,18:72]=math.tanh((1-40)/8)
    f[:,72:75]=1
    f[:,93:111]=.5
    paths=f[:,281:].view(-1,18,7)
    paths[:,:,:2]=.75
    paths[:,:,2:4]=1
    paths[:,:,6]=.75
    prior=torch.ones(n,dtype=torch.long)      # Last actually executed command is left.
    logits=torch.zeros(n,19)
    logits[:,2]=.25                           # Policy currently favours right.
    logits[:,18]=-torch.inf
    scores=torch.full((n,18,3),-1.)
    long_risk=torch.full((n,18,2),-3.)
    prior[1]=-1                               # New episode or new stage.
    f[2,54+1]=math.tanh((1-20)/8)
    f[3,74]=0
    paths[4,1,3]=-.5
    f[5,94]=.05                              # Would reach a wall.
    logits[6,9]=1                            # Preserve chosen stop.
    logits[7,11]=1                           # Preserve chosen change to focused movement.
    logits[8,2]=.8                           # A decisive policy change wins.
    long_risk[9,1,1]=2
    f[10,113]=f[10,117]=1
    f[10,114]=70/368                         # The boss is far right; holding left loses aim.
    f[11,123]=1                              # Nearby valuable item.
    scores[12,1,0]=1
    paths[13:16,1,6]=.25                    # Conditional tails need extra short clearance.
    f[13,54+1]=math.tanh((1-36)/8)
    logits[15,2]=.4                          # Discounted .375 bonus cannot overrule .4.
    paths[16,1,6]=0                          # No long-route evidence.
    out=dict(logits=logits,horizon_scores=scores,corridor_logits=long_risk)
    changed,eligible,base=prefer_safe_heading(out,f,prior,source18_action_descriptors()[:,2:4]*18,.5)
    assert eligible.tolist()==[True,False,False,False,False,False,False,False,True,False,False,False,False,False,True,True,False]
    assert changed.argmax(-1).tolist()==[1,2,2,2,2,2,9,11,2,2,2,2,2,2,1,2,2]
    torch.testing.assert_close(changed[~eligible],logits[~eligible])
    torch.testing.assert_close(changed[:,18],logits[:,18])
    torch.testing.assert_close(base,logits.argmax(-1))


def test_frozen_graph_heading_history_survives_padding_and_chunks():
    import torch
    from fly_touhou.nbnm_cuda import InferenceGraphs
    from fly_touhou.spiking_engineered_control import SpikingState
    class Echo(InferenceGraphs):
        sample=False
        def __init__(self):self.received=[]
        def __call__(self,features,state,previous_actions):
            self.received.append(previous_actions.tolist())
            return {'actions':previous_actions},state,features
    g=Echo()
    x=torch.arange(5,dtype=torch.float32)[:,None]
    state=SpikingState(*(x.clone() for _ in range(4)))
    previous=torch.tensor([3,-1,6,1,8])
    result,_,_=g.frozen_batch(x,state,3,previous_actions=previous)
    assert g.received==[[3,-1,6],[1,8,-1]]
    torch.testing.assert_close(result['actions'],previous)


def test_rare_wall_exit_beats_a_blocked_choice_and_allows_a_later_turn():
    import math
    import torch
    from fly_touhou.compact_story_idle_prior import prefer_wall_escape
    f=torch.zeros(9,407)
    f[:,0]=16/368                            # 16 pixels from the left wall.
    f[:,14]=1
    f[:,72:75]=1
    hints=f[:,18:72].reshape(-1,3,18)
    hints[:]=math.tanh((1.-18.)/8.)
    hints[:,2,2]=math.tanh((1.+4.)/8.)       # Needs to turn after starting right.
    f[:,93+5]=2/64
    f[:,93+2]=34/64
    routes=f[:,281:].reshape(-1,18,7)
    routes[:,2,1]=.18
    routes[:,2,3]=1
    routes[:,2,6]=.25
    z=torch.zeros(9,19)
    z[:,5]=2
    z[:,2]=1
    z[:,18]=-torch.inf
    f[1,0]=.5                                # Away from wall.
    f[2,74]=0                                # Unknown short path.
    routes[3,5,:]=routes[3,2,:]              # The chosen action has a long exit.
    routes[4,7:9,:]=routes[4,2,:]            # Several witnessed exits.
    f[5,93+2]=18/64                          # Little additional lateral room.
    routes[6:9]=0
    hints[6:,0,5]=.05                        # Independently avoid imminent contact.
    hints[7,:, :]=.25                        # No credible alternative.
    z[8,18]=3                                # Leave a legal Bomb selection alone.
    adjusted,reason,original=prefer_wall_escape(z,f)
    assert original.tolist()==[5,5,5,5,5,5,5,5,18]
    assert reason.tolist()==[2,0,0,0,0,0,1,0,0]
    assert adjusted.argmax(-1).tolist()==[2,5,5,5,5,5,2,5,18]
    torch.testing.assert_close(adjusted[reason==0],z[reason==0])
    torch.testing.assert_close(adjusted[:,18],z[:,18])


def test_fixed_frozen_graph_batch_owns_each_real_row():
    import torch
    from fly_touhou.nbnm_cuda import InferenceGraphs
    from fly_touhou.spiking_engineered_control import SpikingState
    class ScratchGraph(InferenceGraphs):
        def __init__(self):
            self.sample=False
            self.scratch=torch.zeros(4,2)
            self.calls=[]
        def __call__(self,x,state):
            self.calls.append(len(x))
            self.scratch.copy_(x+state.rate)
            return {'values':self.scratch},SpikingState(*(self.scratch for _ in range(4))),self.scratch
    x=torch.arange(14).reshape(7,2).float()
    state=SpikingState(*(torch.ones_like(x) for _ in range(4)))
    g=ScratchGraph()
    out,future,dn=g.frozen_batch(x,state,4)
    g.scratch.fill_(1000)
    assert g.calls==[4,4]
    for a in (out['values'],*future.tensors(),dn):
        torch.testing.assert_close(a,x+1)


def script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_only_frozen_zero_contact_story_through_all_six_stages_counts():
    from fly_touhou.nbnm_run import frozen_story_clears
    success=dict(native_seed=61008,native_frames=29000,full_game_NBNM=True,nbnm_success=True,
                 stage=6,task_misses=0,task_Bombs=0,nb_nm_contact=False,
                 nb_nm_reason='native-six-stage-clear',stage_history=list(range(1,7)))
    def report(mode,episodes,**kw):
        return dict(dict(frozen=True,task={'mode':mode},evaluation_split='test',episode_results=episodes),**kw)
    records=[success,dict(success,task_Bombs=1),dict(success,nb_nm_contact=True),
             dict(success,stage_history=[6]),dict(success,nb_nm_reason='native-spell-capture'),
             dict(success,full_story_prefix_played_by_current_policy=False),dict(success,nb_nm_contact=None)]
    found=frozen_story_clears(dict(story=report('full_story',records),practice=report('stage',[success]),
                                  training=report('full_story',[success],frozen=False),
                                  recorded_prefix=report('full_story',[success],fresh_policy_confirmation=False)))
    assert found==[dict(native_seed=61008,split='test',native_frames=29000)]


def test_stable_six_stage_target_uses_every_reserved_seed_once():
    from fly_touhou.nbnm_run import full_story_qualification
    seeds=list(range(100,120))
    spec=dict(full_story_heldout_seeds=seeds,qualification=dict(minimum_attempts=20,minimum_successes=18,minimum_rate=.9))
    episodes=[dict(native_seed=s,native_frames=32000,stage_history=[1,2,3,4,5,6],stage=6,
                   full_game_NBNM=i<18,nbnm_success=i<18,task_misses=0,task_Bombs=0,nb_nm_contact=i>=18,
                   nb_nm_reason='native-six-stage-clear' if i<18 else 'first-native-contact') for i,s in enumerate(seeds)]
    report=dict(frozen=True,task={'mode':'full_story'},episode_results=episodes,evaluation_split='test')
    assert full_story_qualification(report,spec)['qualified']
    assert not full_story_qualification(dict(report,evaluation_split='development'),spec)['qualified']
    assert not full_story_qualification(dict(report,episode_results=episodes[:-1]+[episodes[0]]),spec)['qualified']
    assert not full_story_qualification(dict(report,episode_results=episodes[:-1]),spec)['qualified']


def test_portability_gate_requires_fresh_policy_to_pass_stage_four_on_reserved_seeds():
    from fly_touhou.nbnm_run import full_story_qualification

    seeds=list(range(12))
    gate=dict(stage=4,minimum_entries=10,minimum_clears=9)
    spec=dict(full_story_heldout_seeds=seeds,
              qualification=dict(minimum_attempts=12,minimum_successes=6,minimum_rate=.5,stage_gate=gate))
    episodes=[]
    for i in seeds:
        stage=6 if i<6 else 5 if i<9 else 4 if i==9 else 3
        episodes.append(dict(native_seed=i,native_frames=32000,stage_history=list(range(1,stage+1)),
            stage=stage,full_game_NBNM=i<6,nbnm_success=i<6,task_misses=0,task_Bombs=0,nb_nm_contact=i>=6,
            nb_nm_reason='native-six-stage-clear' if i<6 else 'first-native-contact',
            full_story_prefix_played_by_current_policy=True))
    report=dict(frozen=True,task={'mode':'full_story'},episode_results=episodes,evaluation_split='test',
                soft_movement_scores=True,verified_movement_decisions=12000)
    assert full_story_qualification(report,spec)['qualified']
    for field in ('soft_movement_scores','verified_movement_decisions'):
        unverified={key:value for key,value in report.items() if key!=field}
        assert not full_story_qualification(unverified,spec)['qualified']
    for assistance in (dict(corridor_exit_scoring={'scores_applied':1}),dict(frozen_safe_idle_bonus=.3),
                       dict(frozen_safe_heading_bonus=.2),dict(frozen_wall_escape=True),
                       dict(frozen_route_aware_wait=True),dict(soft_movement_scores=False)):
        assisted=full_story_qualification(dict(report,**assistance),spec)
        assert not assisted['qualified'] and not assisted['autonomous_movement_scoring']
    episodes[8]['stage_history']=[1,2,3,4]
    gate_result=full_story_qualification(report,spec)
    assert gate_result['verified_six_stage_clears']==6
    assert gate_result['stage_gate']['cleared']==8 and not gate_result['qualified']


def test_frozen_movement_proof_rejects_masked_or_replaced_scoring():
    import numpy as np
    from fly_touhou.nbnm_numerics import verified_movement_actions

    scores=np.vstack((np.arange(18),np.arange(17,-1,-1))).astype(float)
    assert verified_movement_actions(scores,[17,0])==2
    with pytest.raises(ValueError,match='18 finite neural scores'):
        verified_movement_actions(scores,[16,0])
    scores[0,0]=-np.inf
    with pytest.raises(ValueError,match='18 finite neural scores'):
        verified_movement_actions(scores,[17,0])


def test_continued_prefix_rebases_loading_ticks_without_mutating_original():
    module=script('prepare_nbnm_start_bank')
    original=dict(transitions=[dict(before=dict(ticks=100),after=dict(ticks=120)),
                              dict(before=dict(ticks=300),after=dict(ticks=330))],
                  status='nbnm_warmup',ticks=500,raw_input_segments=[[1,500]])
    untouched=copy.deepcopy(original)
    receipt=dict(barriers=[dict(recorded_ticks=20,actual_ticks=24),
                           dict(recorded_ticks=30,actual_ticks=28)])
    result=module.continued_record(original,receipt,758)
    assert original==untouched
    assert result['transitions']==[dict(before=dict(ticks=100),after=dict(ticks=124)),
                                  dict(before=dict(ticks=304),after=dict(ticks=332))]
    assert result['ticks']==758 and result['status']=='recording'


def reports(before,after):
    def report(values):
        rows=[dict(native_seed=i,nbnm_success=success,task_native_frames=100+i,
                   end_reason='success' if success else 'native_contact') for i,success in enumerate(values)]
        return dict(task=dict(mode='attack',stage=3,attack='BossCard2'),frozen=True,
                    action_selection='joint_mode',opening_seeds=list(range(len(values))),
                    successes=sum(values),episode_results=rows)
    return report(before),report(after)


def test_paired_comparison_requires_matching_inputs_and_reports_regressions():
    compare=script('compare_nbnm_frozen').comparison
    a,b=reports([False]*8,[True]*8)
    entries={str(i):dict(features=[i],policy_burn_in=None) for i in range(8)}
    result=compare(a,b,entries,entries)
    assert result['improved_pairs']==8 and result['regressed_pairs']==0
    assert result['improvement_supported'] and result['exact_paired_success_p']<.05
    changed=copy.deepcopy(entries);changed['0']['features'][0]=.1
    assert not compare(a,b,entries,changed)['improvement_supported']
    a,b=reports([True,False],[False,True])
    result=compare(a,b,entries,entries)
    assert result['regressed_pairs']==result['improved_pairs']==1
    assert not result['improvement_supported']


def test_reference_stays_original_after_pilot_warmstart():
    choose=script('compare_nbnm_frozen').reference_checkpoint
    spec=dict(initial_checkpoint='pilot.pt',reference_checkpoint='original.pt')
    assert choose(spec)==Path('original.pt')
    assert choose(spec,'explicit.pt')==Path('explicit.pt')
    assert choose(dict(initial_checkpoint='pilot.pt'))==Path('pilot.pt')


def test_all_six_failures_do_not_hide_stage_progress_regression():
    compare=script('compare_nbnm_frozen').comparison
    a,b=reports([False,False],[False,False])
    a['task']=b['task']=dict(mode='full_story')
    for r,stage in zip(a['episode_results'],[3,2]):r['stage']=stage
    for r,stage in zip(b['episode_results'],[1,3]):r['stage']=stage
    entries={str(i):dict(features=[i]) for i in range(2)}
    result=compare(a,b,entries,entries)
    assert result['initial_successes']==result['trained_successes']==0
    assert result['full_story_progress']['stage_arrivals']['2']==dict(
        initial=2,trained=1,improved_pairs=0,regressed_pairs=1)


def test_paired_report_labels_test_split():
    compare=script('compare_nbnm_frozen').comparison
    a,b=reports([False],[True])
    a['evaluation_split']=b['evaluation_split']='test'
    entries={'0':dict(features=[0],policy_burn_in=None)}
    result=compare(a,b,entries,entries)
    assert result['evaluation_split']=='test'
    assert 'Matched test starts' in result['scope']


def test_recording_appends_acknowledged_native_replay_commands():
    from types import SimpleNamespace as NS
    from fly_touhou.frida_controller import OriginalGameController
    posted=[]
    controller=OriginalGameController.__new__(OriginalGameController)
    controller._script=NS(post=posted.append);controller._checkpoint={}
    controller._hook_ready={'native_tape':{'schema':1}}
    controller._memory=NS();controller._record_input_tape=True
    controller._input_tape=[(1,4)]
    controller._wait=lambda name:dict(sequence=2)
    controller.read=lambda:'native snapshot'
    assert controller.advance_tape([(9,4),(17,8)])=='native snapshot'
    assert controller._input_tape==[(1,4),(9,4),(17,8)]
    assert posted[0]['payload']['tape']==[[9,4],[17,8]]


def test_frozen_evaluation_accepts_readonly_vector_snapshots(monkeypatch,tmp_path):
    import numpy as np
    from fly_touhou import nbnm_run,thrl_vector,compact_full_story
    from fly_touhou.nbnm_model import configure_model
    from test_nbnm import model_and_observations
    model,features=model_and_observations()
    configure_model(model,dict(objective={},risk_mode='route'),reset_critics=False)
    initial=np.concatenate((features[:1].numpy(),np.zeros((1,152),np.float32)),axis=1)
    initial.flags.writeable=False
    class Vector:
        _inflight=False
        def __init__(self,*args,**kwargs):pass
        def reset(self,**kwargs):return dict(features=initial),[dict(native_seed=6)]
        def submit_at(self,*args,**kwargs):self._inflight=True
        def receive_ready(self,**kwargs):
            self._inflight=False
            summary=dict(native_seed=6,nbnm_success=True,task_native_frames=4,end_reason='attack_complete')
            return [dict(index=0,operation='step',observation=dict(features=initial[0]),
                info=dict(episode_summary=summary),terminated=True,truncated=False)]
        def close(self):pass
    monkeypatch.setattr(thrl_vector,'THRLNativeVector',Vector)
    monkeypatch.setattr(compact_full_story,'story_kwargs',lambda *a,**k:[])
    monkeypatch.setattr(nbnm_run,'environment',lambda *a,**k:dict(nbnm={}))
    monkeypatch.setattr(nbnm_run,'evaluation_seeds',lambda *a,**k:[6])
    task=dict(mode='attack',stage=3,attack='BossCard1')
    report=nbnm_run.evaluate(model,dict(nbnm=dict(task=task)),tmp_path/'frozen')
    assert report['attempts']==report['successes']==1
    assert not initial.flags.writeable


@pytest.mark.parametrize('through_stage',[None,2])
def test_native_openings_reuse_original_entry_and_count_contact_before_midboss(monkeypatch,tmp_path,through_stage):
    import numpy as np
    from fly_touhou import nbnm_run,thrl_vector,compact_full_story
    from fly_touhou.nbnm_model import configure_model
    from test_nbnm import model_and_observations
    model,base=model_and_observations()
    task=dict(mode='full_story')
    configure_model(model,dict(objective={},risk_mode='route'),reset_critics=False)
    row=np.concatenate((base[0].numpy(),np.zeros(152,np.float32)))
    seen=[]
    class Vector:
        _inflight=False
        def __init__(self,*args,**kwargs):pass
        def reset(self,*,options):
            self.seed=options[0]['native_seed']
            seen.append(self.seed)
            return dict(features=row[None]),[dict(native_seed=self.seed)]
        def submit_at(self,ids,operation,actions=None,options=None):
            self.operation=operation
            if operation=='reset':
                self.seed=options[0]['native_seed']
                seen.append(self.seed)
            self._inflight=True
        def receive_ready(self,**kwargs):
            self._inflight=False
            hit=self.operation=='step' and self.seed==7
            info=dict(native_seed=self.seed,stage=1,stage_frame=2462,nb_nm_phase='s1/MBoss1',
                      nb_nm_contact=hit)
            if hit:info['episode_summary']=dict(end_reason='first-native-contact')
            if through_stage is not None:
                info.update(stage=2 if hit else 3,nb_nm_phase='s2/BossCard2' if hit else 's3/road')
            return [dict(index=0, operation=self.operation, observation=dict(features=row),
                         info=info, terminated=hit, truncated=False)]
        def close(self):pass
    monkeypatch.setattr(thrl_vector,'THRLNativeVector',Vector)
    monkeypatch.setattr(compact_full_story,'story_kwargs',lambda cfg,*args,**kw: [])
    monkeypatch.setattr(nbnm_run,'environment',lambda *args,**kw:dict(nbnm={}))
    monkeypatch.setattr(nbnm_run,'evaluation_seeds',lambda spec,*,split,task:[6,7])
    report=nbnm_run.evaluate(model,dict(nbnm=dict(task=task),evaluation_workers=1),
                             tmp_path/'opening',opening_stage=1 if through_stage is None else None,through_stage=through_stage)
    if through_stage is not None:
        assert seen==[6,7] and report['attempts']==2 and report['stage_passes']==1
        assert [r['stage_passed'] for r in report['outcomes']]==[True,False]
        assert 'successes' not in report and not (tmp_path/'opening'/'episodes.jsonl').exists()
        return
    assert seen==[6,7] and report['attempts']==2 and report['opening_passes']==1
    assert 'successes' not in report and not (tmp_path/'opening'/'episodes.jsonl').exists()
    assert [r['reached_midboss'] for r in report['opening_results']]==[True,False]
