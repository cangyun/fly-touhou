import numpy as np
import pytest
import torch
from fly_touhou import nbnm_corridor as corridors
from fly_touhou.nbnm_corridor_loss import compose,loss,restrict_reference,targets,witnessed_contacts
from fly_touhou.nbnm_sensation import candidate_extension
from test_nbnm_route_preference import scene


def example():
    e=scene(x=10,y=360)
    e.update(frame=20,stage=4)
    e['details']['entities']=1
    return e


def test_short_risk_priority_only_changes_a_conflicting_learned_preference():
    from fly_touhou.nbnm_corridor_loss import prioritize_short_risk

    current = torch.full((3, 19), -3.)
    current[:, :2] = torch.tensor([2., 1.])
    current[:, 18] = -torch.inf
    base = current[:, :18].clone()
    base[:, 0] = 0.
    risk = torch.full((3, 18), .3)
    risk[:2, 1] = .04
    settings = dict(trigger_risk=.2, alternative_risk=.1, remaining_corridor_weight=.25)
    priority = prioritize_short_risk(current, base, risk, torch.tensor([True, False, True]), settings)
    assert priority.argmax(dim=1).tolist() == [1, 0, 0]
    torch.testing.assert_close(priority[1:], current[1:], rtol=0., atol=0.)
    assert torch.isfinite(priority[:, :18]).all()


def test_new_laser_intersection_on_first_command_reaches_long_risk_labels():
    e=example()
    e['details'].update(entities=0,trusted=[False]*3)
    e['source_native_forecasts'].update(reactive_context={'player':[10.,360.]},
        pending_line_half_extents=[1.,1.],pending_line_births=[dict(tick=2,aim_tick=0,
            aim_origin=[0.,360.],angle=0.,origin=[0.,360.],speed=5.,width=14.,
            initial_length=24.,max_length=160.,travel_limit=0.)])
    planned=corridors.plan(e)
    status=planned['features']
    assert status[0,2:4].tolist()==[-1.,-1.] and status[0,6]==1.
    assert (status[:,3]>0).any()
    features=torch.zeros(1,407)
    features[:,281:]=torch.from_numpy(status.reshape(-1))
    danger,weight,_,_=targets(features)
    assert danger[0,0].tolist()==[1.,1.] and weight[0,0].tolist()==[1.,1.]


def test_uncertain_late_laser_keeps_the_covered_48_tick_corridor():
    from fly_touhou.nbnm_corridor_evidence import _covered

    e = example()
    scene = e['source_native_forecasts']
    scene['emitter_projection_horizon'] = 96
    scene['pending_emitter_unknown_from_tick'] = 70
    assert _covered(e,48) is not None
    assert _covered(e,96) is None
    result = corridors.plan(e,refine_uncertain_turns=True)
    assert result['features'][0,2] == 1.
    assert result['features'][0,3] == 0.
    assert len(corridors.commands4(result,0)) == 12


def test_local_turn_adds_a_missing_root_without_replacing_other_proven_courses(monkeypatch):
    from fly_touhou import compact_spiking_sensation

    encoding = example()
    result = corridors.plan(encoding)
    root = 1
    result['features'][root, 3] = -.5
    result['margins'][4:, root] = np.nan
    result['commands'][4:, root] = -1
    baseline = result['features'].copy()
    prefix = corridors.commands4(result, root)
    starts = []

    def clearance(*args, initial_time, **kwargs):
        starts.append(initial_time)
        return np.full(len(args[2]), 10.)

    monkeypatch.setattr(corridors, 'native_window_covered', lambda *args: True)
    monkeypatch.setattr(compact_spiking_sensation, 'point_and_laser_clearances', clearance)
    corridors.refine_local_turns(encoding, result, 1)

    assert len(prefix) == 12
    assert starts == [48, 60, 72]
    assert result['local_turn_refinement']['added'] == [root]
    np.testing.assert_array_equal(corridors.commands4(result, root)[:12], prefix)
    np.testing.assert_array_equal(np.delete(result['features'], root, axis=0), np.delete(baseline, root, axis=0))


def test_point_birth_sector_keeps_the_forward_and_rewind_paths_enclosed():
    from fly_touhou.native_pending_lines import point_birth_clearances

    centered = dict(tick=1, origin=[0., 0.], max_speed=2., bullet_radius=1., muzzle_radius=0.)
    directional = dict(centered, angle=0., spread=.1, aim_tick=0)
    path = np.repeat([[[0., 30.]], [[-8., 0.]], [[25., 2.]]], 25, axis=1)
    circular = point_birth_clearances([centered], path, 0, 24, 1.)
    sector = point_birth_clearances([directional], path, 0, 24, 1.)
    assert circular[0] < 0 < sector[0]
    assert np.all(sector[1:] < 0)


def test_native_deviation_witnesses_only_the_played_first_move_without_a_clearance_label():
    from fly_touhou.nbnm_corridor_loss import recovered_exits,witness_targets,course_targets
    f=torch.zeros(1,407)
    cues=f[:,281:].view(1,18,7)
    cues[:,:,-1]=.25
    cues[:,:,2:4]=-.5
    cues[:,4,:4]=torch.tensor([.55,.56,.75,.75])
    cues[:,5,:4]=torch.tensor([.20,.20,1.,1.])
    act=torch.tensor([7])
    outcome=torch.tensor([0])
    restored=recovered_exits(f,act,torch.tensor([[True,True]]))
    assert restored.sum()==2 and restored[0,7].all()
    y,w=witness_targets(f,act,outcome,recovered=restored)
    assert y[0,7].all() and (w[0,7]>0).all() and not w[0,8].any()
    risk,certainty,utility,eligible=targets(f,act,outcome,recovered=restored)
    assert not certainty[0,7].any() and risk[0,7,1]==1
    assert eligible.item() and utility[0,7]>=.70*utility.max() and utility[0,4]>utility[0,7]
    accept,rows=course_targets(f,act,outcome,allow_replans=True,recovered=restored)
    assert rows.item() and accept[0,7] and accept[0,4] and not accept[0,8]
    restored=recovered_exits(f,act,torch.tensor([[True,False]]))
    assert targets(f,act,outcome,recovered=restored)[2][0,7]==0


def test_legal_turning_witness_from_observed_player_and_focus_lock():
    e=example()
    e['source_features25'][18]=-1
    result=corridors.plan(e)
    assert result['features'].shape==(18,7)
    assert result['features'][:,3].min()==1
    for a in range(18):
        path=corridors.commands4(result,a)
        assert len(path)==24 and path[0]==a
    np.testing.assert_allclose(result['features'][:9],result['features'][9:])
    carry=corridors.carry_after(e,result,1,20)
    assert carry['frame']==24
    e.update(frame=24)
    e['source_features25'][:2]=[(carry['position'][0]+184)/368,(carry['position'][1]-32)/400]
    renewed=corridors.plan(e,carry=carry)
    assert len(renewed['retained'])==24
    assert renewed['features'][renewed['retained'][0],3]==.75
    e['source_features25'][0]+=.1
    assert 'retained' not in corridors.plan(e,carry=carry)


def test_native_spell_scope_changes_only_the_extra_search(monkeypatch):
    from fly_touhou.nbnm_sensation import NBNMEncoder
    config=dict(revision=corridors.REVISION,actor_weight=4.,loss_weight=.3,refinement_phases='boss_spells',
        refinement_first_boss_spell=2,corner_reserve_stages=[6],reactive_retry_stages=[6])
    encoder=NBNMEncoder(corridor_risk=config)
    calls=[]
    def capture(enc,**settings):
        calls.append(settings)
        return {'features':np.zeros((18,7),np.float32)}
    monkeypatch.setattr(corridors,'plan',capture)
    for stage,phase in ((4,'BossCard2'),(6,'BossCard1'),(6,'BossCard2_at'),(6,'Boss3'),(6,'BossCard5')):
        encoder.last_result={'stage':stage};encoder._attack=(stage,phase)
        assert len(encoder.current_corridor())==126
    assert [c['reactive_retry'] for c in calls]==[False,False,True,False,True]
    assert [c['corner_reserve'] for c in calls]==[False,False,True,False,True]

    # Preserve the original room policy on each spell. Recompute alternate
    # player-aimed trajectories only for the fifth spell and its helper routines.
    del config['refinement_first_boss_spell']
    config['reactive_retry_boss_cards']=[5]
    config['refine_reactive_turns']=True
    config['refine_early_turns']=True
    encoder=NBNMEncoder(corridor_risk=config)
    calls.clear()
    for stage,phase in ((6,'BossCard1'),(6,'BossCard4'),(6,'Boss5'),(6,'BossCard5_at'),
                        (6,'BossCard6'),(5,'BossCard5'),(6,'MBossCard5')):
        encoder.last_result={'stage':stage};encoder._attack=(stage,phase)
        encoder.current_corridor()
    assert [c['reactive_retry'] for c in calls]==[False,False,False,True,False,False,False]
    assert [c['refine_uncertain_turns'] for c in calls]==[False,False,False,True,False,False,False]
    assert [c['refine_early_turns'] for c in calls]==[False,False,False,True,False,False,False]
    assert [c['corner_reserve'] for c in calls]==[True,True,False,True,True,False,False]


def test_refines_a_lost_96_frame_exit_and_carries_the_exact_delayed_turn(monkeypatch):
    from fly_touhou import compact_spiking_sensation as sensation
    e=example();e['details']['margins'][0]=-2.;e['details']['margins'][0,0]=10.
    def gap(features,objects,pos,speed,**options):
        begin=options['initial_time']
        if begin<48:
            return np.where(np.all(speed==0.,axis=1),10.,-3.)
        if options['duration']>12:
            return np.full(len(pos),-3.)
        if begin==48:
            return np.where((speed[:,0]>0.)&(speed[:,1]==0.),10.,-3.)
        if begin==60:
            return np.where((speed[:,0]==0.)&(speed[:,1]<0.),10.,-3.)
        return np.where(np.all(speed==0.,axis=1),10.,-3.)
    monkeypatch.setattr(sensation,'point_and_laser_clearances',gap)
    old=corridors.plan(e)
    result=corridors.plan(e,refine_uncertain_turns=True)
    assert old['features'][0,3]<0 and result['features'][0,3]>0
    course=corridors.commands4(result,0)
    assert len(course)==24 and not course[:12].any()
    assert course[12]==course[14] and course[15]!=course[14]
    assert np.array_equal(corridors.pack_course(result,0),course)
    assert np.array_equal(corridors.carry_after(e,result,0,20)['course'],course[1:])
    assert not np.any(result['features'][1:9,3]>0)


def test_early_turn_recovers_and_preserves_the_actual_48_frame_witness(monkeypatch):
    from fly_touhou import compact_spiking_sensation as sensation
    from fly_touhou.thrl_movement_teacher import action_velocities
    e=example();e['details']['margins'][0]=-2.;e['details']['margins'][0,0]=10.
    for blocked_after_48 in (False,True):
        def gap(features,objects,pos,speed,*,initial_time,duration,**options):
            if initial_time<8:
                allowed=(speed[:,0]>0)&(speed[:,1]==0)&(initial_time+duration<=8)
            elif initial_time<16:
                allowed=(speed[:,0]==0)&(speed[:,1]<0)&(initial_time+duration<=16)
            else:
                allowed=(speed==0).all(1)&(not blocked_after_48 or initial_time<48)
            return np.where(allowed,10.,-3.)
        monkeypatch.setattr(sensation,'point_and_laser_clearances',gap)
        old=corridors.plan(e)
        assert not (old['features'][:,2]>0).any()
        row=corridors.plan(e,refine_uncertain_turns=True,refine_early_turns=True)
        path=corridors.commands4(row,0)
        assert len(path)==(12 if blocked_after_48 else 24) and row['refined_early_segments']>0
        assert row['features'][0,2]>0 and (row['features'][0,3]>0)==(not blocked_after_48)
        speeds=action_velocities()[path]
        assert speeds[1,0]>0 and speeds[2,1]<0 and (speeds[4:]==0).all()
        np.testing.assert_allclose(row['features'][0,0],10/18)
        np.testing.assert_array_equal(corridors.pack_course(row,0)[:len(path)],path)


def test_uncovered_laser_keeps_its_segment_in_late_corridor_legs_and_carry():
    from fly_touhou.thrl_laser_observation import bounded
    e=example()
    token=np.zeros((1,29),np.float32)
    token[0,[0,3,13,26]]=1.
    token[0,6:8]=bounded([0.,-33.],[384,448])
    token[0,21:23]=bounded([36.,0.],[384,448])
    token[0,10]=bounded(3.,128)
    e['source_objects29']=token
    current=corridors.plan(e)
    witness=corridors.commands4(current,0)
    assert witness is not None and len(witness)==24
    assert current['features'][0,6]==.25
    carry=corridors.carry_after(e,current,0,20)
    e['frame']=24
    renewed=corridors.plan(e,carry=carry)
    assert len(renewed['retained'])==24 and np.isfinite(renewed['features']).all()
    onward=corridors.plan(e,carry=carry,onward=True)
    assert len(onward['retained'])==24 and onward['onward_command'][onward['retained'][0]]>=0
    assert onward['features'][0,6]==.25


def test_corner_reserve_preserves_the_only_geometrically_possible_exit(monkeypatch):
    from fly_touhou import compact_spiking_sensation as sensation
    from fly_touhou.thrl_movement_teacher import action_velocities
    e=example()
    e['source_features25'][:2]=[0.,1.]
    e['details']['margins'][0]=-3.
    e['details']['margins'][0,0]=20.
    speeds=action_velocities()
    assert np.all(speeds[0]==0.)
    def stationary_exit(features,objects,position,velocity,**options):
        return np.where((velocity==0.).all(-1),20.,-3.)
    monkeypatch.setattr(sensation,'point_and_laser_clearances',stationary_exit)
    found=corridors.plan(e,corner_reserve=True)
    # A side wall alone still leaves room to turn. At the intersection of
    # two walls this margin shrinks, but an actually reachable exit survives.
    np.testing.assert_allclose(corridors._room_to_turn([[-184,432],[0,432],[-184,232]]),[0,184,200])
    assert found['features'][0,3]>0 and found['features'][0,1]<.2
    assert (found['features'][1:9,2:4]==-1).all()
    assert len(corridors.commands4(found,0))==24


def test_reactive_retry_recovers_a_turn_pruned_by_rigid_motion(monkeypatch):
    from fly_touhou import compact_spiking_sensation as sensation
    e=example()
    # The first four ticks have one physical exit. The observed candidate-
    # dependent object only lets a later downward turn through.
    e['details']['margins'][0]=-3.
    e['details']['margins'][0,0]=8.
    e['source_native_forecasts']['reactive_bullets']=[{'supported':True}]
    def clearance(f,objects,pos,speed,**kw):
        if kw['initial_time']!=4:
            return np.full(len(pos),8.)
        down=(speed[:,0]==0)&(speed[:,1]>0)
        if kw.get('native_forecasts') is None:
            return np.where(down,-2.,9.)
        return np.where(down,8.,-2.)
    monkeypatch.setattr(sensation,'point_and_laser_clearances',clearance)
    short=corridors.plan(e)
    retry=corridors.plan(e,reactive_retry=True)
    assert short['features'][0,3]<=0
    assert retry['features'][0,3]>0 and 0<retry['reactive_retry_segments']<20
    witness=corridors.commands4(retry,0)
    assert len(witness)==24 and witness[1] in (4,13)
    assert (retry['features'][1:9,2:4]==-1).all()


def test_onward_probe_keeps_96_separate_from_a_120_frame_exit(monkeypatch):
    from fly_touhou import compact_spiking_sensation as sensation
    e=example()
    for state in ('open','closed'):
        def geometry(features,objects,position,velocity,*,initial_time,**kw):
            if initial_time==96:
                return np.where((velocity==0.).all(-1)&(state=='open'),7.,-4.)
            return np.full(len(position),15.)
        monkeypatch.setattr(sensation,'point_and_laser_clearances',geometry)
        proposal=corridors.plan(e,onward=True)
        a=0
        assert len(corridors.commands4(proposal,a))==24 and proposal['features'][a,3]>0
        assert (proposal['onward_command'][a]>=0)==(state=='open')
        carry=corridors.carry_after(e,proposal,a,20)
        e['frame']=24
        updated=corridors.plan(e,onward=True,carry=carry)
        assert len(updated['retained'])==24
        first=int(updated['retained'][0])
        assert (updated['onward_command'][first]>=0)==(state=='open')


def test_narrow_only_corridor_trains_both_readouts_and_vetoes_unsafe_preference():
    f=torch.zeros(1,407)
    x=f[:,281:].reshape(1,18,7)
    x[:,:,6]=.25
    x[:,:,2:4]=-.5
    x[:,8,:4]=torch.tensor([.01,.02,1.,1.])
    t=targets(f)
    assert t[3].item()
    assert t[1][0,7].sum()==0
    risk=torch.zeros(1,18,2,requires_grad=True)
    actor=torch.zeros(1,19,requires_grad=True)
    s,g=loss(risk,actor,t)
    (s+g).backward()
    assert risk.grad[0,8].sum()>0 and actor.grad[0,8]<0
    # Max gradient descent lowers risk at the narrow exit; actor prefers it.
    final=compose(actor.detach(),(-risk.grad)*6,f,torch.tensor([True]),4)
    assert final[:,:18].argmax()==8
    ref=torch.zeros(1,19);ref[0,2]=1
    _,weight=restrict_reference(ref,torch.ones(1),t)
    assert weight.item()==0
    assert candidate_extension(f,corridor=True).shape==(1,18,23)
    legacy=candidate_extension(f[:,:281],corridor=True)
    assert not legacy[...,-7:].any()


def test_only_a_competitive_revalidated_continuation_is_taught():
    from fly_touhou.nbnm_corridor_loss import compose_course,course_loss,course_targets
    f=torch.zeros(4,407)
    f[:,111]=torch.tensor([1.,1.,1.,2/6])
    z=f[:,281:].reshape(4,18,7)
    z[:,:,:4]=torch.tensor([.75,.75,1.,1.]);z[:,:,6]=.75
    z[:,9,2:4]=.75
    z[1,9,1]=.20
    # The retained idle aliases agree in native displacement. A contact
    # on either invalidates consideration of that same maintained path.
    target,rows=course_targets(f,torch.tensor([6,6,0,7]),torch.tensor([0,0,1,0]))
    assert rows.tolist()==[True,False,False,True]
    assert target[0].nonzero().flatten().tolist()==[0,9]
    score=torch.zeros(4,18,requires_grad=True)
    course_loss(score,target,rows).backward()
    assert score.grad[0,9]<0 and score.grad[0,5]>0 and not score.grad[1:3].any()
    learned=torch.zeros(4,18);learned[:,9]=5.
    policy=compose_course(torch.zeros(4,19),learned,f,torch.tensor([True,False,True,True]),
                         dict(keep_course_stages=[6],keep_course_weight=3.))
    assert policy[0].argmax()==9 and not policy[1].any() and not policy[3].any()
    # Same learned scores, but the old suffix is now narrow or absent.
    z[2,9,3]=1.
    z[0,5,1]=.4
    revised=compose_course(torch.zeros(4,19),learned,f,torch.ones(4,dtype=torch.bool),
        dict(keep_course_stages=[6],keep_course_weight=3.,revalidated_course_only=True))
    assert revised[0].argmax()==9 and not revised[1:].any()
    f[:,274]=torch.tensor([1.,1.,0.,1.])
    f[:,275]=torch.tensor([5.,4.,5.,5.])/32
    scoped=compose_course(torch.zeros(4,19),learned,f,torch.ones(4,dtype=torch.bool),
        dict(keep_course_stages=[6],keep_course_spell_numbers=[5],keep_course_weight=3.))
    assert scoped[0].argmax()==9 and not scoped[1:].any()


def test_course_guidance_accepts_safe_replans_and_censors_only_the_failed_path():
    from fly_touhou.nbnm_corridor_loss import course_loss,course_targets
    f=torch.zeros(3,407)
    z=f[:,281:].reshape(3,18,7)
    z[:,:,6]=.75
    z[:,0,:4]=torch.tensor([.65,.60,.75,.75])
    z[:,9,:4]=torch.tensor([.65,.60,1.,1.])
    z[:,1,:4]=torch.tensor([.65,.60,1.,1.])  # equal, freely interchangeable
    z[:,2,:4]=torch.tensor([.55,.65,1.,1.])  # both near approaches clear 8 px
    z[:,3,:4]=torch.tensor([.47,.65,1.,1.])  # tighter approach, same far gain
    z[:,4,:4]=torch.tensor([.65,.60,1.,.25])  # limited onward room
    actions=torch.tensor([1,0,0])
    outcomes=torch.tensor([0,0,1])
    hits=torch.tensor([[False,False],[False,True],[False,False]])
    accepted,rows=course_targets(f,actions,outcomes,hits,allow_replans=True)
    assert rows.tolist()==[True,False,False]
    assert accepted[0].nonzero().flatten().tolist()==[0,1,2,9]
    logits=torch.zeros(3,18,requires_grad=True)
    course_loss(logits,accepted,rows).backward()
    assert logits.grad[0,1]<0 and logits.grad[0,2]<0 and logits.grad[0,3]>0
    # New witnesses are checked against native outcomes as well. Contact on
    # a later turn invalidates only that course; immediate contact invalidates
    # every encoding of the same first displacement.
    z[:,5,:4]=torch.tensor([.65,.60,.75,.75])
    accepted,rows=course_targets(f,actions,outcomes,hits,allow_replans=True)
    assert rows.all() and accepted[1,9] and not accepted[1,0]
    assert not accepted[2,0] and not accepted[2,9]
    z[:,:,:4]=torch.tensor([.65,.60,1.,1.]);z[:,5,3]=.75
    accepted,rows=course_targets(f,allow_replans=True)
    assert accepted.all() and not rows.any()


def test_continuation_signal_reaches_existing_ports_only_in_the_configured_stage(tmp_path):
    from test_compact_story_ready import build
    from fly_touhou.nbnm_model import configure_model
    _,model=build(tmp_path)
    # This miniature connectome has fewer VP ports than opponent feature
    # types. Give it a representative of the population present in the full brain.
    model.core.input_encoder.base.feature[0]=7
    model.core.input_encoder.base.polarity[0]=1.
    settings=dict(revision=corridors.REVISION,actor_weight=4.,loss_weight=.3,
        keep_course=True,keep_course_weight=1.,keep_course_stages=[6])
    configure_model(model,dict(risk_mode='route',objective={},corridor_risk=settings),reset_critics=False)
    f=torch.zeros(2,407);f[:,111]=torch.tensor([5/6,1.])
    cue=f[:,281:].reshape(2,18,7);cue[:,:,:4]=torch.tensor([.75,.75,1.,1.]);cue[:,:,6]=.75
    cue[:,3,2:4]=.75
    x=model.candidate_features(f).reshape(36,-1)
    encoder=model.core.input_encoder
    baseline=encoder(x).detach()
    configure_model(model,dict(risk_mode='route',objective={},corridor_risk=dict(settings,course_port_gain=1.25)),reset_critics=False)
    changed=(encoder(x).detach()!=baseline).any(-1).reshape(2,18)
    assert not changed[0].any() and changed[1].nonzero().flatten().tolist()==[3]


def test_displaced_student_rejoins_a_rechecked_suffix_using_legal_motion(monkeypatch):
    from fly_touhou.thrl_movement_teacher import action_velocities
    e=example()
    v=np.trunc(action_velocities()*128)/128
    right=int(np.flatnonzero(np.all(v==[4.5,0],axis=1))[0])
    left=int(np.flatnonzero(np.all(v==[-4.5,0],axis=1))[0])
    neutral=int(np.flatnonzero((v==0).all(1))[0])
    origin=np.array([10.,360.]);anchor=origin+4*v[left]
    suffix=np.full(24,neutral)
    immediate=np.full(18,10.)
    matches=corridors._rejoin_courses(origin,anchor,suffix,v,np.arange(18),immediate)
    assert any(path[0]==left and note['endpoint_error']==0 for path,note in matches)
    carry=dict(frame=20,stage=4,position=anchor,course=suffix[:23])
    planned=corridors.plan(e,carry=carry)
    assert planned['rejoined']['frames']==4 and planned['retained'][0]==left
    # A different executed action preserves the old route as a possible future
    # rendezvous. It cannot be used if the swept path now meets new fire.
    following=corridors.carry_after(e,planned,right,20)
    np.testing.assert_allclose(following['alternate']['position'],anchor,atol=2e-5)
    monkeypatch.setattr(corridors,'_revalidate',lambda *args: {})
    denied=corridors.plan(e,carry=carry)
    assert 'rejoined' not in denied and 'retained' not in denied


def test_legacy_restoration_adds_the_current_corridor_once_at_landing(monkeypatch):
    from types import SimpleNamespace
    from fly_touhou.nbnm_sensation import NBNMEncoder
    from fly_touhou.compact_story_sensation import CompactStoryEncoder
    import fly_touhou.nbnm_sensation as sensation
    e=example()
    encoder=NBNMEncoder(risk_mode='route',corridor_risk=dict(revision=corridors.REVISION,
        actor_weight=4.,loss_weight=.3,witness_weight=8.))
    encoder.reset();encoder.base.set_native_episode(1)
    def base(self,snapshot):
        self.last_result=dict(e,frame=snapshot.frame,stage=snapshot.stage)
        return dict(features=np.zeros(129,np.float32))
    monkeypatch.setattr(CompactStoryEncoder,'encode',base)
    monkeypatch.setattr(sensation,'current_attack',lambda snapshot:'BossCard2')
    monkeypatch.setattr(sensation,'route_context',lambda encoding:np.zeros((18,8),np.float32))
    native=SimpleNamespace(stage=4,frame=20)
    calls=[];real=corridors.plan
    def logged(*args,**kwargs):
        calls.append(args[0]['frame'])
        return real(*args,**kwargs)
    monkeypatch.setattr(corridors,'plan',logged)
    encoder.replaying_before_corridor_migration=True
    previous=encoder.encode(native)['features']
    native.frame+=4
    landing=encoder.encode(native)['features']
    assert not calls and not previous[281:].any() and not landing[281:].any()
    encoder.replaying_before_corridor_migration=False
    upgraded=encoder.finish_corridor_migration()['features']
    np.testing.assert_array_equal(landing[:281],upgraded[:281])
    assert calls==[24] and upgraded[281:].any()
    np.testing.assert_array_equal(encoder.encode(native)['features'],upgraded)
    assert calls==[24]


def test_orb_outside_24_frame_reach_can_be_lured_later_on_actual_path(monkeypatch):
    from fly_touhou import native_bullet_release,native_wall_geometry
    def straight(raw,horizon,*,_clearance_arrays=False):
        point=np.array([300.,200.])
        return dict(bounds=np.broadcast_to(point,(horizon+1,2,2)).copy(),
            known_by_frame=[True]*(horizon+1))
    monkeypatch.setattr(native_bullet_release,'predict_bullet_release',straight)
    point=dict(marker=1,native=dict(native_state=1,native_flags=0,native_ex_active_flags=0))
    ctx=dict(player=[0.,200.],bodies=[dict(near_radius=64.)])
    initial=native_wall_geometry._untriggered_prefix(point,ctx,24)
    assert initial['reactive_trigger_unreachable_through_frame']==24
    assert native_wall_geometry._untriggered_prefix(point,ctx,96) is None
    e=example()
    e['source_native_forecasts']=dict(reactive_context=ctx,bullets=[dict(object_index=0,release=initial,reactive_point=point)])
    tail,_=corridors._tail_scene(e,96)
    assert not tail['bullets'] and tail['reactive_bullets'][0]['point'] is point


def test_orb_batch_fast_path_matches_native_scalar_dynamics_bit_for_bit():
    import json,struct
    from fly_touhou.native_wall_motion import OPS,parse_program,simulate
    from fly_touhou.native_wall_geometry import _cached_positions
    rows=bytearray()
    for i,op in enumerate(OPS):
        values=[0.,0.,0.,0.,0,0,0,0,op,0,0]
        if i in (5,14):values[4]=32
        if i==9:values[4]=8
        if i in (6,15):values[4]=2;values[6]=int(i==15)
        if i in (8,17):values[:2]=[.1,999.];values[4]=2
        rows.extend(struct.pack('<4f4i3I',*values))
    program=parse_program(rows.hex())
    point=dict(marker=1,reaction_timer=0,native=dict(native_state=1,native_flags=0,
        native_position=[.1,140.7],native_velocity=[.77,-.38],native_speed=1.1,
        native_angle=.22,native_ex_index=18,native_ex_active_flags=0))
    routes=np.full((4,97,2),350.,np.float32)
    routes[1,4:30]=[2.,142.]
    routes[2]=routes[1]
    routes[3,65:]=[52.,115.]
    result=_cached_positions(json.dumps(point),rows.hex(),64.,85.,routes.tobytes(),routes.shape)
    for j,path in enumerate(routes):
        ref=[point['native']['native_position'], *[row['position'] for row in simulate(point,program,64.,85.,path)]]
        np.testing.assert_array_equal(result[j],np.asarray(ref,np.float32))


def test_a_verified_corner_exit_retains_horizon_credit_but_not_open_route_priority():
    from fly_touhou.nbnm_corridor_loss import witness_targets
    f=torch.zeros(2,407);x=f[:,281:].reshape(2,18,7)
    x[:,1,:4]=torch.tensor([.8,.8,1.,.25]);x[:,1,6]=.75
    x[0,2,:4]=torch.tensor([.8,.8,1.,1.]);x[0,2,6]=.75
    y,w,quality,prefer=targets(f)
    exists,_=witness_targets(f)
    assert exists[0,1,1]==1 and w[0,1,1]>0 and y[0,1,1]>=.75
    assert quality[0,1]<.70*quality[0,2] and prefer[0]
    # No competing route: the sole witnessed exit is still positively trained.
    assert y[1,1,1]<.25 and quality[1,1]>0 and quality[1,1]>quality[0,1]
    # A blocked first move never becomes a route through this preference.
    x[0,2,2:4]=-1
    y,w,quality,_=targets(f)
    assert y[0,1,1]<.25 and w[0,2,1]>0 and quality[0,2]==0


def test_native_contact_overrides_both_idle_encodings_even_without_an_alternative_witness():
    from fly_touhou.nbnm_corridor_loss import witness_targets
    f=torch.zeros(3,407)
    action=torch.tensor([9,12,12]);f[1,11]=-1.
    hit=torch.ones(3,dtype=torch.long)
    values=targets(f,action,hit)
    _,weights=witness_targets(f,action,hit)
    teacher=torch.zeros(3,19);teacher[0,0]=1.;teacher[1:,3]=1.
    probabilities,teacher_weight=restrict_reference(teacher,torch.ones(3),values)
    assert torch.equal(teacher_weight,torch.tensor([0.,0.,1.]))
    assert not probabilities[:2].any() and probabilities[2,3]==1.
    assert (weights[0,[0,9]]>0).all() and (weights[1,[3,12]]>0).all()
    assert not weights[2,3].any()


def test_tactical_reference_and_student_policy_share_competitive_corridor_targets():
    f=torch.zeros(2,407);x=f[:,281:].reshape(2,18,7)
    x[:,1,:4]=torch.tensor([.8,.8,1.,.25]);x[:,1,6]=.75
    x[0,2,:4]=torch.tensor([.8,.8,1.,1.]);x[0,2,6]=.75
    p=torch.zeros(2,19);p[:,1]=.8;p[:,2]=.2
    distribution,weight=restrict_reference(p,torch.ones(2),targets(f))
    assert weight.tolist()==[1.,1.]
    assert distribution[0,2]==1. and distribution[0,1]==0.
    assert distribution[1,1]==1. and distribution[1,2]==0.
    p[0,1]=1.;p[0,2]=0.
    distribution,weight=restrict_reference(p,torch.ones(2),targets(f))
    assert weight[0]==0. and not distribution[0].any()


def test_endpoint_readout_uses_only_neural_predictions_and_declared_spell_scope():
    from fly_touhou.nbnm_corridor_loss import endpoint_adjustment
    f=torch.zeros(4,407);f[:,111]=torch.tensor([1.,5/6,1.,1.]);f[:,274]=torch.tensor([1.,1.,0.,1.])
    p=torch.zeros(4,18,1);p[:,2]=2.;p.requires_grad_()
    options=dict(stages=[6],weight=2.)
    extra=endpoint_adjustment(p,f,options)
    assert extra[0,2]>1. and not extra[1:3].any()
    assert extra[0].sum().abs()<1.e-6 and torch.equal(extra[0],extra[3])
    f[:,281:]=torch.randn(4,126)
    assert torch.equal(extra,endpoint_adjustment(p,f,options))
    extra[:,2].sum().backward()
    assert p.grad[0,2]>0 and not p.grad[1:3].any()


def test_endpoint_preference_only_votes_when_both_types_of_exit_have_evidence():
    from fly_touhou.nbnm_corridor_loss import endpoint_adjustment
    f=torch.zeros(5,407);f[:,111]=1.;f[:,274]=1.
    routes=f[:,281:].reshape(5,18,7)
    routes[:,1,3]=.1875;routes[:,1,6]=.75
    routes[1:,2,3]=1.;routes[2:,2,6]=.75
    routes[3,1,3]=0.
    routes[4]=routes[2].roll(5,0)
    pred=torch.zeros(5,18,1);pred[:,7]=2.
    costs=endpoint_adjustment(pred,f,dict(stages=[6],weight=3.,paired_only=True))
    assert not costs[[0,1,3]].any()
    assert costs[2,7]>2. and torch.equal(costs[2],costs[4])


def test_followed_path_collision_teaches_its_risk_and_leaves_other_branches_unknown():
    T=5
    values=dict(actions=torch.ones(T,1,dtype=torch.long),valid_mask=torch.ones(T,1,dtype=torch.bool),
        unprotected_mask=torch.ones(T,1,dtype=torch.bool),recorded_native_episode=torch.ones(T,1,dtype=torch.long),
        recorded_native_stage=torch.full((T,1),4),recorded_episode_frames=torch.arange(T)[:,None]*4,
        native_action_frames=torch.full((T,1),4),outcomes=torch.zeros(T,1,dtype=torch.int8),
        terminated=torch.zeros(T,1,dtype=torch.bool),recorded_corridor_course=torch.ones(T,1,24,dtype=torch.int8))
    values['outcomes'][-1]=1;values['terminated'][-1]=True
    hits=witnessed_contacts(values)
    assert hits[0,0].all()
    values['recorded_lure_course'] = values['recorded_corridor_course'].clone()
    values['recorded_lure_course'][0,0,1] = 2
    lure_hits = witnessed_contacts(values, course_key='recorded_lure_course')
    assert not lure_hits[0,0].any() and lure_hits[1,0].all()
    f=torch.zeros(T,1,407);x=f[...,281:].reshape(T,1,18,7);x[...,1,:4]=torch.tensor([.4,.4,1.,1.]);x[...,6]=.75
    y,w,_,_=targets(f,values['actions'],values['outcomes'],hits)
    assert (w[0,0,1]==1).all() and (y[0,0,1]==1).all()
    assert (w[-1,0,1]==1).all() and (y[-1,0,1]==1).all()
    assert (w[0,0,2:]==0).all()
    values['actions'][1]=3
    divergent=witnessed_contacts(values)
    assert not divergent[0].any()
    y,w,_,_=targets(f,values['actions'],values['outcomes'],divergent)
    assert (y[0,0,1]<1).all() and (w[0,0,1]<1).all()
    values['actions'][1]=1;values['recorded_native_stage'][2:]=5
    assert not witnessed_contacts(values)[0].any()


def test_native_course_contact_after_48_updates_only_96_and_never_crosses_an_unknown_tick():
    length=14
    actions=torch.ones(length,1,dtype=torch.long)
    tape=dict(actions=actions,unprotected_mask=torch.ones_like(actions,dtype=torch.bool),
        outcomes=torch.zeros_like(actions),recorded_native_episode=torch.ones_like(actions),
        recorded_native_stage=torch.full_like(actions,4),recorded_episode_frames=(4*torch.arange(length))[:,None],
        native_action_frames=torch.full_like(actions,4),terminated=torch.zeros_like(actions,dtype=torch.bool),
        recorded_corridor_course=torch.ones(length,1,24,dtype=torch.int8))
    tape['outcomes'][-1]=1
    tape['terminated'][-1]=True
    f=torch.zeros(length,1,407)
    z=f[...,281:].reshape(length,1,18,7)
    z[...,1,:4]=torch.tensor([.4,.4,1.,1.])
    z[...,1,6]=.75
    hit=witnessed_contacts(tape)
    assert hit[0,0].tolist()==[False,True]
    y,w,_,_=targets(f,actions,tape['outcomes'],hit)
    torch.testing.assert_close(y[0,0,1],torch.tensor([.6,1.]))
    torch.testing.assert_close(w[0,0,1],torch.tensor([.75,1.]))
    tape['outcomes'][3]=-1
    assert not witnessed_contacts(tape)[0].any()


@pytest.mark.parametrize(('scope', 'short_metric', 'long_metric'), [
    ('road_exit_supervision_stages', 'road_short', 'road_exit'),
    ('exit_supervision', 'exit_short', 'exit_corridor'),
])
def test_committed_resume_updates_corridor_head_in_ready_ppo(tmp_path, scope, short_metric, long_metric):
    from test_compact_story_ready import build
    from fly_touhou.compact_story_ppo import optimizer_for,restore_optimizer,update
    from fly_touhou.nbnm_model import configure_model
    collector,model=build(tmp_path)
    config=dict(risk_mode='route',objective={})
    configure_model(model,config,reset_critics=False)
    old=optimizer_for(model)
    weight=old.param_groups[0]['params'][0]
    weight.grad=torch.ones_like(weight);old.step()
    config['corridor_risk']=dict(revision=corridors.REVISION,actor_weight=4.,loss_weight=.3,
        witness_weight=8.,keep_course=True,keep_course_weight=2.,order_supported_exits=True,learn_native_exits=True,
        prefer_full_horizon=True)
    config['corridor_risk'][scope] = True if scope == 'exit_supervision' else [4]
    configure_model(model,config,reset_critics=False)
    opt=optimizer_for(model);assert restore_optimizer(opt,old.state_dict())['corridor_risk_added']
    torch.testing.assert_close(opt.state[weight]['exp_avg'],old.state[weight]['exp_avg'])
    previous=collector.vector.snapshot
    def snapshot(world):
        obs,info=previous(world)
        x=np.pad(obs['features'],(0,407-len(obs['features'])))
        x[0:2] = [.98, .8]
        x[111] = 1. if scope == 'exit_supervision' else 4 / 6
        x[72] = 1.
        x[273] = float(scope == 'exit_supervision')
        x[18:36] = -.5
        x[18] = .5
        cue=x[281:].reshape(18,7);cue[:,6]=.75;cue[:,2:4]=-.5
        cue[2,:4]=(.64,.60,.75,.75);cue[3,:4]=(.67,.45,1.,1.)
        obs['features']=x
        info['corridor_courses']=np.tile(np.arange(18,dtype=np.int8)[:,None],(1,24))
        return obs,info
    collector.vector.snapshot=snapshot
    collector.obs,collector.infos=collector.vector.initial()
    f = torch.as_tensor(collector.obs['features'][:1])
    dn = torch.zeros(1,18,2*len(model.core.output_indices))
    with torch.no_grad():
        enabled = model.heads(dn, f)['logits']
        value = model.nbnm['corridor_risk'].pop(scope)
        ordinary = model.heads(dn, f)['logits']
        model.nbnm['corridor_risk'][scope] = value
    assert torch.equal(enabled, ordinary)
    rollout=collector.rollout(model,total_decisions=256)
    assert rollout['recorded_corridor_course'].shape[-1]==24
    before=model.corridor_head.weight.detach().clone()
    before_course=model.corridor_course_head.weight.detach().clone()
    result=update(model,opt,rollout,epochs=1,micro_scenes=8,checkpoint_chunks=True)
    log=result['logs'][0]
    assert log['corridor_route_rows']>0 and log['corridor_risk_loss']>0 and log['corridor_witness_loss']>0
    assert log['corridor_course_rows']>0 and log['corridor_course_loss']>0
    assert log[f'{short_metric}_action_rows'] > 0 and log[f'{short_metric}_action_loss'] > 0
    assert log[f'{long_metric}_action_rows'] > 0 and log[f'{long_metric}_action_loss'] > 0
    assert log['corridor_exit_order_rows']>0 and log['corridor_exit_order_loss']>0
    assert 'corridor_native_recovered96' in log
    assert log['corridor_full_horizon_rows']>0 and log['corridor_full_horizon_mass']>0
    assert rollout['recorded_corridor_witness_logits'].shape[-2:]==(18,2)
    assert torch.isfinite(model.corridor_head.weight).all() and not torch.equal(before,model.corridor_head.weight)
    assert torch.isfinite(model.corridor_course_head.weight).all() and not torch.equal(before_course,model.corridor_course_head.weight)


def test_actor_prefers_96f_but_keeps_native_replans_and_unresolved_risk_censored():
    from fly_touhou.nbnm_corridor_loss import recovered_exits, full_horizon_preference, _competitive_exits, opening_loss
    f = torch.zeros(3, 407)
    cue = f[:,281:].reshape(3,18,7)
    cue[:,:,6] = .25
    cue[:,:,2:4] = -.5
    cue[:,0,:4] = torch.tensor([.85,0.,1.,-.5])
    cue[:2,8,:4] = torch.tensor([.62,.38,1.,1.])
    observed = recovered_exits(f,torch.tensor([4,4,4]),torch.tensor([[False,False],[True,True],[False,False]]))
    original = targets(f,recovered=observed)
    revised, mass = full_horizon_preference(f,original,observed)
    accepted = _competitive_exits(revised[2])
    assert accepted[0,8] and accepted[0].sum()==1
    assert accepted[1,4] and accepted[1,8] and not accepted[1,0]
    assert accepted[2,0] and mass[0]>mass[1]>mass[2]>0
    assert not original[1][0,0,1] and torch.equal(original[1],revised[1])
    logits = torch.zeros(3,18,requires_grad=True)
    loss = opening_loss(logits,revised,row_mass=mass)
    split = sum(opening_loss(logits[i:i+1],tuple(x[i:i+1] for x in revised),mass.sum(),
                             row_mass=mass[i:i+1]) for i in range(3))
    torch.testing.assert_close(loss,split)
    loss.backward()
    assert logits.grad[0,8]<0 and logits.grad[0,0]>0 and logits.grad[1,4]<0 and logits.grad[2,0]<0


def test_long_risk_orders_two_open_courses_only_if_the_near_path_is_clear():
    from fly_touhou.nbnm_corridor_loss import supported_exit_order,supported_exit_order_loss
    f=torch.zeros(3,407);z=f[:,281:].reshape(3,18,7)
    z[:,8,:4]=torch.tensor([.85,.38,1.,1.])
    z[:,14,:4]=torch.tensor([.76,.44,1.,1.])
    z[:,:,6]=.25
    z[1,14,0]=.1
    z[2,14,3]=-.5
    labels=supported_exit_order(f,targets(f))
    logits=torch.zeros(3,18,requires_grad=True)
    supported_exit_order_loss(logits,labels).backward()
    assert logits.grad[0,14]>0 and logits.grad[0,8]<0 and not logits.grad[1:].any()
    assert torch.count_nonzero(logits.grad)==2


def test_road_supervision_separates_trusted_first_steps_from_exit_preferences():
    from fly_touhou.nbnm_corridor_loss import course_loss, exit_supervision

    f = torch.zeros(6, 407)
    f[:, 0:2] = torch.tensor([1., .8])
    f[:, 111] = 4 / 6
    f[:, 72] = 1.
    f[:, 18:36] = -.5
    f[:, 18] = .5
    cue = f[:, 281:].reshape(6, 18, 7)
    cue[:, :, 6] = .25
    cue[:, 1:3, :4] = torch.tensor([.5,.4,1.,1.])
    cue[1, 1:3, 6] = .75
    f[2, 273] = 1.
    f[3, 111] = 6 / 6
    f[4, 18:36] = .5
    f[5, 72] = 0.
    batch = dict(features=f, actions=torch.ones(6,dtype=torch.long),
                 outcomes=torch.zeros(6, dtype=torch.long),
                 unprotected_mask=torch.ones(6,dtype=torch.bool),
                 corridor_course_hits=torch.zeros(6,2,dtype=torch.bool))
    batch['corridor_course_hits'][0, 1] = True
    batch['outcomes'][1] = 1
    allowed, use_short, exits, weights = exit_supervision(batch, (4,))

    assert use_short.tolist() == [True, True, False, False, False, False]
    assert not allowed[0,0] and not allowed[1,1]
    assert torch.nonzero(exits[0]).flatten().tolist() == [2]
    assert torch.nonzero(exits[1]).flatten().tolist() == [2]
    torch.testing.assert_close(weights, torch.tensor([.25,.75,0.,0.,0.,0.]))
    logits = torch.zeros(6,18,requires_grad=True)
    short = course_loss(logits, allowed, use_short)
    within_short = logits.masked_fill(~allowed, -1e8)
    long = course_loss(within_short, exits, weights > 0, row_weights=weights)
    (short + long).backward()
    assert (logits.grad[:2,0]>0).all() and (logits.grad[:2,2]<0).all()
    assert not logits.grad[2:].any()


def test_exit_supervision_applies_to_all_six_stages_and_attack_phases():
    from fly_touhou.nbnm_corridor_loss import exit_supervision

    f = torch.zeros(13, 407)
    f[:, 0:2] = torch.tensor([.5, .5])
    f[:12, 111] = torch.arange(1,7).repeat_interleave(2) / 6
    f[1:12:2, 273] = 1.
    f[:, 72] = 1.
    f[:, 18:36] = -.5
    f[:, 18] = .5
    cue = f[:, 281:].reshape(13, 18, 7)
    cue[:, 2, :4] = torch.tensor([.6, .5, 1., 1.])
    cue[:, 2, 6] = .25
    batch = dict(features=f, actions=torch.full((13,), 1),
                 outcomes=torch.zeros(13,dtype=torch.long),
                 unprotected_mask=torch.ones(13,dtype=torch.bool),
                 valid_mask=torch.arange(13) < 12,
                 corridor_course_hits=torch.zeros(13,2,dtype=torch.bool))
    allowed, short_rows, exits, weights = exit_supervision(batch)
    assert not allowed[:,0].any()
    assert exits[:12,2].all() and short_rows[:12].all()
    torch.testing.assert_close(weights[:12], torch.full((12,), .25))
    assert weights[-1] == 0 and not short_rows[-1]
    _, legacy_short, _, legacy_weights = exit_supervision(batch, (4,))
    assert torch.nonzero(legacy_short).flatten().tolist() == [6]
    assert torch.nonzero(legacy_weights).flatten().tolist() == [6]



def test_exit_supervision_uses_a_weak_48_frame_target_only_without_a_96_frame_exit():
    from fly_touhou.nbnm_corridor_loss import exit_supervision

    f = torch.zeros(4, 407)
    f[:, 0:2] = torch.tensor([.98, .9])
    f[:, 111] = 4 / 6
    f[:, 72] = 1.
    f[:, 18:36] = -.5
    cue = f[:, 281:].reshape(4, 18, 7)
    cue[:, 2, :4] = torch.tensor([.4, 0., 1., -.5])
    cue[:, 2, 6] = .25
    cue[1, 3, :4] = torch.tensor([.4, .3, 1., 1.])
    cue[1, 3, 6] = .25
    hits = torch.zeros(4, 2, dtype=torch.bool)
    hits[2, 0] = True
    recovered = torch.zeros(4,18,2,dtype=torch.bool)
    recovered[3,4,1] = True
    batch = dict(features=f, actions=torch.full((4,), 2),
                 outcomes=torch.zeros(4,dtype=torch.long),
                 unprotected_mask=torch.ones(4,dtype=torch.bool),
                 corridor_course_hits=hits, corridor_recovered_exits=recovered)
    _, _, exits, weights = exit_supervision(batch, allow_48=True)
    assert torch.nonzero(exits).tolist() == [[0,2], [1,3], [3,4]]
    torch.testing.assert_close(weights, torch.tensor([.125,.25,0.,1.]))
    _, _, _, legacy = exit_supervision(batch, (4,))
    assert legacy[0] == 0.


def test_continuation_preference_wait_switch_and_factual_collision():
    from fly_touhou.nbnm_corridor_loss import continuation_preference, continuation_reference

    features = torch.zeros(8, 407)
    features[:, 18:36] = -.5
    cues = features[:, 281:].reshape(8, 18, 7)
    cues[:, 2, :4] = torch.tensor([.7, .7, .75, .75])
    cues[:, 1, :4] = torch.tensor([.62, .62, 1., 1.])
    cues[:, [1, 2], 6] = .75
    cues[1, 1, :2] = .8  # Clearly narrower routes stay outside the target.
    cues[2, 2, 3] = -.5  # A once-viable path can close as the game changes.
    cues[3:5, 0, :4] = torch.tensor([.7, .7, 1., 1.])
    cues[3:5, 0, 6] = .75
    cues[6:, :, 6] *= 1 / 3  # A truncated long forecast is weak evidence.
    margins = torch.full((8, 3, 18), 24.)
    margins[6, 0, 4] = -1.
    outcomes = torch.zeros(8, dtype=torch.long)
    outcomes[5] = 1
    course_hits = torch.zeros((8, 2), dtype=torch.bool)
    course_hits[4] = True
    safe_wait = torch.zeros((8, 18), dtype=torch.bool)
    safe_wait[3:5, 0] = True
    batch = dict(features=features, actions=torch.tensor([1, 1, 1, 1, 0, 2, 1, 1]), outcomes=outcomes,
                 margins=margins, trusted=torch.ones(8, 3, dtype=torch.bool),
                 unprotected_mask=torch.ones(8, dtype=torch.bool),
                 valid_mask=torch.ones(8, dtype=torch.bool), corridor_course_hits=course_hits,
                 nbnm_calm_moves=safe_wait)
    preferred, strength, waiting = continuation_preference(batch)
    assert strength.tolist() == [1., 1., 0., 1., 1., 0., .25, 0.]
    assert preferred[0].nonzero().flatten().tolist() == [2]
    assert preferred[1].nonzero().flatten().tolist() == [1, 2]
    assert preferred[3].nonzero().flatten().tolist() == [0]
    assert preferred[4].nonzero().flatten().tolist() == [2]
    assert waiting.nonzero().flatten().tolist() == [3]
    reference = torch.zeros((8, 19))
    reference[:, 1] = 1
    adjusted = continuation_reference(reference, preferred, strength)
    torch.testing.assert_close(adjusted.sum(-1), torch.ones(8))
    assert adjusted.argmax(-1).tolist() == [2, 1, 1, 0, 2, 1, 1, 1]
    assert adjusted[6, 2] == .25


def test_carry_after_selecting_a_new_prefix_advances_that_path():
    encoding = dict(source_features25=np.zeros(25, dtype=np.float32), stage=3)
    encoding['source_features25'][:2] = [.5, .8]
    old = np.array([2] * 24, dtype=np.int8)
    new = np.array([1] + [4] * 23, dtype=np.int8)
    plans = dict(retained=old, refined_courses={1: new})
    carry = corridors.carry_after(encoding, plans, 1, 100)
    assert carry['frame'] == 104 and carry['stage'] == 3
    assert np.array_equal(carry['course'], new[1:])
    assert np.array_equal(carry['alternate']['course'], old[1:])


@pytest.mark.parametrize('timed', [False, True])
def test_continuation_supervision_reaches_ready_ppo_without_a_new_readout(tmp_path, timed):
    from test_compact_story_ready import build
    from fly_touhou.compact_story_ppo import optimizer_for, update
    from fly_touhou.nbnm_model import configure_model

    collector, model = build(tmp_path)
    corridor = dict(revision=corridors.REVISION, actor_weight=4., loss_weight=.3, witness_weight=8.)
    config = dict(risk_mode='route', objective={}, corridor_risk=corridor)
    configure_model(model, config, reset_critics=False)
    keys = set(model.state_dict())
    previous = collector.vector.snapshot

    def snapshot(world):
        obs, info = previous(world)
        features = np.pad(obs['features'], (0,407-len(obs['features'])))
        features[18:36] = -.5
        features[72:75] = 1
        cue = features[281:].reshape(18,7)
        cue[2, :4] = [.7,.7,.75,.75]
        cue[2, 6] = .75
        cue[1, :4] = [.62,.62,1.,1.]
        cue[1, 6] = .75
        obs['features'] = features
        info['corridor_courses'] = np.tile(np.arange(18, dtype=np.int8)[:,None], (1,24))
        info['nbnm_verified_corridors'] = np.zeros((18, 2), bool)
        info['nbnm_verified_corridors'][2, 0] = True
        info['nbnm_due_turns'] = np.zeros(18, bool)
        info['nbnm_due_turns'][2] = True
        info['nbnm_previous_heading'] = 1
        info['nbnm_corridor_room'] = np.full(18, 30., np.float32)
        return obs, info

    collector.vector.snapshot = snapshot
    collector.obs, collector.infos = collector.vector.initial()
    features = torch.as_tensor(collector.obs['features'][:1])
    dn = torch.zeros(1, 18, 2*len(model.core.output_indices))
    ordinary = model.heads(dn, features)['logits']
    config['continuation_supervision'] = True
    config['timed_corridor_supervision'] = timed
    configure_model(model, config, reset_critics=False)
    assert set(model.state_dict()) == keys
    assert torch.equal(ordinary, model.heads(dn, features)['logits'])
    rollout = collector.rollout(model, total_decisions=256)
    result = update(model, optimizer_for(model), rollout, epochs=1, micro_scenes=8, checkpoint_chunks=True)
    log = result['logs'][0]
    assert log['corridor_continuation_rows'] > 0
    assert log['corridor_continuation_verified_rows'] == log['corridor_continuation_rows']
    assert log['corridor_wait_rows'] == 0
    assert log['corridor_policy_loss'] > 0.
    if timed:
        assert log['corridor_verified48_rows'] > 0
        assert log['corridor_due_turn_rows'] > 0


def test_independent_48_verification_and_delay_check(monkeypatch):
    from fly_touhou import nbnm_corridor_evidence as evidence

    scene = example()
    scene['details']['margins'][:, 1] = [5., 4., 0.]
    road = {'features': np.zeros((18, 7), np.float32)}
    road['features'][2, :4] = [.5, .4, .75, 1.]
    road['features'][2, 6] = .25
    paths = np.full((18, 24), -1, np.int8)
    paths[2] = 2
    checks = []

    def covered(_, horizon):
        return {'horizon': horizon} if horizon == 48 else None

    def revalidate(encoding, tail, velocity, origin, course, *, clearance_floor):
        checks.append((len(course) * 4, int(course[0]), clearance_floor))
        if course[0] == 1:
            return {}
        return {len(course) * 4: 8.}

    monkeypatch.setattr(evidence, '_covered', covered)
    monkeypatch.setattr(evidence, '_revalidate', revalidate)
    for stage in range(1, 7):
        scene['stage'] = stage
        verified, _, turn = evidence.evidence(scene, road, paths, previous_action=1)
        assert np.argwhere(verified).tolist() == [[2, 0]]
        assert np.flatnonzero(turn).tolist() == [2]
    assert (24, 1, 0.) in checks
    assert (48, 2, evidence.PREFERRED_CLEARANCE) in checks


def test_native_route_measurements_require_a_contiguous_clear_exit():
    from fly_touhou.nbnm_corridor_loss import native_exit_measurements

    T = 28
    times = torch.arange(T).reshape(T, 1) * 4
    data = dict(actions=torch.full((T, 2), 2, dtype=torch.long),
                unprotected_mask=torch.ones(T, 2, dtype=torch.bool),
                terminated=torch.zeros(T, 2, dtype=torch.bool),
                outcomes=torch.zeros(T, 2, dtype=torch.long),
                native_action_frames=torch.full((T, 2), 4, dtype=torch.long),
                recorded_native_stage=torch.full((T, 2), 5),
                recorded_native_episode=torch.ones(T, 2, dtype=torch.long),
                recorded_episode_frames=times.expand(-1, 2).clone(),
                trusted=torch.ones(T, 2, 3, dtype=torch.bool),
                margins=torch.full((T, 2, 3, 18), 9.),
                features=torch.zeros(T, 2, 407))
    data['recorded_episode_frames'][13:, 1] += 4
    data['outcomes'][22, 0] = 1
    observed, _ = native_exit_measurements(data)
    assert observed[0, 0].tolist() == [9., 0.]
    assert observed[0, 1].tolist() == [9., 0.]
    assert observed[1, 1].tolist() == [0., 0.]
    data['trusted'][12, 0, 1] = False
    revised, _ = native_exit_measurements(data)
    assert revised[0, 0].tolist() == [0., 0.]
