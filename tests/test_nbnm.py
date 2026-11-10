from types import SimpleNamespace as NS
import copy
import numpy as np
import pytest
import torch
from fly_touhou.nbnm_objective import Objective, prefer_candidate, movement_reference
from fly_touhou.nbnm_task import TaskTracker, attack_key


def snapshot(*, frame=100, stage=3, name='BossCard1', hp=100, state=1, misses=0, bombs=0, protected=0):
    role=dict(ecl_sub_name=name, hp=hp)
    return NS(frame=frame,stage=stage,player=NS(state=state,invulnerable_frames=protected),
        extra={'thrl_events':dict(misses=misses,bombs_used=bombs,
            ecl_roles=dict(observed=True,main_bosses=[role] if name else [],midbosses=[]))})


def test_failure_cannot_save_time_penalty_and_attack_does_not_buy_success():
    objective=Objective()
    assert objective.weights==(1.,0.,0.)
    early=objective.rewards(terminal=True,success=False,native_frames=100,hp_progress=100)
    late=objective.rewards(terminal=True,success=False,native_frames=10000)
    assert early@objective.weights==late@objective.weights==-1
    fast=objective.rewards(terminal=True,success=True,native_frames=100)
    slow=objective.rewards(terminal=True,success=True,native_frames=10000)
    assert fast@objective.weights==slow@objective.weights==1
    speed=Objective(phase='speed')
    assert fast@speed.weights>slow@speed.weights>early@speed.weights
    assert objective.rewards(terminal=False,success=False,native_frames=999)[0]==0


def test_native_step_elapsed_reaches_nbnm_info(monkeypatch):
    from fly_touhou.compact_full_story import CompactFullStoryEnv
    from fly_touhou.nbnm_env import NBNMEnv
    monkeypatch.setattr(CompactFullStoryEnv,'_info',lambda self,elapsed=0:dict(elapsed_frames=elapsed))
    env=NBNMEnv.__new__(NBNMEnv)
    env.nbnm=dict(task=dict(mode='attack',stage=3,attack='BossCard1'))
    assert env._info()['elapsed_frames']==0
    info=env._info(4)
    assert info['elapsed_frames']==4 and info['episode_mode']=='NBNM-attack'
    assert info['nb_nm_task']==env.nbnm['task']


def test_checkpoint_selection_success_has_priority_over_speed():
    def report(times):
        return dict(frozen=True,task='s3',opening_seeds=list(range(len(times))),action_selection='joint_mode',
            episode_results=[dict(nbnm_success=t is not None,task_native_frames=t or 10) for t in times])
    assert not prefer_candidate(report([1000,1000]),report([100,None]))
    assert prefer_candidate(report([1000,1000]),report([900,900]))


def test_speed_qualification_requires_different_frozen_native_seeds():
    from fly_touhou.nbnm_objective import qualifies_speed
    rows=[dict(native_seed=7,nbnm_success=True) for _ in range(50)]
    report=dict(frozen=True,episode_results=rows)
    assert not qualifies_speed(report)
    for i,row in enumerate(rows):row['native_seed']=i
    assert qualifies_speed(report)


def test_prefix_comparison_retains_physics_and_script_offsets_not_addresses():
    from fly_touhou.nbnm_replay_state import gameplay_state
    a=dict(enemies=[dict(resource=dict(hp=200,native_motion=dict(data_address=1000,
        ecl_contexts=[dict(address=1008,instruction=10444,opcode=44,time=7)],
        interrupts=[dict(time_sub=10000,hp_sub=10000)])))],rng=dict(gameplay_seed=1))
    b=copy.deepcopy(a);motion=b['enemies'][0]['resource']['native_motion']
    motion['data_address']+=700;motion['ecl_contexts'][0]['address']+=700
    motion['ecl_contexts'][0]['instruction']+=50000
    for k in ('time_sub','hp_sub'):motion['interrupts'][0][k]+=50000
    assert gameplay_state(a)==gameplay_state(b)
    motion['ecl_contexts'][0]['instruction']+=4
    assert gameplay_state(a)!=gameplay_state(b)


def test_native_contact_ends_before_miss_counter_and_before_deathbomb():
    tracker=TaskTracker(dict(mode='attack',stage=3,attack='BossCard1'))
    tracker.begin(snapshot())
    s=snapshot(frame=104,state=4)
    assert tracker.terminal(s)==1
    assert not tracker.report(s,terminal=1)['nbnm_success']


def test_helper_and_disappearance_are_not_success_but_next_attack_is():
    tracker=TaskTracker(dict(mode='attack',stage=3,attack='BossCard1'))
    tracker.begin(snapshot())
    assert tracker.terminal(snapshot(name='BossCard1Move'))==0
    assert tracker.terminal(snapshot(name=None))==0
    assert tracker.terminal(snapshot(name='BossHide'))==0
    assert tracker.terminal(snapshot(name='Boss2'))==2
    assert tracker.report(snapshot(name='Boss2'),terminal=2)['full_game_NBNM'] is False


def test_bomb_and_actual_miss_invalidate_attack_success():
    for update in [dict(bombs=1),dict(misses=1)]:
        tracker=TaskTracker(dict(mode='attack',stage=3,attack='BossCard1'))
        tracker.begin(snapshot())
        assert tracker.terminal(snapshot(name='Boss2',**update))==1


def test_no_teacher_bomb_supervision_is_preserved():
    label=movement_reference(dict(probabilities=[0.]*18+[1.],guidance_weight=1.))
    assert label['probabilities'][18]==0 and label['guidance_weight']==0


def test_extended_clearance_does_not_reuse_the_old_64_pixel_clip():
    from test_compact_spiking import scene
    from fly_touhou.compact_spiking_sensation import encode_geometry
    from fly_touhou.nbnm_sensation import route_context,NBNMEncoder
    f,o=scene(0);_,details=encode_geometry(f,o,complete=True)
    details['unclipped_margins']=np.full((3,18),128.,np.float32)
    r=dict(source_features25=f,source_objects29=o,details=details,source_native_forecasts=None)
    a=route_context(r)
    details['unclipped_margins'][:]=256.
    b=route_context(r)
    assert np.isfinite(b).all() and (abs(b)<=1).all()
    assert (b[:,0]>a[:,0]).all()
    assert NBNMEncoder().specification()['features']['shape']==[281]


def test_nbnm_parameter_checkpoint_restores_new_ports_and_masks(monkeypatch):
    import fly_touhou.compact_story_model as module
    from test_candidate_spiking import model as base
    from fly_touhou.dn_residual_risk import DNResidualRisk
    from fly_touhou.nbnm_model import configure_model
    m=module.CompactStoryPolicy(base());width=2*len(m.core.output_indices)
    m.risk=DNResidualRisk(torch.zeros(width),torch.ones(width),torch.zeros(3,width),torch.zeros(3))
    configure_model(m,dict(objective={},risk_mode='route'))
    with torch.no_grad():m.core.input_encoder.extended_context.weight.fill_(.01)
    payload=m.parameter_payload({})
    monkeypatch.setattr(module,'build_compact_model',lambda *a,**k:base())
    restored=module.from_story_checkpoint(None,payload,device='cpu')
    assert restored.bombs_disabled and not restored.lower_field_default
    assert restored.core.sensory_width==63
    for key,value in m.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)


def test_nbnm_ready_collection_and_actual_ppo_update_remain_finite(tmp_path):
    from test_compact_story_ready import build
    from fly_touhou.nbnm_model import configure_model
    from fly_touhou.compact_story_ppo import update,optimizer_for
    collector,model=build(tmp_path)
    old_snapshot=collector.vector.snapshot
    def extended(world):
        obs,info=old_snapshot(world)
        obs['features']=np.concatenate((obs['features'],np.zeros(152,np.float32)))
        # Synthetic native terminal reward, not an old resource-play reward.
        info['reward_vector']=[-1. if collector.vector.needs_reset[world] else 0.,0.,0.]
        return obs,info
    collector.vector.snapshot=extended
    collector.obs,collector.infos=collector.vector.initial()
    configure_model(model,dict(objective={},risk_mode='route'))
    collector.worker_groups=['card','card','full','full']
    collector.group_weights=dict(card=.75,full=.25)
    data=collector.rollout(model,total_decisions=256)
    assert data['metadata']['task_group_decisions']==dict(card=192,full=64)
    assert not data['deathbomb_mask'].any()
    assert (data['actions'][data['valid_mask']]<18).all()
    result=update(model,optimizer_for(model),data,epochs=1,micro_scenes=16,
                  checkpoint_chunks=True,target_kl=None)
    assert result['optimizer_updates']==1
    assert result['unoptimized_student_rows']==0
    assert np.isfinite(result['logs'][0]['loss'])


def model_and_observations():
    from test_candidate_spiking import model as base
    from fly_touhou.compact_story_model import CompactStoryPolicy
    from test_compact_story_model import observations
    m=CompactStoryPolicy(base()).eval();m.bombs_disabled=True;m.pad_candidate_batch=False
    f=observations()
    return m,f


def test_migration_preserves_graph_and_movement_then_learns_extended_ports():
    from fly_touhou.nbnm_model import configure_model
    m,f=model_and_observations();old=copy.deepcopy(m)
    before={n:p.clone() for n,p in m.core.named_parameters()}
    configure_model(m,dict(objective={},risk_mode='route'),reset_critics=False)
    for n,p in before.items():torch.testing.assert_close(dict(m.core.named_parameters())[n],p,rtol=0,atol=0)
    x=torch.cat((f,torch.rand(len(f),152)),1)
    with torch.no_grad():
        a,sa,_=old.step(f,old.initial_state(len(f)))
        b,sb,_=m.step(x,m.initial_state(len(f)))
    torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=1e-5)
    for first,second in zip(sa.tensors(),sb.tensors()):torch.testing.assert_close(first,second,rtol=0,atol=1e-6)
    m.train();out,_,_=m.step(x,m.initial_state(len(f)))
    out['horizon_scores'].square().mean().backward()
    assert m.core.input_encoder.extended_context.weight.grad is not None
    assert m.core.edge_gain_logits.grad is not None
    assert torch.isneginf(out['logits'][:,18]).all()


def test_cached_inference_invalidates_after_update():
    m,f=model_and_observations();core=m.core;core.cache_inference_parameters=True
    with torch.no_grad():
        first=core.dynamic_parameters();same=core.dynamic_parameters()
        assert first[0] is same[0]
        core.edge_gain_logits.add_(.01)
        updated=core.dynamic_parameters()
        assert first[0] is not updated[0]
    core.train();values=core.dynamic_parameters()[0]
    assert values.requires_grad


def test_recurrent_prefix_reset_survives_learner_replay():
    from fly_touhou.nbnm_prefix import replay_resets, apply_resets
    m,_=model_and_observations()
    initial=m.initial_state(2);warm=m.initial_state(1)
    warm.membrane.fill_(.25)
    saved={(17,4):warm}
    reset=replay_resets(saved,[(16,4),(32,5)],16,'cpu')
    assert set(reset)=={1}
    result=apply_resets(initial,reset[1])
    assert torch.all(result.membrane[0]==.25) and torch.all(result.membrane[1]==0)


def test_new_schema_names_and_candidate_dimensions():
    from fly_touhou.nbnm_sensation import NBNMEncoder,SCHEMA,candidate_extension
    from fly_touhou.thrl_inputs import observation_shapes
    assert len(NBNMEncoder.feature_names)==281
    assert observation_shapes(SCHEMA)=={'features':(281,)}
    assert candidate_extension(torch.zeros(16,2,281)).shape==(16,2,18,16)


def test_all_direction_teacher_has_no_vertical_standing_bias():
    from fly_touhou.nbnm_teacher import label_current
    encoding=dict(details=dict(margins=np.full((3,18),30.),trusted=np.ones(3,bool),normal=True))
    s=snapshot();s.stage_frame=100
    p=np.asarray(label_current(encoding,s,1)['probabilities'])
    np.testing.assert_allclose(p[:18],np.ones(18)/18)
    assert p[18]==0


def test_practice_replay_ignores_only_unavailable_inventory_and_visual_rng():
    from copy import deepcopy
    from fly_touhou.nbnm_replay_state import practice_variation
    reference=dict(bombs=4,lives=3,items=[1],bullets=[2],score=500,
        rng=dict(gameplay_seed=3,gameplay_counter=100,visual_seed=2,visual_counter=7))
    actual=deepcopy(reference);actual['bombs']=3;actual['rng']['visual_seed']=8
    assert practice_variation(reference,actual)==[]
    for key in ('gameplay_seed','gameplay_counter'):
        changed=deepcopy(actual);changed['rng'][key]+=1
        assert practice_variation(reference,changed)==['rng']
    for key in ('lives','bullets','score','items'):
        changed=deepcopy(actual);changed[key]=None
        assert practice_variation(reference,changed)==[key]
    actual['items']=[9]
    assert practice_variation(reference,actual,loading=True)==[]
