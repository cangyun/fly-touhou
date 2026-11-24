import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from fly_touhou.nbnm_demonstration_routes import scene_signature,apply_advice,build_bank,DemonstrationRouteGuide,SCHEMA
from fly_touhou.state import Snapshot,Player,Bullet,Laser


def test_scene_signature_observes_position_motion_and_lasers_without_becoming_inputs():
    a=Snapshot(frame=1,stage_frame=1,player=Player(0,400),bullets=(Bullet(1,0,200,1,0,3),))
    b=Snapshot(frame=1,stage_frame=1,player=Player(0,400),bullets=(Bullet(1,0,200,-1,0,3),))
    assert scene_signature(a).shape==(256,)
    assert not np.array_equal(scene_signature(a),scene_signature(b))
    c=Snapshot(frame=1,stage_frame=1,player=Player(0,400),lasers=(Laser(1,0,0,100,1.,200,16,True),))
    assert np.any(scene_signature(c))


def test_successful_demonstration_never_authorizes_a_live_unsafe_move():
    p=np.ones(18)/18;safe=np.ones(18,bool);safe[0]=False;p[0]=0;p/=p.sum()
    advice=dict(available=True,confidence=1.,scores=[10,1]+[0]*16)
    q=apply_advice(p,safe,advice)
    assert q[0]==0 and q[1]>p[1] and np.isclose(q.sum(),1)
    no=np.zeros(18,bool)
    assert np.array_equal(apply_advice(p,no,advice),p)
    assert np.array_equal(apply_advice(p,safe,dict(available=False)),p)


def test_route_bank_requires_complete_game_and_does_not_cross_phase_boundaries(tmp_path):
    capture=tmp_path/'capture';capture.mkdir()
    report=dict(full_game_NBNM=False,Misses=0,Bombs=0,feature_kind='route-matching-only')
    (capture/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):build_bank(capture,tmp_path/'bank')
    report.update(full_game_NBNM=True,phase_names=['s1/road','s1/BossCard1'],
        stages=[dict(file='stage01.npz')],source='example.rpy')
    (capture/'report.json').write_text(json.dumps(report))
    (capture/'source.json').write_text(json.dumps(dict(metadata=dict(stages=[dict(native_seed=50000)]))))
    np.savez(capture/'stage01.npz',frame=np.array([6,18,30,42]),phase=np.array([0,0,1,1]),
        phase_clock=np.array([6,18,0,12]),position=np.array([[0,400],[8,400],[100,200],[120,200]]),
        features=np.zeros((4,3)),scene_signature=np.zeros((4,256)))
    result=build_bank(capture,tmp_path/'bank');assert result['added_neural_features']==0
    guide=DemonstrationRouteGuide(tmp_path/'bank/manifest.json')
    assert np.all(guide.data['waypoints'][0]==[8,400])
    assert guide.data['phase_remaining'][0]==12
    assert result['PPO_rows']==0


def test_merge_requires_success_and_preserves_partial_task_scope(tmp_path):
    from fly_touhou.nbnm_demonstration_routes import merge_verified_sources
    positions=np.array([[0.,400.],[4.,400.],[100.,300.],[108.,300.]])
    np.savez(tmp_path/'teacher.npz',phase=np.array(['s1/Boss1']*2+['s1/BossCard1']*2),
        frame=np.array([10,14,18,22]),clock=np.array([100,104,0,4]),position=positions,
        power=np.ones(4),signature=np.zeros((4,256)))
    record=dict(file='teacher.npz',nbnm_success=True,split='train',native_seed=50000,
        task=dict(mode='stage',stage=1))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(histories=[record,dict(record,nbnm_success=False)])))
    out=merge_verified_sources([], [manifest], tmp_path/'merged')
    assert out['rows']==4 and len(out['sources'])==1
    assert out['all_source_paths_reach_task_NBNM'] and not out['all_source_paths_reach_full_game_NBNM']
    guide=DemonstrationRouteGuide(tmp_path/'merged/manifest.json')
    assert np.all(guide.data['waypoints'][0]==positions[1])
    assert not guide.data['source_full_game'].any()


def test_new_successes_publish_separate_data_and_refresh_live_guide(tmp_path):
    from fly_touhou.nbnm_demonstration_routes import RouteBankRefresh
    np.savez(tmp_path/'teacher.npz',phase=np.array(['s1/road']*2),frame=np.array([10,14]),clock=np.array([10,14]),
        position=np.array([[0.,400.],[4.,400.]]),power=np.ones(2),signature=np.zeros((2,256)))
    record=dict(file='teacher.npz',nbnm_success=True,split='train',native_seed=50000,task=dict(mode='stage',stage=1))
    manifest=tmp_path/'teacher.json';manifest.write_text(json.dumps(dict(histories=[record])))
    refresh=RouteBankRefresh(dict(demonstrations=[],teacher_manifests=[str(manifest)],output=str(tmp_path/'bank')))
    assert refresh.refresh()['rows']==2
    guide=DemonstrationRouteGuide(tmp_path/'bank/manifest.json');old_data=guide.manifest['data']
    assert refresh.refresh() is None
    manifest.write_text(json.dumps(dict(histories=[record,dict(record,native_seed=50001)])))
    assert refresh.refresh()['rows']==4
    guide.queries=63
    # An unknown stage avoids irrelevant geometry work after the refresh.
    guide.query(Snapshot(frame=1,stage=2,stage_frame=1,player=Player(0,400)),{})
    assert guide.manifest['data']!=old_data and len(guide.data['clock'])==4
    assert (tmp_path/'bank'/old_data).exists()  # Existing readers retain a complete old version.


def test_card_practice_transition_cannot_become_story_road_guidance(tmp_path):
    from fly_touhou.nbnm_demonstration_routes import merge_verified_sources
    phases=np.array(['s4/road','s4/BossCard2','s4/BossCard2','s4/road'])
    for name in ('card','stage'):
        np.savez(tmp_path/(name+'.npz'),phase=phases,frame=np.arange(4)*4,
            clock=np.array([1900,0,4,2100]),position=np.zeros((4,2)),
            power=np.ones(4),signature=np.zeros((4,256)))
    common=dict(nbnm_success=True,split='train',native_seed=50000)
    rows=[dict(common,file='card.npz',task=dict(mode='attack',stage=4,attack='BossCard2')),
          dict(common,file='stage.npz',task=dict(mode='stage',stage=4))]
    source=tmp_path/'source.json';source.write_text(json.dumps(dict(histories=rows)))
    result=merge_verified_sources([],[source],tmp_path/'bank')
    guide=DemonstrationRouteGuide(tmp_path/'bank/manifest.json')
    road=guide.data['phase']==result['phases'].index('s4/road')
    assert result['rows']==6 and road.sum()==2
    assert (guide.data['source_id'][road]==1).all()
    assert (guide.data['source_id'][~road]==0).sum()==2


def test_reference_continuity_prevents_small_match_jitter_without_changing_confidence(tmp_path):
    np.savez(tmp_path/'routes.npz',phase=np.zeros(4,dtype=int),clock=np.array([100,100,50,50]),
        position=np.array([[0.,400.],[1.,400.],[0.,400.],[1.,400.]]),
        power=np.ones(4),signature=np.zeros((4,256)),source_id=np.array([0,1,0,1]),
        waypoints=np.zeros((4,4,2)),phase_remaining=np.ones(4)*1000)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA,data='routes.npz',phases=['s4/road'],
        all_source_paths_reach_task_NBNM=True,training_split_only=True)))
    plain=DemonstrationRouteGuide(manifest)
    stable=DemonstrationRouteGuide(manifest,source_switch_penalty=.5)
    encoding={'source_features25':np.zeros(25)}
    def state(x,clock):return Snapshot(frame=clock,stage=4,stage_frame=clock,player=Player(x,400),power=400)
    assert stable.query(state(0,100),encoding)['source_id']==0
    assert plain.query(state(.6,104),encoding)['source_id']==1
    kept=stable.query(state(.6,104),encoding)
    assert kept['source_id']==0 and kept['prior_source_id']==0
    assert kept['confidence']==pytest.approx(np.exp(-(.6/96+4/48)))
    # A new episode/clock reset releases the old reference.
    assert stable.query(state(1,50),encoding)['source_id']==1
    stable.query(state(0,100),encoding)
    # A source outside the existing scene-support radius cannot be retained.
    assert stable.query(state(97,104),encoding)['source_id']==1
    scoped=DemonstrationRouteGuide(manifest,source_switch_penalty=.5,source_switch_stages=[1])
    scoped.query(state(0,100),encoding)
    outside=scoped.query(state(.6,104),encoding)
    assert outside['source_id']==1 and outside['source_switch_penalty']==0
    stable.last_reference=None
    stable.query(state(0,100),encoding)
    saved=stable.export_reference()
    restored=DemonstrationRouteGuide(manifest,source_switch_penalty=.5,source_switch_stages=[4])
    assert restored.restore_reference(saved)
    assert restored.query(state(.6,104),encoding)['source_id']==0
    assert not restored.restore_reference(dict(saved,data='new-bank-generation.npz'))
    assert restored.query(state(.6,104),encoding)['source_id']==1
    from fly_touhou.nbnm_demonstration_routes import guide_for_environment
    settings=dict(enabled=True,manifest=str(manifest))
    env=NS(nbnm={'demonstration_route_teacher':settings})
    before=guide_for_environment(env)
    assert guide_for_environment(env) is before
    settings.update(source_switch_penalty=.5,source_switch_stages=[4])
    after=guide_for_environment(env)
    assert after is not before and after.source_switch_penalty==.5
    assert after.source_switch_stages==(4,)

