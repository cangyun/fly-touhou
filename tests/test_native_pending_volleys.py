import numpy as np
from fly_touhou.native_pending_volleys import _stationary_prefix, volley_clearances
from fly_touhou.native_shot_rules import Shooter


def test_fixed_stage3_volley_applies_at_native_arming_and_expires_with_validated_prefix():
    table=np.array([[10,16,5.,100.,8.5],[16,22,60.,100.,8.5]])
    paths=np.broadcast_to([[5.,100.],[60.,100.]],(24,2,2)).transpose(1,0,2).copy()
    assert np.isposinf(volley_clearances(table,paths,0,9,2.)).all()
    near,far=volley_clearances(table,paths,0,10,2.)
    assert near<0<far
    assert (volley_clearances(table,paths,15,16,2.)<0).all()
    near,far=volley_clearances(table,paths,16,20,2.)
    assert far<0<near
    assert np.isposinf(volley_clearances(table,paths,22,23,2.)).all()


def test_emitter_requires_fixed_aim_and_the_stationary_ex_prefix():
    s=Shooter(style=18,color=3,aim=1,speed=0.,speed2=0.)
    s.ex[:4]=[dict(command=c,condition=f,ints=[k]) for c,f,k in
        [(2,0,1),(1048576,0,1),(1,1,0),(0x80000000,0,10)]]
    assert _stationary_prefix(s)
    s.aim=0
    assert not _stationary_prefix(s)
    s.aim=1;s.ex[3]['condition']=1
    assert not _stationary_prefix(s)


def test_many_stationary_births_match_the_tickwise_reference():
    from fly_touhou.native_pending_volleys import _numpy_volley_clearances
    rng=np.random.default_rng(60)
    birth=rng.integers(3,90,48)
    rows=np.column_stack([birth,birth+6,rng.normal(0,75,(48,2)),rng.uniform(2,12,48)])
    paths=rng.normal(0,60,(18,97,2)).astype(np.float32)
    for first,last in [(0,4),(0,12),(12,24),(24,48),(48,72),(72,96)]:
        np.testing.assert_allclose(volley_clearances(rows,paths,first,last,2.),
            _numpy_volley_clearances(rows,paths,first,last,2.),rtol=1e-13,atol=1e-13)
