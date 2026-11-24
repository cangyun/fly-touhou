import numpy as np
import pytest
from fly_touhou.thrl_movement_teacher import swept_clearances, _numpy_swept_clearances


def test_kernel_matches_numpy_for_turns_walls_ties_and_mixed_hitboxes():
    pytest.importorskip("numba")
    rng=np.random.default_rng(29)
    starts=rng.uniform([-184.,32.],[184.,432.],(35,2))
    starts[:4]=[[-184,32],[184,432],[184,32],[-184,432]]
    velocity=rng.uniform(-6,6,starts.shape)
    velocity[4:8]=[[0,0],[0,4],[-2,0],[3.,3.]]
    pos=rng.uniform([-190,20],[200,440],(23,2))
    motions=rng.uniform(-4,4,pos.shape)
    motions[:4]=velocity[:4]
    radii=rng.uniform(.1,17,len(pos))
    ext=rng.uniform(.1,19,pos.shape)
    for sel in (np.zeros(23,bool),np.ones(23,bool),rng.random(23)>.5):
        args=(starts,velocity,pos,motions,sel,radii,ext,np.array([.75,1.25]))
        for start,dt in [(0.,4.),(3.25,0.),(13.,12.),(37.5,48.),(0.,96.)]:
            kw=dict(initial_time=start,duration=dt)
            np.testing.assert_allclose(swept_clearances(*args,**kw),
                _numpy_swept_clearances(*args,**kw),rtol=1.e-13,atol=1.e-13)


def test_nonfinite_motion_and_per_object_surface_retain_original_behavior():
    starts=np.zeros((18,2))
    pos=np.arange(12,dtype=float).reshape(6,2)
    motion=np.ones_like(pos)
    motion[2,0]=np.nan
    args=(starts,starts,pos,motion,np.ones(6,bool),np.ones(6),motion,np.zeros(2))
    for per_object in (False,True):
        np.testing.assert_equal(swept_clearances(*args,per_object=per_object),
            _numpy_swept_clearances(*args,per_object=per_object))
