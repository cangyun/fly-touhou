"""Sound reach sets for not-yet-drawn shared-RNG turn angles.

A per-bullet seed cannot establish the next shared draw. For angle-only
randomness, timing and speed are independent of the draw: enclosing the sum
of travelled lengths covers EVERY possible seed/call interleaving, not a
handful of sampled angles. Stored, already drawn angles still run exactly.
"""
import numpy as np


def random_turn_envelope(raw,horizon,*,player_path=None):
    from .native_bullet_program import decode_program,predict_program
    all_commands=decode_program(bytes.fromhex(raw['native_ex_program_hex']))
    commands=[];seen=set();index=raw['native_ex_index']
    while 0<=index<18 and index not in seen:
        seen.add(index);record=all_commands[index]
        if not record[2]:break
        commands.append(record);index=record[1][0] if record[2]==65536 else index+1
    # These effects preserve draw-independent speed/timing. Edge-dependent
    # dispatch and vector acceleration require a different reach calculation.
    allowed={0,1,2,8,16,128,512,1024,2048,65536,1048576,4194304,33554432,67108864,2147483648}
    if any(r[2]==2 for r in commands) and not (raw.get('native_ex_animation') or {}).get('supported'):return None
    if raw['native_state']!=1 or any(op not in allowed for _,_,op,_,_ in commands):return None
    if raw['native_ex_active_flags']&~(1|8|16|4194304|67108864|2147483648):return None
    if any(op==16 and ints[2] in (1,2,6) for _,ints,op,_,_ in commands) and player_path is None:return None
    nominal=predict_program(raw,horizon,player_path=player_path,rng_seed=0)
    if nominal is None or not nominal['random_draw_frames'] or not all(nominal['known_by_frame']):return None
    first=nominal['random_draw_frames'][0]
    origin=np.asarray(nominal['positions'][first-1])
    radius=np.cumsum(np.asarray(nominal['movement_bound'][first:],float))
    # Cancellation adds a half-step before the final movement update.
    if any(op==1024 for _,_,op,_,_ in commands):radius*=1.5
    # Float32 accumulation roundoff is bounded in addition to path length.
    radius+=1e-4*(np.arange(len(radius))+1)*(1+radius)
    result=dict(nominal,reason='covered_all_shared_rng_turn_angles',exact_center_trajectory=False,
        random_prediction='all_angle_reach_set_not_a_claim_about_shared_rng_order')
    for t,r in enumerate(radius,start=first):
        result['positions'][t]=origin.tolist();result['bounds'][t]=[origin.tolist(),origin.tolist()]
        if 'collision_bounds' in result:result['collision_bounds'][t]=[origin.tolist(),origin.tolist()]
        result['extra_radii'][t]=float(r)
    result.pop('growth_collision',None)
    result.pop('conditional_rng_state',None)
    return result
