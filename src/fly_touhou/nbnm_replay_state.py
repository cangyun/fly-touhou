"""Compare gameplay state without comparing per-process native addresses.

Keep positions, geometry, timers, velocities, HP, flags, forecasts and RNG.
Predictor version labels are implementation metadata, not native game state.
Normalize ECL instruction/interrupt addresses together, preserving offsets.
"""
from copy import deepcopy


def practice_variation(expected, actual, *, loading=False):
    """Separate non-task loading variation from a changed gameplay trajectory.

    A disabled Bomb's inventory and visual RNG are not NBNM task conditions.
    Native Bomb uses, Misses and point pickups are checked separately at the
    landing. Loading items may move asynchronously, but must match at landing.
    Never ignore gameplay RNG, player physics, hazards, HP or score.
    """
    first, second = deepcopy(expected), deepcopy(actual)
    allowed = ('bombs', 'items') if loading else ('bombs',)
    for key in allowed:
        first.pop(key, None); second.pop(key, None)
    for row in (first, second):
        for key in ('visual_seed', 'visual_counter'):
            row.get('rng', {}).pop(key, None)
    return [k for k in sorted(set(first) | set(second)) if first.get(k) != second.get(k)]


def gameplay_state(state):
    result=deepcopy(state)
    code_keys={'instruction_pointer','stopped_instruction_pointer','instruction','time_sub','hp_sub'}
    address_keys={'enemy_base','data_address','owner_address','context','address','manager','shot_definition'}
    for enemy in result.get('enemies',[]):
        resource=enemy.get('resource',{})
        contact=resource.get('native_contact',{})
        for key in ('image_base','ecl_revision','tick_callback_effects_revision'):
            contact.pop(key,None)
        resource.get('native_motion_forecast',{}).pop('ecl_revision',None)
        ecl=resource.get('native_motion',{}).get('ecl_forecast',{})
        for key in ('revision','tick_callback_effects_revision'):
            ecl.pop(key,None)
        roots=[resource[k] for k in ('native_contact','native_motion','native_motion_forecast') if k in resource]
        pointers=[]
        def gather(x):
            if isinstance(x,dict):
                for k,v in x.items():
                    if k in code_keys and isinstance(v,int) and v:pointers.append(v)
                    else:gather(v)
            elif isinstance(x,list):
                for v in x:gather(v)
        for root in roots:gather(root)
        base=min(pointers) if pointers else 0
        def normalize(x):
            if isinstance(x,dict):
                for k,v in x.items():
                    if k in code_keys and isinstance(v,int):x[k]=v-base if v else None
                    elif k in address_keys and isinstance(v,int):x[k]=bool(v)
                    elif k=='id' and isinstance(v,int):x[k]=bool(v)  # nested native animation identity
                    else:normalize(v)
            elif isinstance(x,list):
                for v in x:normalize(v)
        for root in roots:normalize(root)
    return result
