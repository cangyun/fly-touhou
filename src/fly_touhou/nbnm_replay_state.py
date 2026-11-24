"""Compare gameplay state without comparing per-process native addresses.

Keep positions, geometry, timers, velocities, HP, flags, forecasts and RNG.
Predictor version labels are implementation metadata, not native game state.
Instruction offsets share bytecode allocations. Interrupt target strings do
not: each target may be allocated independently, so retain target presence
and the native HP/time gates, never differences between their addresses.
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
    # Old native start banks never recorded the new instruction capture.
    # Compare their original measurements exactly. When BOTH sides recorded
    # EX semantics, also require the pending bytecode and simulation scalars.
    import json
    from .nbnm_geometry_revision import canonical_hazard
    for state in (first,second):
        for row in state.get('bullets') or ():
            if isinstance(row,dict) and 'hazard' in row:row['hazard']=canonical_hazard(row['hazard'])
    versions=[any(isinstance(r,dict) and 'live_ex_semantics' in r.get('hazard',{}) for r in (s.get('bullets') or ())) for s in (first,second)]
    if versions[0]!=versions[1]:
        for state in (first,second):
            for row in state.get('bullets') or ():
                if isinstance(row,dict):row.get('hazard',{}).pop('live_ex_semantics',None)
    for state in (first,second):
        if isinstance(state.get('bullets'),list):state['bullets']=sorted(state['bullets'],key=lambda row:json.dumps(row,sort_keys=True))
    allowed = ('bombs', 'items') if loading else ('bombs',)
    for key in allowed:
        first.pop(key, None); second.pop(key, None)
    for row in (first, second):
        for key in ('visual_seed', 'visual_counter'):
            row.get('rng', {}).pop(key, None)
    return [k for k in sorted(set(first) | set(second)) if first.get(k) != second.get(k)]


def gameplay_state(state):
    result=deepcopy(state)
    from .nbnm_geometry_revision import canonical_hazard
    for row in result.get('bullets',()):
        if isinstance(row,dict) and 'hazard' in row:row['hazard']=canonical_hazard(row['hazard'])
    # Interrupt target references need not even share an allocation with one
    # another. A common-base subtraction was the cause of failed live resumes.
    code_keys={'instruction_pointer','stopped_instruction_pointer','instruction'}
    interrupt_keys={'time_sub','hp_sub'}
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
                    elif k in interrupt_keys:continue
                    else:gather(v)
            elif isinstance(x,list):
                for v in x:gather(v)
        for root in roots:gather(root)
        base=min(pointers) if pointers else 0
        def normalize(x):
            if isinstance(x,dict):
                for k,v in x.items():
                    if k in code_keys and isinstance(v,int):x[k]=v-base if v else None
                    elif k in interrupt_keys and isinstance(v,int):x[k]=True if v else None
                    elif k in address_keys and isinstance(v,int):x[k]=bool(v)
                    elif k=='id' and isinstance(v,int):x[k]=bool(v)  # nested native animation identity
                    else:normalize(v)
            elif isinstance(x,list):
                for v in x:normalize(v)
        for root in roots:normalize(root)
    return result


def upgrade_continuation_interrupts(state):
    """Read existing v1 capsules without discarding their saved game progress.

    v1 already replaced null pointers by None; numeric zero was a *present*
    target at the minimum address, not null. New capsules use True/None.
    This migration is only for normalized continuation states, not raw states.
    """
    result = deepcopy(state)
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('hp_sub', 'time_sub'):
                    value[key] = None if item is None else True
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(result.get('enemies', []))
    return result
