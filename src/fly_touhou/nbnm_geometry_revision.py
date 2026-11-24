"""Version geometry separately from unchanged brain parameters/input width."""
from copy import deepcopy
import numpy as np

REVISION='th14-live-ex-program-v1'


def validate(config):
    if not config:return None
    if config.get('revision')!=REVISION or config.get('horizon') not in (24,96):
        raise ValueError('Use the verified TH14 live program revision with horizon24 or96')
    if set(config)!={'revision','horizon'}:raise ValueError('Unknown native rule geometry option')
    return dict(config)


def schema(config):
    config=validate(config)
    return ('connectome-nbnm-route281-live-ex'+str(config['horizon'])+'-v1' if config
            else 'connectome-nbnm-route281-v1')


def legacy_snapshot(snapshot):
    """Discard only the newly read EX contract, preserving original raw state."""
    result=deepcopy(snapshot)
    metadata=result.extra.get('hazard_metadata',{})
    metadata.pop('native_rule_geometry',None)
    metadata['bullets']=[legacy_hazard(raw) for raw in metadata.get('bullets',())]
    for row in result.extra.get('native_line_lasers',{}).get('lines',()):
        row.pop('ex_program_revision',None);row.pop('ex_state_hex',None)
    for row in result.extra.get('native_curve_lasers',{}).get('curves',()):
        row.get('motion_records',{}).pop('compiled_ex_terminal_supported',None)
        row.get('motion_records',{}).pop('ex_program_hex',None)
    return result


def legacy_hazard(raw):
    keys=raw.get('native_ex_program_legacy_keys')
    if keys is not None:return {k:deepcopy(raw[k]) for k in keys if k in raw}
    return {k:deepcopy(v) for k,v in raw.items() if not k.startswith('native_ex_program')
        and k not in ('native_ex_full_program_start_index','native_ex_style_radii',
                     'native_ex_reaction_selector','native_ex_bounce_global_bounds','native_ex_animation')}


def canonical_hazard(raw):
    if not raw.get('native_ex_program_revision'):return deepcopy(raw)
    # Native timers carry absolute float-pointer fields and the broad block
    # also contains animation handles. Retain program bytes and the actual
    # simulation scalars, never allocator addresses.
    from .native_bullet_program import BASE
    data=bytes.fromhex(raw['native_ex_program_state_hex'])
    offsets=[0xbc0,0xbc4,0xbcc,0xbd0,0xbd8,0xbdc,0xbe0,0xbe8,0xc04,0xc20,0x12b8,0x13bc]
    effects={1:(0xf44,0xf48),4:(0xf8c,0xf90,0xf9c,0xfb0,0xfb4,0xfbc),
        8:(0xfd4,0xfd8,0xfe4,0xfe8,0x1004),16:(0x101c,0x1020,0x102c,0x1030,0x104c,0x1050,0x1054,0x1058),
        64:(0x1074,0x107c,0x1080,0x1094,0x1098,0x10a0),0x80000000:(0x10ac,0x10b0),
        0x1000:(0x1124,0x1128,0x112c),0x200000:(0x1214,0x1218,0x1224,0x1238,0x123c,0x1244),
        0x100:(0x125c,0x1260,0x128c),0x4000000:(0x12ec,0x12f0),
        0x400000:(0x138c,0x1390,0x1394,0x1398,0x13a4,0x13a8,0x13b4,0x13b8)}
    for flag,fields in effects.items():
        if raw.get('native_ex_active_flags',0)&flag:offsets.extend(fields)
    result=legacy_hazard(raw)
    result['live_ex_semantics']=dict(program=raw['native_ex_program_hex'],
        scalars={hex(k):data[k-BASE:k-BASE+4].hex() for k in offsets},
        unit_clocks=raw.get('native_ex_program_unit_clocks'),
        bounce_bounds=raw.get('native_ex_bounce_global_bounds'),style_radii=raw.get('native_ex_style_radii'),
        animation=deepcopy(raw.get('native_ex_animation')))
    return result


def verify_restored_features(env,capsule,observation):
    """A migration must reproduce the old input before using the improved one.

    Re-encoding this one landing state is not a new rollout. Native gameplay
    has already been compared by restore_world; its historical actions and
    the saved SNN/Adam state are retained.
    """
    before=capsule.get('native_rule_geometry');after=validate(env.nbnm.get('native_rule_geometry'))
    if before==after:return None
    if before is not None or after is None:
        raise ValueError('Only legacy to live-EX geometry continuation is implemented')
    encoder=deepcopy(env.encoder)
    encoder.native_rule_geometry=None;encoder.schema=schema(None);encoder._nbnm_cache=None
    legacy=encoder.encode(legacy_snapshot(env.snapshot))['features']
    np.testing.assert_allclose(legacy,capsule['features'],rtol=1e-5,atol=1e-6)
    fresh=np.asarray(observation['features'])
    if fresh.shape!=np.asarray(capsule['features']).shape:
        raise ValueError('Geometry migration must preserve the existing281-column layout')
    return dict(before=before,after=after,legacy_input_reproduced=True,
        changed_columns=int(np.count_nonzero(np.abs(fresh-legacy)>1e-6)),
        old_rollout_relabelled=False,native_world_and_saved_brain_preserved=True)
