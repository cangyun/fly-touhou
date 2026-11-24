"""Resolve original TH14 actor geometry on the same movement and firing tape."""

import numpy as np

from .native_enemy_contact import predict_contact_ticks, read_enemy_contact
from .native_enemy_ecl import CONTACT_OPS, CALLBACK_EFFECTS_REVISION, REVISION as ECL_REVISION
from .native_enemy_motion import _phase_boundary
from .native_player_damage_forecast import UnsupportedNativeFire, forecast_continuous_fire, targets_along_actor_tracks
from .native_shot_rules import forecast_emissions


def capture_contact_origins(reader, states, horizon):
    """Read native shape, flags and timers without guessing any future ECL branch."""
    original_script = dict(
        revision=ECL_REVISION, horizon=horizon, events=[], threads=[],
        tick_callback_effects_revision=CALLBACK_EFFECTS_REVISION,
    )
    return {
        state['owner_address']: read_enemy_contact(
            reader.mem, state['owner_address'], state['data_address'],
            image_base=state['image_base'], horizon=horizon,
            motion_state=dict(state, ecl_forecast=original_script),
        )
        for state in states
    }


def _emissions(reader, context, paths, rng, horizon, **kwargs):
    return forecast_emissions(
        reader.mem, context['states'], paths, horizon=horizon,
        seed=rng['gameplay_seed'], counter=rng['gameplay_counter'],
        capture_actor_tracks=True, **kwargs,
    )


def _damage_agreement_frames(initial, corrected, lifetimes, horizon):
    common = horizon
    for previous, revised in zip(initial, corrected, strict=True):
        keep = min(horizon, lifetimes.get(previous['owner'], horizon)) + 1
        for key in ('positions', 'damage_flags_by_tick', 'damage_dimensions_by_tick'):
            matched = 0
            for before, after in zip(previous[key][:keep], revised[key][:keep]):
                if before != after:
                    break
                matched += 1
            if matched < keep:
                common = min(common, matched - 1)
    return common


def replay_candidate_actors(reader, context, paths, courses, rng):
    """Couple path-specific ECL and player damage for courses of one length."""
    horizon = len(paths[0]) - 1
    results, supported = [], []
    initial = _emissions(reader, context, paths, rng, horizon)
    for path, course, forecast in zip(paths, courses, initial, strict=True):
        row = {}
        results.append(row)
        targets = targets_along_actor_tracks(context['targets'], forecast['actor_tracks'], horizon)
        try:
            life, _ = forecast_continuous_fire(
                context['initial'], targets, context['viewport'], path, course, require_sprites=True,
            )
        except UnsupportedNativeFire as error:
            row['reason'] = str(error)
            continue
        removed = {**life['edge_removals'], **life['defeat_ticks']}
        keys = {target['owner']: target['key'] for target in targets}
        ends = {owner: removed[key] for owner, key in keys.items() if key in removed}
        row.update(life=life, ends=ends, keys=keys)
        supported.append((path, targets, row))
    if not supported:
        return results
    complete = _emissions(reader, context, [path for path, _, _ in supported], rng, horizon,
        actor_last_ticks=[row['ends'] for _, _, row in supported],
        actor_rng_draws=[row['life']['gameplay_draws_after_actor'] for _, _, row in supported],
    )
    for (_, targets, row), forecast in zip(supported, complete, strict=True):
        joined = targets_along_actor_tracks(context['targets'], forecast['actor_tracks'], horizon)
        row['forecast'] = forecast
        row['coherent_through'] = _damage_agreement_frames(targets, joined, row['ends'], horizon)
    return results


def _uncovered_phase(state, actor, damage, key, until):
    """Only certify ticks governed by the captured attack's original program."""
    check = dict(
        state, damage_bound_enabled=True,
        damage_bound=dict(per_frame=0, source='candidate exact incoming shots'),
        ecl_forecast=dict(damage_phase_change=actor.get('damage_phase_change')),
    )
    boundary, reason, phase = _phase_boundary(check, horizon=until)
    if boundary is not None:
        return dict(reason=reason, tick=boundary)
    trigger = phase['hp_interrupt']
    if trigger is None:
        return None
    if key in damage['defeat_ticks'] and damage['defeat_ticks'][key] <= until:
        return dict(reason='candidate_defeats_phase_actor', tick=damage['defeat_ticks'][key])
    for item in damage['rows'][:until]:
        if item['health'].get(key, trigger['hp'] + 1) <= trigger['hp']:
            return dict(reason='candidate_reaches_phase_hp_threshold', tick=item['tick'])
    return None


def conditional_body_encoding(encoding, context, origins, combined, horizon):
    """Replace covered native bodies while retaining every captured projectile."""
    from .compact_spiking_sensation import encode_geometry

    if combined['coherent_through'] < horizon:
        return None, dict(reason='native_firing_and_actor_motion_disagree',
                          coherent_through=combined['coherent_through'])
    scene = dict(encoding['source_native_forecasts'])
    original_bodies = {row['native_id']: row for row in scene.get('enemies', ())}
    tracks = combined['forecast']['actor_tracks']
    bodies = []
    for state in context['states']:
        owner = state['owner_address']
        end = min(horizon, combined['ends'].get(owner, horizon))
        actor = tracks.get(owner)
        key = combined['keys'][owner]
        if not actor or len(actor['samples']) <= end:
            return None, dict(reason='incomplete_body_motion', actor=key)
        boundary = _uncovered_phase(state, actor, combined['life'], key, end)
        if boundary:
            return None, dict(boundary, actor=key)
        initial = origins[owner]
        contact = dict(initial, ecl_forecast_horizon=end,
                       ecl_events=[event for event in actor['events'] if event['opcode'] in CONTACT_OPS],
                       ecl_threads=actor['threads'])
        forecast = predict_contact_ticks(contact, end)
        ticks = forecast['ticks']
        unknown = next((tick for tick in ticks if not tick['known']
                        or (tick['contactable'] and not tick['geometry_known'])), None)
        if unknown is not None:
            return None, dict(reason='body_contact_gate_unknown', actor=key, first=unknown)
        matched = original_bodies.pop(owner, None)
        if matched is None:
            if any(tick['contactable'] for tick in ticks):
                return None, dict(reason='body_not_in_observation', actor=key)
            continue
        positions = [sample['position'] for sample in actor['samples'][:end + 1]]
        if end < horizon:
            positions.extend([positions[-1]] * (horizon - end))
            ticks.extend([dict(known=True, contactable=False)] * (horizon - end))
        motion = dict(matched['motion'], positions=positions, unknown_from_frame=None,
                      phase_unknown_from_frame=None, pending_enemy_spawns=[],
                      reachable_radius_by_frame=np.zeros(horizon + 1))
        bodies.append(dict(matched, motion=motion, contact=dict(contact, forecast=forecast)))

    if original_bodies:
        return None, dict(reason='observed_bodies_not_in_actor_inventory', count=len(original_bodies))
    scene['enemies'] = bodies
    scene['emitter_projection_horizon'] = horizon
    _, details = encode_geometry(
        encoding['source_features25'], encoding['source_objects29'],
        complete=encoding['details']['full_observed_hazards'], anticipate_arming=True,
        growth_motion_envelope=True, unknown_laser_player_motion=True, native_forecasts=scene,
    )
    return dict(encoding, source_native_forecasts=scene, details=details), None
