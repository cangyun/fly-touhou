"""Phase-conditioned native route memories, not an oracle of future bullets.

Only clean, unprotected, no-Bomb training trajectories enter this atlas. A
query matches phase, native clock, position, power and the current hazard
signature. Its waypoints are advisory; live collision labels remain separate.
"""
from pathlib import Path
import json
import numpy as np

SCHEMA = 'th14-nbnm-native-route-atlas-v1'
WIDTH = 6
NAMES = ('recorded_next_step_agreement', 'waypoint24_alignment',
         'waypoint96_alignment', 'waypoint240_alignment',
         'route_match_confidence', 'recorded_clean_horizon')
HORIZONS = (4, 24, 96, 240)


def phase_progress(snapshot):
    """Native phase clock survives mid-attack starts and does not use RNG."""
    from .nbnm_sensation import current_attack
    from .nbnm_task import attack_key
    attack = attack_key(current_attack(snapshot))
    if attack is None:
        return f's{snapshot.stage}/road', int(snapshot.stage_frame)
    phase = attack[0] + ('Card' if attack[1] else '') + str(attack[2])
    roles = snapshot.extra.get('thrl_events', {}).get('ecl_roles', {})
    # Match the same primary role that current_attack selected. A lingering
    # midboss/helper must not supply the main boss's clock.
    role = next((r for k in ('main_bosses', 'midbosses') for r in roles.get(k, ())
                 if r.get('ecl_sub_name')), None)
    selected_id = role.get('id') if role is not None else None
    for row in snapshot.extra.get('resources', {}).get('enemies', ()):
        if selected_id is not None and row.get('id') == selected_id:
            timer = row.get('native_motion', {}).get('phase_timer', {})
            if 'timer' in timer:
                return f's{snapshot.stage}/{phase}', int(timer['timer'])
    # An absent native clock is not replaced with time since loading a prefix.
    return f's{snapshot.stage}/{phase}', None


def route_windows(rows):
    """Retain only actually executed clean continuations; never bridge a hit.

    Each row's `clean` describes its transition to the following observation.
    A phase boundary ends the horizon. Finishing a resource-based teacher run
    is not evidence of NBNM; these individual intervals are the evidence.
    """
    records = []
    for i, row in enumerate(rows):
        if row['clock'] is None or not row['clean']:
            continue
        end = i
        while end+1 < len(rows) and end-i < 60:
            a, b = rows[end], rows[end+1]
            if (not a['clean'] or b['phase'] != row['phase'] or b['clock'] is None
                    or b['frame']-a['frame'] != 4 or b['clock']-a['clock'] != 4):
                break
            end += 1
        horizon = (end-i)*4
        if horizon < 24:
            continue
        points = [rows[i+h//4]['position'] if h <= horizon else row['position'] for h in HORIZONS]
        records.append(dict(phase=row['phase'], clock=row['clock'], position=row['position'],
            power=row['power'], signature=row['signature'], waypoints=points,
            horizon=horizon, native_seed=row['native_seed']))
    return records


def write_atlas(records, output, *, sources):
    from .thrl_training_mode import seed_split
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError('No native-verified clean route intervals')
    if any(seed_split(r['native_seed']) != 'train' for r in records):
        raise ValueError('Development and test routes cannot enter the training atlas')
    phases = sorted({r['phase'] for r in records})
    phase_ids = {p: i for i, p in enumerate(phases)}
    np.savez_compressed(output/'routes.npz',
        phase=np.array([phase_ids[r['phase']] for r in records], np.int32),
        clock=np.array([r['clock'] for r in records], np.int32),
        position=np.array([r['position'] for r in records], np.float32),
        power=np.array([r['power'] for r in records], np.float32),
        signature=np.array([r['signature'] for r in records], np.float32),
        waypoints=np.array([r['waypoints'] for r in records], np.float32),
        horizon=np.array([r['horizon'] for r in records], np.int16),
        native_seed=np.array([r['native_seed'] for r in records], np.int32))
    coverage = {p: dict(rows=sum(r['phase']==p for r in records),
        max_clean_horizon=max(r['horizon'] for r in records if r['phase']==p)) for p in phases}
    manifest = dict(schema=SCHEMA, data='routes.npz', phases=phases, rows=len(records),
        coverage=coverage, sources=sources, training_split_only=True,
        maximum_horizon=240, guaranteed_future_spawns=False,
        interpretation='Recorded clean routes conditional on a matching live scene, not alternative-action collision labels',
        contains_raw_game_assets=False, network_training=False)
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


class RouteAtlas:
    def __init__(self, manifest, *, phase_landmarks=False):
        path = Path(manifest)
        self.manifest = json.loads(path.read_text(encoding='utf-8'))
        if self.manifest['schema'] != SCHEMA or not self.manifest['training_split_only']:
            raise ValueError('Training-only native route atlas required')
        with np.load(path.parent/self.manifest['data'], allow_pickle=False) as z:
            self.data = {k: z[k].copy() for k in z.files}
        from .thrl_training_mode import seed_split
        if any(seed_split(int(s)) != 'train' for s in np.unique(self.data['native_seed'])):
            raise ValueError('Atlas contains held-out trajectories')
        self.groups = {p: np.flatnonzero(self.data['phase']==i) for i, p in enumerate(self.manifest['phases'])}
        self.phase_landmarks = bool(phase_landmarks)
        from .thrl_movement_teacher import MovementSafetyTeacher
        self.velocities = MovementSafetyTeacher().velocities.copy()

    def query(self, phase, clock, features, *, focused_available=True):
        """All 18 first actions receive advisory scores; none executes here."""
        result = np.zeros((18, WIDTH), np.float32)
        info = dict(available=False, phase=phase, clock=clock, confidence=0.,
            future_spawns_known=False, native_risk_labels=False, teacher_actions=0)
        rows = self.groups.get(phase)
        if clock is None or rows is None:
            return result, info
        d = self.data
        rows = rows[np.abs(d['clock'][rows]-clock) <= 12]
        if not len(rows):
            return result, info
        f = np.asarray(features)
        origin = f[:2]*[368, 400]+[-184, 32]
        distance = np.linalg.norm(d['position'][rows]-origin, axis=1)
        hazard_error = np.sqrt(np.mean((d['signature'][rows]-f[18:72])**2, axis=1))
        power_error = np.abs(d['power'][rows]-f[2])
        valid = (distance <= 48) & (hazard_error <= .22) & (power_error <= .25)
        cost = distance/48 + hazard_error/.22 + power_error/.25 + np.abs(d['clock'][rows]-clock)/24
        cost[~valid] = np.inf
        k = int(np.argmin(cost)); row = int(rows[k])
        confidence = float(np.exp(-cost[k]))
        local = bool(confidence >= .15)
        # No claim that a different seed follows exactly the stored route.
        if not local:
            if not self.phase_landmarks:
                return result, info
            # A distant player's exact next action is inapplicable. At most
            # use a weak stage-timing landmark, explicitly not a safe route
            # from the current position. Live planning handles that path.
            valid=(power_error<=.25) & (d['horizon'][rows]>=96)
            if not valid.any():return result, info
            cost=distance/192+hazard_error/.5+power_error/.25+np.abs(d['clock'][rows]-clock)/24
            cost[~valid]=np.inf;k=int(np.argmin(cost));row=int(rows[k])
            confidence=float(.1/(1+cost[k]))
        velocity = self.velocities.copy()
        if not focused_available:
            velocity[9:] = velocity[:9]
        endpoints = np.clip(origin+velocity*4, [-184, 32], [184, 432])
        goals = d['waypoints'][row]
        if local:
            result[:, 0] = 1-2*np.minimum(1., np.linalg.norm(endpoints-goals[0], axis=1)/36)
        for j, h in enumerate(HORIZONS[1:], 1):
            if d['horizon'][row] >= h:
                delta = goals[j]-origin
                distance0 = np.linalg.norm(delta)
                result[:, j] = np.clip((distance0-np.linalg.norm(goals[j]-endpoints, axis=1))/18, -1, 1)
        result[:, 4] = confidence
        result[:, 5] = d['horizon'][row]/240
        info.update(available=True, confidence=confidence, matched_row=row,
            match_kind='local_continuation' if local else 'phase_landmark',
            current_path_verified=False,
            clean_horizon=int(d['horizon'][row]), position_error=float(distance[k]),
            hazard_signature_error=float(hazard_error[k]), waypoints=goals.tolist(),
            # Broad routes still defer to live one- and two-turn geometry.
            scores=(result[:,0]+.25*result[:,1]+.15*result[:,2]+.1*result[:,3]).tolist())
        return result, info


def guide_with_atlas(probabilities, safe, prior):
    """A soft movement target, not a safety label or execution override."""
    p = np.asarray(probabilities, float).copy()
    if not prior or not prior.get('available') or not np.any(safe):
        return p
    scores = np.asarray(prior['scores'], float)
    scores -= scores[safe].max()
    q = np.where(safe, np.exp(np.clip(3*scores, -60, 0)), 0.)
    q /= q.sum()
    weight = .5*prior['confidence']
    return (1-weight)*p + weight*q
