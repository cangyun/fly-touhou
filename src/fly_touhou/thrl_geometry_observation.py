"""Reviewed current-state input: independent globals, objects and spatial maps.

Only current native fields enter. Closest-approach ranking is a linear-velocity
heuristic, not a prediction of the game's future script or a teacher action.
Objects are NEVER flattened into global numeric features. The original 273
representation remains available separately as the controlled reference.
"""
from __future__ import annotations

import math
import numpy as np

from .thrl_observation import THRLObservationEncoder, THRL_MAP_SHAPE

GEOMETRY_SCHEMA = 'th14-thrl-semantic-separated-v1'
OBJECT_COUNT = 96
OBJECT_DIM = 21
GLOBAL_FIELDS = ('player_x', 'player_y', 'power', 'lives', 'bombs', 'invincible',
    'character', 'stage', 'difficulty', 'graze', 'score', 'misses', 'point_items',
    'player_circle_component', 'player_bounds_half_x', 'player_bounds_half_y',
    'player_bounds_offset_x', 'player_bounds_offset_y', 'focused', 'deathbomb_timer')
FEATURE_DIM = len(GLOBAL_FIELDS)
OBJECT_FIELDS = ('present', 'bullet', 'enemy', 'line_laser', 'curve_laser', 'item',
    'dx', 'dy', 'vx', 'vy', 'radius_or_half_width', 'half_height', 'circle', 'collidable',
    'boss', 'hp', 'subtype', 'spawn_delay', 'age', 'cancel_protection_counter', 'growth_phase')


def shared_object_candidates(snapshot,*,hp_independent_contacts=False):
    """Same unmodified token construction used by production and passive audit.

    Candidate identity is metadata only; ranking, tie-breaking and clipping do
    not inspect it. This helper requires native geometry/resources, not events.
    """
    if type(hp_independent_contacts) is not bool:raise ValueError('Explicit versioned enemy-contact flag required')
    snapshot.validate()
    # Import lazily: source/shape inspection starts no native process.
    from .native_geometry import collision_geometry
    metadata = snapshot.extra.get('hazard_metadata', {})
    if metadata.get('frame') != snapshot.frame or not isinstance(metadata.get('bullets'), list):
        raise ValueError('Geometry observations require same-frame raw hazard metadata')
    raw = {row['id']: row for row in metadata['bullets']}
    shapes, player = collision_geometry(snapshot, raw)
    if snapshot.player is None or player is None:
        raise ValueError('Geometry observation requires a measured live player')
    geometry = {}
    raw_geometry = {row['id']: row for row in snapshot.extra['native_collision_geometry']['bullets']}
    for b in snapshot.bullets:
        row, g = raw_geometry[b.id], shapes[b.id]
        circle = g['shape'] == 'circle'
        # Native rectangle dimensions before expanding by player bounds.
        scale = float(np.float32(row['native_scale'])) if row['native_flags'] & 0x40 else 1.
        width = 2*g['bullet_radius'] if circle else float(np.float32(np.float32(row['native_hitbox_value'])*np.float32(scale)))
        height = width if circle else float(np.float32(np.float32(row['native_height_value'])*np.float32(scale)))
        geometry[b.id] = {'width': width, 'height': height, 'circle': circle,
            'visible': raw[b.id]['native_state'] in (1,2) and not b.intangible,
            'collidable': bool(row['native_flags'] & 2) and not b.intangible
                and (raw[b.id]['native_state']==1 or
                     (raw[b.id]['native_state']==2 and raw[b.id]['native_timer']>=8)),
            'scale_applies': bool(row['native_flags'] & 0x40), 'scale': row['native_scale']}
    resources=snapshot.extra.get('resources',{})
    if (resources.get('schema')!='original-th14-resources-v1' or resources.get('observed') is not True
            or not isinstance(resources.get('items'),list) or not isinstance(resources.get('enemies'),list)):
        raise ValueError('Measured resources must contain full items/enemies lists')
    rows={int(row['id']):row for row in resources['enemies']}
    if len(rows)!=len(resources['enemies']) or set(rows)!={e.id for e in snapshot.enemies}:
        raise ValueError('Resource enemy identities do not match this full snapshot')
    for e in snapshot.enemies:
        row=rows[e.id]
        THRLObservationEncoder._finite(e.x,e.y,e.hp,row['vx'],row['vy'])
        if (row['x'],row['y'],row['hp'])!=(e.x,e.y,e.hp):
            raise ValueError('Resource enemy measurements do not match the same snapshot')
    p=snapshot.player
    objects = []

    def add(kind, x, y, vx, vy, hx, hy, circle, applies=False, scale=1., collidable=True,
            state=1,age=0,protection=0,animation=0,subtype=0,boss=False,hp=0.,identity=None):
        values = [x,y,vx,vy,hx,hy,scale]
        if not np.isfinite(values).all() or hx < 0 or hy < 0 or scale < 0:
            raise ValueError('Invalid measured object geometry')
        dx,dy = x-p.x,y-p.y
        speed2 = vx*vx+vy*vy
        t = np.clip(-(dx*vx+dy*vy)/speed2, 0, 60) if speed2 > 1e-12 else 0.
        extent = hx if circle else math.hypot(hx,hy)
        rank = max(0.,math.hypot(dx+vx*t,dy+vy*t)-extent)+.1*math.hypot(dx,dy)
        token = [1.] + [float(kind==i) for i in range(5)] + [dx/384,dy/448,vx/12,vy/12,
            hx/128,0. if circle else hy/128,float(circle),float(collidable),float(boss),hp/30000,subtype,
            max(0.,8.-age)/8 if kind==0 and state==2 else 0.,
            age/120,protection/120,float(kind==0 and state==2)]
        group = 'boss' if boss else ('item' if kind==4 else ('enemy' if kind==1 else ('laser' if kind in (2,3) else 'bullet')))
        objects.append((rank, tuple(token), token, group, identity))

    for b in snapshot.bullets:
        if geometry[b.id]['visible']:
            g = geometry[b.id]
            r=raw[b.id]
            add(0,b.x,b.y,b.vx,b.vy,g['width']/2,g['height']/2,g['circle'],
                g['scale_applies'],g['scale'],g['collidable'],r['native_state'],
                r['native_timer'],r['native_cancel_protection_counter'],r['native_animation_gate'],
                abs(b.kind)%256/255,identity=('bullet',b.id))
    rows = {row['id']: row for row in snapshot.extra['resources']['enemies']}
    for e in snapshot.enemies:
        row=rows[e.id]
        # A currently intangible body can be enabled by ECL on the next tick.
        # Preserve its measured identity/shape for the native gate forecast.
        native_body = hp_independent_contacts and 'native_contact' in row
        if native_body or (not e.intangible and (e.hp>0 or (hp_independent_contacts and row['contactable'] is True))):
            add(1,e.x,e.y,row['vx'],row['vy'],max(0,e.radius_x)/2,max(0,e.radius_y)/2,
                row.get('native_shape')=='circle',collidable=bool(row['contactable']),boss=e.boss,hp=e.hp,identity=('enemy',e.id))
    if hp_independent_contacts:
        # These native bodies are absent from Snapshot.enemies because they do
        # not have an animation. Only the contact-aware geometry receives them:
        # they must not become ordinary enemies or attack targets.
        for row in resources.get('hidden_enemy_contacts', ()):
            ticks=row['native_contact']['forecast']['ticks']
            if ticks and all(tick['contactable'] is False for tick in ticks):
                continue
            add(1,row['x'],row['y'],row['vx'],row['vy'],max(0,row['radius_x'])/2,
                max(0,row['radius_y'])/2,row.get('native_shape')=='circle',
                collidable=bool(row['contactable']),boss=False,hp=0.,identity=('enemy',row['id']))
    for laser in snapshot.lasers:
        if not laser.active or laser.width <= 0:
            continue
        points,arc=THRLObservationEncoder._polyline(laser)
        length=float(arc[-1])
        count=min(16,max(2,math.ceil(length/32)+1))
        for sample_index,d in enumerate(np.linspace(0,length,count)):
            add(3 if laser.nodes else 2,float(np.interp(d,arc,points[:,0])),
                float(np.interp(d,arc,points[:,1])),0.,0.,laser.width/2,laser.width/2,True,
                age=laser.age,subtype=laser.kind/3,identity=('laser',laser.id,sample_index))
    for item in snapshot.extra['resources']['items']:
        add(4,item['x'],item['y'],item.get('vx',0.),item.get('vy',0.),0.,0.,False,
            collidable=False,state=item.get('state',1),age=item.get('age',0),subtype=item['kind']/12,identity=('item',item['id']))
    return {'candidates':objects,'geometry':geometry,'player_geometry':player,
            'collision_shapes':shapes,'raw_bullets':raw}


def select_object_candidates(candidates):
    """Frozen v1/v2 selector; retained for historical input/checkpoint semantics."""
    objects=sorted(candidates,key=lambda row:(row[0],row[1]))
    selected=set()
    for kind,quota in (('boss',2),('item',8),('enemy',6),('laser',16)):
        selected.update([i for i,row in enumerate(objects) if row[3]==kind][:quota])
    for i in range(len(objects)):
        if len(selected)>=OBJECT_COUNT:break
        selected.add(i)
    return [objects[index] for index in sorted(selected)]


def select_nonitem_fill_candidates(candidates):
    """Preserve ranking/reserves; surplus items cannot evict non-item objects."""
    objects=sorted(candidates,key=lambda row:(row[0],row[1]))
    selected=set()
    for kind,quota in (('boss',2),('item',8),('enemy',6),('laser',16)):
        selected.update([i for i,row in enumerate(objects) if row[3]==kind][:quota])
    for is_item in (False,True):
        for i,row in enumerate(objects):
            if len(selected)>=OBJECT_COUNT:break
            if (row[3]=='item')==is_item:selected.add(i)
    return [objects[index] for index in sorted(selected)]


class THRLGeometryObservationEncoder:
    feature_shape = (FEATURE_DIM,)
    object_shape = (OBJECT_COUNT, OBJECT_DIM)
    map_shape = THRL_MAP_SHAPE
    feature_names = GLOBAL_FIELDS

    def __init__(self, *, require_events=True, nonitem_first=False,hp_independent_contacts=False):
        if type(nonitem_first) is not bool or type(hp_independent_contacts) is not bool:
            raise ValueError('Explicit object allocation and enemy-contact flags required')
        self.nonitem_first=nonitem_first
        self.hp_independent_contacts=hp_independent_contacts
        self.prefix = THRLObservationEncoder(require_events=require_events)

    def specification(self):
        spec = self.prefix.specification()
        spec.update(schema=GEOMETRY_SCHEMA, checkpoint_compatible_with_v1=False,
                    geometry_required=True, not_encoded=['native RNG', 'future scripts/trajectories',
                    'teacher action risks', 'raw process addresses'])
        spec['features'] = {'shape':[FEATURE_DIM], 'names':list(self.feature_names),
            'dtype':'float32', 'bounds':[-1,1], 'semantics':'independent run/player scalars only',
            'objects_flattened_into_features':False}
        spec['objects'] = {'shape':[OBJECT_COUNT,OBJECT_DIM], 'fields':list(OBJECT_FIELDS),
            'dtype':'float32', 'bounds':[-1,1], 'mask':'column0 present',
            'selection':'reserve up to2 bosses,8 items,6 other enemies,16 laser samples; fill remaining by linear closest approach within60frames +0.1 current distance; semantic ties',
            'count_is_total_entities_not_bullets':True, 'map_has_no_top_k_limit':True}
        if self.nonitem_first:
            spec['objects'].update(selection_revision='nonitem-first-fill-v2',
                selection='reserve up to2 bosses,8 items,6 enemies,16 laser samples; fill non-items by unchanged closest-approach rank before surplus items; output original rank order',
                surplus_items_cannot_displace_nonitems=True)
        if self.hp_independent_contacts:
            spec['objects'].update(enemy_retention='HP>0 targets or measured contactable enemies regardless of HP',
                zero_HP_contactable_enemies_retained=True,HP_is_not_collision_gate=True)
            spec['maps']['enemy_retention']='same HP-independent measured contactability for nonboss enemy footprints;boss geometry remains in object tokens'
        spec['constant_zero_unavailable'] = ['enemy_map_type_id']
        spec['mapping']['player_geometry'] = 'effective native collision components; no redundant raw scale/large flags'
        spec['mapping']['object_motion'] = 'current stored native velocity, not guaranteed future displacement; growth_phase retains state2 motion distinction; laser velocity unavailable/zero'
        spec['maps']['footprints'] = {'bullet': 'measured circle or native axis-aligned rectangle; scaling only with flag0x40',
            'enemy': 'measured contact rectangle', 'laser': 'measured current beam width at sampled centers',
            'player_radius': 'encoded separately, no dilation by player size'}
        return spec

    def encode(self, snapshot):
        prepared=shared_object_candidates(snapshot,hp_independent_contacts=self.hp_independent_contacts)
        geometry,player=prepared['geometry'],prepared['player_geometry']
        observation = self.prefix.encode(snapshot, _geometry=geometry,
            _hp_independent_contacts=self.hp_independent_contacts)
        p = snapshot.player
        base = observation['features']
        player_features = [base[0],base[1],base[8],base[9],base[10],base[11],
            base[12],base[13],base[16],*base[32:36],player['circle_component']/64,
            *(player['bounds_half_extents']/64),*(player['bounds_offset']/64),
            float(p.focused),p.deathbomb_timer/60]
        tokens=np.zeros((OBJECT_COUNT,OBJECT_DIM),np.float32)
        selector=select_nonitem_fill_candidates if self.nonitem_first else select_object_candidates
        for i,row in enumerate(selector(prepared['candidates'])):
            tokens[i]=np.clip(row[2],-1,1)
        features=np.clip(player_features,-1,1).astype(np.float32)
        if features.shape != self.feature_shape or not np.isfinite(features).all():
            raise ValueError('Geometry feature contract failed')
        return {'features': features, 'objects':tokens, 'maps': observation['maps']}
