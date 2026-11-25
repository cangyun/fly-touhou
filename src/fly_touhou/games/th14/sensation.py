"""Compact111 hazard sensing plus18 nonduplicated full-story task scalars."""
import numpy as np
from ...compact_spiking_sensation import CompactObservationEncoder,FEATURE_NAMES
from ..item_collection import NAMES as COLLECTION_NAMES, augment, input_schema
from .items import collection_state

SCHEMA='connectome-compact129-full-story-v1'
TASK_NAMES=('stage','resources_observed','attack_target_present','attack_target_dx','attack_target_dy','attack_target_hp',
    'attack_target_is_boss','power_item_count','nearest_power_dx','nearest_power_dy','power_centroid_dx','power_centroid_dy',
    'valuable_item_present','valuable_item_dx','valuable_item_dy','valuable_item_is_life','valuable_item_is_bomb','poc_line_offset')


def task_features(snapshot,*,primary_attack_target=False):
    p=snapshot.player;x=np.zeros(18,np.float32);x[0]=snapshot.stage/6
    resources=snapshot.extra.get('resources',{});x[1]=resources.get('observed') is True
    enemies=[e for e in snapshot.enemies if e.hp>0]
    if primary_attack_target:
        # Use the same observed ECL primary identities as the HP reward.
        # Auxiliary boss-flag heads remain collision hazards in the base
        # encoder, but should not displace the main shooting target.
        from ...th14_completion_v3 import ResourceCompletionReward
        primary,error=ResourceCompletionReward._bosses(snapshot)
        enemies=[e for e in enemies if not e.intangible]
        if error is None:
            ids={key[1] for key in primary}
            targets=[e for e in enemies if e.id in ids]
            enemies=targets or [e for e in enemies if not e.boss]
    if enemies:
        target=min(enemies,key=lambda e:(not e.boss,abs(e.x-p.x),abs(e.y-p.y)))
        x[2:7]=[1,(target.x-p.x)/368,(target.y-p.y)/400,target.hp/(target.hp+1000),target.boss]
    items=[i for i in resources.get('items',[]) if i.get('active',True) and not i.get('autocollecting',False) and i.get('state')!=3] if x[1] else []
    pickup_context(x, (p.x, p.y), (368, 400), items,
                   resources.get('poc_line_y') if x[1] else None)
    return np.clip(x,-1,1)


def pickup_context(x, position, span, items, poc_line_y=None):
    """Fill the shared power, life, Bomb and collection-line features."""
    px, py = position
    width, height = span
    powers=[i for i in items if i.get('kind') in (1,3,8)]
    if powers:
        nearest=min(powers,key=lambda i:(i['x']-px)**2+(i['y']-py)**2)
        centre=np.mean([[i['x'],i['y']] for i in powers],axis=0)
        x[7:12]=[len(powers)/(len(powers)+10),(nearest['x']-px)/width,(nearest['y']-py)/height,
                  (centre[0]-px)/width,(centre[1]-py)/height]
    valuable=[i for i in items if i.get('kind') in (4,5,6,7)]
    if valuable:
        i=min(valuable,key=lambda i:(i['x']-px)**2+(i['y']-py)**2)
        x[12:17]=[1,(i['x']-px)/width,(i['y']-py)/height,i['kind'] in (4,5),i['kind'] in (6,7)]
    if poc_line_y is not None:
        x[17]=(poc_line_y-py)/height


class CompactStoryEncoder:
    schema=SCHEMA;feature_shape=(129,);map_shape=None;feature_names=(*FEATURE_NAMES,*TASK_NAMES)
    def __init__(self, *, item_collection_input=False):
        self.base=CompactObservationEncoder();self.last_result=None;self.primary_attack_target=False
        self.item_collection_input = item_collection_input
        if item_collection_input:
            self.schema = input_schema(self.schema, True)
            self.feature_names = (*self.feature_names, *COLLECTION_NAMES)
            self.feature_shape = (len(self.feature_names),)
    def reset(self):self.base.reset();self.last_result=None
    def set_native_episode(self,episode):self.base.set_native_episode(episode)
    def specification(self):
        return dict(schema=self.schema,features=dict(shape=list(self.feature_shape),names=self.feature_names),maps=False,objects=False,
            primary_attack_target=self.primary_attack_target,
            task_source='current native enemies/resources only;no teacher action,RNG,reward or future state',
            inherited_hazards=self.base.specification())
    def encode(self,snapshot):
        base=self.base.encode(snapshot)['features']
        features=np.concatenate((base,task_features(snapshot,primary_attack_target=self.primary_attack_target)))
        if self.item_collection_input:
            features = augment(features, collection_state(snapshot), field_height=400.)
        self.last_result={**self.base.last_result,'features':features}
        return {'features':features.copy()}
