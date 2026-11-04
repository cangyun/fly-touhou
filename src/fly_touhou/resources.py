"""Read-only original TH14 items, attack availability and resource inventories.

Extra fields are versioned inside Snapshot.extra, preserving old raw traces.
Offsets and association names are inherited from the pinned TH14 profile and
ParaKit extraction source; no item, power, damage or inventory is written.
"""
from __future__ import annotations

from dataclasses import replace
import ctypes as C
import math
import struct
import numpy as np

from .native import NativeError, NativeReader
from .state import Enemy

RESOURCE_SCHEMA = 'original-th14-resources-v1'
ITEM_TYPES = {1:'power',2:'point',3:'big_power',4:'life_piece',5:'life',
              6:'bomb_piece',7:'bomb',8:'full_power',9:'green',10:'cancel',
              11:'big_cancel',12:'undifferentiated_piece'}
POWER_KINDS = frozenset((1,3,8))
SURVIVAL_KINDS = frozenset((4,5,6,7))


class ResourceNativeReader(NativeReader):
    def _resource_items(self):
        offsets = self.o.get('items')
        if not offsets:
            raise NativeError('Resource observations require pinned item offsets')
        manager = self.pointer('items','item_manager_pointer')
        if not manager:
            return [],0,[]
        count = self.mem.integer(manager+offsets['zItemManager_item_cnt'])
        length,stride = offsets['zItemManager_array_len'],offsets['zItem_len']
        if not (0 <= count <= length and 0 < length <= 8192 and 0 < stride <= 8192):
            raise NativeError('Invalid native item array metadata')
        if count == 0:
            return [],0,[]
        # One bounded bulk read avoids thousands of separate kernel calls.
        start = manager+offsets['zItemManager_array']
        size=length*stride
        if size>16*1024*1024:raise NativeError('Native item array exceeds the bounded read limit')
        if callable(getattr(self.mem,'read_into',None)):
            # TH14's pinned table is 14,538,816 bytes, mostly unused animation
            # fields. Reuse storage and decode directly instead of clearing a
            # fresh allocation and copying all bytes into an immutable object.
            data=getattr(self,'_item_read_buffer',None)
            if data is None or C.sizeof(data)!=size:
                data=self._item_read_buffer=C.create_string_buffer(size)
            self.mem.read_into(start,data)
        else:
            data=self.mem.read(start,size)
        items,unknown = [],set()
        states=np.ndarray((length,),dtype='<u4',buffer=data,
            offset=offsets['zItem_state'],strides=(stride,))
        for index in np.flatnonzero(states):
            offset = int(index)*stride
            state = int(states[index])
            kind = struct.unpack_from('<I',data,offset+offsets['zItem_type'])[0]
            if kind not in ITEM_TYPES:
                unknown.add(kind)
                continue
            x,y = struct.unpack_from('<ff',data,offset+offsets['zItem_pos'])
            vx,vy = struct.unpack_from('<ff',data,offset+offsets['zItem_vel'])
            if not all(math.isfinite(value) for value in (x,y,vx,vy)):
                raise NativeError(f'Non-finite item geometry at {start+offset:#x}')
            age = struct.unpack_from('<i',data,offset+offsets['zItem_timer'])[0]
            items.append({'id':start+offset,'x':x,'y':y,'vx':vx,'vy':vy,
                'kind':kind,'kind_name':ITEM_TYPES[kind],'state':state,'age':age,
                'autocollecting':state==offsets['zItemState_autocollect'],
                'attracted':state==offsets['zItemState_attracted']})
        return items,count,sorted(unknown)

    def _resource_enemies(self,snapshot,*,entities=None):
        offsets = self.o['enemies']
        native_forecast = (self.profile.hooks or {}).get('native_enemy_forecast') == 'th14-native-enemy-v1'
        enemies = []
        for enemy in snapshot.enemies if entities is None else entities:
            address = enemy.id+offsets['zEnemy_data']
            flags = self.mem.integer(address+offsets['zEnemyData_flags'],size=8)
            vx,vy = self.mem.pair(address+offsets['zEnemyData_vel'])
            if not all(math.isfinite(v) for v in (vx,vy)):
                raise NativeError('Non-finite enemy velocity in resource observation')
            intangible = bool(flags & offsets['zEnemyFlags_intangible'])
            row = {'id':enemy.id,'x':enemy.x,'y':enemy.y,'vx':vx,'vy':vy,
                'hp':enemy.hp,'radius_x':enemy.radius_x,'radius_y':enemy.radius_y,
                'boss':enemy.boss,'intangible':intangible,
                'damageable':enemy.hp>0 and not bool(flags & (
                    offsets['zEnemyFlags_no_hurtbox']|offsets['zEnemyFlags_invincible']
                    |offsets['zEnemyFlags_intangible'])),
                'contactable':not bool(flags & (offsets['zEnemyFlags_no_hitbox']
                    |offsets['zEnemyFlags_intangible'])),
                'native_flags':flags}
            if native_forecast:
                from .native_enemy_contact import read_enemy_contact, CONTACT_MASK
                from .native_enemy_motion import read_enemy_motion, predict_enemy_positions
                motion = read_enemy_motion(self.mem, address, forecast_damage_bound=bool(
                    (self.profile.hooks or {}).get('native_enemy_damage_bound')))
                contact = read_enemy_contact(self.mem, enemy.id, address, image_base=self.mem.base,motion_state=motion)
                prediction = predict_enemy_positions(motion)
                timer = contact['contact_timer']['integer']
                # Native contact uses a different timer from damage immunity.
                row.update(contactable=not bool(flags & CONTACT_MASK) and timer is not None and timer <= 0,
                    native_contact=contact, native_motion=motion, native_motion_forecast=prediction,
                    native_shape='rectangle' if flags & 0x1000 else 'circle',
                    stored_vx=vx, stored_vy=vy)
                # Data+78 is BEFORE native clamp. Use the actual next movement
                # for tokens; the full trajectory and script boundary travel
                # separately to the compact geometry calculation.
                if len(prediction['displacements']) > 1:
                    row['vx'], row['vy'] = prediction['displacements'][1][:2]
            enemies.append(row)
        return enemies

    def _hidden_contact_enemies(self,snapshot):
        """Read non-visual native bodies separately from attackable targets.

        NativeReader deliberately skips non-boss enemies without an animation.
        Some have a hitbox and can change their contact gate in ECL. They must
        not become attack targets, but the compact hazard encoder can use their
        actual contact schedule. This is enabled only by the new profile hook.
        """
        offsets=self.o['enemies']
        manager=self.pointer('enemies','enemy_manager_pointer')
        if not manager:return [],0
        visible={enemy.id for enemy in snapshot.enemies}
        hidden=[];unreadable_geometry=0
        for identity in self._entries(self.mem.integer(manager+offsets['zEnemyManager_list']),limit=2048):
            if identity in visible:continue
            address=identity+offsets['zEnemy_data']
            flags=self.mem.integer(address+offsets['zEnemyData_flags'],size=8)
            x,y=self.mem.pair(address+offsets['zEnemyData_pos'])
            width,height=self.mem.pair(address+offsets['zEnemyData_hitbox'])
            if not all(math.isfinite(value) for value in (x,y,width,height)):
                unreadable_geometry+=1
                continue
            hidden.append(Enemy(identity,x,y,self.mem.integer(address+offsets['zEnemyData_hp'],signed=True),
                width,height,bool(flags&offsets['zEnemyFlags_is_boss']),
                bool(flags&offsets['zEnemyFlags_intangible'])))
        return self._resource_enemies(snapshot,entities=hidden),unreadable_geometry

    def read(self,retries=6):
        for _ in range(retries):
            snapshot = super().read(retries=retries)
            if snapshot.player is None:
                return replace(snapshot,extra={**snapshot.extra,'resources':{
                    'schema':RESOURCE_SCHEMA,'observed':False,'items':[],'enemies':[]}})
            items,count,unknown = self._resource_items()
            enemies = self._resource_enemies(snapshot)
            pointer = self.pointer('player','player_pointer')
            flags = self.mem.integer(pointer+self.o['player']['zPlayer_flags'])
            can_shoot = bool(snapshot.player.state==1 and not snapshot.dialogue
                and not flags & self.o['player']['zPlayerFlags_cant_shoot']
                and not snapshot.extra.get('game_thread_flags',0) & 0x4000)
            resources = {'schema':RESOURCE_SCHEMA,'observed':True,
                'items':items,'native_item_count':count,'unknown_item_kinds':unknown,
                'enemies':enemies,'life_pieces':self.static('statics','life_pieces'),
                'bomb_pieces':self.static('statics','bomb_pieces'),
                'poc_line_y':148.0 if snapshot.character==1 else 128.0,
                'power_max':400,'player_can_shoot':can_shoot,'player_flags':flags,
                'source':'pinned TH14 offsets and ParaKit native extraction associations',
                'poc_rule':'native upper-screen entry; Shift is not a separate collect command'}
            if (self.profile.hooks or {}).get('native_hidden_enemy_contacts')=='th14-hidden-enemy-contact-v1':
                if (self.profile.hooks or {}).get('native_enemy_forecast')!='th14-native-enemy-v1':
                    raise NativeError('Hidden enemy contact observations require native enemy forecasts')
                hidden,unreadable=self._hidden_contact_enemies(snapshot)
                resources.update(hidden_enemy_contacts=hidden,
                    hidden_enemy_unreadable_geometry=unreadable,
                    hidden_enemy_observation='th14-hidden-enemy-contact-v1')
            if self.global_frame()==snapshot.frame:
                return replace(snapshot,extra={**snapshot.extra,'resources':resources})
        raise NativeError('Resource read crossed native frames; no partial item observation returned')
