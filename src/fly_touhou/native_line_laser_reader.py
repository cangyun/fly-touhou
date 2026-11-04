"""Same-checkpoint TH14 line-laser motion fields, including first observation."""
from __future__ import annotations
import struct

SCHEMA='th14-native-line-laser-motion-v1'


def read_native_line_lasers(memory,snapshot,*,offsets,player_address=None):
    """Read the state consumed by 43be20, rather than infer it from scalar speed.

    At 43bf78 a line grows to max_length with a stationary origin. Only on a
    subsequent tick does its origin translate. The same update then clips the
    remaining length against the travel limit. Pending EX instructions remain
    explicit: the ordinary update cannot predict their future interventions.
    """
    lines=[]
    for laser in snapshot.lasers:
        if laser.kind!=0 or laser.nodes:continue
        data=memory.read(laser.id+0x54,0x2c)
        fields=struct.unpack('<11f',data)
        tail=memory.read(laser.id+0x594,0x60)
        ex_index,ex_flags=struct.unpack_from('<II',tail,0)
        next_ex=memory.integer(laser.id+0x5f4+ex_index*0x2c+0x20) if ex_index<18 else 0
        lines.append(dict(id=laser.id,origin=list(fields[:2]),velocity=list(fields[3:5]),
            angle=fields[6],length=fields[7],width=fields[8],speed=fields[9],travelled=fields[10],
            max_length=struct.unpack_from('<f',tail,0x5d0-0x594)[0],
            travel_limit=struct.unpack_from('<f',tail,0x5d8-0x594)[0],
            native_state=laser.native_state,native_iframes=laser.native_iframes,age=laser.age,
            ex_index=ex_index,ex_active_flags=ex_flags,next_ex_type=next_ex))
    speed=memory.floating(memory.base+offsets['statics']['game_speed']) if lines else 1.
    result=dict(schema=SCHEMA,frame=snapshot.frame,game_speed=speed,lines=lines)
    if player_address is not None:
        result['player_half_extents']=list(struct.unpack('<ff',memory.read(player_address+0x648,8)))
    return result
