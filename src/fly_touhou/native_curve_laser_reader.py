"""Read TH14 curve-node collision geometry at the existing game checkpoint."""
from __future__ import annotations
import struct

SCHEMA='th14-native-curve-laser-geometry-v1'


def read_native_curve_lasers(memory,snapshot,*,player_address,offsets):
    """Capture the fields consumed by native curve collision, not future motion.

    Nodes are a linear array; node +0x0c is intentionally not interpreted as
    current velocity. Caller performs the surrounding same-frame snapshot read.
    """
    from .native_curve_laser_motion import read_curve_motion_records
    half_x,half_y=struct.unpack('<ff',memory.read(player_address+0x648,8))
    base=offsets['laser_base'];curve=offsets['laser_curve'];node=offsets['laser_curve_node']
    rows=[]
    for laser in snapshot.lasers:
        if laser.kind!=2 or not laser.nodes:continue
        derived=laser.id+base['zLaserBaseClass_len']
        count=memory.integer(derived+curve['zLaserCurve_max_length'])
        array=memory.integer(derived+curve['zLaserCurve_array'])
        data=memory.read(array,count*node['zLaserCurveNode_size']) if array and count else b''
        nodes=[]
        for i in range(count if array else 0):
            start=i*node['zLaserCurveNode_size']
            x,y=struct.unpack_from('<ff',data,start+node['zLaserCurveNode_pos'])
            angle=struct.unpack_from('<f',data,start+node['zLaserCurveNode_angle'])[0]
            length=struct.unpack_from('<f',data,start+node['zLaserCurveNode_speed'])[0]
            nodes.append((x,y,angle,length))
        rows.append(dict(id=laser.id,width=laser.width,native_state=laser.native_state,
            native_iframes=laser.native_iframes,curve_time=memory.floating(laser.id+0x48),nodes=nodes,
            motion_records=read_curve_motion_records(memory,laser.id)))
    return dict(schema=SCHEMA,frame=snapshot.frame,player_half_extents=[half_x,half_y],curves=rows)
