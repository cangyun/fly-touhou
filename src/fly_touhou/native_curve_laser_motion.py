"""Read TH14's curve trajectory records; no forecast is certified here.

440910 chooses a trajectory record for each node time. 440f60 evaluates
the leading node, with modes0/1/2 for linear/vector/polar motion. Keeping
the actual records avoids inferring an unseen future program from a few
past node positions.
"""
from __future__ import annotations

import math
import struct

import numpy as np

REVISION = 'native-curve-trajectory-records-v1'


def read_curve_motion_records(memory, laser_address):
    """Optional same-checkpoint telemetry. Existing collision inputs stay intact."""
    try:
        header = memory.read(laser_address + 0x594, 8)
        ex_index, active_flags = struct.unpack('<iI', header)
        pending = memory.read(laser_address + 0x5ec + ex_index * 0x2c, 0x2c) if 0 <= ex_index < 18 else None
        initial = memory.read(laser_address + 0x5c0, 0x18)
        clock_pointer = memory.integer(laser_address + 0x4c)
        rate = memory.floating(clock_pointer) if clock_pointer else 1.
        number = lambda value: value if math.isfinite(value) else None
        records = []
        address = laser_address + 0x14b4
        seen = set()
        while address and address not in seen and len(records) < 64:
            seen.add(address)
            data = memory.read(address, 0x3c)
            following, previous = struct.unpack_from('<II', data)
            start, end = struct.unpack_from('<ff', data, 8)
            records.append(dict(address=address, previous=previous, start=number(start), end=number(end),
                mode=struct.unpack_from('<i', data, 0x10)[0],
                unit_vector=[number(x) for x in struct.unpack_from('<fff', data, 0x14)],
                origin=[number(x) for x in struct.unpack_from('<fff', data, 0x20)],
                angle=number(struct.unpack_from('<f', data, 0x2c)[0]),
                speed=number(struct.unpack_from('<f', data, 0x30)[0]),
                speed_delta=number(struct.unpack_from('<f', data, 0x34)[0]),
                angle_delta=number(struct.unpack_from('<f', data, 0x38)[0]), raw_hex=data.hex()))
            address = following
        return dict(revision=REVISION, available=True, record_chain_complete=address == 0,
            clock_rate=number(rate), ex_index=ex_index, active_ex_flags=active_flags,
            pending_ex_record_hex=pending.hex() if pending is not None else None,
            initial_position=[number(x) for x in struct.unpack_from('<fff', initial)],
            initial_angle=number(struct.unpack_from('<f', initial, 0xc)[0]),
            initial_speed=number(struct.unpack_from('<f', initial, 0x14)[0]), records=records,
            forecast_enabled=False)
    except Exception as error:
        # This is additional diagnostic data, not a prerequisite for the
        # already working native collision reader or the running evaluation.
        return dict(revision=REVISION, available=False, error=repr(error), forecast_enabled=False)


def trajectory_record_nodes(record, times):
    """Offline transcription of 440f60 modes0 and2; mode1 stays unsupported."""
    times=np.asarray(times,dtype=np.float32)
    dt=(times-np.float32(record['start'])).astype(np.float32)
    if np.any(dt<0):raise ValueError('Record evaluated before its native start')
    origin=np.asarray(record['origin'],dtype=np.float32)
    angle=np.float32(record['angle']);speed=np.float32(record['speed'])
    if record['mode']==0:
        unit=np.asarray(record['unit_vector'],dtype=np.float32)
        position=(unit[None,:]*dt[:,None]*speed+origin).astype(np.float32)
        return np.column_stack((position[:,:2],np.full(len(times),angle),np.full(len(times),speed)))
    if record['mode']!=2:raise ValueError('Unimplemented native trajectory mode')
    da=np.float32(record['angle_delta']);ds=np.float32(record['speed_delta'])
    steps=dt.astype(np.int64)
    if steps.max(initial=0)>10000:raise ValueError('Offline trajectory prefix exceeds10000 ticks')
    positions=[origin.copy()];angles=[angle];speeds=[speed]
    pi=np.float32(math.pi);tau=np.float32(2*math.pi)
    for _ in range(int(steps.max(initial=0))):
        velocity=np.array((math.cos(float(angle))*float(speed),math.sin(float(angle))*float(speed),0.),np.float32)
        origin=(origin+velocity).astype(np.float32)
        angle=np.float32(angle+da)
        while angle>pi:angle=np.float32(angle-tau)
        while angle<-pi:angle=np.float32(angle+tau)
        speed=np.float32(speed+ds)
        positions.append(origin.copy());angles.append(angle);speeds.append(speed)
    positions=np.asarray(positions);angles=np.asarray(angles);speeds=np.asarray(speeds)
    angles=angles[steps];speeds=speeds[steps]
    velocity=np.column_stack((np.cos(angles.astype(float))*speeds,np.sin(angles.astype(float))*speeds)).astype(np.float32)
    position=(positions[steps,:2]+velocity*(dt-steps.astype(np.float32))[:,None]).astype(np.float32)
    return np.column_stack((position,angles,speeds))


def curve_node_sequence(curve, ahead):
    """Offline geometry candidate; does not enable trust or change live inputs.

    The native clock increments after creating the node array. Each following
    node samples one earlier trajectory tick; negative times retain the
    initial shot position/angle/length.
    """
    motion=curve.get('motion_records') or {}
    if not motion.get('available') or not motion.get('record_chain_complete'):
        raise ValueError('Complete original trajectory records required')
    if motion.get('clock_rate')!=1.:raise ValueError('Only unit native time is transcribed')
    count=len(curve['nodes']);ahead=np.atleast_1d(ahead).astype(np.float32)
    times=np.float32(curve['curve_time']-1)+ahead[:,None]-np.arange(count,dtype=np.float32)[None,:]
    initial=[*motion['initial_position'][:2],motion['initial_angle'],motion['initial_speed']]
    result=np.broadcast_to(np.asarray(initial,np.float32),(len(ahead),count,4)).copy()
    indices=np.full(times.shape,-1,np.int32)
    for j,record in enumerate(motion['records']):
        if record['start'] is not None and record['end'] is not None:
            mask=(indices<0)&(times>=record['start'])&(times<record['end'])
            indices[mask]=j
    if np.any((times>=0)&(indices<0)):raise ValueError('Some native node times have no captured trajectory record')
    pi=np.float32(math.pi);tau=np.float32(2*math.pi)
    for i in range(count):
        for j in np.unique(indices[:,i]):
            if j<0:continue
            mask=(indices[:,i]==j)&(times[:,i]>=0)
            if not mask.any():continue
            record=motion['records'][j]
            if i==0:
                result[mask,i]=trajectory_record_nodes(record,times[mask,i])
            elif record['mode']==0:
                velocity=(np.asarray(record['unit_vector'][:2],np.float32)*np.float32(record['speed'])).astype(np.float32)
                result[mask,i,:2]=result[mask,i-1,:2]-velocity
                result[mask,i,2:]=record['angle'],record['speed']
            elif record['mode']==2:
                # 441430 walks backward from the preceding freshly updated
                # node; batch independent ticks while keeping this recurrence.
                fraction=(times[mask,i]-np.floor(times[mask,i])).astype(np.float32)
                previous=result[mask,i-1]
                angle64=previous[:,2].astype(float)
                velocity=(np.column_stack((np.cos(angle64),np.sin(angle64)))*previous[:,3,None]).astype(np.float32)
                position=(previous[:,:2]-fraction[:,None]*velocity).astype(np.float32)
                angle=(previous[:,2]-np.float32(record['angle_delta'])).astype(np.float32)
                while np.any(angle>pi):angle=np.where(angle>pi,angle-tau,angle).astype(np.float32)
                while np.any(angle<-pi):angle=np.where(angle<-pi,angle+tau,angle).astype(np.float32)
                speed=(previous[:,3]-np.float32(record['speed_delta'])).astype(np.float32)
                angle64=angle.astype(float)
                velocity=(np.column_stack((np.cos(angle64),np.sin(angle64)))*speed[:,None]).astype(np.float32)
                result[mask,i,:2]=position-(np.float32(1.)-fraction[:,None])*velocity
                result[mask,i,2]=angle;result[mask,i,3]=speed
            else:raise ValueError('Unimplemented native trajectory mode')
    if not np.isfinite(result).all():raise ValueError('Nonfinite native trajectory candidate')
    return result


def curve_nodes_from_records(curve, *, ahead=0):
    return curve_node_sequence(curve,[ahead])[0]
