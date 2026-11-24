"""Upcoming TH14 offset rings, including the original random turn between volleys."""
import math
import struct

import numpy as np

from .native_enemy_ecl import ECLIncomplete, apply_tick_callback_effects
from .native_enemy_motion import ALREADY_UPDATED_FLAG, _step_motion
from .native_shot_rules import ReadCache, RuleWorld, emission_vectors


def _angular_update(world, thread, local):
    """Recognize angle -= constant + signed_random * coefficient in the firing loop."""
    address=thread.ip
    expected=((44,1),(44,0),(44,1),(44,0),(55,0),(51,0),(53,0),(45,1))
    args=[]
    for opcode,mask in expected:
        address+=len(world.instruction(address))
        ins=world.instruction(address)
        if struct.unpack_from('<H',ins,4)[0]!=opcode or struct.unpack_from('<H',ins,8)[0]!=mask:
            return None
        if opcode in (44,45):
            args.append(struct.unpack_from('<f',ins,16)[0])
        if len(args)==3 and opcode==44:
            random_site=address
    variable,step,random,coefficient,destination=args
    if variable!=local or destination!=local or random!=-9987. or not all(map(math.isfinite,(step,coefficient))):
        return None
    return random_site,step,abs(coefficient)


def _native_ring(shooter):
    active=[(i,p) for i,p in enumerate(shooter.ex) if p]
    return (shooter.style==21 and shooter.color==3 and shooter.aim==12 and
        shooter.layers==1 and shooter.radius==0. and len(active)==2 and
        [(i,p['command'],p['condition'],p['ints'][0]) for i,p in active]==[(0,2,0,1),(1,33554432,0,4)])


class _RingWorld(RuleWorld):
    def __init__(self,memory,state,instructions):
        super().__init__(memory,state,player_path=(),rng=None,events=[],shared_instructions=instructions)
        self.local_angle={}
        self.angle_history={}
        self.updates={}

    def global_value(self,variable,kind):
        if variable==-9987 and self.site in self.updates:
            self.updates[self.site]+=1
            return 0.  # Center of the signed draw. Its full angular range is retained below.
        if variable in (-10000,-9999,-9998,-9987,-9991,-9990,-9989):
            raise ECLIncomplete('unresolved ring emitter input')
        return super().global_value(variable,kind)

    def rule_instruction(self,thread,ins,opcode,order):
        self.site=(id(thread),thread.ip)
        if opcode in (613,615,616):
            raise ECLIncomplete('subsequent global projectile change')
        if opcode==604:
            self.local_angle.pop(id(thread),None)
            if struct.unpack_from('<H',ins,8)[0]==2:
                bank,angle,_=struct.unpack_from('<iff',ins,16)
                if 0<=bank<16 and angle>=0:
                    self.local_angle[id(thread)]=(thread.ip+len(ins),bank,angle)
        return super().rule_instruction(thread,ins,opcode,order)

    def emit(self,shooter,kind,thread):
        if kind!='bullet' or not _native_ring(shooter):
            return
        branch=self.local_angle.get(id(thread))
        if branch is None or thread.ip!=branch[0] or self.shooter(branch[1]) is not shooter:
            return
        loop=_angular_update(self,thread,branch[2])
        if loop is None:
            return
        address,step,jitter=loop
        site=id(thread),address
        draws=self.updates.setdefault(site,0)
        previous=self.angle_history.get(site)
        if previous is not None and (draws!=previous[1]+1 or abs(shooter.angle-(previous[0]-step))>2e-5):
            raise ECLIncomplete('ring angle changed outside the recognized local update')
        self.angle_history[site]=(shooter.angle,draws)
        center=self.state['final']['position'][:2] if shooter.absolute is None else shooter.absolute
        origin=np.add(center,shooter.offset,dtype=np.float32)
        radius=struct.unpack('<f',self.memory.read(self.base+0xd1f94+shooter.style*0x114,4))[0]
        if not np.isfinite(origin).all() or not (math.isfinite(radius) and radius>0):
            return
        velocity=np.asarray(emission_vectors(shooter,origin,(0.,0.),None),np.float64)[:,2:]
        if np.isfinite(velocity).all():
            self.future_events.append((self.frame,origin,velocity,draws*jitter,radius))


def ring_discs(events,horizon,player_radius):
    """Enclose every allowed rotation, including both collision checks on the release tick."""
    counts=np.zeros(horizon+1,np.int32)
    for tick,_,velocities,_,_ in events:
        counts[tick+7:]+=len(velocities)
    discs=np.zeros((horizon+1,int(counts.max()),3),np.float64)
    counts.fill(0)
    for tick,origin,velocities,jitter,radius in events:
        first=tick+7
        if first>horizon:
            continue
        times=np.arange(first,horizon+1)
        age=times-tick+1
        travel=np.where(age<=14,.5*age-4.,age-10.5)
        travel[age==15]=4.  # half-speed check at 3.5 and normal-speed check at 4.5
        speed=np.linalg.norm(velocities,axis=1)
        angular=2.*math.sin(.5*min(jitter,math.pi))
        radii=(math.hypot(radius,player_radius)+.05
            +(abs(travel[:,None])*angular+.5*(age[:,None]==15))*speed)
        ids=counts[times,None]+np.arange(len(velocities))
        discs[times[:,None],ids,:2]=origin+travel[:,None,None]*velocities
        discs[times[:,None],ids,2]=radii
        counts[times]+=len(velocities)
    return discs,counts


def ring_arcs(events, horizon, player_radius):
    """Angular randomness rotates a ray. It leaves the speed and radial distance fixed."""
    counts = np.zeros(horizon + 1, np.int32)
    for tick, _, velocities, _, _ in events:
        counts[tick + 7:] += len(velocities)
        if tick + 14 <= horizon:
            counts[tick + 14] += len(velocities)
    arcs = np.empty((horizon + 1, int(counts.max()), 9), np.float64)
    counts.fill(0)
    for tick, origin, velocities, jitter, radius in events:
        times = np.arange(tick + 7, horizon + 1)
        if not len(times):
            continue
        age = times - tick + 1
        travel = np.where(age <= 14, .5 * age - 4., age - 10.5)
        # The original checks 3.5 and 4.5 on the release tick.
        release_tick = tick + 14
        legs = [(times, travel)]
        if release_tick <= horizon:
            legs.append((np.array([release_tick]), np.array([3.5])))
        speed = np.linalg.norm(velocities, axis=1)
        unit = velocities / np.where(speed > 0, speed, 1.)[:, None]
        unit[speed == 0] = [1., 0.]
        angle = min(abs(jitter), math.pi)
        contact = math.hypot(radius, player_radius) + .05
        for frames, distance in legs:
            rows = frames[:, None]
            cols = counts[frames, None] + np.arange(len(velocities))
            length = distance[:, None] * speed
            arcs[rows, cols, :2] = origin + distance[:, None, None] * velocities
            arcs[rows, cols, 2:4] = unit
            arcs[rows, cols, 4] = length
            arcs[rows, cols, 5:8] = [math.cos(angle), math.sin(angle), contact]
            arcs[rows, cols, 8] = contact + 2. * math.sin(angle / 2.) * length
            counts[frames] += len(velocities)
    return arcs, counts


def forecast_rotating_rings(memory,snapshot,phase,horizon,player_radius,*,arc=False):
    from .nbnm_task import attack_key
    if snapshot.stage!=6 or attack_key(phase)!=('Boss',True,3):
        return None
    fields=snapshot.extra.get('resources',{})
    cache=ReadCache(memory)
    events=[]
    instructions={}
    for enemy in fields.get('enemies',()):
        state=enemy.get('native_motion')
        if not enemy.get('boss') or not state or state['effective_game_speed']!=1.:
            continue
        try:
            world=_RingWorld(cache,state,instructions)
        except ECLIncomplete:
            continue
        cutoff=horizon+1
        for tick in range(1,horizon-6):
            if world.state.get('ecl_time_step')!=1. or world.motion_unknown:
                cutoff=tick
                break
            skip=bool(world.state['flags']&ALREADY_UPDATED_FLAG)
            try:
                _step_motion(world.state)
                if not skip:
                    world.tick(tick)
                    apply_tick_callback_effects(world)
            except (ECLIncomplete,NotImplementedError):
                cutoff=tick
                break
        events.extend(e for e in world.future_events if e[0]<cutoff)
    geometry = ring_arcs if arc else ring_discs
    return geometry(events,horizon,player_radius) if events else None


def archive_ring_geometry(rings):
    """Only initialized, active arcs or discs belong in the contact record."""
    data, counts = rings
    return dict(format='original-ring-geometry-v1', width=int(data.shape[-1]),
                frames=[frame[:n].tolist() for frame, n in zip(data, counts)])


def restore_ring_geometry(record):
    width = record['width']
    if record['format'] != 'original-ring-geometry-v1' or width not in (3, 9):
        raise ValueError('Unknown original ring geometry archive')
    frames = record['frames']
    counts = np.asarray([len(frame) for frame in frames], np.int32)
    data = np.empty((len(counts), int(counts.max(initial=0)), width), np.float64)
    for t, frame in enumerate(frames):
        if counts[t]:
            data[t, :counts[t]] = frame
    return data, counts


def ring_clearances(rings,paths,begin,end):
    if isinstance(rings, dict):
        rings = restore_ring_geometry(rings)
    if not 0<=begin<end<paths.shape[1] or end>=len(rings[0]):
        raise ValueError('Ring forecast and reachable path must cover the same native ticks')
    from .swept_clearance_kernel import moving_ring_clearances, moving_arc_clearances
    kernel = moving_arc_clearances if rings[0].shape[-1] == 9 else moving_ring_clearances
    return kernel(*rings,paths,int(begin),int(end))
