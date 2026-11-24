"""Early contact window of fixed-angle, zero-speed TH14 spawn-animation volleys."""
import math
import struct

import numpy as np

from .native_enemy_ecl import ECLIncomplete, apply_tick_callback_effects, f32
from .native_enemy_motion import ALREADY_UPDATED_FLAG, _step_motion
from .native_shot_rules import ReadCache, RuleWorld, emission_vectors


def _stationary_prefix(shooter):
    # Verified native style/color and EX order. The center remains at the
    # muzzle during ages 8..14. The post-animation EX8 and child emissions
    # have different motion and are not represented by this stationary part.
    prefix=shooter.ex[:4]
    return (shooter.style==18 and shooter.color==3 and shooter.aim==1 and
        shooter.speed==0. and shooter.speed2==0. and
        all(prefix) and [(p['command'],p['condition']) for p in prefix]==
        [(2,0),(1048576,0),(1,1),(0x80000000,0)] and
        prefix[0]['ints'][0]==1 and prefix[1]['ints'][0]==1 and prefix[3]['ints'][0]>=10)


class _VolleyWorld(RuleWorld):
    def global_value(self,variable,kind):
        if variable in (-10000,-9999,-9998,-9991,-9990,-9989,-9987):
            raise ECLIncomplete('emitter trajectory depends on the candidate or a random draw')
        return super().global_value(variable,kind)

    def rule_instruction(self,thread,ins,opcode,order):
        if opcode in (613,615,616):
            raise ECLIncomplete('subsequent global modification of spawned projectiles')
        return super().rule_instruction(thread,ins,opcode,order)

    def emit(self,shooter,kind,thread):
        if kind!='bullet' or not _stationary_prefix(shooter):
            return
        center=self.state['final']['position'][:2] if shooter.absolute is None else shooter.absolute
        origin=[f32(p+o) for p,o in zip(center,shooter.offset)]
        radius=struct.unpack('<f',self.memory.read(self.base+0xd1f94+shooter.style*0x114,4))[0]
        if not all(math.isfinite(v) for v in (*origin,radius,shooter.radius,shooter.angle,shooter.spread)) or radius<=0:
            return
        for angle,_,_,_ in emission_vectors(shooter,origin,(0.,0.),None):
            x=f32(origin[0]+f32(math.cos(angle)*shooter.radius))
            y=f32(origin[1]+f32(math.sin(angle)*shooter.radius))
            self.future_events.append((self.frame+7,self.frame+13,x,y,radius))


def forecast_stationary_volleys(memory,snapshot,phase,horizon):
    from .nbnm_task import attack_key
    # A known issuing attack is recognizable before the first shot. At
    # other stages a matching projectile appearance can trigger inspection,
    # but only the live shooter and its EX program can add a hazard.
    anticipatory=snapshot.stage==3 and attack_key(phase)==('Boss',True,1)
    if not anticipatory and not any(b.kind==18 and b.age<=14 and b.vx==0. and b.vy==0. for b in snapshot.bullets):
        return None
    cache=ReadCache(memory)
    resources=snapshot.extra.get('resources',{})
    instructions={};rows=[];seen=set()
    for enemy in (*resources.get('enemies',()),*resources.get('hidden_enemy_contacts',())):
        state=enemy.get('native_motion')
        if not state or state['owner_address'] in seen or state['effective_game_speed']!=1.:
            continue
        seen.add(state['owner_address'])
        try:
            world=_VolleyWorld(cache,state,player_path=(),rng=None,events=[],shared_instructions=instructions)
        except ECLIncomplete:
            continue
        cutoff=horizon+1
        for tick in range(1,horizon-6):
            if world.state.get('ecl_time_step') is None or world.motion_unknown:
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
        for first,last,x,y,radius in world.future_events:
            if first<cutoff:
                rows.append((first,min(last,cutoff-1),x,y,radius))
    return np.asarray(rows,np.float64) if rows else None


def volley_clearances(volleys,paths,begin,end,player_radius):
    """Native circle test at each reachable player tick while the newborn is stationary."""
    volleys = np.asarray(volleys, dtype=np.float64).reshape(-1, 5)
    if len(paths)*len(volleys)>=64:
        try:
            from .swept_clearance_kernel import stationary_volley_clearances
        except ImportError:
            pass
        else:
            if (0<=begin<end<paths.shape[1] and np.isfinite(paths).all()
                    and np.isfinite(volleys).all() and np.isfinite(player_radius)):
                return stationary_volley_clearances(volleys,paths,int(begin),int(end),float(player_radius))
    return _numpy_volley_clearances(volleys,paths,begin,end,player_radius)


def _numpy_volley_clearances(volleys,paths,begin,end,player_radius):
    first,last,x,y,radius=volleys.T
    applicable=(first<=end)&(last>begin)
    gaps=np.full(len(paths),np.inf)
    if not applicable.any():
        return gaps
    first,last,x,y,radius=(q[applicable] for q in (first,last,x,y,radius))
    contact_radius=np.sqrt(radius*radius+player_radius*player_radius)+.05
    for t in range(max(begin+1,int(first.min())),min(end,int(last.max()))+1):
        active=(first<=t)&(t<=last)
        if active.any():
            dx=paths[:,t,0,None]-x[active]
            dy=paths[:,t,1,None]-y[active]
            gaps=np.minimum(gaps,(np.sqrt(dx*dx+dy*dy)-contact_radius[active]).min(axis=1))
    return gaps
