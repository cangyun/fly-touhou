"""TH14 shooter state and forward ECL emission events (read-only).

Native references: shooter setup 427a43..4286b2, emission 4191a0/417660,
RNG 403310/403470/4034b0, line/curve creation 428a49/429014.
Animation and EX dynamics are handled separately, never assumed linear merely
because a shooter has supplied an initial velocity.
"""
from dataclasses import dataclass, field
import copy
import math
import struct
import numpy as np
from .native_enemy_ecl import ECLWorld, ECLThread, ECLIncomplete, f32, _convert

REVISION = 'th14-native-future-shooters-v1'
EX_NAMES = {1:'speedup',2:'spawn_animation',4:'vector_acceleration',8:'polar_acceleration',
    16:'step_turn',64:'bounce',128:'bullet_cancel_protection',256:'offscreen_retention',
    512:'change_sprite_and_hitbox',1024:'delete',2048:'sound',4096:'wrap',
    8192:'emit_child_shots',16384:'child_shot_type_record',32768:'native_reaction_selector',65536:'program_jump',
    1048576:'animation_flags',2097152:'velocity_interpolation',4194304:'collision_scale',
    33554432:'draw_layer',67108864:'hold',2147483648:'wait'}
AIM_NAMES = {0:'aimed_fan',1:'fixed_fan',2:'aimed_ring',3:'fixed_ring',4:'aimed_offset_ring',
    5:'fixed_offset_ring',6:'random_angle',7:'random_speed_ring',8:'random_angle_and_speed',
    9:'aimed_pyramid',10:'fixed_pyramid',11:'peanut',12:'offset_peanut'}


@dataclass
class NativeRNG:
    seed: int
    counter: int = 0

    def u32(self):
        a = ((self.seed ^ 0x9630) - 0x6553) & 0xffff
        rotated = ((a >> 14) + (a << 2)) & 0xffff
        b = ((rotated ^ 0x9630) - 0x6553) & 0xffff
        self.seed = ((b >> 14) + (b << 2)) & 0xffff
        self.counter += 2
        return (a << 16) | b

    def uniform(self): return f32(f32(self.u32()) * 2**-32)
    def signed(self): return f32(f32(f32(self.u32()) * 2**-31) - 1.)


@dataclass
class Shooter:
    style: int = 0
    color: int = 0
    angle: float = 0.
    spread: float = 0.
    speed: float = 2.
    speed2: float = 0.
    radius: float = 0.
    count: int = 1
    layers: int = 1
    aim: int = 0
    offset: list = field(default_factory=lambda:[0.,0.])
    absolute: list | None = None
    ex: list = field(default_factory=lambda:[None]*18)
    ex_cursor: int = 0
    laser: list = field(default_factory=lambda:[0.,0.,0.,0.])
    laser_times: list = field(default_factory=lambda:[0]*5)

    @classmethod
    def read(cls, memory, data, index):
        raw = memory.read(data + 0x598 + index * 0x380, 0x380)
        style,color=struct.unpack_from('<2i',raw)
        angle,spread,speed,speed2,radius=struct.unpack_from('<5f',raw,0x14)
        count,layers,aim=struct.unpack_from('<3h',raw,0x364)
        result=cls(style,color,angle,spread,speed,speed2,radius,count,layers,aim)
        result.offset=list(struct.unpack('<2f',memory.read(data+0x3dd8+index*12,8)))
        absolute=struct.unpack('<3f',memory.read(data+0x3e98+index*12,12))
        if absolute[2]>.5: result.absolute=list(absolute[:2])
        result.ex_cursor=struct.unpack('<i',memory.read(data+0x3d98+index*4,4))[0]
        for i in range(18):
            floats=struct.unpack_from('<4f',raw,0x28+i*44)
            ints=struct.unpack_from('<4i',raw,0x38+i*44)
            command,condition=struct.unpack_from('<2I',raw,0x48+i*44)
            if command:result.ex[i]=dict(command=command,condition=condition,ints=list(ints),floats=list(floats))
        result.laser=[*struct.unpack_from('<3f',raw,8),struct.unpack_from('<f',raw,0x340)[0]]
        result.laser_times=list(struct.unpack_from('<5i',raw,0x350))
        return result


def emission_vectors(shooter, origin, player, rng):
    """All original 0..12 emission modes; preserve native loop/RNG order."""
    s=shooter;n=s.count;layers=s.layers
    if not (0<=n<=8192 and 0<=layers<=8192 and n*layers<=8192):
        raise ECLIncomplete('invalid_or_excessive_shooter_count')
    if s.aim not in AIM_NAMES:raise ECLIncomplete(f'unknown_aim_mode_{s.aim}')
    aim=f32(math.atan2(f32(player[1]-origin[1]),f32(player[0]-origin[0])))
    if player[0]==origin[0] and player[1]==origin[1]:aim=f32(math.pi/2)
    tau=f32(2*math.pi);pi=f32(math.pi)
    rows=[]
    for layer in range(layers):
        speed=f32(s.speed-f32(f32(f32(s.speed-s.speed2)*layer)/(layers-1))) if layers>1 else s.speed
        for index in range(n):
            mode=s.aim;v=speed
            if mode in (0,1):
                offset=f32(((index+1)//2 if n%2 else index//2+.5)*s.spread)
                if index%2:offset=f32(-offset)
                angle=f32(s.angle+f32(offset+(aim if mode==0 else 0.)))
            elif mode in (2,3,4,5):
                angle=f32(f32(aim if mode in (2,4) else 0.)+
                    f32(pi/n if mode in (4,5) else 0.))
                angle=f32(angle+f32(f32(index*tau)/n))
                angle=f32(angle+f32(s.angle+f32(layer*s.spread)))
            elif mode in (6,8):
                angle=f32(s.angle+f32(rng.signed()*s.spread))
                if mode==8:v=f32(s.speed+f32(rng.uniform()*s.speed2))
            elif mode==7:
                v=f32(s.speed+f32(rng.uniform()*s.speed2))
                angle=f32(f32(f32(index*tau)/n)+f32(s.angle+f32(layer*s.spread)))
            elif mode in (9,10):
                ring=f32(f32(index*tau)/n)
                offset=f32((((layer+1)//2) if layers%2 else layer//2+.5)*s.spread+ring)
                if layer%2:offset=f32(-offset)
                if mode==9:offset=f32(offset+aim)
                angle=f32(s.angle+offset)
                if layers>1:
                    k=((layer+1)&~1) if layers%2 else (layer&~1)
                    v=f32(s.speed+f32(f32(f32(s.speed2-s.speed)*k)/(layers-1)))
            else:
                offset=f32(f32(index*tau)/n)
                if mode==12:offset=f32(offset+f32(pi/n))
                angle=f32(s.angle+offset)
                # The offset peanut ring has its fast lobes on the zero-angle
                # axis. All 32 native velocities were matched by ray index.
                axis=math.sin(offset) if mode==12 else math.cos(offset)
                v=f32(s.speed*f32(1-f32(abs(f32(axis))*s.speed2)))
            angle=f32((angle+pi)%tau-pi)
            rows.append([angle,v,f32(math.cos(angle)*v),f32(math.sin(angle)*v)])
    return rows


class _YieldNativeThread(Exception):
    """The native interpreter leaves this instruction and clock in place."""


class RuleWorld(ECLWorld):
    def __init__(self,memory,state,*,player_path,rng,events,shared_instructions=None,pending=None,library=None):
        super().__init__(memory,copy.deepcopy(state))
        self.player_path=np.asarray(player_path,float)
        self.rng=rng;self.future_events=events;self.shooters={}
        self.rule_unknown=[];self.used_rng=False
        self.actor_track = None
        self.actor_lifetimes = None
        self.pending=pending;self.library=library
        if shared_instructions is not None:self.instructions=shared_instructions

    def subroutine(self,name):
        return self.library.subroutine(name) if self.library is not None else super().subroutine(name)

    def global_value(self,variable,kind):
        if self.state.get('virtual_enemy'):
            if variable in self.globals:return _convert(self.globals[variable],kind)
            if -9985<=variable<=-9978 or -9935<=variable<=-9932:return _convert(0,kind)
            if variable==-9954:return _convert(self.state['health']['hp'],kind)
        if variable in (-9991, -9990, -9965, -9964, -9989, -9944):
            if not len(self.player_path):
                raise ECLIncomplete('path_or_random_dependent_emitter_control')
            player = self.player_path[min(self.frame, len(self.player_path) - 1)]
            if variable in (-9991, -9965):
                return _convert(player[0], kind)
            if variable in (-9990, -9964):
                return _convert(player[1], kind)
            x, y = self.state['final']['position'][:2]
            if variable == -9944:
                # The native getter rounds both products and their sum before sqrt.
                dx, dy = f32(x - player[0]), f32(y - player[1])
                distance = f32(math.sqrt(f32(f32(dx * dx) + f32(dy * dy))))
                return _convert(distance, kind)
            return _convert(math.atan2(player[1] - y, player[0] - x), kind)
        if variable == -9998 and kind == 'i':
            return 0
        if variable in (-10000,-9999,-9998,-9987):
            self.used_rng=True
            if variable == -10000:
                value = self.rng.u32() & 0x7fffffff
            elif variable == -9999:
                value = self.rng.uniform()
            else:
                value = self.rng.signed()
                if variable == -9998:
                    value = f32(value * f32(math.pi))
            return _convert(value,kind)
        return super().global_value(variable,kind)

    def shooter(self,index):
        if not 0<=index<16:raise ECLIncomplete('shooter_index_outside_native_bank')
        if index not in self.shooters:
            self.shooters[index]=(Shooter() if self.state.get('virtual_enemy') else
                Shooter.read(self.memory,self.state['data_address'],index))
        return self.shooters[index]

    def arguments(self,thread,ins,signature):
        values=[self.argument(thread,ins,i,kind) for i,kind in enumerate(signature)]
        if any(v is None or not math.isfinite(v) for v in values):
            raise ECLIncomplete(f'unresolved_shooter_argument_{struct.unpack_from("<H",ins,4)[0]}')
        return values

    def emit(self,s,kind,thread):
        position=self.state['final']['position'][:2] if s.absolute is None else s.absolute
        origin=[f32(p+o) for p,o in zip(position,s.offset)]
        player=self.player_path[min(self.frame,len(self.player_path)-1)]
        event=dict(tick=self.frame,kind=kind,owner=self.owner,instruction=thread.ip,
            position=origin,shooter=copy.deepcopy(vars(s)),rng_conditional=self.used_rng or s.aim in (6,7,8),
            phase_conditional=True)
        if kind == 'bullet':
            event['vectors'] = emission_vectors(s, origin, player, self.rng)
            event['unit_point_motion'] = (self.state['global_game_speed'] == 1.
                                          and self.state['ecl_time_step'] == 1.)
            commands = [row for row in s.ex if row is not None]
            if (event['unit_point_motion'] and len(commands) == 1
                    and (commands[0]['command'], commands[0]['condition'], commands[0]['ints'][0]) == (2, 0, 1)):
                from .native_bullet_animation import initial_ex2_gate_updates
                updates = initial_ex2_gate_updates(self.memory, s.style, s.color)
                if updates >= 2:
                    event['fresh_ex2_release_age'] = updates - 1
        elif kind == 'line_laser':
            event['unit_line_motion'] = self.state['global_game_speed'] == 1.
        self.future_events.append(event)

    def run_thread(self, thread, order):
        try:
            super().run_thread(thread, order)
        except _YieldNativeThread:
            if thread.variable_interpolators:
                raise ECLIncomplete('active_ecl_variable_interpolator')

    def rule_instruction(self,t,ins,op,order):
        if op == 520:
            # 42886e: this is an enemy-manager guard, not a write to HP.
            # Returning -1 ends this ECL invocation when a manager gate is set.
            if not self.manager:
                raise ECLIncomplete('native_enemy_manager_guard_unreadable')
            gates = [self.read(self.manager + offset, 'I') for offset in (0x5c, 0x60, 0x64)]
            if any(gate is None for gate in gates):
                raise ECLIncomplete('native_enemy_manager_guard_unreadable')
            if self.actor_lifetimes and any(self.actor_lifetimes.get(gate, self.frame) < self.frame for gate in gates):
                raise ECLIncomplete('native_enemy_manager_guard_changed_by_actor_departure')
            if any(gates):
                # 483c01 bypasses both instruction cleanup and clock advance.
                raise _YieldNativeThread
            return True
        if op==556:
            length=struct.unpack_from('<I',ins,16)[0]
            self.death_program=ins[20:20+length].split(b'\0',1)[0]
            # Registration is not a death or an immediate emission. HP/damage
            # scheduling remains a distinct conditional part of the forecast.
            self.rule_unknown.append(dict(tick=self.frame,opcode=op,
                reason='death_trigger_not_simulated',program=self.death_program.decode('cp932')))
            return True
        if op==641:
            s=self.shooter(self.arguments(t,ins,'i')[0]);s.ex_cursor=max(0,s.ex_cursor-1);return True
        if op==613:
            self.future_events.append(dict(tick=self.frame,kind='clear_projectiles',owner=self.owner,
                instruction=t.ip,phase_conditional=True));return True
        if op==600:
            index=self.arguments(t,ins,'i')[0];self.shooters[index]=Shooter();return True
        signatures={601:'i',602:'iii',603:'iff',604:'iff',605:'iff',606:'iii',607:'ii',608:'iii',
            614:'ii',624:'iffffffff',625:'iiiiiiiii',626:'iff',627:'if',628:'iff',
            700:'iffff',701:'iiiiii',702:'i',703:'i',711:'i'}
        if op in signatures:
            a=self.arguments(t,ins,signatures[op]);index=a[0]
            if op==614:self.shooters[index]=copy.deepcopy(self.shooter(a[1]));return True
            s=self.shooter(index)
            if op in (601,702,703,711):
                self.emit(s,{601:'bullet',702:'line_laser',703:'infinite_laser',711:'curve_laser'}[op],t)
            elif op==602:s.style,s.color=a[1:]
            elif op==603:s.offset=a[1:]
            elif op==604:s.angle,s.spread=a[1:]
            elif op==605:s.speed,s.speed2=a[1:]
            elif op==606:s.count,s.layers=a[1:]
            elif op==607:s.aim=a[1]
            elif op==608:pass # audio only
            elif op in (624,625):
                rank=self.global_value(-9959,'i')
                if rank is None:raise ECLIncomplete('unknown_shooter_difficulty')
                if op==624:s.speed,s.speed2=a[1+rank],a[5+rank]
                else:s.count,s.layers=a[1+rank],a[5+rank]
            elif op==626:s.offset=[f32(math.cos(a[1])*a[2]),f32(math.sin(a[1])*a[2])]
            elif op==627:s.radius=a[1]
            elif op==628:s.absolute=None if a[1]<-990 else a[1:]
            elif op==700:s.laser=a[1:]
            elif op==701:s.laser_times=a[1:]
            return True
        if op in (609,610,611,612):
            append=op in (611,612);long=op in (610,612)
            signature=('iii' if append else 'iiii')+('iiiiffff' if long else 'iiff')
            a=self.arguments(t,ins,signature);s=self.shooter(a[0]);index=s.ex_cursor if append else a[1]
            start=1 if append else 2
            condition,command=a[start:start+2];rest=a[start+2:]
            ints=rest[:4] if long else rest[:2]+[-999999,-999999]
            floats=rest[4:] if long else rest[2:]+[-999999.,-999999.]
            if not 0<=index<18:raise ECLIncomplete('EX_program_index_outside_18')
            s.ex[index]=dict(command=command&0xffffffff,condition=condition,ints=ints,floats=floats)
            s.ex_cursor=index+1
            return True
        if op in (615,616):
            # These alter already existing projectiles globally. Mark the
            # forecast boundary instead of silently preserving old trajectories.
            self.rule_unknown.append(dict(tick=self.frame,opcode=op,reason='global_projectile_mutation'))
            return True
        if op>=600 and op not in (632,633,634,637,638,639):
            raise ECLIncomplete(f'unimplemented_future_rule_{op}')
        if op in (300,301,304,305,309,310,311,312) and self.pending is not None:
            self.spawn(t,ins,op);return True
        if op in (300,301,304,305,309,310,311,312,556):
            self.rule_unknown.append(dict(tick=self.frame,opcode=op,
                reason='new_enemy_or_death_program_requires_world_scheduler'))
        return False

    def spawn(self,t,ins,opcode):
        length=struct.unpack_from('<I',ins,16)[0];name=ins[20:20+length].split(b'\0',1)[0]
        header=bytearray(ins[:16]);struct.pack_into('<H',header,8,struct.unpack_from('<H',ins,8)[0]>>1)
        a=self.arguments(t,bytes(header)+ins[20+length:],'ffiii');x,y,hp,score,item=a
        if opcode in (300,304,309,311):
            x=f32(x+self.state['final']['position'][0]);y=f32(y+self.state['final']['position'][1])
        mirror=bool(self.state['flags']&0x80000)
        if mirror:x=f32(-x)
        work=copy.deepcopy(self.state);work.pop('ecl_forecast',None)
        work.pop('damage_bound',None);work['damage_bound_enabled']=False
        work['interrupts']=[]
        work.update(ecl_contexts=[],flags=0x80000 if mirror else 0,tick_callback=0,
            virtual_enemy=True,owner_address=-(len(self.future_events)+1),ecl_time_step=1.,local_slowdown=0.,
            effective_game_speed=work['global_game_speed'])
        for key in ('previous','final','absolute','relative'):
            motion=work[key]
            for field,value in list(motion.items()):motion[field]=[0.]*len(value) if isinstance(value,list) else 0
            if key!='relative':motion['position']=[x,y,0.]
        for interp in work['interpolators'].values():interp.update(duration=0,timer=0,elapsed=0.)
        work['movement_limit']['enabled']=False
        work['health'].update(hp=hp,hp_max=hp);work['phase_timer'].update(timer=0,elapsed=0.)
        child=RuleWorld(self.memory,work,player_path=self.player_path,rng=self.rng,events=self.future_events,
            shared_instructions=self.instructions,pending=self.pending,library=self)
        # 42a361..42a3c9 copies all twelve parent user variables to the
        # construction record. A new actor does not receive twelve zeroes.
        for variable in (*range(-9985,-9977),*range(-9935,-9931)):
            child.globals[variable]=self.global_value(variable,'i' if variable<=-9982 else 'f')
        thread=ECLThread();thread.ip=self.subroutine(name);thread.sp=16;thread.rank=t.rank
        child.threads=[thread];child.all_threads=[thread]
        self.pending.append(child)
        self.future_events.append(dict(tick=self.frame,kind='enemy',owner=self.owner,instruction=t.ip,
            subroutine=name.decode('cp932'),position=[x,y],opcode=opcode,phase_conditional=True,
            new_enemy_tick_order_validated=False))


def capture_enemy_states(reader,snapshot):
    """Include invisible stage controllers, not just collision/attack targets."""
    from .native_enemy_motion import read_enemy_motion
    resources=snapshot.extra.get('resources',{})
    rows=[*resources.get('enemies',[]),*resources.get('hidden_enemy_contacts',[])]
    states={e['id']:e['native_motion'] for e in rows if e.get('native_motion')}
    offsets=reader.o['enemies'];manager=reader.pointer('enemies','enemy_manager_pointer')
    ordered=[]
    if manager:
        for identity in reader._entries(reader.mem.integer(manager+offsets['zEnemyManager_list']),limit=2048):
            if identity not in states:
                states[identity]=read_enemy_motion(reader.mem,identity+offsets['zEnemy_data'],forecast_ecl=False)
            ordered.append(states[identity])
    return ordered if manager else list(states.values())


class ReadCache:
    """One paused observation; no cached dynamic memory crosses game frames."""
    def __init__(self,memory):self.memory=memory;self.base=memory.base;self.cache={}
    def read(self,address,size):
        key=address,size
        if key not in self.cache:self.cache[key]=self.memory.read(address,size)
        return self.cache[key]


def preload_programs(memory,states,catalog,stage):
    """Bind complete local rule programs to the currently loaded ECL addresses.

    Runtime bytecode takes precedence (e.g. a different local patch). Unmatched
    functions fall back to the existing live instruction reader. No game writes.
    """
    candidates={}
    for program in catalog['programs']:
        if program['file']!='default.ecl' and not program['file'].startswith(f'st0{stage}'):continue
        for sub in program['subroutines']:
            candidates.setdefault(sub['name'].encode('cp932'),[]).append(sub)
    instructions={};tables=set();matched=[];fallback=[]
    for state in states:
        owner=state['owner_address']
        manager=struct.unpack('<I',memory.read(owner+0x11cc,4))[0]
        if not manager or manager in tables:continue
        tables.add(manager)
        count=struct.unpack('<I',memory.read(manager+8,4))[0]
        table=struct.unpack('<I',memory.read(manager+0x8c,4))[0]
        if not table or not 0<=count<=10000:raise ECLIncomplete('invalid_preload_subroutine_table')
        entries=memory.read(table,count*8) if count else b''
        for index in range(count):
            name_ptr,sub_ptr=struct.unpack_from('<II',entries,index*8)
            chars=bytearray()
            for n in range(128):
                char=memory.read(name_ptr+n,1)
                if char==b'\0':break
                chars+=char
            name=bytes(chars);found=False
            for sub in candidates.get(name,[]):
                code=b''.join(bytes.fromhex(r['raw_hex']) for r in sub['instructions'])
                if code and memory.read(sub_ptr+16,len(code))==code:
                    for row in sub['instructions']:
                        instructions[sub_ptr+16+row['relative_offset']]=bytes.fromhex(row['raw_hex'])
                    matched.append(name.decode('cp932'));found=True;break
            if not found:fallback.append(name.decode('cp932',errors='replace'))
    return instructions,dict(stage=stage,matched_subroutines=len(matched),
        instructions=len(instructions),live_fallback_subroutines=fallback)


def _actor_track_frame(world, tick):
    return dict(tick=tick, position=world.state['final']['position'][:],
                flags=world.state['flags'])


def _advance_emitters(worlds, pending, begin, end, actor_last_ticks=None, actor_rng_draws=None):
    from .native_enemy_motion import _step_motion, ALREADY_UPDATED_FLAG
    from .native_enemy_ecl import apply_tick_callback_effects

    for tick in range(begin, end + 1):
        draws_here = actor_rng_draws.get(tick, {}) if actor_rng_draws else {}
        for world in worlds:
            if world.motion_unknown:
                continue
            if actor_last_ticks and tick > actor_last_ticks.get(world.owner, end):
                continue
            try:
                skip = bool(world.state['flags'] & ALREADY_UPDATED_FLAG)
                _step_motion(world.state)
                if not skip:
                    world.actor_lifetimes = actor_last_ticks
                    world.tick(tick)
                    apply_tick_callback_effects(world)
            except (NotImplementedError, ECLIncomplete) as error:
                world.motion_unknown = (tick, str(error))
            if world.actor_track is not None and not world.motion_unknown:
                world.actor_track.append(_actor_track_frame(world, tick))
            for _ in range(draws_here.get(world.owner, 0)):
                world.rng.u32()
        if pending:
            worlds.extend(pending)
            pending.clear()
        if len(worlds) > 2048:
            raise ECLIncomplete('future_enemy_population_exceeds_native_limit')


def _common_player_ticks(paths, horizon):
    """The last tick before any candidate reads different player coordinates."""
    if len(paths) < 2:
        return 0
    length = min(horizon + 1, min(len(path) for path in paths))
    if length < 2:
        return 0
    # Compare the actual float64 inputs, including the sign of zero.
    reference = paths[0][:length].view(np.uint64)
    for path in paths[1:]:
        different = np.flatnonzero((reference != path[:length].view(np.uint64)).any(axis=1))
        if len(different):
            length = int(different[0])
            if length < 2:
                return 0
            reference = reference[:length]
    return length - 1


def forecast_emissions(memory, states, player_paths, *, seed, counter=0, horizon=60,
                       preloaded_instructions=None, actor_last_ticks=None, actor_rng_draws=None,
                       capture_actor_tracks=False):
    """Project emissions along candidates under an optional actor lifetime.

    Each supplied table maps an original actor address to the last tick on
    which its current script can run. Derive it for the same candidate path.
    Known draws from actor damage and item drops run after that actor's ECL
    in native manager order. Counts are of 32-bit calls.
    Retain same-tick emissions. The interpreter reports registered defeat
    scripts separately until their new behavior is supported.
    Optional actor tracks contain the same candidate's ECL decisions and its
    observable body position and flags while the simulation remains covered.
    """
    cache = ReadCache(memory)
    instructions = dict(preloaded_instructions or {})
    paths = tuple(np.ascontiguousarray(path, dtype=np.float64) for path in player_paths)
    lifetimes = tuple(actor_last_ticks) if actor_last_ticks is not None else ({},) * len(paths)
    rng_schedules = tuple(actor_rng_draws) if actor_rng_draws is not None else ({},) * len(paths)
    if len(lifetimes) != len(paths) or len(rng_schedules) != len(paths):
        raise ValueError('Actor lifetime and RNG tables refer to individual candidate paths')
    results = []

    def start(path):
        rng = NativeRNG(seed, counter)
        events, pending = [], []
        worlds = [RuleWorld(cache, state, player_path=path, rng=rng, events=events,
                           shared_instructions=instructions, pending=pending) for state in states]
        if capture_actor_tracks:
            for world in worlds:
                world.actor_track = [_actor_track_frame(world, 0)]
        return worlds, rng, events, pending

    common = _common_player_ticks(paths, horizon)
    if any(lifetimes):
        # Identical positions can have different focus inputs and damage.
        # Share only ticks before the first different emitter omission.
        reference = lifetimes[0]
        for table in lifetimes[1:]:
            for owner in table.keys() | reference.keys():
                first, other = reference.get(owner, horizon), table.get(owner, horizon)
                if first != other:
                    common = min(common, first, other)
    if any(rng_schedules):
        reference_draws = rng_schedules[0]
        for draws in rng_schedules[1:]:
            for tick in draws.keys() | reference_draws.keys():
                if draws.get(tick, {}) != reference_draws.get(tick, {}):
                    common = min(common, tick - 1)
    shared = None
    if common >= 16:
        shared = start(paths[0])
        _advance_emitters(shared[0], shared[3], 1, common, lifetimes[0], rng_schedules[0])

    for path, lifetime, draws in zip(paths, lifetimes, rng_schedules, strict=True):
        if shared is None:
            worlds, rng, events, pending = start(path)
            begin = 1
        else:
            # The RNG, contexts, pending children and already issued events
            # form one mutable snapshot. Native reads and code bytes are shared.
            memo = {id(cache): cache, id(instructions): instructions, id(paths[0]): path}
            worlds, rng, events, pending = copy.deepcopy(shared, memo)
            for world in worlds:
                world.player_path = path
            begin = common + 1
        _advance_emitters(worlds, pending, begin, horizon, lifetime, draws)
        unknown = [dict(owner=w.owner, tick=w.motion_unknown[0], reason=w.motion_unknown[1])
                   for w in worlds if w.motion_unknown]
        unknown += [dict(owner=w.owner, **row) for w in worlds for row in w.rule_unknown]
        result = dict(events=events, unknown=unknown, horizon=horizon,
            rule_complete=not unknown, global_rng_schedule_complete=False,
            rng_draws=rng.counter-counter, source='paused_native_ECL_state')
        if capture_actor_tracks:
            result['actor_tracks'] = {
                world.owner: dict(samples=world.actor_track, events=world.events,
                                  threads=[thread.initial for thread in world.all_threads],
                                  damage_phase_change=world.state.get('ecl_damage_phase_change'))
                for world in worlds if world.actor_track is not None
            }
        if lifetime:
            result['conditional_actor_last_ticks'] = dict(lifetime)
        if actor_rng_draws is not None:
            result['modeled_final_rng_seed'] = rng.seed
        results.append(result)
    return results
