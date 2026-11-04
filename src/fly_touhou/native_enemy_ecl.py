"""Read-only short-horizon TH14 1.00b ECL execution for enemy geometry.

The game runs movement BEFORE ECL. A 401 position interpolation issued at
tick 1 therefore first moves the enemy at tick 2. The old reader stopped at
42 (push), 14 (conditional jump), or 15 (async call), missing this movement.

Native evidence: interpreter 482380, stack/arguments 483f30..484477, calls
482080, thread scheduler 483e70, sub lookup 4848b0, enemy handler 425270.
Only copies of state are changed. Projectile/animation commands are not
simulated; unknown values propagate through arithmetic and cannot select a
branch or become a fabricated geometry argument. The same resolved events
are consumed by movement and body-contact forecasts.
"""
from __future__ import annotations

import copy
import math
import struct


REVISION = 'th14-native-ecl-stack-motion-contact-polar-v3'
CONTACT_OPS = {500, 501, 502, 503, 515, 541, 563, 564, 632, 634, 639}
MOVEMENT_SIGNATURES = {
    400: 'ff', 402: 'ff', 401: 'iiff', 403: 'iiff',
    404: 'ff', 406: 'ff', 405: 'iiff', 407: 'iiff',
    412: 'iif', 413: 'iif',
    416: 'fff', 417: 'fff', 425: 'iffffff', 426: 'iffffff',
    428: 'ff', 430: 'ff', 429: 'iiff', 431: 'iiff',
    434: 'iiiff', 435: 'iiiff', 436: 'iiff', 437: 'iiff',
    438: 'iiiff', 439: 'iiiff', 440: 'f', 442: 'f',
    441: 'iif', 443: 'iif', 444: 'f', 446: 'f',
    445: 'iif', 447: 'iif', 504: 'ffff', 505: '',
}
# These affect shots, effects, sound, drops, boss UI, or receiving damage,
# not this enemy's movement/contact. Their expression temporaries still pop.
# 535/536 write ECL variables and are handled separately below.
GEOMETRY_NEUTRAL = {
    302, 303, 306, 307, 308, 313, 314, 315, 316, 317, 318, 319,
    320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330,
    506, 507, 508, 509, 510, 512, 516, 517, 518, 519, 520, 521,
    522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
    533, 534, 537, 538, 539, 540, 542, 543, 544, 545,
}
SPAWN_OPS = {300, 301, 304, 305, 309, 310, 311, 312}


class ECLIncomplete(Exception):
    pass


def f32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def normalize_native_angle(value):
    """403590: float32 subtraction/addition, with the original 34-loop cap."""
    if value is None or not math.isfinite(value):
        return None
    value = f32(value)
    pi, tau = f32(math.pi), f32(2 * math.pi)
    iterations = 0
    for sign in (1, -1):
        while sign * value > pi:
            value = f32(value - sign * tau)
            previous = iterations
            iterations += 1
            if previous > 32:
                break
    return value


def i32(value):
    return (int(value) + 0x80000000) % 0x100000000 - 0x80000000


def _bits(value, kind):
    if value is None:
        return None
    return struct.unpack('<I', struct.pack('<f' if kind == 'f' else '<i',
        f32(value) if kind == 'f' else i32(value)))[0]


def _value(bits, kind):
    if bits is None:
        return None
    return struct.unpack('<f' if kind == 'f' else '<i', struct.pack('<I', bits))[0]


def _convert(value, kind):
    return None if value is None else f32(value) if kind == 'f' else i32(value)


class ECLThread:
    def __init__(self, address=0, raw=None):
        self.address = address
        self.ip = 0
        self.time = 0.
        self.sp = self.local_base = self.thread_id = 0
        self.rank = 0xff
        self.cells = [0] * 1024
        self.stopped = False
        self.flags = self.secondary_id = 0
        self.unknown = None
        self.variable_interpolators = False
        if raw is not None:
            self.time, self.ip = struct.unpack_from('<fI', raw)
            self.cells = list(struct.unpack_from('<1024I', raw, 8))
            self.sp, self.local_base, self.thread_id = struct.unpack_from('<3i', raw, 0x1008)
            self.rank = raw[0x101c]
            self.secondary_id = struct.unpack_from('<i', raw, 0x1018)[0]
            self.flags = struct.unpack_from('<I', raw, 0x11c0)[0]
            self.variable_interpolators = any(struct.unpack_from('<i', raw, 0x1048 + j * 0x30)[0]
                                               for j in range(8))
        self.initial = dict(context=address, time=self.time, instruction_pointer=self.ip,
                            rank_mask=self.rank, events=[], spawn_instructions=[],
                            unknown_from_tick=None, unmodeled=[])

    def cell(self, offset, kind='i'):
        if offset < 0 or offset >= 4096 or offset % 4:
            raise ECLIncomplete(f'invalid_stack_offset_{offset}')
        return _value(self.cells[offset // 4], kind)

    def put(self, offset, value, kind='i'):
        if offset < 0 or offset >= 4096 or offset % 4:
            raise ECLIncomplete(f'invalid_stack_offset_{offset}')
        self.cells[offset // 4] = _bits(value, kind)

    def push(self, value, kind):
        self.put(self.sp, ord(kind)); self.put(self.sp + 4, value, kind)
        self.sp += 8

    def pop(self, kind='i'):
        self.sp -= 8
        source_kind = 'f' if self.cell(self.sp) & 0xff == ord('f') else 'i'
        return _convert(self.cell(self.sp + 4, source_kind), kind)

    def raw_push(self, value, kind='i'):
        self.put(self.sp, value, kind); self.sp += 4

    def raw_pop(self, kind='i'):
        self.sp -= 4
        return self.cell(self.sp, kind)

    def clear_frame(self):
        old_base = self.local_base
        self.local_base = self.raw_pop()
        self.sp = old_base


class ECLWorld:
    def __init__(self, memory, state):
        self.memory, self.state = memory, state
        self.base = state['image_base']
        self.owner = state['owner_address']
        self.instructions = {}
        self.subs = {}
        self.sub_table = None
        self.globals = {}
        self.threads = []
        self.events = []
        self.frame = 0
        self.read_errors = []
        self.motion_unknown = None
        self.contact_unknown = None
        self.steps = 0
        self.spawn_events = []
        self.unresolved_variables = set()
        self.manager = self.read(self.base + 0xdb544, 'I')
        self.player = self.read(self.base + 0xdb67c, 'I')
        for context in state.get('ecl_contexts', ()):
            try:
                thread = ECLThread(context['address'], memory.read(context['address'], 0x11c4))
                self.threads.append(thread)
            except Exception as exc:
                raise ECLIncomplete(f'unreadable_ecl_context_{context["address"]:#x}: {exc}') from exc
        self.all_threads = self.threads[:]

    def read(self, address, fmt):
        try:
            return struct.unpack('<' + fmt, self.memory.read(address, struct.calcsize('<' + fmt)))[0]
        except Exception:
            return None

    def instruction(self, address):
        if address not in self.instructions:
            try:
                h = self.memory.read(address, 16)
                size = struct.unpack_from('<H', h, 6)[0]
                if not 16 <= size <= 4096:
                    raise ECLIncomplete(f'invalid_ecl_instruction_size_{size}_at_{address:#x}')
                # Return, arithmetic and other zero-operand instructions are
                # exactly one header. ProcessMemory rejects zero-byte reads.
                body = self.memory.read(address + 16, size - 16) if size > 16 else b''
                self.instructions[address] = h + body
            except ECLIncomplete:
                raise
            except Exception as exc:
                raise ECLIncomplete(f'unreadable_ecl_instruction_{address:#x}') from exc
        return self.instructions[address]

    def subroutine(self, name):
        if name in self.subs:
            return self.subs[name]
        if self.sub_table is None:
            manager = self.read(self.owner + 0x11cc, 'I')
            count = self.read(manager + 8, 'i') if manager else None
            table = self.read(manager + 0x8c, 'I') if manager else None
            if count is None or not 0 <= count <= 10000 or not table:
                raise ECLIncomplete('unreadable_ecl_subroutine_table')
            self.sub_table = (table, count)
        table, count = self.sub_table
        lo, hi = 0, count
        try:
            while lo < hi:
                middle = (lo + hi) // 2
                name_ptr, sub_ptr = struct.unpack('<II', self.memory.read(table + middle * 8, 8))
                try:
                    found = self.memory.read(name_ptr, 128).split(b'\0', 1)[0]
                except Exception:
                    chars = bytearray()
                    for i in range(128):
                        char = self.memory.read(name_ptr + i, 1)
                        if char == b'\0': break
                        chars += char
                    found = bytes(chars)
                if found == name:
                    self.subs[name] = sub_ptr + 16
                    return sub_ptr + 16
                if found < name: lo = middle + 1
                else: hi = middle
        except Exception as exc:
            raise ECLIncomplete('unreadable_ecl_subroutine_names') from exc
        raise ECLIncomplete(f'unresolved_ecl_subroutine_{name!r}')

    def global_value(self, variable, kind):
        """Verified owner getters 42a920/42b230; values shared by all threads."""
        state = self.state
        dynamic = {
            -9997: state['final']['position'][0], -9996: state['final']['position'][1],
            -9995: state['absolute']['position'][0], -9994: state['absolute']['position'][1],
            -9993: state['relative']['position'][0], -9992: state['relative']['position'][1],
            -9977: state['final']['position'][0], -9976: state['final']['position'][1],
            -9975: state['absolute']['position'][0], -9974: state['absolute']['position'][1],
            -9973: state['relative']['position'][0], -9972: state['relative']['position'][1],
            -9971: state['absolute']['angle'], -9970: state['relative']['angle'],
            -9969: state['absolute']['speed'], -9968: state['relative']['speed'],
            -9967: state['absolute']['radius'], -9966: state['relative']['radius'],
            -9958: math.atan2(state['final']['velocity'][1], state['final']['velocity'][0]),
            -9957: 1, -9986: (state['flags'] >> 24) & 1,
            -9906: (state['flags'] >> 19) & 1,
        }
        if variable in dynamic:
            # Once an unseen RNG draw has chosen the angle, nominal future
            # coordinates/angles must not be reintroduced as exact arguments.
            if state.get('_ecl_reachable_radius', 0.) > 0 and variable in (
                    -9997, -9996, -9995, -9994, -9993, -9992,
                    -9977, -9976, -9975, -9974, -9973, -9972, -9958):
                return None
            if variable in (-9971, -9970):
                part = 'absolute' if variable == -9971 else 'relative'
                if state[part].get('_unknown_ecl_heading'):
                    return None
            return _convert(dynamic[variable], kind)
        if (variable == -9954 and state.get('damage_bound_enabled')
                and self.frame > 1):
            # Incoming damage is not an exact future HP value. Do not reuse
            # the observed HP in later conditional branches.
            self.unresolved_variables.add(variable)
            return None
        if variable in self.globals:
            return _convert(self.globals[variable], kind)
        address, storage_kind = None, 'i'
        if -9985 <= variable <= -9982:
            address = self.owner + 0x147c + (variable + 9985) * 4
        elif -9981 <= variable <= -9978:
            address, storage_kind = self.owner + 0x148c + (variable + 9981) * 4, 'f'
        elif -9935 <= variable <= -9932:
            address, storage_kind = self.owner + 0x149c + (variable + 9935) * 4, 'f'
        elif -9949 <= variable <= -9947 and self.manager:
            address = self.manager + 0x10 + (variable + 9949) * 4
        elif -9926 <= variable <= -9923 and self.manager:
            address = self.manager + 0x2c + (variable + 9926) * 4
        elif -9922 <= variable <= -9915 and self.manager:
            address, storage_kind = self.manager + 0x3c + (variable + 9922) * 4, 'f'
        elif variable in (-9960, -9959, -9930, -9907):
            address = self.base + {-9960: 0xf583c, -9959: 0xf5834,
                                   -9930: 0xf5858, -9907: 0xf5844}[variable]
        elif variable == -9954:
            address = self.owner + 0x5164
        elif variable in (-9961, -9929, -9928, -9913, -9912):
            return _convert(0, kind)
        elif -9953 <= variable <= -9950:
            difficulty = self.global_value(-9959, 'i')
            return None if difficulty is None else _convert(difficulty != variable + 9953, kind)
        # RNG and future player/other-enemy positions are deliberately not
        # read as fixed future truth. They may still flow into neutral shots.
        value = self.read(address, storage_kind) if address else None
        if value is None:
            self.unresolved_variables.add(variable)
        if address:
            self.globals[variable] = value
        return _convert(value, kind)

    def set_global(self, variable, value):
        if variable == -9954 and self.state.get('damage_bound_enabled'):
            self.state.setdefault('ecl_damage_phase_change', dict(tick=self.frame, opcode='HP-variable-write'))
        self.globals[variable] = value

    def argument(self, thread, ins, index, kind='i', *, pop=False):
        try:
            raw = struct.unpack_from('<f' if kind == 'f' else '<i', ins, 16 + index * 4)[0]
        except struct.error as exc:
            raise ECLIncomplete('truncated_ecl_argument') from exc
        if not struct.unpack_from('<H', ins, 8)[0] & (1 << index):
            return raw
        if raw >= 0:
            return thread.cell(thread.local_base + int(raw), kind)
        if -100 <= raw <= -1:
            if pop:
                return thread.pop(kind)
            offset = thread.sp + int(raw) * 8
            source_kind = 'f' if thread.cell(offset) & 0xff == ord('f') else 'i'
            return _convert(thread.cell(offset + 4, source_kind), kind)
        return self.global_value(int(raw), kind)

    def target(self, thread, ins, index, value, kind='i'):
        if not struct.unpack_from('<H', ins, 8)[0] & (1 << index):
            raise ECLIncomplete('unmasked_ecl_variable_write')
        variable = struct.unpack_from('<f' if kind == 'f' else '<i', ins, 16 + index * 4)[0]
        if variable >= 0:
            thread.put(thread.local_base + int(variable), value, kind)
        else:
            self.set_global(int(variable), value)

    def call(self, caller, ins, *, asynchronous=False, thread_id=-1, extra=0):
        name_size = struct.unpack_from('<i', ins, 16)[0]
        name = ins[20:20 + name_size].split(b'\0', 1)[0]
        target_ip = self.subroutine(name)
        callee = ECLThread() if asynchronous else caller
        original_sp = callee.sp
        destination = original_sp + 12 if original_sp else 16
        offset = 20 + name_size + extra * 4
        argument_index = 1 + extra
        values = []
        while argument_index < ins[11]:
            if offset + 8 > len(ins):
                raise ECLIncomplete('truncated_ecl_call_argument')
            source_kind = 'f' if ins[offset] in (ord('f'), ord('g')) else 'i'
            target_kind = 'f' if ins[offset + 1] == ord('f') else 'i'
            # Call arguments use the parameter bit for logical argument,
            # whereas their data occupy a type/value pair.
            raw = struct.unpack_from('<f' if source_kind == 'f' else '<i', ins, offset + 4)[0]
            if struct.unpack_from('<H', ins, 8)[0] & (1 << argument_index):
                if raw >= 0:
                    raw = caller.cell(caller.local_base + int(raw), source_kind)
                elif -100 <= raw <= -1:
                    raw = caller.pop(source_kind)
                else:
                    raw = self.global_value(int(raw), source_kind)
            values.append((_convert(raw, target_kind), target_kind))
            offset += 8; argument_index += 1
        remaining_sp = callee.sp
        if original_sp:
            marker = callee.raw_pop()
            callee.sp = original_sp
            callee.put(original_sp - 4, marker)
        else:
            callee.put(0, 0); callee.sp = 4
        callee.raw_push(remaining_sp)
        callee.raw_push(caller.time if original_sp else 0., 'f')
        callee.raw_push(caller.ip if original_sp else 0)
        for i, (value, kind) in enumerate(values):
            callee.put(destination + i * 4, value, kind)
        callee.time, callee.ip = 0., target_ip
        if asynchronous:
            callee.rank, callee.thread_id = caller.rank, thread_id
            callee.initial.update(time=0., instruction_pointer=target_ip, rank_mask=callee.rank)
            self.threads.insert(1, callee)
            self.all_threads.append(callee)

    def event(self, thread, ins, opcode, signature, order):
        args = [self.argument(thread, ins, i, kind) for i, kind in enumerate(signature)]
        if any(v is None or not math.isfinite(v) for v in args):
            raise ECLIncomplete(f'unresolved_geometry_argument_opcode_{opcode}')
        event = dict(tick=self.frame, thread_order=order,
                     instruction_order=len(self.events), address=thread.ip,
                     time=struct.unpack_from('<i', ins)[0], opcode=opcode, arguments=args)
        if opcode in (632, 634, 639):
            table_rva = 0xdb548 if opcode == 634 else 0xd4cec
            event['callback'] = self.read(self.base + table_rva + int(args[0]) * 4, 'I')
            if event['callback'] is None:
                raise ECLIncomplete('unreadable_ecl_callback')
        self.events.append(event)
        thread.initial['events'].append(event)
        apply_geometry_event(self.state, event)
        if opcode in (412, 413) and self.motion_unknown is None:
            # The random angle affects movement on the following tick. Its
            # speed curve is known and a sound displacement bound is retained.
            self.motion_unknown = (self.frame + 1, f'native_random_heading_opcode_{opcode}_bounded')

    def run_thread(self, thread, order):
        if thread.stopped or not thread.ip:
            return
        count = 0
        while thread.ip:
            ins = self.instruction(thread.ip)
            due, opcode, size = struct.unpack_from('<ihH', ins)
            if thread.time < due:
                break
            count += 1; self.steps += 1
            if count > 4096:
                raise ECLIncomplete('ecl_non_yielding_loop')
            if not ins[10] & thread.rank:
                thread.ip += size
                continue
            cleanup = True
            if opcode in (0, 22, 30, 31):
                pass
            elif opcode == 1:
                thread.ip = 0
                return
            elif opcode == 10:
                thread.clear_frame()
                if thread.sp == 0:
                    thread.ip = 0
                    return
                thread.ip = thread.raw_pop()
                thread.time = thread.raw_pop('f')
                thread.sp = thread.raw_pop()
                if not thread.ip:
                    return
                # A return advances the saved call instruction, not this 10.
                old = self.instruction(thread.ip)
                thread.ip += struct.unpack_from('<H', old, 6)[0]
                continue
            elif opcode in (11, 15, 16):
                if opcode == 11:
                    self.call(thread, ins)
                    continue
                extra = int(opcode == 16)
                identifier = -1
                if extra:
                    name_size = struct.unpack_from('<i', ins, 16)[0]
                    identifier = self.argument(thread, ins, (name_size + 4) // 4, 'i')
                self.call(thread, ins, asynchronous=True, thread_id=identifier, extra=extra)
                cleanup = False
            elif opcode in (12, 13, 14):
                take = True
                if opcode != 12:
                    value = thread.pop()
                    if value is None:
                        raise ECLIncomplete(f'unresolved_ecl_condition_{opcode}')
                    take = (value == 0) if opcode == 13 else (value != 0)
                if take:
                    jump, clock = struct.unpack_from('<ii', ins, 16)
                    thread.ip += jump
                    thread.time = f32(clock)
                    continue
            elif opcode in (17, 18, 19, 20):
                identifier = self.argument(thread, ins, 0)
                if identifier is None:
                    raise ECLIncomplete('unresolved_ecl_thread_identifier')
                for other in self.threads[1:]:
                    if other.thread_id == identifier:
                        if opcode == 17: other.ip = 0
                        elif opcode == 18: other.flags |= 1
                        elif opcode == 19: other.flags &= ~1
                        else: other.secondary_id = self.argument(thread, ins, 1)
            elif opcode == 21:
                for other in self.threads[1:]:
                    other.ip = 0
            elif opcode in (23, 24):
                delay = self.argument(thread, ins, 0, 'i' if opcode == 23 else 'f')
                if delay is None:
                    raise ECLIncomplete('unresolved_ecl_wait')
                thread.time = f32(thread.time - f32(delay))
            elif opcode == 40:
                count = self.argument(thread, ins, 0)
                if count is None:
                    raise ECLIncomplete('unresolved_ecl_frame_size')
                old = thread.sp
                thread.sp += count
                thread.raw_push(thread.local_base)
                thread.local_base = old
            elif opcode == 41:
                thread.clear_frame()
            elif opcode in (42, 44):
                kind = 'i' if opcode == 42 else 'f'
                thread.push(self.argument(thread, ins, 0, kind, pop=True), kind)
                cleanup = False
            elif opcode in (43, 45):
                kind = 'i' if opcode == 43 else 'f'
                self.target(thread, ins, 0, thread.pop(kind), kind)
                cleanup = False
            elif 50 <= opcode <= 77:
                self.arithmetic(thread, opcode)
                cleanup = False
            elif opcode == 78:
                value = self.argument(thread, ins, 0)
                self.target(thread, ins, 0, None if value is None else value - 1)
                thread.push(value, 'i'); cleanup = False
            elif opcode == 81:
                # 4836cf -> 403590 -> 484630: normalize angle, FSINCOS,
                # multiply by radius, then write the two float variables.
                # Neither the native game nor its RNG is changed by this VM.
                angle = normalize_native_angle(self.argument(thread, ins, 2, 'f'))
                radius = self.argument(thread, ins, 3, 'f')
                x = y = None
                if angle is not None and radius is not None and math.isfinite(radius):
                    x, y = f32(math.cos(angle) * radius), f32(math.sin(angle) * radius)
                self.target(thread, ins, 0, x, 'f')
                self.target(thread, ins, 1, y, 'f')
            elif opcode == 82:
                value = normalize_native_angle(self.argument(thread, ins, 0, 'f'))
                self.target(thread, ins, 0, value, 'f')
            elif opcode in (83, 84):
                kind = 'i' if opcode == 83 else 'f'
                value = thread.pop(kind)
                thread.push(None if value is None else -value, kind)
                cleanup = False
            elif opcode in (535, 536):
                kind = 'i' if opcode == 535 else 'f'
                rank = self.global_value(-9959, 'i')
                if rank is None or not 0 <= rank <= 3:
                    raise ECLIncomplete('unknown_difficulty_for_ecl_selection')
                self.target(thread, ins, 0, self.argument(thread, ins, rank + 1, kind), kind)
            elif opcode in MOVEMENT_SIGNATURES:
                self.event(thread, ins, opcode, MOVEMENT_SIGNATURES[opcode], order)
            elif opcode in CONTACT_OPS:
                signature = 'ff' if opcode in (500, 501) else 'f' if opcode == 564 else 'i'
                self.event(thread, ins, opcode, signature, order)
                if opcode == 502 and int(self.events[-1]['arguments'][0]) & 0x800:
                    self.state.setdefault('ecl_damage_phase_change', dict(tick=self.frame, opcode=opcode))
            elif opcode in (511, 513, 514):
                # These modify subsequent phase interruptions. They do not
                # jump now; the motion forecast reports that independent limit.
                self.state['ecl_changed_phase_interrupt'] = self.frame
                self.state.setdefault('ecl_damage_phase_change', dict(tick=self.frame, opcode=opcode))
            elif opcode in SPAWN_OPS:
                event = dict(tick=self.frame, opcode=opcode, address=thread.ip,
                             parameter_mask=struct.unpack_from('<H', ins, 8)[0],
                             arguments_hex=ins[16:].hex())
                self.spawn_events.append(event)
                thread.initial['spawn_instructions'].append(event)
            elif opcode in GEOMETRY_NEUTRAL or (600 <= opcode <= 631) or opcode in (633, 636, 637, 638):
                if 506 <= opcode <= 545 or opcode in (633, 636, 637, 638):
                    self.state.setdefault('ecl_damage_phase_change', dict(tick=self.frame, opcode=opcode))
            else:
                raise ECLIncomplete(f'unimplemented_ecl_opcode_{opcode}')
            if cleanup:
                thread.sp -= ins[12]
            thread.ip += size
        thread.time = f32(thread.time + self.state['ecl_time_step'])
        if thread.variable_interpolators:
            raise ECLIncomplete('active_ecl_variable_interpolator')

    @staticmethod
    def arithmetic(thread, opcode):
        if opcode in (71, 72):
            x = thread.pop('i' if opcode == 71 else 'f')
            thread.push(None if x is None else int(x == 0), 'i'); return
        kind = 'f' if opcode in (51, 53, 55, 57, 60, 62, 64, 66, 68, 70) else 'i'
        b, a = thread.pop(kind), thread.pop(kind)
        out_kind = kind if opcode <= 58 else 'i'
        if a is None or b is None:
            thread.push(None, out_kind); return
        if opcode in (50, 51): value = a + b
        elif opcode in (52, 53): value = a - b
        elif opcode in (54, 55): value = a * b
        elif opcode in (56, 57): value = None if b == 0 else a / b
        elif opcode == 58: value = None if b == 0 else a - math.trunc(a / b) * b
        elif opcode in (59, 60): value = int(a == b)
        elif opcode in (61, 62): value = int(a != b)
        elif opcode in (63, 64): value = int(a < b)
        elif opcode in (65, 66): value = int(a <= b)
        elif opcode in (67, 68): value = int(a > b)
        elif opcode in (69, 70): value = int(a >= b)
        elif opcode == 73: value = int(bool(a) or bool(b))
        elif opcode == 74: value = int(bool(a) and bool(b))
        elif opcode == 75: value = a ^ b
        elif opcode == 76: value = a | b
        elif opcode == 77: value = a & b
        else: raise ECLIncomplete(f'unimplemented_arithmetic_{opcode}')
        thread.push(value, out_kind)

    def tick(self, frame):
        self.frame = frame
        # Native scheduler captures next BEFORE a thread runs. New async
        # threads inserted after the head start on the following native tick.
        scheduled = self.threads[:]
        for order, thread in enumerate(scheduled):
            if thread.unknown:
                continue
            try:
                self.run_thread(thread, order)
            except ECLIncomplete as exc:
                thread.unknown = str(exc)
                thread.initial.update(unknown_from_tick=frame, unmodeled=[str(exc)],
                                      stopped_instruction_pointer=thread.ip)
                if self.motion_unknown is None:
                    self.motion_unknown = (frame, str(exc))
                if self.contact_unknown is None:
                    self.contact_unknown = (frame, str(exc))
        self.threads = [t for i, t in enumerate(self.threads) if i == 0 or t.ip]


def _reset_interp(state, name, start, end, duration, mode, *, axis_modes=None, controls=None):
    interp = state['interpolators'][name]
    if duration <= 0:
        interp['duration'] = 0
        return
    if not interp['timer_flags'] & 1:
        interp['timer_flags'] |= 1
        interp['scale_pointer'] = state['image_base'] + 0xd8f58
        interp['timer_scale'] = state['effective_game_speed']
    interp.update(start=list(start), end=list(end), control1=[0.] * len(start),
                  control2=[0.] * len(start), duration=int(duration), mode=int(mode),
                  timer=0, previous_timer=-1, elapsed=0.)
    if controls:
        interp['control1'], interp['control2'] = controls
    if interp['dimension'] == 3:
        interp['flags'] = (interp['flags'] & ~1) | int(axis_modes is not None)
        if axis_modes is not None: interp['axis_modes'] = list(axis_modes)


def apply_geometry_event(state, event):
    """Replay a resolved event; no game memory or external calls required."""
    from .native_enemy_motion import _add, _sub, _wrap_angle
    op, args = event['opcode'], event['arguments']
    absolute_ops = {400, 401, 404, 405, 412, 416, 425, 428, 429, 434, 436, 438, 440, 441, 444, 445}
    name = 'absolute' if op in absolute_ops else 'relative'
    motion = state[name]
    if op in (400, 402, 416, 417):
        for axis, value in enumerate(args):
            if op in (416, 417): motion['position'][axis] = _add(motion['position'][axis], value)
            elif value > -999999: motion['position'][axis] = value
        if op in (400, 402): motion['mode'] = 0
        state['final']['position'] = [_add(a, b) for a, b in zip(
            state['absolute']['position'], state['relative']['position'])]
        limit = state['movement_limit']
        if limit['enabled']:
            for axis, size in enumerate((limit['width'], limit['height'])):
                p = state['final']['position'][axis]
                state['final']['position'][axis] = min(max(p, limit['center'][axis] - size * .5),
                                                       limit['center'][axis] + size * .5)
            state['absolute']['position'] = [_sub(a, b) for a, b in zip(
                state['final']['position'], state['relative']['position'])]
    elif op in (401, 403, 434, 435, 436, 437, 438, 439, 425, 426):
        duration = int(args[0]); controls = None; axis_modes = None
        if op in (425, 426):
            mode, xy = 8, args[3:5]
            controls = [list(args[1:3]) + [0.], list(args[5:7]) + [0.]]
        elif op in (434, 435, 438, 439):
            mode, xy, axis_modes = 0, args[3:5], [int(args[1]), int(args[2]), 0]
        else:
            mode, xy = int(args[1]), args[2:4]
        target = list(xy) + [0.]
        if op in (436, 437, 438, 439):
            target[0] = _add(motion['position'][0], -target[0] if state['flags'] & 0x80000 else target[0])
            target[1] = _add(motion['position'][1], target[1])
        target[:2] = [v if v > -999999 else motion['position'][i] for i, v in enumerate(target[:2])]
        _reset_interp(state, name + '_position', motion['position'], target, duration, mode,
                      axis_modes=axis_modes, controls=controls)
        if duration > 0: motion['mode'] = 0
    elif op in (404, 406, 428, 430, 440, 442, 444, 446):
        if op in (444, 446):
            if args[0] > -999999: motion['speed'] = args[0]
        else:
            if args[0] > -999999 or op in (440, 442):
                angle = args[0]
                if op in (404, 406, 440, 442) and state['flags'] & 0x80000:
                    angle = _wrap_angle(_sub(f32(math.pi), angle))
                motion['angle'] = _wrap_angle(angle)
                motion['_unknown_ecl_heading'] = False
            if len(args) > 1 and args[1] > -999999: motion['speed'] = args[1]
        motion['mode'] = 0
    elif op in (405, 407, 429, 431, 441, 443, 445, 447):
        duration, mode = int(args[0]), int(args[1])
        fields = ['speed'] if op in (445, 447) else ['angle'] if op in (441, 443) else ['angle', 'speed']
        for field, end in zip(fields, args[2:]):
            start = motion[field]
            if end <= -999999: end = 0. if mode == 7 else motion[field]
            if field == 'angle':
                if op in (405, 407) and state['flags'] & 0x80000 and mode != 7:
                    end = _wrap_angle(_sub(f32(math.pi), end))
                if abs(_sub(start, end)) >= f32(math.pi):
                    # 4265f6 and 42687e use distinct endpoint adjustments.
                    if op in (441, 443):
                        if end > start: end = _add(end, f32(2 * math.pi))
                        else: start = _add(start, f32(2 * math.pi))
                    elif end > start: start = _add(start, f32(2 * math.pi))
                    else: end = _add(end, f32(2 * math.pi))
            _reset_interp(state, name + '_' + field, [start], [end], duration, mode)
        if duration > 0: motion['mode'] = 0
    elif op in (412, 413):
        # 427289..427596 samples a bounded random angle, optionally reflects
        # it against movement limits/player side, then initializes angle with
        # identical endpoints and speed with (argument2 -> 0), duration arg0.
        # Native 427569..42756d writes the ANGLE mode twice and leaves the
        # SPEED mode unchanged. Real sf3302 confirms angle=4, speed=0.
        # All possible angles have unit direction norm. Angle0 below is only
        # a nominal representative; reachable_radius_by_frame covers every
        # draw without assuming other entities' shared-RNG consumption.
        duration, mode, speed = int(args[0]), int(args[1]), args[2]
        speed_mode = state['interpolators'][name + '_speed']['mode']
        _reset_interp(state, name + '_angle', [0.], [0.], duration, mode)
        _reset_interp(state, name + '_speed', [speed], [0.], duration, speed_mode)
        motion['mode'] = 0
        motion['_unknown_ecl_heading'] = duration > 0
    elif op in (502, 503):
        state['flags'] = state['flags'] | (int(args[0]) & 0xffffffff) if op == 502 else state['flags'] & ~(int(args[0]) & 0xffffffff)
        state['movement_limit']['enabled'] = bool(state['flags'] & 0x20000)
    elif op == 504:
        state['flags'] |= 0x20000
        state['movement_limit'].update(enabled=True, center=args[:2], width=args[2], height=args[3])
    elif op == 505:
        state['flags'] &= ~0x20000
        state['movement_limit']['enabled'] = False
    elif op in (632, 639):
        state['tick_callback'] = event.get('callback')


def forecast_ecl(memory, state, horizon=24):
    """Resolve ECL on the coherent paused observation; return JSON-safe events."""
    from .native_enemy_motion import _step_motion, ALREADY_UPDATED_FLAG
    work = copy.deepcopy(state)
    answer = dict(revision=REVISION, horizon=horizon, events=[], threads=[],
                  unknown_from_tick=None, unknown_reason=None, instruction_count=0,
                  contact_unknown_from_tick=None, contact_unknown_reason=None,
                  spawn_instructions=[], position_samples=[], reachable_radius_by_frame=[0.])
    try:
        world = ECLWorld(memory, work)
    except ECLIncomplete as exc:
        answer.update(unknown_from_tick=1, unknown_reason=str(exc))
        return answer
    for frame in range(1, horizon + 1):
        skip = bool(work['flags'] & ALREADY_UPDATED_FLAG)
        interpolating = {name: work['interpolators'][name + '_position']['duration'] > 0
                         for name in ('absolute', 'relative')}
        try:
            _step_motion(work)
        except NotImplementedError as exc:
            answer.update(unknown_from_tick=frame, unknown_reason=str(exc))
            break
        radius = answer['reachable_radius_by_frame'][-1]
        if not skip:
            for name in ('absolute', 'relative'):
                if work[name].get('_unknown_ecl_heading') and not interpolating[name]:
                    # Triangle inequality between true and nominal paths.
                    # Native scalar interpolation supplies the exact speed;
                    # 0.06 > 4*sqrt(2)*.01 covers two per-axis centipixel
                    # floors on each of the true/nominal paths per tick.
                    radius += 2.00001 * abs(work[name]['speed'] * work['effective_game_speed']) + .06
        work['_ecl_reachable_radius'] = radius
        answer['reachable_radius_by_frame'].append(radius)
        if not skip:
            if work['ecl_time_step'] is None:
                answer.update(unknown_from_tick=frame, unknown_reason='unknown_ecl_clock')
                break
            world.tick(frame)
        answer['position_samples'].append(work['final']['position'][:])
        if work.get('tick_callback'):
            if world.motion_unknown is None:
                world.motion_unknown = (frame, 'unmodeled_native_tick_callback')
    answer.update(events=world.events, threads=[t.initial for t in world.all_threads],
                  instruction_count=world.steps, spawn_instructions=world.spawn_events,
                  unresolved_variables=sorted(world.unresolved_variables))
    answer['damage_phase_change'] = work.get('ecl_damage_phase_change')
    if world.motion_unknown and answer['unknown_from_tick'] is None:
        answer['unknown_from_tick'], answer['unknown_reason'] = world.motion_unknown
    if world.contact_unknown:
        answer['contact_unknown_from_tick'], answer['contact_unknown_reason'] = world.contact_unknown
    answer['random_motion_events'] = [e for e in world.events if e['opcode'] in (412, 413)]
    if answer['random_motion_events']:
        answer['reachable_scope'] = ('conditional on captured phase/callback program; native 412/413 '
            'unknown heading with exact speed interpolation; radius bounds true-vs-nominal displacement')
    return answer
