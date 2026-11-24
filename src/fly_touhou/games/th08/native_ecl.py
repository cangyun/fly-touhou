"""Script and emitter state at an original TH08 1.00d input checkpoint."""

from dataclasses import dataclass
import struct

from ...native import NativeError
from ...th08_enemies import MANAGER_RVA


# Retail layouts cross-referenced against EclManager.hpp and EnemyManager.hpp
# in N0zoM1z0/th08 at 55e4aad7cb7d5707f40df86df97530ab5b3830a0.
ECL_RUNTIME_RVA = 0xECCB8
RNG_RVA = 0x124D520
ECL_FILE_HEADER = 0x48
ECL_MAX_BYTES = 16 * 1024 * 1024
ECL_SUB_HEADER = struct.Struct("<iHHBBH")
ECL_TIMELINE_HEADER = struct.Struct("<iHBB")
TIMER = struct.Struct("<ifi")
CONTEXT_SIZE = 0x228
MANAGER_STATE_OFFSET = 0x9DCDC0
MANAGER_STATE_SIZE = 0x150


@dataclass(frozen=True)
class TH08RngState:
    """Live generator state. The replay manager resets draws each calculation."""
    seed: int
    saved_seed: int
    draws: int

    def after_draws(self, count):
        seed = self.seed
        for _ in range(count):
            difference = ((seed ^ 0x9630) - 0x6553) & 0xFFFF
            seed = ((difference << 2) | (difference >> 14)) & 0xFFFF
        return TH08RngState(seed, self.saved_seed, (self.draws + count) & 0xFFFFFFFF)


@dataclass(frozen=True)
class TH08EclProgram:
    address: int
    data: bytes
    subroutines: tuple[int, ...]
    timelines: tuple[int, ...]

    def instruction(self, address, *, timeline=False):
        if not address:
            return None
        header = ECL_TIMELINE_HEADER if timeline else ECL_SUB_HEADER
        offset = address - self.address
        if not 0 <= offset <= len(self.data) - header.size:
            raise NativeError(f"TH08 live ECL pointer {address:#x} is outside the loaded script")
        values = header.unpack_from(self.data, offset)
        end = offset + _record_bytes(values, header)
        if end > len(self.data):
            raise NativeError(f"TH08 live ECL instruction crosses the loaded script at {offset:#x}")
        return {"offset": offset, "time": values[0], "opcode": values[1], "difficulty": values[3 if timeline else 4],
                "operand_flags": 0 if timeline else values[5], "size": end - offset,
                "operands": self.data[offset + header.size:end]}

    def instructions(self, *, timeline):
        for pc in self.timelines if timeline else self.subroutines:
            while True:
                instruction = self.instruction(pc, timeline=timeline)
                if instruction["time"] < 0 and (timeline or instruction["opcode"] == 0xFFFF):
                    break
                yield instruction
                pc += instruction["size"]


def _record_bytes(values, header):
    length = values[2]
    if header == ECL_TIMELINE_HEADER and values[0] < 0:
        return header.size
    if length < header.size or length % 4:
        raise NativeError(f"Invalid TH08 ECL record length {length} for opcode {values[1]:#x}")
    return length


def _program_end(memory, address, subroutines, timelines):
    entry = max(subroutines + timelines)
    header = ECL_TIMELINE_HEADER if entry in timelines else ECL_SUB_HEADER
    cursor = entry
    while cursor - address <= ECL_MAX_BYTES - header.size:
        values = header.unpack(memory.read(cursor, header.size))
        cursor += _record_bytes(values, header)
        if values[0] < 0 and (header == ECL_TIMELINE_HEADER or values[1] == 0xFFFF):
            return cursor
    raise NativeError("The loaded TH08 ECL script has no final sentinel")


def _context(program, raw, offset):
    instruction, = struct.unpack_from("<I", raw, offset)
    callback, callback_pc = struct.unpack_from("<II", raw, offset + 0x10)
    return {
        "subroutine": struct.unpack_from("<h", raw, offset + 0x224)[0],
        "timer": TIMER.unpack_from(raw, offset + 4),
        "instruction": program.instruction(instruction),
        "callback": callback,
        "callback_instruction": program.instruction(callback_pc) if callback else None,
        "raw": raw[offset:offset + CONTEXT_SIZE],
    }


def _stack(program, raw, offset, depth):
    if not 0 <= depth <= 16:
        raise NativeError(f"TH08 ECL stack depth {depth} exceeds the native layout")
    return tuple(_context(program, raw, offset + index * CONTEXT_SIZE) for index in range(depth))


def _emitter(memory, program, raw):
    children = []
    for pointer in struct.unpack_from("<4I", raw, 0x3384):
        if not pointer:
            children.append(None)
            continue
        child = memory.read(pointer, 0x24B0)
        depth = struct.unpack_from("<h", child, 6)[0]
        children.append({"pointer": pointer, "context": _context(program, child, 8),
                         "stack": _stack(program, child, 0x230, depth)})
    main_depth = struct.unpack_from("<h", raw, 0x2CE8)[0]
    return {
        "main": _context(program, raw, 0x7F8),
        "stack": _stack(program, raw, 0xA20, main_depth),
        "children": tuple(children),
        "shot_interval": struct.unpack_from("<i", raw, 0x3060)[0],
        "shot_timer": TIMER.unpack_from(raw, 0x3064),
        # Also retains movement, flags, native shot/laser descriptors and
        # pending instructions, damage callbacks, and emitter attachments.
        "raw": raw,
    }


class TH08LiveEclReader:
    """Load the stage bytecode once and capture all occupied script contexts."""

    def __init__(self):
        self.program = None
        self.program_key = None

    def _loaded_program(self, memory, stage, runtime):
        address, sub_table = struct.unpack_from("<II", runtime)
        if not address or sub_table != address + ECL_FILE_HEADER:
            raise NativeError("TH08 has no loaded stage ECL at this checkpoint")
        header = memory.read(address, ECL_FILE_HEADER)
        sub_count, timeline_count = struct.unpack_from("<hh", header, 4)
        if sub_count < 0 or not 0 <= timeline_count <= 16 or sub_count + timeline_count == 0:
            raise NativeError("TH08 loaded ECL contains an invalid program table")
        key = stage, address, header
        if self.program_key != key:
            subroutines = struct.unpack(f"<{sub_count}I", memory.read(sub_table, 4 * sub_count)) if sub_count else ()
            timelines = struct.unpack_from(f"<{timeline_count}I", header, 8)
            first_code = sub_table + 4 * sub_count
            if any(not first_code <= entry < address + ECL_MAX_BYTES for entry in subroutines + timelines):
                raise NativeError("TH08 ECL table points outside the loaded script")
            last = _program_end(memory, address, subroutines, timelines)
            self.program = TH08EclProgram(address, memory.read(address, last - address), subroutines, timelines)
            self.program_key = key
        return self.program

    def read(self, memory, stage, enemies):
        runtime = memory.read(memory.base + ECL_RUNTIME_RVA, 0x188)
        program = self._loaded_program(memory, stage, runtime)
        manager = memory.read(memory.base + MANAGER_RVA + MANAGER_STATE_OFFSET, MANAGER_STATE_SIZE)
        timelines = []
        for index, start in enumerate(program.timelines):
            offset = 0x10 + index * 0x10
            pc = struct.unpack_from("<I", manager, offset + 0xC)[0]
            timelines.append({"timer": TIMER.unpack_from(manager, offset),
                              "instruction": program.instruction(pc or start, timeline=True)})
        rng = read_live_rng(memory)
        return {"program": program, "runtime": runtime, "manager": manager, "timelines": tuple(timelines), "rng": rng,
                "emitters": {slot: _emitter(memory, program, raw) for slot, raw in enemies.items()}}


def read_live_rng(memory):
    return TH08RngState(*struct.unpack("<HHI", memory.read(memory.base + RNG_RVA, 8)))
