"""Preload original TH14 programs, including unreachable/conditional branches.

This is a bytecode catalogue, not a replay-derived trajectory database. Original
game bytes stay in the user's external cache, never in the source distribution.
Every instruction retains its difficulty mask, variable mask and source offset.
"""
from pathlib import Path
import collections
import json
import re
import struct

SCHEMA = 'th14-complete-ecl-rule-catalog-v1'
EMIT_OPS = {601, 702, 703, 705, 711}
EX_OPS = {609, 610, 611, 612}
SPAWN_OPS = {300, 301, 304, 305, 309, 310, 311, 312}


def _strings(data, offset, count):
    result = []
    for _ in range(count):
        end = data.index(b'\0', offset)
        result.append(data[offset:end].decode('cp932'))
        offset = end + 1
    return result, offset


def decode_program(path):
    path = Path(path)
    data = path.read_bytes()
    if data[:4] != b'SCPT':
        raise ValueError(f'{path}: expected TH14 SCPT')
    include_size, include_offset = struct.unpack_from('<HI', data, 6)
    count = struct.unpack_from('<I', data, 16)[0]
    cursor, includes = include_offset, {}
    while cursor < include_offset + include_size:
        kind = data[cursor:cursor + 4].decode('ascii')
        n = struct.unpack_from('<I', data, cursor + 4)[0]
        names, cursor = _strings(data, cursor + 8, n)
        cursor = (cursor + 3) & ~3
        includes[kind] = names
    table = include_offset + include_size
    offsets = struct.unpack_from('<' + 'I' * count, data, table)
    names, _ = _strings(data, table + 4 * count, count)
    subs = []
    for index, (name, start) in enumerate(zip(names, offsets)):
        if data[start:start + 4] != b'ECLH':
            raise ValueError(f'{path}:{name}: expected ECLH')
        body = start + struct.unpack_from('<I', data, start + 4)[0]
        end = offsets[index + 1] if index + 1 < count else len(data)
        instructions = []
        cursor = body
        while cursor < end:
            time, opcode, size, mask, rank, argc, cleanup = struct.unpack_from('<iHHHBBI', data, cursor)
            if size < 16 or cursor + size > end:
                raise ValueError(f'{path}:{name}:{cursor}: malformed instruction')
            raw = data[cursor:cursor + size]
            instructions.append(dict(offset=cursor, relative_offset=cursor - body,
                time=time, opcode=opcode, size=size, parameter_mask=mask,
                difficulty_mask=rank, argument_count=argc,
                stack_cleanup_bytes=cleanup, raw_hex=raw.hex()))
            cursor += size
        subs.append(dict(name=name, offset=start, code_offset=body, end_offset=end,
                         instructions=instructions))
    return dict(file=path.name, bytes=len(data), includes=includes, subroutines=subs)


def instruction_string(row, argument=0):
    raw = bytes.fromhex(row['raw_hex']); offset = 16 + argument * 4
    length = struct.unpack_from('<I', raw, offset)[0]
    if length > len(raw) - offset - 4:
        return None
    return raw[offset + 4:offset + 4 + length].split(b'\0', 1)[0].decode('cp932')


def literal(row, index, kind='i'):
    if row['parameter_mask'] & (1 << index):
        return None
    raw = bytes.fromhex(row['raw_hex'])
    return struct.unpack_from('<' + kind, raw, 16 + 4 * index)[0]


def summarize(programs, *, difficulty_mask=8):
    rows, totals = [], collections.Counter()
    for program in programs:
        for sub in program['subroutines']:
            live = [r for r in sub['instructions'] if r['difficulty_mask'] & difficulty_mask]
            ops = collections.Counter(r['opcode'] for r in live)
            totals.update(ops)
            emit = [r for r in live if r['opcode'] in EMIT_OPS]
            extensions, modes, styles = [], [], []
            dependencies = set()
            for r in live:
                op = r['opcode']
                if op in EX_OPS:
                    index = 3 if op in (609, 610) else 2
                    value = literal(r, index)
                    extensions.append(dict(offset=r['offset'], opcode=op,
                        command=None if value is None else value & 0xffffffff,
                        dynamic=value is None))
                if op == 607:
                    modes.append(literal(r, 1))
                if op == 602:
                    styles.append(literal(r, 1))
                raw = bytes.fromhex(r['raw_hex'])
                for index in range(min(16, (len(raw) - 16) // 4)):
                    if r['parameter_mask'] & (1 << index):
                        iv = struct.unpack_from('<i', raw, 16 + 4 * index)[0]
                        fv = struct.unpack_from('<f', raw, 16 + 4 * index)[0]
                        for value in (iv, fv):
                            if -10000 <= value <= -9900 and int(value) == value:
                                dependencies.add(int(value))
            calls = []
            for r in live:
                if r['opcode'] in {11, 15, 16, 556} | SPAWN_OPS:
                    try: name = instruction_string(r)
                    except (ValueError, struct.error, UnicodeDecodeError): name = None
                    if name: calls.append(dict(opcode=r['opcode'], target=name, offset=r['offset']))
            rows.append(dict(file=program['file'], subroutine=sub['name'],
                instruction_count=len(live), opcodes=dict(sorted(ops.items())),
                emit_sites=[dict(offset=r['offset'], opcode=r['opcode'], time=r['time']) for r in emit],
                aim_modes=sorted(set(modes), key=lambda v: -1 if v is None else v),
                bullet_styles=sorted(set(styles), key=lambda v: -1 if v is None else v),
                ex_commands=extensions, variable_dependencies=sorted(dependencies), calls=calls))
    return dict(difficulty='Lunatic', functions=rows, opcodes=dict(sorted(totals.items())),
        function_count=len(rows), instruction_count=sum(totals.values()),
        emission_sites=sum(len(r['emit_sites']) for r in rows),
        ex_command_types=sorted({e['command'] for r in rows for e in r['ex_commands'] if e['command'] is not None}),
        aim_modes=sorted({m for r in rows for m in r['aim_modes'] if m is not None}),
        bullet_styles=sorted({m for r in rows for m in r['bullet_styles'] if m is not None}))


def build_catalog(directory):
    root = Path(directory)
    files = sorted(p for p in root.glob('*.ecl') if re.fullmatch(r'(default|st0[1-6][a-z0-9]*)\.ecl', p.name))
    required = {'default.ecl'} | {f'st0{s}{part}.ecl' for s in range(1, 7) for part in ('', 'bs', 'mbs')} | {'st04bs2.ecl','st04mbs2.ecl'}
    missing = required - {p.name for p in files}
    if missing: raise ValueError(f'Missing original six-stage programs: {sorted(missing)}')
    programs = [decode_program(p) for p in files]
    return dict(schema=SCHEMA, programs=programs, summary=summarize(programs),
        source='original game bytecode; no teacher trajectories',
        includes_both_stage4_branches=True, all_difficulties_retained=True,
        bytecode_coverage_complete=True, forward_simulator_complete=False)


def write_catalog(directory, output):
    result = build_catalog(directory)
    Path(output).write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    return result
