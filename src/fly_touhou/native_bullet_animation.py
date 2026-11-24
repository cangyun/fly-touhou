"""Physical parts of TH14 bullet ANM: release gate and sprite edge extents.

The original VM executes after wrap/collision. Its visual scale is not the
collision scale. 46fe50, 46f1e0 and sprite callback41bb70 define this contract.
Only read the original script; no game asset is bundled here.
"""
import struct
import numpy as np
from numba import njit


def read_animation(memory,address):
    vm=address+0x28;block=memory.read(vm,0x5cc)
    pointer=struct.unpack_from('<I',block,0x38)[0];current=struct.unpack_from('<I',block,0x28)[0]
    resource=struct.unpack_from('<I',block,0x30)[0]
    style,color=struct.unpack('<hh',memory.read(address+0x13ec,4))
    callback=struct.unpack_from('<I',block,0x5ac)[0]
    contract=_read_animation_contract(memory,pointer,resource,style,color,callback)
    pc=contract['offsets'].index(current-pointer) if current-pointer in contract['offsets'] else -1
    return dict(contract,pc=pc,clock=struct.unpack_from('<i',block,0x54c)[0],
        gate=struct.unpack_from('<i',block,0x4ac)[0],interrupt=struct.unpack_from('<i',block,0x4a0)[0])


def _read_animation_contract(memory,pointer,resource,style,color,callback):
    cache=getattr(memory,'_bullet_anm_cache',None)
    if cache is None:cache={};memory._bullet_anm_cache=cache
    key=(pointer,resource,style,color,callback)
    if key not in cache:
        data=memory.read(pointer,4096);rows=[];offsets=[];offset=0
        while offset+8<=len(data):
            op,size,t,mask=struct.unpack_from('<HHhH',data,offset)
            if op==65535:break
            if size<8 or offset+size>len(data):raise ValueError('Incomplete native ANM program')
            args=struct.unpack_from('<'+'i'*((size-8)//4),data,offset+8)
            rows.append([op,t,mask,*(list(args[:2])+[0,0])[:2],len(rows)+1,-1]);offsets.append(offset);offset+=size
        else:raise ValueError('Native bullet ANM script exceeds captured block')
        locations={v:i for i,v in enumerate(offsets)};supported=True;sizes={}
        table=struct.unpack('<I',memory.read(resource+0x11c,4))[0]
        for row in rows:
            op=row[0]
            if op==200:row[6]=locations.get(row[3],-1);supported &= row[6]>=0 and row[2]==0
            elif op==300:
                supported &= row[2]==0
                index=row[3]
                if callback==memory.base+0x1bb70:
                    definition=memory.base+0xd1e94+style*0x114
                    first=struct.unpack('<i',memory.read(definition,4))[0]
                    if first>=0:index=struct.unpack('<i',memory.read(definition+color*16+index*4,4))[0]
                elif callback:supported=False
                if index>=0:
                    h,w=struct.unpack('<ff',memory.read(table+index*0x48+0x34,8));sizes[str(row[3])]=[w,h]
                else:supported=False
            elif op==100:supported &= row[2]==1 and row[3]==10000
            elif op not in (1,3,5,302,303,304,401,402,403,409,412,415,424,436):supported=False
        cache[key]=dict(rows=rows,offsets=offsets,sprite_sizes=sizes,supported=bool(supported))
    return cache[key]


def initial_ex2_half_steps(memory, style, color):
    """Minimum half-speed steps for a fresh EX2 argument-1 shot in unit time."""
    return max(0, initial_ex2_gate_updates(memory, style, color) - 3)


def initial_ex2_gate_updates(memory, style, color):
    """Read the number of initial ANM updates until the EX2 gate opens."""
    from .native import NativeError

    cache = getattr(memory, '_initial_ex2_gate_updates', None)
    if cache is None:
        cache = memory._initial_ex2_gate_updates = {}
    key = style, color
    if key in cache:
        return cache[key]
    try:
        index, = struct.unpack('<i', memory.read(memory.base + 0xd1e90 + style * 0x114, 4))
        manager, = struct.unpack('<I', memory.read(memory.base + 0xdb530, 4))
        resource, = struct.unpack('<I', memory.read(manager + 0x9bf6cc, 4))
        table, _, state = struct.unpack('<3I', memory.read(resource + 0x120, 12))
        if state or not table or not 0 <= index < 4096:
            return 0
        pointer, = struct.unpack('<I', memory.read(table + index * 4, 4))
        contract = _read_animation_contract(memory, pointer, resource, style, color,
                                            memory.base + 0x1bb70) if pointer else None
    except (NativeError, OSError, struct.error, ValueError):
        contract = None
    gate_updates = 0
    if contract and contract['supported']:
        program = np.asarray(contract['rows'], dtype=np.int64)
        sizes = np.zeros((len(program), 2), np.float32)
        for i, row in enumerate(program):
            if row[0] == 300:
                sizes[i] = contract['sprite_sizes'][str(row[3])]
        pc = clock = gate = 0
        width = height = 16.
        for calls in range(1, 121):
            pc, clock, gate, width, height, dead = step(program, sizes, pc, clock, gate, 0, width, height)
            if dead:
                break
            if gate:
                # 46f637 loads and ticks the VM, 4180b5 ticks after the
                # EX2 rewind, and 416d31 ticks on the regular update.
                # 417fe9 skips the interrupt for EX2 argument 1.
                gate_updates = calls
                break
    cache[key] = gate_updates
    return gate_updates


@njit(cache=True)
def step(program,sizes,pc,clock,gate,interrupt,width,height):
    if interrupt:
        for i in range(len(program)):
            if program[i,0]==5 and program[i,3]==interrupt:
                pc=i+1;clock=program[i,1];break
        interrupt=0
    for _ in range(512):
        if pc<0 or pc>=len(program):return pc,clock,gate,width,height,True
        op,t,mask,a,b,nxt,jump=program[pc]
        if t>clock:break
        if op==1:return -1,clock,gate,width,height,True
        if op==3:break
        if op==200:pc=jump;clock=b;continue
        if op==100:gate=b
        if op==300:width=sizes[pc,0];height=sizes[pc,1]
        pc=nxt
    return pc,clock+1,gate,width,height,False
