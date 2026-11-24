"""Read TH14 1.00b replay metadata and its original per-frame input stream.

Format reference: n-rook/thscoreboard ref/threp-ksy/th14.ksy (Apache-2.0).
Decryption uses the separately installed Unlicense package tsadecode.
Replay metadata alone does not certify a native no-Miss completion.
"""
from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np


@dataclass
class ReplayStage:
    metadata: dict
    inputs: np.ndarray


def read_replay(path):
    import tsadecode
    raw=Path(path).read_bytes()
    if raw[:4]!=b't13r':raise ValueError('Expected the TH13/14 replay container')
    packed_length,unpacked_length=struct.unpack_from('<II',raw,28)
    packed=bytearray(raw[36:36+packed_length])
    tsadecode.decrypt(packed,0x400,0x5c,0xe1)
    tsadecode.decrypt(packed,0x100,0x7d,0x3a)
    data=bytes(tsadecode.unlzss(packed))
    if len(data)!=unpacked_length:raise ValueError('Replay decompressed length differs')
    stage_count,shot,subshot,difficulty,cleared,_,spell_id=struct.unpack_from('<7I',data,120)
    metadata=dict(name=data[:12].split(b'\0')[0].decode('shift_jis').strip(),
        timestamp=struct.unpack_from('<Q',data,12)[0],score=10*struct.unpack_from('<I',data,20)[0],
        slowdown=struct.unpack_from('<f',data,116)[0],stage_count=stage_count,
        shot=shot,subshot=subshot,difficulty=difficulty,cleared=cleared,spell_id=spell_id,
        native_revalidated=False)
    stages=[];offset=148
    for _ in range(stage_count):
        stage,seed,frames,length=struct.unpack_from('<HHII',data,offset)
        if length!=frames*6+(frames+29)//30:
            raise ValueError('Unknown TH14 input/FPS payload layout')
        payload=data[offset+220:offset+220+length]
        inputs=np.frombuffer(payload[:frames*6],dtype='<u2').reshape(frames,3).copy()
        held,pressed,released=inputs.T
        if (not np.array_equal(pressed[1:],held[1:] & ~held[:-1])
                or not np.array_equal(released[1:],held[:-1] & ~held[1:])):
            raise ValueError('Recorded key transitions do not match held inputs')
        desc=dict(stage=stage,native_seed=seed,frame_count=frames,
            continues=struct.unpack_from('<I',data,offset+36)[0],
            power=struct.unpack_from('<I',data,offset+68)[0],
            lives=struct.unpack_from('<I',data,offset+80)[0],
            bombs=struct.unpack_from('<I',data,offset+92)[0],
            Bomb_pressed_frames=int(np.count_nonzero(held&2)),
            shot_released_frames=int(np.count_nonzero((held&1)==0)))
        stages.append(ReplayStage(desc,inputs));offset+=220+length
    if offset!=len(data):raise ValueError('Unparsed bytes after TH14 replay stages')
    metadata['stages']=[s.metadata for s in stages]
    return metadata,stages


def movement_distribution(masks):
    """Four human frames may change direction; preserve that soft target.

    This is demonstration supervision, never a held-four-frame PPO action.
    Opposing directions cancel as in the original movement input.
    """
    from ...state import decode_action
    from ...thrl_native_env import SOURCE19_TO_NATIVE36
    lookup={decode_action(native):a for a,native in enumerate(SOURCE19_TO_NATIVE36[:18])}
    result=np.zeros(18,np.float32)
    for raw in masks:
        m=int(raw);direction=(int(bool(m&128))-int(bool(m&64)),
            int(bool(m&32))-int(bool(m&16)),bool(m&8))
        result[lookup[direction]]+=1
    if len(masks):result/=len(masks)
    return result
