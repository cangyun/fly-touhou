"""TH14's stage-selectable RPY container for a private, student-controlled Stage 6 branch."""
import struct
import time
from pathlib import Path

import numpy as np


def _literal_stream(body):
    # TH14 ends the stream at its declared packed length.
    raw=np.frombuffer(body,dtype=np.uint8)
    tokens=np.ones((len(raw),9),dtype=np.uint8)
    tokens[:,1:]=np.unpackbits(raw).reshape(-1,8)
    return np.packbits(tokens.ravel()).tobytes()


def _encrypt(packed):
    data=np.frombuffer(packed,dtype=np.uint8)
    for size,key,step in ((256,0x7d,0x3a),(1024,0x5c,0xe1)):
        span=len(data)-(len(data)&1)
        remainder=len(data)%size
        if remainder<size//4:span=max(0,span-remainder)
        packed=data.copy()
        for start in range(0,span,size):
            count=min(size,span-start)
            order=np.r_[np.arange(count-1,-1,-2),np.arange(count-2,-1,-2)]
            keys=(np.arange(start,start+count,dtype=np.uint32)*step+key).astype(np.uint8)
            packed[start:start+count]=data[start+order]^keys
        data=packed
    return data.tobytes()


def write_stage_replay(path,record,frames=131072):
    """Build a one-stage recording from the original game's own captured headers."""
    if (record['stage'],record['difficulty'],record['shot'],record['subshot'],record['continues'])!=(6,3,0,0,0):
        raise ValueError('Expected an original Lunatic ReimuA sixth-stage recording')
    inner=bytearray.fromhex(record['replay_header'])
    stage=bytearray.fromhex(record['stage_header'])
    if len(inner)!=148 or len(stage)!=220 or struct.unpack_from('<HH',stage)!=(6,record['stage_rng_seed']):
        raise ValueError('Invalid TH14 stage header')
    initial=bytes.fromhex(record['initial_key_frames'])
    if not initial or len(initial)%6 or len(initial)//6>frames:
        raise ValueError('The original opening key transitions are required')
    inner[:8]=b'FLYSTAGE'
    struct.pack_into('<Q',inner,12,int(time.time()))
    struct.pack_into('<I',inner,120,1)
    clock_log=bytes([60])*((frames+29)//30)
    struct.pack_into('<II',stage,4,frames,6*frames+len(clock_log))
    # Complete the recorded key transition before the remaining empty frames.
    actions=bytearray(6*frames);actions[:len(initial)]=initial
    if len(initial)<len(actions):
        struct.pack_into('<H',actions,len(initial)+4,struct.unpack_from('<H',initial,len(initial)-6)[0])
    body=inner+stage+actions+clock_log
    encrypted=_encrypt(_literal_stream(body))
    outer=bytearray.fromhex(record['container_header'])
    if len(outer)!=36 or outer[:4]!=b't13r':raise ValueError('Invalid TH14 container')
    struct.pack_into('<I',outer,12,len(outer)+len(encrypted))
    struct.pack_into('<II',outer,28,len(encrypted),len(body))
    user=('東方輝針城 リプレイファイル情報\r\nVersion 1.00b\r\nName FLYSTAGE\r\n'
          'Chara ReimuA\r\nRank Lunatic\r\nStage 6\r\n').encode('cp932')+b'\x00\x00'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(outer+encrypted+b'USER'+struct.pack('<II',12+len(user),0)+user)


class StageSixKeys:
    """Supply ordinary legal movement keys to the next TH14 replay ticks of an owned branch."""
    def __init__(self,controller,record):
        from ...native import ProcessMemory
        self.controller=controller
        self.seed=record['stage_rng_seed']
        self.memory=ProcessMemory(controller.pid,writable=True)

    def close(self):
        self.memory.close()

    def supply(self,mask,ticks):
        from ...native import NativeError
        if mask&~0xf9 or not 1<=ticks<=128:
            raise ValueError('Stage branch accepts only movement, focus and shot keys')
        m=self.memory
        rec=m.integer(m.base+0xdb688)
        if m.integer(m.base+0xf58a4)!=6 or not rec or m.integer(rec+0x10)!=1:
            raise NativeError('The owned TH14 replay has left its sixth stage')
        stage=m.integer(rec+0x218)
        if stage!=6:raise NativeError('Stage-six input stream is not active')
        slot=rec+40*stage+0xc8
        base=m.integer(slot);current=m.integer(slot+4);header=m.integer(slot+16)
        if not header or struct.unpack('<HH',m.read(header,4))!=(6,self.seed):
            raise NativeError('Replay header does not match the archived entry')
        frames=m.integer(header+4)
        if not base or not base<=current or (current-base)%6 or current+6*ticks>base+6*frames:
            raise NativeError('No complete input interval remains in the native replay')
        held=m.integer(current-6,size=2) if current>base else 0
        first=struct.pack('<HHH',mask,mask&~held,held&~mask)
        m.write(current,first+struct.pack('<HHH',mask,0,0)*(ticks-1))
