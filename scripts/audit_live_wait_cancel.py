"""Passive WAIT->EX0x400 observations inside one already-owned native worker."""
import os
import struct
import sys
from pathlib import Path


def arm(output,expected_pid,limit=16):
    from fly_touhou.hazard_reader import VersionedHazardReader
    from fly_touhou.thrl_durable_records import write_json
    if os.getpid()!=expected_pid:raise ValueError('Wrong owned worker')
    cls=VersionedHazardReader
    if getattr(cls,'_wait_cancel_probe',False):raise RuntimeError('Probe already armed')
    original=cls._bullet_metadata;pending={};pairs=[];calls=0;qualifying_calls=0;done=False
    path=Path(output)

    def save(status,error=None):
        nonlocal done
        done=True
        if cls._bullet_metadata is observed:cls._bullet_metadata=original
        cls._wait_cancel_probe=False
        write_json(path,dict(status=status,PID=os.getpid(),calls=calls,pairs=pairs,pending=len(pending),
            error=error,reader_restored=True,extra_native_steps=0,
            policy_inputs_changed=False,scope='Same original-game snapshots; no policy or teacher changes'))

    def context():
        frame=sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name=='read' and Path(frame.f_code.co_filename).name=='hazard_reader.py':
                snapshot=frame.f_locals.get('snapshot')
                if snapshot is not None:return snapshot.stage,snapshot.frame
            frame=frame.f_back
        return None

    def observed(self,bullet):
        nonlocal calls,qualifying_calls
        raw=original(self,bullet)
        if done:return raw
        calls+=1
        try:
            previous=pending.get(bullet.id)
            record=raw.get('native_ex_current_record_hex')
            eligible=(raw['native_state']==1 and raw.get('native_ex_active_flags')==0x80000000
                and type(raw.get('native_ex_wait_remaining')) is int
                and 0<=raw['native_ex_wait_remaining']<=3 and record is not None
                and struct.unpack_from('<II',bytes.fromhex(record),32)==(0x400,0))
            if previous or eligible:
                qualifying_calls+=1
                here=context()
                if here is not None:
                    stage,frame=here
                    if previous and frame-previous['frame']>=previous['raw']['native_ex_wait_remaining']+1:
                        pairs.append(dict(before=previous,after=dict(frame=frame,stage=stage,
                            kind=bullet.kind,raw=dict(raw),
                            active_ex_flags=struct.unpack('<I',self.mem.read(bullet.id+0xc04,4))[0]),
                            native_frames=frame-previous['frame']))
                        pending.pop(bullet.id,None)
                        write_json(path,dict(status='sampling',PID=os.getpid(),pairs=pairs,
                            calls=calls,reader_restored=False,extra_native_steps=0,policy_inputs_changed=False))
                    if eligible and bullet.id not in pending and len(pending)+len(pairs)<limit:
                        index=raw['native_ex_index']
                        following=struct.unpack('<II',self.mem.read(bullet.id+0xc28+(index+1)*44+32,8)) if index+1<18 else (0,0)
                        rate=struct.unpack('<f',self.mem.read(self.mem.base+0xd8f58,4))[0]
                        pending[bullet.id]=dict(id=bullet.id,stage=stage,frame=frame,kind=bullet.kind,
                            raw=dict(raw,native_ex_after_cancel_opcode=following[0],
                                native_ex_after_cancel_condition=following[1],native_motion_time_rate=rate))
            if len(pairs)>=limit:save('completed')
            elif qualifying_calls>=24000:save('sample_limit')
        except Exception as error:save('diagnostic_error',repr(error))
        return raw

    cls._wait_cancel_probe=True
    cls._bullet_metadata=observed
    write_json(path,dict(status='armed',PID=os.getpid(),pairs=[],calls=0,
        reader_restored=False,extra_native_steps=0,policy_inputs_changed=False,
        call_bound_starts_on_relevant_wait_cancel=True))
