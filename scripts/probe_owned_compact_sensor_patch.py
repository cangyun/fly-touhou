"""Read the new focus and curve-laser inputs from existing owned worlds.

No game is started, suspended, resumed, injected into, or sent an action.
"""
from pathlib import Path
import argparse
import json
from types import SimpleNamespace

import numpy as np
import psutil

from fly_touhou.native import NativeReader, ProcessMemory, Profile
from fly_touhou.native_curve_laser_reader import read_native_curve_lasers
from fly_touhou.native_curve_laser_contact import curve_laser_contact
from fly_touhou.native_focus_gate import read_native_focus_gate
from fly_touhou.thrl_durable_records import write_json
from fly_touhou.thrl_study_io import read_shared_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ownership',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--world',type=int,help='Read only this owned world if others have finished')
    a=p.parse_args()
    ownership=read_shared_json(a.ownership)
    profile=Profile.load(a.profile)
    reports=[]
    for entry in ownership['workers']:
        if a.world is not None and entry['index']!=a.world:
            continue
        identity=entry['native_identity'];pid=identity['pid']
        process=psutil.Process(pid)
        if abs(process.create_time()-identity['create_time'])>.01:
            continue
        memory=ProcessMemory(pid,writable=False,can_suspend=False)
        try:
            # Reuse only the original reader's laser methods.  This probe does
            # not instantiate or alter the training environment.
            reader=object.__new__(NativeReader)
            reader.mem=memory;reader.profile=profile;reader.o=profile.offsets
            before=reader.global_frame()
            lasers=reader._lasers()
            player=reader.pointer('player','player_pointer')
            position=memory.pair(player+profile.offsets['player']['zPlayer_pos'])
            geometry=read_native_curve_lasers(memory,SimpleNamespace(frame=before,lasers=lasers),
                player_address=player,offsets=profile.offsets)
            focus=read_native_focus_gate(memory)
            after=reader.global_frame()
            original={row.id:row for row in lasers}
            curves=[]
            for curve in geometry['curves']:
                nodes=np.asarray(curve['nodes'],float).reshape(-1,4)
                legacy=np.asarray(original[curve['id']].nodes,float).reshape(-1,2)
                matching=nodes.shape[0]==legacy.shape[0] and np.array_equal(nodes[:,:2],legacy)
                result=curve_laser_contact(curve['nodes'],curve['width'],position,
                                           geometry['player_half_extents'])
                curves.append(dict(id=curve['id'],nodes=len(nodes),same_positions=matching,
                    width=curve['width'],geometric_collision=result['geometric_collision'],
                    signed_clearance=result['signed_clearance'] if np.isfinite(result['signed_clearance']) else None))
            reports.append(dict(world=entry['index'],frame=before,frame_after=after,
                laser_count=len(lasers),laser_kinds={str(k):sum(l.kind==k for l in lasers) for k in {l.kind for l in lasers}},
                focus_available=focus['available'],enemy_count=focus['enemy_count'],
                elapsed_player_timer=focus['player_elapsed']['integer'] if focus['player_elapsed'] else None,
                player_half_extents=geometry['player_half_extents'],curves=curves))
        finally:
            memory.close()
    result=dict(scope='Read-only snapshots of currently owned worlds; no actions, writes or native collision outcome claim',
                worlds=reports)
    write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
