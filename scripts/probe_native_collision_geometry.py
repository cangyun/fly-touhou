"""Bounded owned-EXE collision-function test plus unmodified reader smoke test.

The function tests temporarily set geometry/iframes in this owned disposable
process and invoke its ORIGINAL collision helper on scratch native memory.
They are controlled mechanism probes, never training/clear evidence. All native
player bytes are restored before the subsequent ordinary game steps.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.geometry_environment import GeometryHazardResourceCampaignEnv
from fly_touhou.hazard_reader import FULL_GAME_SHA256, COLLISION_GEOMETRY_SCHEMA
from fly_touhou.hazard_observation import HazardObservationEncoder
from fly_touhou.native import sha256_file
from fly_touhou.native_hazard_planner import GeometryNativeHazardPlanner, NativeHazardPlanner
from fly_touhou.resource_teacher_v5 import ResourceTeacherV5
from fly_touhou.state import Bullet, Player

PROBE_JS=r'''
'use strict';
const base=Process.getModuleByName(MODULE_NAME).base;
const player=base.add(0xdb67c).readPointer();
const helper=base.add(0x16d70);
const actual=Array.from(new Uint8Array(helper.readByteArray(8))).map(x=>x.toString(16).padStart(2,'0')).join('');
if(actual!=='558bec83ec105356')throw new Error('Pinned collision helper bytes differ');
const test=new NativeFunction(helper,'int',['pointer','int'],{abi:'thiscall'});
const scratch=Memory.alloc(0x1500);
const cases=[
 {name:'circle_visual_scale_ignored_without_40',flags:0x16,scale:4,w:2,h:0,dx:5,dy:0,ps:1,pflag:false,hit:false},
 {name:'circle_collision_scale_enabled_40',flags:0x56,scale:4,w:2,h:0,dx:5,dy:0,ps:1,pflag:false,hit:true},
 {name:'circle_collision_shrinks_with_gate',flags:0x56,scale:.5,w:2,h:0,dx:2.5,dy:0,ps:1,pflag:false,hit:false},
 {name:'large_player_circle_extra_multiplier',flags:0x16,scale:1,w:2,h:0,dx:8,dy:0,ps:3,pflag:true,hit:true},
 {name:'player_circle_flag_clear_control',flags:0x16,scale:1,w:2,h:0,dx:8,dy:0,ps:3,pflag:false,hit:false},
 {name:'native_circle_sum_of_squares',flags:0x16,scale:1,w:2,h:0,dx:3.5,dy:0,ps:1,pflag:false,hit:false},
 {name:'normal_circle_small_bullet_legacy_underestimate',flags:0x16,scale:1,w:1,h:0,dx:2.1,dy:0,ps:1,pflag:false,hit:true},
 {name:'normal_circle_small_bullet_boundary_outside',flags:0x16,scale:1,w:1,h:0,dx:2.3,dy:0,ps:1,pflag:false,hit:false},
 {name:'rectangle_scale_ignored_without_40',flags:0x06,scale:2,w:4,h:20,dx:0,dy:15,ps:1,pflag:false,hit:false},
 {name:'rectangle_scaled_both_dimensions',flags:0x46,scale:2,w:4,h:20,dx:0,dy:15,ps:1,pflag:false,hit:true},
 {name:'rectangle_independent_height',flags:0x06,scale:1,w:4,h:20,dx:0,dy:8,ps:1,pflag:false,hit:true},
 {name:'large_player_rectangle_bounds',flags:0x06,scale:1,w:2,h:2,dx:3.5,dy:0,ps:3,pflag:true,hit:true},
 {name:'collision_enable_gate_clear',flags:0x54,scale:4,w:2,h:0,dx:0,dy:0,ps:1,pflag:false,hit:false}
];
rpc.exports={probe(){
 if(player.isNull()||player.add(0x684).readU32()!==1)throw new Error('Requires initialized alive player');
 const original=[0x630,24,0x18308,4,0x182d4,4,0x182c4,4];
 const saved=[];
 for(let n=0;n<original.length;n+=2)saved.push([original[n],player.add(original[n]).readByteArray(original[n+1])]);
 const x=player.add(0x5e0).readFloat(),y=player.add(0x5e4).readFloat();
 const bx=player.add(0x648).readFloat(),by=player.add(0x64c).readFloat();
 const definition=player.add(0x688).readPointer().add(4).readFloat();
 const flags=player.add(0x182d4).readU32();
 const result=[];
 try{
  // Suppress the native life-loss procedure, without replacing the original
  // helper or its contact return value. Scratch cancellation counter prevents
  // bullet deletion; graze-already flag suppresses a graze-side-effect call.
  player.add(0x182c4).writeS32(600);
  for(const c of cases){
   player.add(0x18308).writeFloat(c.ps);
   const pf=(flags&~0x10)|(c.pflag?0x10:0);
   player.add(0x182d4).writeU32(pf);
   player.add(0x630).writeFloat(x-bx*c.ps);player.add(0x634).writeFloat(y-by*c.ps);
   player.add(0x63c).writeFloat(x+bx*c.ps);player.add(0x640).writeFloat(y+by*c.ps);
   scratch.add(0x20).writeU32(c.flags);scratch.add(0x24).writeS32(1);
   scratch.add(0xbc0).writeFloat(x+c.dx);scratch.add(0xbc4).writeFloat(y+c.dy);
   scratch.add(0xbe0).writeFloat(c.w);scratch.add(0xbe4).writeFloat(c.h);
   scratch.add(0x13bc).writeFloat(c.scale);
   const nativeResult=test(scratch,0);
   result.push({...c,native_result:nativeResult,native_contact:nativeResult===1,
    player:{x:x,y:y,base_x:bx,base_y:by,definition_radius:definition,flags:pf,
      bounds_min:[player.add(0x630).readFloat(),player.add(0x634).readFloat()],
      bounds_max:[player.add(0x63c).readFloat(),player.add(0x640).readFloat()]},
    scratch_address:scratch.toUInt32()});
  }
 }finally{for(const [offset,data] of saved)player.add(offset).writeByteArray(data);}
 const restored=saved.every(([offset,data])=>{
  const before=new Uint8Array(data),after=new Uint8Array(player.add(offset).readByteArray(before.length));
  return before.every((v,n)=>v===after[n]);
 });
 return {pid:Process.id,helper_address:helper.toString(),helper_pristine_bytes:actual,
    control_intervention:true,temporary_geometry_and_invulnerability_overrides:true,live_bullet_objects_modified:false,
    player_bytes_restored:restored,cases:result};
}};
'''


def check_model(snapshot, case):
    """Compare V5 with controlled native memory, not a natural trace sample."""
    p=case['player']; radius=p['base_x']*case['ps']
    player=Player(p['x'],p['y'],radius,0,False,1,0)
    bullet=Bullet(case['scratch_address'],p['x']+case['dx'],p['y']+case['dy'],0,0,case['w'])
    extra=json.loads(json.dumps(snapshot.extra))
    extra.update(player_scale=case['ps'],base_player_radius=p['base_x'])
    extra['resources']['player_flags']=p['flags']
    extra['hazard_metadata']['lasers']=[]
    raw={'id':bullet.id,'native_state':1,'native_flags':case['flags'],'native_timer':0,
         'native_cancel_protection_counter':0,'native_animation_gate':0,
         'native_position':[bullet.x,bullet.y],'native_velocity':[0,0],
         'native_speed':0,'native_angle':0,'native_hitbox_value':bullet.radius,'native_scale':case['scale']}
    extra['hazard_metadata']['bullets']=[raw]
    extra['native_collision_geometry']={'schema':COLLISION_GEOMETRY_SCHEMA,'frame':snapshot.frame,
        'exe_sha256':FULL_GAME_SHA256,'observed':True,
        'player':{'native_bounds_min':p['bounds_min'],'native_bounds_max':p['bounds_max'],
            'native_scale':case['ps'],'native_base_radius':p['base_x'],
            'native_definition_radius':p['definition_radius'],'native_flags':p['flags']},
        'bullets':[{'id':bullet.id,'native_flags':case['flags'],'native_scale':case['scale'],
            'native_hitbox_value':case['w'],'native_height_value':None if case['flags']&0x10 else case['h']}]}
    controlled=replace(snapshot,player=player,bullets=(bullet,),lasers=(),extra=extra)
    path=[(0.,0.,np.asarray((p['x'],p['y'])),np.zeros(2))]
    clearance=GeometryNativeHazardPlanner().path_clearance(controlled,path)
    case['v5_clearance']=None if np.isinf(clearance) else clearance
    case['v5_contact']=bool(clearance<=0)
    old=NativeHazardPlanner().path_clearance(controlled,path)
    case['v4_clearance']=None if np.isinf(old) else old
    case['v4_contact']=bool(old<=0)
    case['matches_expected_and_native']=(case['v5_contact']==case['native_contact']==case['hit'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args()
    opt.output.mkdir(parents=True,exist_ok=False)
    assert sha256_file(opt.exe)==FULL_GAME_SHA256
    env=GeometryHazardResourceCampaignEnv(opt.exe,opt.profile,difficulty=3,max_episode_frames=1200,
        private_appdata=opt.output/'userdata',dialogue_skip=True,timeout=30,
        window_title='原作尺寸机制验证 · 非训练非通关验收')
    report={'schema':'original-th14-collision-geometry-probe-v1','exe_sha256':sha256_file(opt.exe),
        'source_sha256':sha256_file(__file__),'desktop_input':False,'status':'running',
        'control_intervention':True,'excluded_from_training_and_clear_acceptance':True,
        'scope':'Owned disposable original process. Controlled helper tests followed by natural short reader/teacher smoke test. Not full stage6 or student acceptance.'}
    probe=None
    try:
        _,info=env.reset()
        before=env.snapshot
        report['pid']=env.pid
        report['initial_snapshot']=before.to_dict()
        probe=env.controller._session.create_script('const MODULE_NAME='+json.dumps(opt.exe.name)+';\n'+PROBE_JS)
        probe.load()
        report['controlled_native_function_tests']=probe.exports_sync.probe()
        for case in report['controlled_native_function_tests']['cases']:check_model(before,case)
        restored=env.controller.read()
        report['native_player_snapshot_restored']=restored.player==before.player
        legacy=replace(restored,extra={k:v for k,v in restored.extra.items() if k!='native_collision_geometry'})
        report['v3_vector_identical_with_and_without_extension']=bool(np.array_equal(
            HazardObservationEncoder().encode(restored),HazardObservationEncoder().encode(legacy)))
        if not report['controlled_native_function_tests']['player_bytes_restored']:
            raise RuntimeError('Native player bytes not restored')
        if not all(c['matches_expected_and_native'] for c in report['controlled_native_function_tests']['cases']):
            raise RuntimeError('Some native shape/scale cases disagreed with V5')
        teacher=ResourceTeacherV5(GeometryNativeHazardPlanner(env.encoder.planner.config))
        env.snapshot=restored
        ordinary_decisions=hazard_decisions=0
        durations=[]
        trace=opt.output/'natural_reader_smoke.jsonl.gz'
        with gzip.open(trace,'xt',encoding='utf-8') as f:
            for step in range(250):
                before=env.snapshot
                start=time.perf_counter()
                action=int(teacher.probabilities(before).argmax())
                durations.append(time.perf_counter()-start)
                _,reward,terminated,truncated,info=env.step(action)
                f.write(json.dumps({'record_type':'transition','step':step,'action':action,
                    'snapshot':before.to_dict(),'next_snapshot':env.snapshot.to_dict(),
                    'temporary_overrides_in_this_natural_segment':False,
                    'control_intervention':False,'excluded_from_training_and_clear_acceptance':True},ensure_ascii=False,allow_nan=False)+'\n')
                ordinary_decisions+=1
                hazard_decisions+=bool(before.bullets)
                if terminated or truncated or hazard_decisions>=30:break
        report.update(status='passed',natural_decisions=ordinary_decisions,
            natural_hazard_decisions=hazard_decisions,natural_native_frames=env.episode_frames,
            teacher_seconds_median=float(np.median(durations)),teacher_seconds_p95=float(np.quantile(durations,.95)),
            natural_trace_sha256=sha256_file(trace))
    except Exception as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        if probe is not None:
            try:probe.unload()
            except Exception:pass
        env.close()
        report['exe_sha256_after']=sha256_file(opt.exe)
        (opt.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('initial_snapshot','controlled_native_function_tests')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
