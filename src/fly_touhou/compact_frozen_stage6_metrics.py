"""Passive whole-stage occupancy/damage totals, distinct from contact tails."""
from pathlib import Path
from .thrl_durable_records import write_json


def observe(env,info,done):
    path=getattr(env,'frozen_stage6_metrics',None)
    s=env.snapshot
    if not path or s.stage!=6:return
    record=getattr(env,'_frozen_stage6_motion',None)
    if s.player is None:
        if done and record is not None:
            record.update(status='completed',terminal_player_absent=True,stage_frame=s.stage_frame)
            write_json(Path(path),record)
        return
    if record is None:
        record=dict(schema='frozen-stage6-passive-motion-v1',status='recording',
            source='Every coherent post-action native observation, not selected contact tails',
            native_frames=0,decisions=0,controllable_frames=0,vulnerable_frames=0,
            edge_frames=0,corner_frames=0,boss_target_frames=0,above_boss_frames=0,
            off_axis80_frames=0,aligned40_below_frames=0,nonidle_no_displacement_frames=0,
            observed_boss_hp_damage=0,phases={},entry=dict(stage_frame=s.stage_frame,
                lives=s.lives,bombs=s.bombs,misses=s.extra['thrl_events']['misses'],
                bombs_used=s.extra['thrl_events']['bombs_used']),
            teacher_actions=0,optimizer_updates=0,actions_changed=False)
        env._frozen_stage6_motion=record;env._frozen_stage6_motion_last=None
        env._frozen_stage6_motion_written=s.stage_frame-1000
    previous=env._frozen_stage6_motion_last
    f=env._last_measurable_observation['features'];p=s.player
    boss=next((r for r in s.extra['thrl_events']['ecl_roles']['enemies'] if r['boss'] and r['hp_max']>0),None)
    if previous is not None:
        dt=max(0,s.frame-previous['frame']);record['native_frames']+=dt;record['decisions']+=1
        if f[14]>.5:
            record['controllable_frames']+=dt
            edge=abs(p.x)>150 or p.y<60 or p.y>410
            corner=abs(p.x)>150 and (p.y<60 or p.y>410)
            record['edge_frames']+=int(edge)*dt;record['corner_frames']+=int(corner)*dt
            record['vulnerable_frames']+=int(p.invulnerable_frames<=0)*dt
            if f[113]>.5 and f[117]>.5:
                record['boss_target_frames']+=dt
                dx,dy=float(f[114])*368,float(f[115])*400
                record['above_boss_frames']+=int(dy>0)*dt
                record['off_axis80_frames']+=int(abs(dx)>80)*dt
                record['aligned40_below_frames']+=int(abs(dx)<=40 and dy<0)*dt
            action=info.get('policy_action',0)
            if action<18 and action%9 and abs(p.x-previous['x'])+abs(p.y-previous['y'])<1e-6:
                record['nonidle_no_displacement_frames']+=dt
        old=previous['boss']
        if boss is not None and old is not None and (boss['id'],boss['ecl_sub_name'],boss['hp_max'])==(old['id'],old['ecl_sub_name'],old['hp_max']):
            damage=max(0,old['hp']-boss['hp']);record['observed_boss_hp_damage']+=damage
            phase=record['phases'].setdefault(boss['ecl_sub_name'],dict(native_frames=0,hp_damage=0))
            phase['native_frames']+=dt;phase['hp_damage']+=damage
    env._frozen_stage6_motion_last=dict(frame=s.frame,x=p.x,y=p.y,boss=boss)
    if done or s.stage_frame-env._frozen_stage6_motion_written>=1000:
        record.update(status='completed' if done else 'recording',stage_frame=s.stage_frame,
            lives=s.lives,bombs=s.bombs,misses=s.extra['thrl_events']['misses'],
            bombs_used=s.extra['thrl_events']['bombs_used'])
        write_json(Path(path),record);env._frozen_stage6_motion_written=s.stage_frame
