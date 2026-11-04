"""Native stage exposure, independent of PPO chunk counts or evaluation games."""
import numpy as np


class StoryCoverage:
    def __init__(self,workers):
        self.workers=workers
        self.completed=np.zeros(workers,np.int64)
        self.rows={s:dict(decisions=0,native_frames=0,normal_rows=0,
            guided_normal_rows=0,trusted_normal_rows_4_12_24=[0,0,0],
            partial_danger_pairs_4_12_24=[0,0,0],teacher_Bomb_reference_corrections=0,
            contacts=0,Bombs=0,deathbombs=0) for s in range(1,7)}

    def observe(self,before,after,record,done,*,world_indices=None):
        for w,(old,new) in enumerate(zip(before,after)):
            stage=int(old['stage'])
            if stage in self.rows:
                row=self.rows[stage]
                row['decisions']+=1
                row['native_frames']+=max(0,int(new['episode_frames']-old['episode_frames']))
                normal=bool(record['unprotected_mask'][w])
                row['normal_rows']+=int(normal)
                row['guided_normal_rows']+=int(normal and record['rule_weights'][w]>0)
                for h in range(3):
                    row['trusted_normal_rows_4_12_24'][h]+=int(normal and record['trusted'][w,h])
                    if normal and 'partial_point_danger' in record and not record['trusted'][w,h]:
                        row['partial_danger_pairs_4_12_24'][h]+=int(record['partial_point_danger'][w,h].sum())
                if 'recorded_teacher_Bomb_corrected' in record:
                    row['teacher_Bomb_reference_corrections']+=int(record['recorded_teacher_Bomb_corrected'][w])
                row['contacts']+=int(record['recorded_observed_contacts'][w])
                bombs=max(0,int(new['bombs_used']-old['bombs_used']))
                row['Bombs']+=bombs
                row['deathbombs']+=bombs*int(record['deathbomb_mask'][w])
            self.completed[w if world_indices is None else world_indices[w]]+=int(done[w])

    def reached(self,games_per_world):
        return games_per_world>0 and bool((self.completed>=games_per_world).all())

    def summary(self,*,games_per_world=0):
        return dict(scope='student-controlled training only; excludes frozen evaluation and teacher games',
            stages={str(s):{k:list(v) if isinstance(v,list) else v for k,v in r.items()}
                    for s,r in self.rows.items()},
            completed_games_per_world=self.completed.tolist(),
            complete_games=int(self.completed.sum()),target_games_per_world=games_per_world,
            game_target_met=self.reached(games_per_world) if games_per_world else None,
            zero_sample_stages=[s for s,r in self.rows.items() if not r['decisions']])
