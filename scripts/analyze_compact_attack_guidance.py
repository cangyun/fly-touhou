"""Compare existing teacher and student attack choices on recorded game states."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import psutil,torch
    from fly_touhou.thrl_durable_records import write_json
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher
    proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);proc.cpu_affinity(proc.cpu_affinity()[-2:])
    torch.set_num_threads(2)
    vx=torch.tensor(MovementSafetyTeacher().velocities[:,0],dtype=torch.float32)
    reports=[]
    for path in a.rollout:
        d=torch.load(path,map_location='cpu',weights_only=False);f=d['features'];dx=f[...,114]*368
        margin=d['margins'][...,0,:];safe=margin>1
        progress=dx.abs()[...,None]-(dx[...,None]-4*vx).abs()
        closing=progress>1e-5
        teacher=d['rule_probabilities'][...,:18];mass=teacher.sum(-1,keepdim=True)
        teacher=teacher/mass.clamp_min(1e-9)
        student=d['recorded_policy_logits'][...,:18].softmax(-1)
        behavior=d.get('recorded_behavior_policy_logits',d['recorded_policy_logits'])[...,:18].softmax(-1)
        action=d['actions'];chosen_closing=closing.gather(-1,action.clamp_max(17)[...,None])[...,0]&(action<18)
        normal=d['unprotected_mask']&(d['rule_weights']>0)&d['trusted'][...,0]&(mass[...,0]>1e-6)
        boss=normal&(f[...,113]>.5)&(f[...,117]>.5)&(dx.abs()>32)
        possible=(safe&closing).any(-1)
        result={}
        for stage in range(1,7):
            stage_mask=boss&(torch.round(f[...,111]*6)==stage)
            if not bool(stage_mask.any()):continue
            result[str(stage)]={}
            for name,mask in (('all_immediate_directions_safe',stage_mask&safe.all(-1)),
                              ('some_safe_closing_direction',stage_mask&possible)):
                n=int(mask.sum())
                if not n:continue
                value=lambda t:float(t[mask].mean())
                result[str(stage)][name]=dict(states=n,worlds=int(mask.any(0).sum()),
                    teacher_closing_probability=value((teacher*closing).sum(-1)),
                    student_closing_probability=value((student*closing).sum(-1)),
                    behavior_closing_probability=value((behavior*closing).sum(-1)),
                    actual_closing_fraction=value(chosen_closing.float()),
                    teacher_safe_closing_probability=value((teacher*safe*closing).sum(-1)),
                    behavior_safe_closing_probability=value((behavior*safe*closing).sum(-1)))
        reports.append(dict(rollout=str(path),stage_results=result,metadata=d['metadata']))
        del d,f
    report=dict(schema='compact-native-recorded-boss-attack-guidance-v1',reports=reports,
        scope='Recorded student training states;conditional teacher probabilities and one-step geometric closing,not damage or a frozen evaluation.',
        optimizer_updates=0,native_game_actions=0)
    write_json(a.output,report)
    print(json.dumps([dict(rollout=Path(x['rollout']).name,stage_results=x['stage_results']) for x in reports],ensure_ascii=False))


if __name__=='__main__':main()
