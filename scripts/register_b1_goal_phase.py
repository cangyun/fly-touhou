"""Attach an actually live B1 CUDA phase to the existing local control panel."""
from pathlib import Path
import argparse,sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_control import ControlCenter,controller_lock,alive
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def register(folder,jobid,phase='teacher_pretraining',*,control_root=None):
    folder=Path(folder)
    center=Path(control_root) if control_root is not None else ROOT/'artifacts/engineering/training-control-center-v1'
    if not (center/'configuration.json').is_file():
        raise FileNotFoundError('Existing training control center missing; frozen jobs must pass its explicit control_root')
    c=ControlCenter(center)
    identity=read(folder/'launch.json');identity={k:identity[k] for k in ('pid','create_time')}
    if not alive(identity):raise RuntimeError('Cannot register a stopped phase as running')
    config=read(folder/'study_config.json')
    workers=config.get('workers',30)
    offline=(phase=='calibration' and config.get('offline_neural_calibration') is True and workers==0)
    if type(workers) is not int or not (1<=workers<=30 or offline):raise ValueError('B1 goal phase must fit the30-native-world ceiling')
    with controller_lock(c.root):
        reg=c.registry()
        if c.branch_activity('B1',reg)['running']:raise RuntimeError('B1 already live')
        control=c.root/'jobs'/jobid;control.mkdir(parents=True,exist_ok=False)
        write(control/'registry_before.json',reg)
        reg['jobs'].append(dict(id=jobid,keys=['B1'],mode='authorized_goal_'+phase,output=str(folder),
            control=str(control),submitted_unix=time.time(),submitted_utc=utc_now(),dispatch_identity=identity))
        label=('B1 · 点风险网络尺寸校准（0原作、0PPO）' if offline else
            'B1 · GRU · 冻结模型避弹评估' if phase=='evaluation' else 'B1 · GRU · 避弹专项训练')
        reg['active']['B1']=dict(job_id=jobid,output=str(folder),label=label,attention='both',
            workers=workers,teacher_pretraining=phase=='teacher_pretraining',input_schema=config['input_schema'],config=config,operator_stop_path=str(folder/'STOP'))
        write(folder/'comparison_status.json',dict(branches={'B1':dict(status=phase,launch_identity=identity)},updated_utc=utc_now()))
        c._publish(reg)
    return dict(job_id=jobid,running=c.branch_activity('B1')['running'],identity=identity)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--folder',type=Path,required=True);p.add_argument('--job-id',required=True)
    p.add_argument('--phase',choices=['teacher_pretraining','training','evaluation'],default='teacher_pretraining')
    p.add_argument('--control-root',type=Path)
    a=p.parse_args();print(json.dumps(register(a.folder,a.job_id,a.phase,control_root=a.control_root)))
