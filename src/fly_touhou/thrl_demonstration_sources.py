"""Read sealed equal-length native trajectory pools without copying map files."""
from pathlib import Path
import hashlib
import numpy as np
from .thrl_study_io import read_shared_json


class WorldConcatenation:
    def __init__(self,arrays):
        self.arrays=list(arrays);first=self.arrays[0]
        if any(a.shape[0]!=first.shape[0] or a.shape[2:]!=first.shape[2:] or a.dtype!=first.dtype for a in arrays):
            raise ValueError('Native pools must share time length, tail shape and precision')
        self.ends=np.cumsum([a.shape[1] for a in arrays]);self.dtype=first.dtype
        self.shape=(first.shape[0],int(self.ends[-1]),*first.shape[2:])
    def __getitem__(self,key):
        time,world=key
        if not isinstance(world,(int,np.integer)):raise TypeError('Index one trajectory world at a time')
        if not 0<=world<self.shape[1]:raise IndexError('World outside native pools')
        pool=int(np.searchsorted(self.ends,world,side='right'))
        start=0 if pool==0 else int(self.ends[pool-1])
        return self.arrays[pool][time,world-start]


def demonstration_sources(folders):
    reports=[];pools=[];schema=None
    keys=('features','objects','maps','targets','dones','native_seeds','native_episodes')
    for folder in map(Path,folders):
        report=read_shared_json(folder/'demonstrations.json');config=read_shared_json(folder/'config.json')
        if report.get('status') not in ('complete-budget-prefix','completed'):
            raise ValueError('Only completed, sealed native data can be replayed')
        if schema is not None and schema!=config['input_schema']:raise ValueError('Mixed raw input schemas')
        schema=config['input_schema'];pool={k:np.load(folder/'demonstrations'/f'{k}.npy',mmap_mode='r') for k in keys}
        if int(np.prod(pool['features'].shape[:2]))!=report['samples']:raise ValueError('Native sample accounting differs')
        reports.append(report);pools.append(pool)
    arrays={k:(np.concatenate([p[k] for p in pools],axis=1) if k in ('native_seeds','native_episodes')
        else WorldConcatenation([p[k] for p in pools])) for k in keys}
    return arrays,reports,schema


def verify_supervision_action_accounting(result,samples):
    """Teacher labels and teacher-executed actions are different quantities."""
    mode=result.get('teacher_collection_mode','teacher-prefix-v1')
    if mode=='teacher-prefix-v1':return result.get('teacher_actions_executed')==samples
    if mode!='teacher-labelled-student-state-mixture-v1':raise ValueError('Unknown teacher collection provenance')
    teacher=result.get('teacher_actions_executed');student=result.get('student_actions_executed')
    if (type(teacher) is not int or type(student) is not int or min(teacher,student)<0
            or teacher+student!=samples or result.get('teacher_labels_queried')!=samples):return False
    records=result.get('source_collection_manifests')
    if not isinstance(records,list) or not records:raise ValueError('Student-state labels require source manifests')
    totals=[0,0,0]
    for item in records:
        path=Path(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Native label-source manifest identity changed')
        report=read_shared_json(path)
        if report.get('status') not in ('complete-budget-prefix','completed'):raise ValueError('Native source incomplete')
        totals[0]+=report['samples'];totals[1]+=report.get('teacher_actions_executed',report['samples'])
        totals[2]+=report.get('student_actions_executed',0)
    return totals==[samples,teacher,student]
