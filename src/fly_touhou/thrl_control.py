"""Local native training control. The UI imports no Torch and uses no shell.

Every launch gets its own immutable request, source pin, output and PID receipt.
Resume inspection (including Torch) happens in the detached executor, never Tk.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import hashlib
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from .atomic_json_io import atomic_write_json as write_json
from .thrl_study_io import read_shared_json, utc_now
from .thrl_limits import MAX_NATIVE_WORKERS

DEFAULT_SETTINGS = {'workers':8, 'decisions':5_000_000, 'optuna':True,
    'candidates':2, 'development_interval':100_000, 'development_episodes':6,
    'validation_episodes':20}
GROUPS = ('A0','A1','B0','B1')
ALL_GROUPS = (*GROUPS,'M0','M1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings_plan(values,training_recipe=None):
    settings={**DEFAULT_SETTINGS, **values}
    for name,lo,hi in (('workers',1,MAX_NATIVE_WORKERS),('decisions',2048,1_000_000_000),
            ('candidates',1,8),('development_interval',2048,1_000_000_000),
            ('development_episodes',1,20),('validation_episodes',1,20)):
        value=settings[name]
        if type(value) is not int or not lo<=value<=hi:
            raise ValueError(f'{name} 必须是 {lo}–{hi} 的整数')
    if type(settings['optuna']) is not bool:
        raise ValueError('Optuna 开关必须为布尔值')
    workers=settings['workers']
    rollout=workers*64*math.ceil(2048/(workers*64))
    if training_recipe=='th14-completion-v3':rollout=max(rollout,workers*128)
    target=rollout*math.ceil(settings['decisions']/rollout)
    interval=rollout*math.ceil(settings['development_interval']/rollout)
    nodes=list(range(interval,target,interval))+[target]
    count=settings['candidates'] if settings['optuna'] else 1
    minibatch=256 if training_recipe=='th14-completion-v3' else 64
    return {'settings':settings,'rollout':rollout,'target_decisions':target,
        'development_decisions':nodes,'candidates':count,
        'max_PPO_decisions_per_group':target*count,
        'expected_Adam_per_candidate':(target//rollout)*math.ceil(rollout/minibatch)*3,
        'Adam_estimate_is_upper_bound':training_recipe=='th14-completion-v3',
        'rollout_ram_gib_per_group':rollout*16*92*96*4/2**30,
        'wall_cap_seconds':None}


def protocol_difference(source,target):
    fields=('curriculum_protocol','input_schema','training_recipe','reward_schema','training_mode')
    from .thrl_learning_revision import revision_values,upgrade_record
    differences=[field for field in fields if source.get(field)!=target.get(field)]
    from .thrl_movement_learning import contract as movement_contract
    revision_changed=(revision_values(source)!=revision_values(target)
                      or movement_contract(source)!=movement_contract(target))
    if revision_changed or any(name in differences for name in ('curriculum_protocol','training_mode')):
        try:
            migration=upgrade_record(source,target)
            if migration:
                differences=[name for name in differences if name not in ('curriculum_protocol','training_mode')]
        except ValueError:
            if revision_changed:differences+=['learning_revision_or_chain_horizon']
    return differences


def alive(identity):
    """Only a recorded PID AND creation time establish a live owned process."""
    if not identity or not identity.get('pid') or identity.get('create_time') is None:
        return False
    import psutil
    try:
        process=psutil.Process(int(identity['pid']))
        return abs(process.create_time()-identity['create_time'])<.002 and process.is_running()
    except (psutil.Error,ValueError):
        return False


def process_identity(pid):
    import psutil
    process=psutil.Process(int(pid))
    return {'pid':process.pid,'create_time':process.create_time()}


def read_optional(path):
    try:return read_shared_json(path)
    except FileNotFoundError:return {}


@contextmanager
def controller_lock(root):
    """OS lock released even after a panel crash; no stale lock stealing."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('另一个控制面板正在提交操作，请稍后重试') from error
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_UN)


class ControlCenter:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.config=read_shared_json(self.root/'configuration.json')
        if self.config.get('schema')!='thrl-control-center-v1':
            raise ValueError('控制面板配置版本不兼容')

    def settings(self):
        return settings_plan(read_optional(self.root/'settings.json'))['settings']

    def plan(self,values):
        self.config=read_shared_json(self.root/'configuration.json')
        template=next(iter(self.config['template_configs'].values()))
        recipe=read_shared_json(template['path']).get('training_recipe')
        return settings_plan(values,recipe)

    def save_settings(self,values):
        plan=self.plan(values)
        with controller_lock(self.root):write_json(self.root/'settings.json',plan['settings'])
        return plan

    def registry(self):
        return read_optional(self.root/'registry.json') or {'jobs':[],'active':{}}

    def recovery_source(self,key,registry):
        """A failed submission must not hide the last model behind a missing folder."""
        row=registry['active'].get(key);seen=set()
        for _ in range(32):
            if not row or row['output'] in seen:break
            seen.add(row['output']);output=Path(row['output'])
            if (output/'effective_study_config.json').is_file() or (output/'study_config.json').is_file():
                return row
            job=next((job for job in registry['jobs'] if job['id']==row['job_id']),None)
            request=read_optional(Path(job['control'])/'request.json') if job else {}
            row=request.get('previous',{}).get(key)
        raise ValueError(f'{key} 尚无可继续的已初始化作业，请使用新训练')

    def branch_activity(self,key,registry=None):
        registry=registry or self.registry()
        descriptor=registry['active'].get(key)
        if not descriptor:return {'running':False,'reason':'尚未启动'}
        job=next(row for row in registry['jobs'] if row['id']==descriptor['job_id'])
        status=read_optional(Path(job['output'])/'comparison_status.json')
        branch=status.get('branches',{}).get(key)
        if branch is not None:
            running=alive(branch.get('launch_identity'))
            return {'running':running,'reason':branch.get('status','运行中' if running else '已停止'),
                'identity':branch.get('launch_identity'),'job':job}
        running=alive(job.get('dispatch_identity'))
        pending=(time.time()-job.get('submitted_unix',0)<90 and not job.get('dispatch_error')
                 and not read_optional(Path(job['control'])/'status.json').get('finished_unix'))
        return {'running':bool(running or pending),'reason':'启动准备中' if running or pending else '已停止',
                'job':job}

    def decorate(self,snapshot):
        self.config=read_shared_json(self.root/'configuration.json')
        registry=self.registry()
        snapshot['control']={'root':str(self.root),'settings':self.settings(),'jobs':registry['jobs']}
        for key,branch in snapshot['branches'].items():
            activity=self.branch_activity(key,registry)
            branch['control']=activity
            template=read_shared_json(self.config['template_configs'][key]['path'])
            branch['next_training_recipe']=template.get('training_recipe')
            branch['next_input_schema']=template.get('input_schema')
            branch['next_training_mode']=template.get('training_mode')
            if key in registry['active']:
                folder=Path(registry['active'][key]['output'])
                previous=read_optional(folder/'effective_study_config.json') or read_optional(folder/'study_config.json')
                changes=protocol_difference(previous,template) if previous else []
                branch['resume_blocked_reason']=('输入或训练协议已升级，请点新训练；旧模型及日志保留' if changes else None)
            if key not in registry['active'] and key in self.config.get('standby_groups',[]):
                branch['phase']='ready'
                continue
            job=activity.get('job',{})
            failure=read_optional(Path(job['control'])/'action_failure.json') if job else {}
            if failure:
                branch['failures'].append({'summary':failure.get('error','启动失败'),
                    'path':str(Path(job['control'])/'action_failure.json')})
                if not activity['running']:branch['phase']='failed'
            elif not activity['running'] and branch['phase'] not in (
                    'finished','completed','failed','stopped','operator_or_budget_stop','pruned'):
                # A power loss cannot leave an old "collecting" label alive forever.
                branch['phase']='interrupted'
        return snapshot

    def launch(self,keys,values,mode='new'):
        keys=list(dict.fromkeys(keys));plan=self.plan(values)
        if self.config.get('maintenance',{}).get('active'):
            raise RuntimeError(self.config['maintenance']['reason'])
        if not keys or any(key not in self.config['template_configs'] for key in keys):raise ValueError('请选择有效的实验组')
        if mode not in ('new','resume'):raise ValueError('未知启动操作')
        from .thrl_training_mode import validate_training_mode
        for key in keys:
            validate_training_mode(read_shared_json(self.config['template_configs'][key]['path']))
        with controller_lock(self.root):
            registry=self.registry()
            busy=[key for key in keys if self.branch_activity(key,registry)['running']]
            if busy:raise RuntimeError('这些组仍在运行或收尾，不能重复启动：'+', '.join(busy))
            if mode=='resume' and any(key not in registry['active'] for key in keys):
                raise ValueError('尚无可继续的历史模型，请使用新训练')
            previous={key:registry['active'].get(key) for key in keys}
            if mode=='resume':
                for key in keys:
                    row=self.recovery_source(key,registry);folder=Path(row['output'])
                    source=read_optional(folder/'effective_study_config.json') or read_shared_json(folder/'study_config.json')
                    template=read_shared_json(self.config['template_configs'][key]['path'])
                    changes=protocol_difference(source,template)
                    if 'curriculum_protocol' in changes:
                        raise ValueError('历史作业的课程协议不同，请使用新训练；旧模型和日志仍保留')
                    if changes:raise ValueError('新旧训练协议不同（'+', '.join(changes)+'），请使用新训练；旧模型和日志仍保留')
                    if source['workers']!=plan['settings']['workers'] or source['rollout_total_decisions']!=plan['rollout']:
                        raise ValueError(f'继续 {key} 须保持原来的 {source["workers"]} 个实例；更改实例数请使用新训练')
                    progress=read_optional(folder/'trial_000_both/progress.json')
                    if progress.get('candidate_optimized_decisions',0)>=plan['target_decisions']:
                        raise ValueError('目标不超过已完成决策，请提高继续训练的总目标')
                    previous[key]=row
            job_id=datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
            control=self.root/'jobs'/job_id;control.mkdir(parents=True,exist_ok=False)
            output=Path(self.config['output_root'])/job_id
            request={'schema':'thrl-control-request-v1','job_id':job_id,'keys':keys,'mode':mode,
                'settings':plan['settings'],'control_root':str(self.root),'control':str(control),
                'output':str(output),'engine':self.config['engine'],
                'template_configs':self.config['template_configs'],
                'previous':previous,
                'submitted_utc':utc_now()}
            write_json(control/'request.json',request)
            runtime=Path(self.config['engine']['runtime'])
            spec={'schema':'fly-touhou-background-job-v1','python':self.config['python'],
                'script':str(runtime/'scripts/control_thrl_training.py'),
                'arguments':['--execute',str(control/'request.json')],
                'cwd':str(runtime),'control_directory':str(control),
                'environment':{'PYTHONUTF8':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                    'FLY_TOUHOU_DESKTOP_NAME':'fly-touhou-job-'+job_id}}
            temporary=Path(self.config['output_root'])/'temporary'/job_id
            temporary.mkdir(parents=True,exist_ok=False)
            spec['environment'].update(TEMP=str(temporary),TMP=str(temporary))
            write_json(control/'background_spec.json',spec)
            job={'id':job_id,'keys':keys,'mode':mode,'output':str(output),'control':str(control),
                 'submitted_unix':time.time(),'submitted_utc':utc_now(),'plan':plan}
            registry['jobs'].append(job)
            for key in keys:
                template=self.config['template_configs'][key]
                registry['active'][key]={'job_id':job_id,'output':str(output/f'branch_{key}'),
                    'label':template['label'],'attention':'both','workers':plan['settings']['workers'],
                    'teacher_pretraining':key.endswith('1'),'input_schema':template['input_schema'],
                    'config':{'trial_decisions':plan['target_decisions'],
                        'development_decisions':plan['development_decisions'],
                        'development_episodes':plan['settings']['development_episodes'],
                        'validation_episodes_per_policy':plan['settings']['validation_episodes'],
                        'optimization_trials':plan['candidates']},
                    'operator_stop_path':str(control/f'STOP_{key}')}
            self._publish(registry)
            write_json(self.root/'settings.json',plan['settings'])
            try:
                result=subprocess.run([self.config['python'],str(runtime/'scripts/background_job.py'),
                    '--spec',str(control/'background_spec.json')],capture_output=True,text=True,
                    encoding='utf-8',timeout=30,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if result.returncode:raise RuntimeError(result.stderr[-2000:] or '后台派发失败')
                receipt=read_shared_json(control/'launch.json')
                job['dispatch_identity']=process_identity(receipt['worker_launcher_pid'])
                job['dispatch_receipt']=receipt
            except Exception as error:
                job['dispatch_error']=str(error)
                write_json(control/'action_failure.json',{'error':str(error),'stage':'dispatch','utc':utc_now()})
                self._publish(registry)
                raise
            self._publish(registry)
            return {'job_id':job_id,'output':str(output),'control':str(control),'plan':plan,'groups':keys,'mode':mode}

    def _publish(self,registry):
        write_json(self.root/'registry.json',registry)
        branches=copy.deepcopy(registry['active'])
        for key in self.config.get('standby_groups',[]):
            if key in branches:continue
            template=self.config['template_configs'][key]
            cfg=read_shared_json(template['path'])
            branches[key]=dict(output=str(Path(self.config['output_root'])/'not-started'/key),
                label=template['label'],attention=cfg['attention'],workers=self.settings()['workers'],
                teacher_pretraining=bool(cfg.get('teacher_pretraining')),input_schema=cfg['input_schema'],
                config=cfg,phase='ready',parameter_counts=cfg.get('parameter_counts',{}),prepared_only=True)
        write_json(self.root/'comparison_manifest.json',{'schema':'thrl-control-center-manifest-v1',
            'status':'control_center','branches':branches,'updated_utc':utc_now()})

    def stop(self,keys=None):
        with controller_lock(self.root):
            registry=self.registry();keys=list(registry['active']) if keys is None else keys
            receipts={}
            for key in keys:
                row=registry['active'].get(key)
                if row is None:continue
                branch=Path(row['output'])
                path=Path(row['operator_stop_path']) if row.get('operator_stop_path') else branch/'STOP'
                if not path.parent.is_dir():
                    receipts[key]={'requested':False,'error':'输出目录未就绪'};continue
                if not path.exists():write_json(path,{'source':'local_control_panel','utc':utc_now(),'branch':key})
                # A running frozen older engine understands only branch/STOP.
                if branch.is_dir() and not (branch/'STOP').exists():
                    write_json(branch/'STOP',{'source':'local_control_panel','utc':utc_now(),'branch':key})
                receipts[key]={'requested':True,'path':str(path)}
            return receipts


def prepare_execution(request):
    """Executed in a detached process. A resume retains exact Adam/weights/RNG."""
    runtime=Path(request['engine']['runtime']);control=Path(request['control']);output=Path(request['output'])
    if digest(runtime/'source_manifest.json')!=request['engine']['manifest_sha256']:
        raise ValueError('已验证引擎的清单发生变化，拒绝启动')
    recipe=read_shared_json(next(iter(request['template_configs'].values()))['path']).get('training_recipe')
    plan=settings_plan(request['settings'],recipe);branches={};audit={}
    for key in request['keys']:
        template=request['template_configs'][key];template_path=Path(template['path'])
        if digest(template_path)!=template['sha256']:raise ValueError('模板身份变化：'+key)
        config=copy.deepcopy(read_shared_json(template_path))
        execution_options={field:config[field] for field in ('curriculum_eval_backend','curriculum_eval_workers',
            'curriculum_eval_batch_wait_ms','curriculum_eval_min_batch_size') if field in config}
        target_learning_config=copy.deepcopy(config)
        config.pop('baseline_resume',None);config.pop('previous_study_output',None)
        config.pop('teacher_pretraining_result',None)
        original_nodes=[]
        previous=request['previous'].get(key)
        if request['mode']=='resume':
            previous_output=Path(previous['output'])
            source=read_optional(previous_output/'effective_study_config.json') or read_shared_json(previous_output/'study_config.json')
            if protocol_difference(source,config):raise ValueError('不能跨输入、奖励或PPO协议继续训练；请使用新训练')
            if source['workers']!=plan['settings']['workers'] or source['rollout_total_decisions']!=plan['rollout']:
                raise ValueError(f'继续 {key} 须保持原来的 {source["workers"]} 个实例；更改实例数请使用新训练')
            if source.get('input_schema')!=config['input_schema']:raise ValueError('继续训练不能改变输入结构')
            config=copy.deepcopy(source);config.pop('baseline_resume',None)
            from .thrl_learning_revision import upgrade_record,apply_learning_upgrade
            migration=upgrade_record(source,target_learning_config)
            if migration:
                config=apply_learning_upgrade(config,target_learning_config,migration)
                config['learning_migration']={**migration,'source_study':str(previous_output),'utc':utc_now()}
            config['previous_study_output']=str(previous_output)
            # Scores/nodes from a segmented protocol are not evidence for the
            # full-story policy. Same-protocol continuation keeps its schedule.
            original_nodes=([] if source.get('training_mode')!=config.get('training_mode')
                else source['development_decisions'])
            run=previous_output/'trial_000_both'
            if run.is_dir():
                import torch
                candidates=[]
                for name in ('interrupted.pt','final.pt','latest.pt','initial.pt'):
                    checkpoint=run/name
                    if not checkpoint.is_file():continue
                    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    prefix=payload.get('candidate_optimized_decisions',0)
                    if payload.get('optimizer_transaction_complete',True):
                        candidates.append((prefix,checkpoint,payload.get('candidate_paid_decisions_lower_bound',prefix)))
                    del payload
                if not candidates:raise ValueError('没有完整 Adam 事务的恢复点：'+key)
                prefix,checkpoint,paid=max(candidates,key=lambda item:item[0])
                if prefix>=plan['target_decisions']:raise ValueError('目标不超过已完成决策，请提高继续训练的总目标')
                config['baseline_resume']={'checkpoint':str(checkpoint),'sha256':digest(checkpoint),
                    'optimized_decisions':prefix,'paid_decisions':paid,'source_run':str(run),
                    'unknown_inflight_decisions':True}
                audit[key]=config['baseline_resume']
                if migration:
                    config['learning_migration']['boundary_optimized_decisions']=prefix
                    audit[key]={**audit[key],'learning_migration':config['learning_migration']}
                if config.get('curriculum_manifest'):
                    config['curriculum_inherited_pools']=list(dict.fromkeys(
                        source.get('curriculum_inherited_pools',[])+[str(run/'curriculum_pool')]))
            elif not source.get('teacher_pretraining'):
                raise ValueError('没有可继续的 PPO 模型：'+key)
            # Completed BC is reusable; incomplete collection starts afresh and is labeled.
            teacher=read_optional(previous_output/'teacher_pretraining'/'result.json')
            if teacher and teacher.get('checkpoint'):
                config['teacher_pretraining_result']=teacher
        config.update(source_runtime=str(runtime),source_manifest_sha256=request['engine']['manifest_sha256'],
            profile=str(runtime/'profiles/th14_1.00b_original.json'),
            profile_sha256=digest(runtime/'profiles/th14_1.00b_original.json'),
            workers=plan['settings']['workers'],rollout_total_decisions=plan['rollout'],
            trial_decisions=plan['target_decisions'],development_decisions=sorted(set(
                plan['development_decisions']+[n for n in original_nodes if n<=plan['target_decisions']])),
            development_episodes=plan['settings']['development_episodes'],
            validation_episodes_per_policy=plan['settings']['validation_episodes'],
            optimization_trials=plan['candidates'],trial_wall_cap_seconds=None,study_wall_cap_seconds=None,
            api_port=0,comparison_stop_path=str(output/'STOP'),
            operator_stop_path=str(control/f'STOP_{key}'),
            study_name=f'TH14-panel-{request["job_id"]}-{key}',
            control_request=str(control/'request.json'),requested_trial_decisions=plan['settings']['decisions'],
            new_initialization=request['mode']=='new',old_checkpoints_used=request['mode']=='resume')
        config.update(execution_options)
        from .thrl_training_mode import validate_training_mode
        validate_training_mode(config)
        if config.get('learning_revision'):
            from .th14_completion_protocol import learning_overrides
            config['ppo_learning_overrides']=learning_overrides(config)
        if config.get('teacher_pretraining') and not config.get('teacher_pretraining_result'):
            teacher=config['teacher_pretraining'];limit=min(teacher.get('workers',4),config['workers'])
            teacher['workers']=max(n for n in range(1,limit+1) if teacher['samples']%(n*teacher.get('seq_len',16))==0)
        if config.get('training_recipe')=='th14-completion-v2' and request['mode']=='new':
            config['entropy_decay_decisions']=int(.6*plan['target_decisions'])
        if config.get('training_recipe')=='th14-completion-v3' and request['mode']=='new':
            config['entropy_decay_decisions']=min(1_000_000,plan['target_decisions'])
            from .th14_completion_protocol import baseline_parameters,exploration_parameters
            config['ppo_exploration_schedule']=exploration_parameters(config,
                baseline_parameters(config)['entropy_coeff_start'])
        from .thrl_branching import validate_parallel_config
        config['capacity']=validate_parallel_config(config)
        path=control/f'branch_{key}_config.json';write_json(path,config)
        branches[key]={'directory':f'branch_{key}','label':template['label'],'config':str(path),
            'workers':config['workers'],'attention':config['attention'],
            'teacher_pretraining':bool(config.get('teacher_pretraining')),'input_schema':config['input_schema'],
            'script':str(runtime/'scripts/train_thrl_branch.py')}
    comparison={'schema':'thrl-control-comparison-v1','branches':branches,'python':request['python'],
        'cwd':str(runtime),'api_port':0,'branch_launch_stagger_seconds':5,
        'source_runtime':str(runtime),'trial_decisions':plan['target_decisions'],
        'primary_comparison':'protected trial000; later Optuna trials exploratory',
        'formal_acceptance':False}
    write_json(control/'comparison_config.json',comparison)
    write_json(control/'preparation_receipt.json',{'status':'prepared','plan':plan,'recovery':audit,
        'reward_or_PPO_changed':any(read_shared_json(b['config']).get('training_recipe') in ('th14-completion-v2','th14-completion-v3') for b in branches.values()),
        'wall_cap':None,'utc':utc_now()})
    return comparison
