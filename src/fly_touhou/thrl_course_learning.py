"""Connect one on-policy learner to course scheduling and independent gates."""
from pathlib import Path
import json
import time
import numpy as np
from .thrl_policy_actions import policy_logits

from .atomic_json_io import atomic_write_json
from .thrl_course_store import CourseStore
from .thrl_curriculum import (Curriculum,SCHEMA,qualified,resource_qualified,schema_for,
                              chain_horizon,CHAIN_LESSON_BOUNDARY)
from .th14_completion_protocol import reward_options
from .thrl_recurrent import PolicyHistory
from .thrl_inputs import policy_is_recurrent


class CourseMaintenance:
    """At most one low-priority native verifier per policy, with a small queue."""
    def __init__(self,config,output):
        self.config=config;self.output=Path(output);self.pending=[];self.process=None
        self.logs=[];self.count=0;self.results=[]
        self.stop_path=self.output/'STOP_curriculum_verifier'

    def enqueue(self,path):
        if len(self.pending)<3 and path not in self.pending:self.pending.append(path)

    def poll(self):
        import os,subprocess,sys
        if self.process is not None:
            if self.process.poll() is None:return
            self.results.append(dict(exit_code=self.process.returncode,output=str(self.folder)))
            self.process=None
            for stream in self.logs:stream.close()
            self.logs=[]
        if not self.pending or self.stop_path.exists():return
        candidate=self.pending.pop(0);self.count+=1
        self.folder=self.output/f'course_verifier_{self.count:05d}_{int(time.time())}'
        script=Path(__file__).resolve().parents[2]/'scripts/verify_th14_course_candidate.py'
        command=[sys.executable,'-B',str(script),'--candidate',str(candidate),
            '--executable',self.config['game_exe'],'--profile',self.config['profile'],
            '--output',str(self.folder),'--publish',str(self.output/'curriculum_pool'),
            '--stop-file',str(self.stop_path)]
        for key in ('operator_stop_path','comparison_stop_path'):
            if self.config.get(key):command+=['--stop-file',str(self.config[key])]
        command+=['--stop-file',str(self.output.parent/'STOP')]
        self.logs=[(self.output/f'course_verifier_{self.count:05d}.{suffix}.log').open('wb') for suffix in ('stdout','stderr')]
        self.process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=self.logs[0],stderr=self.logs[1],
            creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.stop_path.write_text('owned course verifier stop\n','utf-8')
            self.process.wait(timeout=60)
        for stream in self.logs:stream.close()

    def summary(self):
        return dict(queued=len(self.pending),running=self.process is not None and self.process.poll() is None,
                    completed=self.results[-10:],completed_count=len(self.results),maximum_extra_native_verifiers=1,training_updates=0)


def environment_options(config,output,count,role):
    if not config.get('curriculum_manifest'):return None
    split='train' if role in ('train','teacher') else 'test' if role=='test' else 'development'
    overlay=config.get('curriculum_overlay_path',str(Path(output)/'curriculum_pool'))
    # Only learner rollouts traverse the entire native game. Teacher capture
    # and independent gates retain their bounded, comparable target episodes.
    horizon=chain_horizon(config.get('curriculum_chain_horizon',CHAIN_LESSON_BOUNDARY))
    return [dict(curriculum_manifest=config['curriculum_manifest'],
        curriculum_manifest_sha256=config['curriculum_manifest_sha256'],
        curriculum_overlay=overlay,curriculum_split=split,
        curriculum_options=config.get('curriculum_options'),
        curriculum_chain_horizon=horizon if role=='train' else CHAIN_LESSON_BOUNDARY,
        curriculum_recurrent=policy_is_recurrent(config),
        curriculum_inherited_pools=config.get('curriculum_inherited_pools',[]),
        curriculum_seed=config['seed']+1009*i+(0 if split=='train' else 200000),
        curriculum_default_mode='chain' if role=='teacher' else 'current',
        harvest=role=='train' and i==0,**reward_options(config)) for i in range(count)]


class CourseRuntime:
    def __init__(self,config,output,count,state=None):
        self.output=Path(output)
        self.config={**config,'curriculum_overlay_path':str(self.output/'curriculum_pool')}
        self.store=CourseStore(config['curriculum_manifest'],config['curriculum_manifest_sha256'],
                               self.output/'curriculum_pool',config.get('curriculum_inherited_pools'))
        self.scheduler=Curriculum(self.store.lessons(),seed=config['seed'],
                                  options=config.get('curriculum_options'),state=state,
                                  curriculum_chain_horizon=config.get('curriculum_chain_horizon',CHAIN_LESSON_BOUNDARY))
        self.history=PolicyHistory(count,self.scheduler.options['burn_in'],enabled=policy_is_recurrent(config))
        self.assignments=[None]*count
        self.gate_number=0
        self.maintenance=CourseMaintenance(config,self.output)

    def reset_options(self,index):
        self.store.refresh()
        latest=self.store.lessons()
        for old,new in zip(self.scheduler.lessons,latest):old.update(new)
        assignment=self.scheduler.sample()
        self.assignments[index]=assignment
        return dict(assignment=assignment,split='train')

    def acknowledge(self,index,info):
        assignment=info.get('curriculum_assignment')
        if assignment!=self.assignments[index]:raise ValueError('Acknowledged course assignment differs from the dispatched lesson')
        self.scheduler.acknowledge(assignment,info)
        if info.get('curriculum_harvest_candidate'):self.maintenance.enqueue(info['curriculum_harvest_candidate'])

    def initial_history(self,vector):
        for i,payload in enumerate(vector.burn_in):self.history.reset(i,payload)

    def summary(self):
        return {**self.scheduler.summary(), 'store':self.store.summary(),'recurrent_history':self.history.summary(),
                'burn_in':self.scheduler.options['burn_in'] if self.history.enabled else 0,
                'temporal_core':'gru' if self.history.enabled else 'mlp',
                'maintenance':self.maintenance.summary()}

    def pending_gate_request(self,initialization,decisions):
        if (initialization or {}).get('source_checkpoint_phase')!='curriculum_gate':return None
        if decisions!=self.scheduler.last_gate_decisions:
            raise ValueError('Interrupted course gate must match the recovered optimized prefix')
        return dict(lesson=self.scheduler.lesson['id'],frontier=self.scheduler.frontier,
            depth=self.scheduler.depth,episodes=self.scheduler.options['gate_episodes'],split='development',
            full_span=self.scheduler.depth>=self.scheduler.lesson['max_depth'])

    def gate(self,torch,model,decisions,check_stop,on_progress=None,retry_request=None):
        request=retry_request if retry_request is not None else self.scheduler.request_gate(decisions)
        if request is None:return None
        self.gate_number+=1
        folder=self.output/f'curriculum_gate_{decisions:09d}_{self.gate_number:03d}'
        report=evaluate_course_gate(torch,model,self.config,request,folder,check_stop,
                                    on_progress=on_progress)
        action=self.scheduler.apply_gate(report,decisions)
        atomic_write_json(folder/'decision.json',dict(action=action,curriculum=self.scheduler.summary()))
        with (self.output/'curriculum_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(decisions=decisions,action=action,report=str(folder/'report.json')))+'\n')
        return dict(action=action,report=report)


def gate_cases(cfg,request,lessons):
    cases=[('current',request['lesson'],None if request['full_span'] else request['depth'],request['episodes'])]
    if request['full_span'] or cfg.get('version')==2:
        cases.append(('chain',request['lesson'],None,request['episodes']))
    if request['full_span']:
        for index in sorted({0,max(0,request['frontier']-1)}):
            if index<request['frontier']:
                cases.append(('review',lessons[index]['id'],None,max(5,request['episodes']//2)))
    return cases


def evaluate_course_gate(torch,model,config,request,output,check_stop,*,on_progress=None):
    """Current frozen policy, native development seeds, no learning or teacher.

    A full-span gate includes a real opening-to-frontier chain plus opening
    and previous-lesson retention. Intermediate window gates test that window
    only. Resource statistics accompany every outcome; no fake native clear.
    """
    if config.get('curriculum_eval_backend')=='ready-parallel-v1':
        from .thrl_course_evaluation import evaluate_parallel_gate
        return evaluate_parallel_gate(torch,model,config,request,output,check_stop,on_progress=on_progress)
    from .thrl_vector import THRLNativeVector
    from .thrl_inputs import forward_observation
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=Curriculum(CourseStore(config['curriculum_manifest']).lessons(),options=config.get('curriculum_options')).options
    cases=gate_cases(cfg,request,CourseStore(config['curriculum_manifest']).lessons())
    # Separate RNG scope: gates must not consume PPO action-sampling RNG.
    model.eval();groups={};vector=None;begun=time.monotonic()
    with torch.random.fork_rng(devices=[next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []):
        torch.manual_seed(config['seed']+300000+int(output.name.split('_')[2]))
        kwargs=environment_options(config,output,1,'development')[0]
        kwargs.update(executable=config['game_exe'],profile_path=config['profile'],
            private_appdata=str(output/'native/userdata'),policy_action_count=19,
            input_schema=config.get('input_schema','semantic-separated-v1'),headless=True,
            frame_skip=4,hide_window=True,timeout=25,require_complete_reward_mapping=True)
        try:
            vector=THRLNativeVector([kwargs],output/'workers',timeout=900)
            device=next(model.parameters()).device
            for mode,lesson,depth,count in cases:
                results=[]
                assignment=dict(schema=schema_for(cfg),mode=mode,lesson=lesson,depth=depth,
                    frontier=request['frontier'],lesson_index=request['frontier'])
                for game in range(count):
                    check_stop();obs,infos=vector.reset(options=[dict(assignment=assignment,split='development')])
                    history=PolicyHistory(1,cfg['burn_in'],enabled=getattr(model,'is_recurrent',True))
                    history.reset(0,vector.burn_in[0])
                    hidden=history.refresh(torch,model)
                    while True:
                        check_stop()
                        with torch.no_grad():
                            logits,_,hidden=forward_observation(torch,model,obs,hidden)
                            action=torch.distributions.Categorical(logits=policy_logits(logits)).sample().cpu().numpy()
                        obs,_,term,trunc,infos=vector.step(action)
                        if term[0] or trunc[0]:
                            row=infos[0]['episode_summary'];row['qualified_pass']=qualified(row,cfg)
                            row['resource_qualified_pass']=resource_qualified(row,cfg)
                            results.append(row)
                            with (output/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                                stream.write(json.dumps({**row,'case':mode,'lesson':lesson,'game':game})+'\n')
                            break
                    if on_progress:on_progress(dict(status='curriculum_gate',case=mode,lesson=lesson,completed=game+1,target=count))
                groups[mode+':'+lesson]=dict(episodes=count,passes=sum(r['qualified_pass'] for r in results),
                    pass_rate=sum(r['qualified_pass'] for r in results)/count,
                    mean_misses=float(np.mean([r['segment_misses'] for r in results])),
                    mean_bombs_used=float(np.mean([r['segment_bombs_used'] for r in results])),
                    reached=sum(r['segment_reached'] for r in results),
                    resource_pass_rate=float(np.mean([r['resource_qualified_pass'] for r in results])),
                    mean_remaining_lives=float(np.mean([r.get('lives',-1) for r in results])),
                    mean_remaining_bombs=float(np.mean([r.get('bombs_remaining',0) for r in results])))
            report=dict(complete=True,split='development',lesson=request['lesson'],depth=request['depth'],
                pass_rate=groups['current:'+request['lesson']]['pass_rate'],
                chain_pass_rate=groups.get('chain:'+request['lesson'],{}).get('pass_rate'),
                chain_resource_pass_rate=groups.get('chain:'+request['lesson'],{}).get('resource_pass_rate'),
                retention={key.split(':',1)[1]:row['pass_rate'] for key,row in groups.items() if key.startswith('review:')},
                groups=groups,wall_seconds=time.monotonic()-begun,original_game_rng_controlled=True,
                policy_frozen=True,PPO_updates=0,formal_six_stage_clear=False)
            atomic_write_json(output/'report.json',report)
            return report
        finally:
            if vector is not None:
                vector.close()
                atomic_write_json(output/'native_cleanup.json',dict(close_errors=vector.close_errors))
                if vector.close_errors:raise RuntimeError('Course gate has unresolved native cleanup')
