"""Independent full-brain original-game campaigns with retained native evidence."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import time

import torch
from .campaign import CampaignTouhouEnv, FULL_GAME_SHA256
from .experiment import ExperimentLogger, file_sha256
from .observation import ObservationConfig
from .resource_environment import ResourceCampaignEnv
from .resource_observation import ResourceObservationConfig
from .frida_controller import OriginalGameController
from .resource_environment import ResourceGameController
from .dialogue_controller import (DialogueGameController, DialogueResourceGameController,
                                  label_owned_game_windows)


class _IsolatedControllerMixin:
    def __init__(self, *args, private_appdata, dialogue_skip, parallel_startup=False, responsive_wait=True, **kwargs):
        self._private_appdata = Path(private_appdata).resolve()
        self._dialogue_skip = bool(dialogue_skip)
        self._parallel_startup = bool(parallel_startup)
        self._responsive_wait = bool(responsive_wait and parallel_startup)
        super().__init__(*args, **kwargs)

    def _create_controller(self):
        if self._parallel_startup:
            from .parallel_controller import (ParallelDialogueResourceGameController,
                ParallelResourceGameController, ParallelDialogueGameController, ParallelGameController)
            if self._responsive_wait:
                from .responsive_controller import (
                    ResponsiveParallelDialogueResourceGameController as ParallelDialogueResourceGameController,
                    ResponsiveParallelResourceGameController as ParallelResourceGameController,
                    ResponsiveParallelDialogueGameController as ParallelDialogueGameController,
                    ResponsiveParallelGameController as ParallelGameController)
            if isinstance(self, ResourceCampaignEnv):
                controller = ParallelDialogueResourceGameController if self._dialogue_skip else ParallelResourceGameController
            else:
                controller = ParallelDialogueGameController if self._dialogue_skip else ParallelGameController
            return controller(self.executable,self.profile_path,timeout=self.timeout,
                              private_appdata=self._private_appdata,worker_name='wholebrain-evaluation')
        if isinstance(self, ResourceCampaignEnv):
            controller = DialogueResourceGameController if self._dialogue_skip else ResourceGameController
        else:
            controller = DialogueGameController if self._dialogue_skip else OriginalGameController
        return controller(self.executable, self.profile_path, timeout=self.timeout,
                          private_appdata=self._private_appdata)


class _IsolatedResourceEnv(_IsolatedControllerMixin, ResourceCampaignEnv):
    pass


class _IsolatedCampaignEnv(_IsolatedControllerMixin, CampaignTouhouEnv):
    pass


def evaluate_whole_brain(policy, executable, profile_path, checkpoint, run_dir, *,
                         source_dir, ports_manifest, episodes=1, difficulty=3,
                         max_episode_frames=120000, deterministic=True,
                         dialogue_skip=True, window_title='果蝇全脑学生 · Lunatic', parallel_startup=False,
                         action_seed=42, responsive_wait=True, frame_skip=4):
    if episodes < 1:
        raise ValueError('At least one complete native episode must be requested')
    if not isinstance(frame_skip, int) or isinstance(frame_skip, bool) or frame_skip < 2:
        raise ValueError('Evaluation frame_skip must be an integer >=2 for a native Bomb press and release')
    model = policy.student
    if model.n_neurons != 138639 or model.graph.n_edges != 15091983:
        raise ValueError('Final-goal evaluation requires the complete pinned brain')
    specification=policy.encoder.specification()
    if specification.get('schema') in ('native-resource-hazard-observation-v3','native-collision-geometry-observation-v4') and not (
            parallel_startup and responsive_wait):
        raise ValueError('Measured v3 evaluation requires parallel_startup and responsive_wait')
    source_dir, run_dir = Path(source_dir), Path(run_dir)
    config = {'phase':'independent-whole-brain-campaign-evaluation',
        'game_exe_sha256':FULL_GAME_SHA256,'checkpoint_sha256':file_sha256(checkpoint),
        'difficulty':difficulty,'episodes':episodes,'max_episode_frames':max_episode_frames,
        'frame_skip':frame_skip,
        'deterministic_actions':deterministic,'actor':'complete-brain-descending-spikes',
        'action_sampling_seed_base':int(action_seed),
        'teacher_action_calls':0,'safety_takeovers':0,'neural_state_reset':'once per run only',
        'game_rng_seed_applied':False,'intervention':'none',
        'private_appdata':str((run_dir/'game_userdata').resolve()),
        'native_dialogue_skip':bool(dialogue_skip),
        'parallel_startup_mutex_namespace':bool(parallel_startup),
        'responsive_checkpoint_wait':bool(parallel_startup and responsive_wait),
        'non_gameplay_input':'one SHOT release tick per native message object; held Z thereafter' if dialogue_skip else 'none',
        'window_title':window_title,
        'model_config':model.config(),'encoder':policy.encoder.specification()}
    if model.inference_execution_specification().get('kind') != 'eager':
        config['inference_execution'] = model.inference_execution_specification()
    files={'checkpoint':checkpoint,'game_executable':executable,'game_profile':profile_path,
           'evaluation_cli':Path(__file__).resolve().parents[2]/'scripts/evaluate_whole_brain.py',
           'ports':ports_manifest,'completeness':source_dir/'Completeness_783.csv',
           'connectivity':source_dir/'Connectivity_783.parquet',
           'upstream_model':source_dir/'model.py','evaluation_code':Path(__file__)}
    for module in ('whole_brain','whole_brain_learning','cached_cusparse','brain_cuda_capture','connectome','campaign','background','native',
                   'frida_controller','state','observation','planner','resources',
                   'resource_environment','resource_observation','dialogue_controller','experiment'):
        files[f'code_{module}']=Path(__file__).with_name(f'{module}.py')
    files['input_hook']=Path(__file__).parent/'hooks/th14.js'
    if specification.get('schema') in ('native-resource-hazard-observation-v3','native-collision-geometry-observation-v4'):
        for module in ('hazard_reader','hazard_observation','hazard_environment',
                       'isolated_campaign','parallel_campaign'):
            files[f'code_{module}']=Path(__file__).with_name(module+'.py')
    if specification.get('schema')=='native-collision-geometry-observation-v4':
        from .geometry_migration import validate_geometry_deployment
        data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        validate_geometry_deployment(data)
        binding=data.get('geometry_validation_proof',data['training_config'].get('geometry_validation_proof'))
        config['geometry_validation_proof']=binding
        files['geometry_validation_proof']=Path(binding['path'])
        for module in ('geometry_environment','geometry_observation','geometry_migration','native_geometry'):
            files[f'code_{module}']=Path(__file__).with_name(module+'.py')
    if parallel_startup:
        files['parallel_controller']=Path(__file__).with_name('parallel_controller.py')
        files['parallel_startup_hook']=Path(__file__).parent/'hooks/th14_parallel_startup.js'
        if responsive_wait:
            files['responsive_controller']=Path(__file__).with_name('responsive_controller.py')
            files['input_hook']=Path(__file__).parent/'hooks/th14_responsive.js'
    logger=ExperimentLogger(run_dir,config,files=files)
    logger.bind_whole_brain(model.graph,model,source_dir,ports_manifest=ports_manifest)
    if specification.get('schema') in ('native-resource-hazard-observation-v3','native-collision-geometry-observation-v4'):
        from .hazard_environment import HazardResourceCampaignEnv
        from .hazard_observation import HazardObservationConfig
        environment_type=HazardResourceCampaignEnv
        if specification.get('schema')=='native-collision-geometry-observation-v4':
            from .geometry_environment import GeometryObservationCampaignEnv
            environment_type=GeometryObservationCampaignEnv
        env=environment_type(executable,profile_path,difficulty=difficulty,frame_skip=frame_skip,
            private_appdata=run_dir/'game_userdata',dialogue_skip=dialogue_skip,
            window_title=window_title, max_episode_frames=max_episode_frames,
            hazard_observation_config=HazardObservationConfig(**specification['observation']))
    elif specification.get('schema')=='native-resource-observation-v2':
        env=_IsolatedResourceEnv(executable,profile_path,difficulty=difficulty,frame_skip=frame_skip,
            private_appdata=run_dir/'game_userdata',dialogue_skip=dialogue_skip,parallel_startup=parallel_startup,
            responsive_wait=responsive_wait,
            max_episode_frames=max_episode_frames,
            resource_observation_config=ResourceObservationConfig(**specification['observation']))
    else:
        env=_IsolatedCampaignEnv(executable,profile_path,difficulty=difficulty,frame_skip=frame_skip,
            private_appdata=run_dir/'game_userdata',dialogue_skip=dialogue_skip,parallel_startup=parallel_startup,
            responsive_wait=responsive_wait,
            max_episode_frames=max_episode_frames,
            observation_config=ObservationConfig(**specification['observation']))
    if env.frame_skip != frame_skip or env.encoder.specification()!=policy.encoder.specification():
        env.close()
        raise ValueError('Evaluation cadence or trained policy observations differ')
    reports=[]
    try:
        with logger.phase('independent-native-whole-brain-campaigns'):
            for episode in range(episodes):
                trajectory=Path(run_dir)/f'campaign_{episode:03d}.jsonl.gz'
                if trajectory.exists():
                    raise FileExistsError('Campaign evidence must never be overwritten')
                policy.reset()
                torch.manual_seed(int(action_seed)+episode)
                if model.device.type=='cuda':
                    torch.cuda.manual_seed_all(int(action_seed)+episode)
                torch.save({'episode':episode,'action_sampling_seed':int(action_seed)+episode,
                    'cpu_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state(model.device) if model.device.type=='cuda' else None,
                    'native_game_rng_fixed':False},Path(run_dir)/f'campaign_{episode:03d}.action_rng.pt')
                observation,info=env.reset(seed=100000+episode)
                if window_title:
                    logger.event('owned-native-window-label',
                                 label_owned_game_windows(env.controller,window_title))
                    env.snapshot=env.controller.read()
                logger.record_native_context(env.snapshot,info)
                phase_start=time.monotonic()
                latencies=[]
                with gzip.open(trajectory,'xt',encoding='utf-8') as stream:
                    def emit(row):
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    emit({'record_type':'metadata','config':config,'episode':episode,
                          'initial_snapshot':env.snapshot.to_dict()})
                    for step in range(max_episode_frames//env.frame_skip+1):
                        before=env.snapshot
                        prediction_start=time.monotonic()
                        action=int(policy.predict(observation,deterministic=deterministic)[0])
                        latency=(time.monotonic()-prediction_start)*1000
                        latencies.append(latency)
                        neural=model.diagnostics(policy.state)
                        observation,reward,terminated,truncated,info=env.step(action)
                        emit({'record_type':'transition','step':step,'snapshot':before.to_dict(),
                              'action':action,'action_source':'whole-brain-descending-spikes',
                              'action_evidence':policy.last_action_evidence,
                              'neural':neural,'neural_noise_counter':int(policy.state.noise_counter[0]),
                              'inference_ms':latency,'reward':reward,'elapsed_frames':info['elapsed_frames'],
                              'next_snapshot':env.snapshot.to_dict(),'terminated':terminated,'truncated':truncated,
                              'brain_state_reset':False,'teacher_action_calls':0})
                        if step%100==0 or terminated or truncated:
                            stream.flush()
                            progress={'episode':episode,'step':step,'native_frames':env.episode_frames,
                                'stage':env.snapshot.stage,'lives':env.snapshot.lives,'bombs':env.snapshot.bombs,
                                'score':env.snapshot.score,'neural':neural,'terminated':terminated,'truncated':truncated}
                            logger.event('whole-brain-native-progress',progress)
                            print('BRAIN CAMPAIGN',episode,step,env.snapshot.stage,env.snapshot.lives,
                                  env.snapshot.bombs,env.snapshot.score,flush=True)
                        if terminated or truncated:
                            ending_verified = bool(info['native_ending_flag']
                                and env.snapshot.extra.get('native_gameover') is False)
                            result={'exe_sha256':FULL_GAME_SHA256,'difficulty':env.snapshot.difficulty,
                                'frame_skip':env.frame_skip,
                                'completed_stages':list(env.stage_history) if ending_verified else [],
                                'visited_stages':list(env.stage_history),
                                'ending_verified':ending_verified,
                                'native_gameover':env.snapshot.extra.get('native_gameover'),
                                'finite_native_lives':True,'resource_overrides':False,
                                'continues_used':env.snapshot.extra.get('continues'),
                                'teacher_action_calls':0,'safety_takeovers':0,
                                'brain_state_resets_between_stages':0,'native_resource_resets_between_stages':0,
                                'run_mode':env.snapshot.extra.get('run_mode'),
                                'replay_playback':env.snapshot.extra.get('replay_playback'),
                                'native_frames':env.episode_frames,'steps':step+1,'life_decrements':env.life_decrements,
                                'score':env.snapshot.score,'terminated':terminated,'truncated':truncated,
                                'native_dialogue_input_events':getattr(env.controller,'dialogue_input_events',[]),
                                'native_startup_isolation':env.snapshot.extra.get('parallel_startup_isolation'),
                                'wall_seconds':time.monotonic()-phase_start,
                                'mean_inference_ms':sum(latencies)/len(latencies),
                                'last_snapshot':env.snapshot.to_dict()}
                            emit({'record_type':'campaign_result','result':result})
                            break
                gate=logger.record_campaign_result(result,evidence_files={'trajectory':trajectory})
                reports.append({'result':result,'acceptance_gate':gate})
        summary={'campaigns':reports,'all_requested_campaigns_passed':all(
            row['acceptance_gate']['reported_measurements_pass_gate'] for row in reports)}
        (Path(run_dir)/'evaluation_summary.json').write_text(
            json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        return summary
    finally:
        env.close()
