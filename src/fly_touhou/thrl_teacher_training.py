"""Fresh V6 demonstrations and bounded recurrent soft-CE pretraining for C.

Torch is passed by the owning branch, never imported at module/worker startup.
The memmap stores current FP32 THRL observations, not legacy flat features.
PPO and BC optimizers/counters remain separate in the resulting checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import uuid

import numpy as np

from .atomic_json_io import atomic_write_json
from .thrl_study_io import read_shared_json, utc_now
from .thrl_teacher import TEACHER_SCHEMA, teacher_env_factory, teacher_specification,teacher_schema
from .thrl_limits import MAX_NATIVE_WORKERS


@dataclass(frozen=True)
class TeacherPretrainingConfig:
    samples: int = 8192
    workers: int = 4
    epochs: int = 4
    seq_len: int = 16
    seqs_per_batch: int = 4
    learning_rate: float = .0001
    max_grad_norm: float = .5
    validation_enabled: bool = False
    validation_fraction: float = .25
    validation_seed: int = 11
    minimum_epochs: int = 4
    patience: int = 4
    minimum_kl_improvement: float = .005
    validation_max_kl: float = .5
    mirror_probability: float = 0.
    context_mode: str = 'chunk-zero'

    def __post_init__(self):
        for name in ('samples', 'workers', 'epochs', 'seq_len', 'seqs_per_batch'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('Positive integer required: ' + name)
        if self.workers > MAX_NATIVE_WORKERS or self.samples % (self.workers * self.seq_len):
            raise ValueError('Demonstration budget must contain whole per-world sequences')
        if self.samples % (self.seq_len * self.seqs_per_batch):
            raise ValueError('Demonstration sequences must form complete batches')
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError('Finite positive learning rate and gradient bound required')
        if type(self.validation_enabled) is not bool:
            raise ValueError('Explicit validation_enabled boolean required')
        if self.context_mode not in ('chunk-zero','episode-refresh'):
            raise ValueError('Explicit teacher recurrent context mode required')
        if isinstance(self.mirror_probability,bool) or not math.isfinite(self.mirror_probability) or not 0<=self.mirror_probability<=1:
            raise ValueError('Mirror probability must be finite in [0,1]')
        if self.validation_enabled:
            if (type(self.minimum_epochs) is not int or not math.isfinite(self.validation_fraction)
                    or not 0 < self.validation_fraction < .5 or not 4 <= self.minimum_epochs <= self.epochs <= 200):
                raise ValueError('Validated BC requires 4..200 epochs and a holdout below one half')
            if type(self.validation_seed) is not int or type(self.patience) is not int or self.patience < 1:
                raise ValueError('Deterministic validation seed and positive patience required')
            if not math.isfinite(self.minimum_kl_improvement) or self.minimum_kl_improvement <= 0:
                raise ValueError('Positive finite validation improvement required')
            if not math.isfinite(self.validation_max_kl) or self.validation_max_kl <= 0:
                raise ValueError('Positive finite validation KL diagnostic threshold required')

    @property
    def updates(self):
        return self.epochs * self.samples // (self.seq_len * self.seqs_per_batch)


from .thrl_geometry_observation import FEATURE_DIM, GEOMETRY_SCHEMA
from .thrl_inputs import observation_shapes, SEPARATED_INPUT, make_encoder, is_separated_input

def data_fields(schema=SEPARATED_INPUT, *, validation=False):
    return {**{key:(np.float32,shape) for key,shape in observation_shapes(schema).items()},
    'targets': (np.float32, (19,)),
    'actions': (np.int64, ()), 'dones': (np.bool_, ()),
    'episode_starts': (np.bool_, ()), 'native_episodes': (np.int64, ()),
    'before_frames': (np.int64, ()), 'after_observation_frames': (np.int64, ()),
    **({'native_seeds':(np.int64,())} if validation else {})}

DATA_FIELDS = data_fields()


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def restore_initial_sampling_rng(torch,checkpoint):
    """BC changes actor weights, but not the controlled initial PPO RNG stream."""
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    torch.set_rng_state(source['torch_rng']);torch.cuda.set_rng_state_all(source['cuda_rng'])
    np.random.set_state(source['numpy_rng']);random.setstate(source['python_rng'])


def allocate_demonstrations(directory, cfg, schema=SEPARATED_INPUT):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    shape = (cfg.samples // cfg.workers, cfg.workers)
    required = sum(np.dtype(dtype).itemsize * int(np.prod((*shape, *tail)))
                   for dtype, tail in data_fields(schema,validation=cfg.validation_enabled).values())
    # Refuse before creating original processes. No large in-RAM rollout copy.
    if shutil.disk_usage(directory).free < required + 256 * 1024**2:
        raise OSError('Insufficient space for fresh FP32 teacher maps and checkpoint reserve')
    arrays = {}
    try:
        for name, (dtype, tail) in data_fields(schema,validation=cfg.validation_enabled).items():
            arrays[name] = np.lib.format.open_memmap(directory / (name + '.npy'), mode='w+',
                        dtype=dtype, shape=(*shape, *tail))
    except BaseException:
        _close_arrays(arrays)
        raise
    return arrays


def _close_arrays(arrays):
    for value in arrays.values():
        value.flush()
        value._mmap.close()


def _env_kwargs(config, output, count):
    from .th14_completion_protocol import reward_options
    from .thrl_training_mode import full_story_environment_options
    full_story_options=full_story_environment_options(config,count,'teacher')
    extra=({'start_bank':config['start_bank'],'segment_end':config.get('segment_end'),
            'start_bank_sha256':config.get('start_bank_sha256'),
            'full_story_fraction':config.get('full_story_fraction',0.)}
           if config.get('start_bank') else {})
    from .thrl_course_learning import environment_options
    courses=None if full_story_options else environment_options(config,output,count,'teacher')
    return [dict(executable=config['game_exe'], profile_path=config['profile'],
        private_appdata=str(output / 'native' / f'worker{i}' / 'appdata'),
        difficulty=3, frame_skip=4, wall_time_limit_seconds=None,
        require_complete_reward_mapping=True, hide_window=True,
        window_title=f'THRL C V6 demonstration {i}', timeout=25,
        policy_action_count=19,headless=config.get('headless',True),
        input_schema=config.get('input_schema',SEPARATED_INPUT),
        **({'teacher_revision':config['teacher_revision']} if config.get('teacher_revision') else {}),
        **(full_story_options[i] if full_story_options else courses[i] if courses else
           {**reward_options(config),**extra,'curriculum_seed':config.get('seed',11)+1009*i} if extra else reward_options(config))) for i in range(count)]


def _validate_label(info,expected_schema=TEACHER_SCHEMA):
    label = info.get('teacher_label')
    if (not label or label.get('schema') != expected_schema
            or label.get('timing') != 'current-before-action'
            or label['native_episode'] != info.get('native_episode')
            or label['frame'] != info.get('observation_native_frame')):
        raise ValueError('Missing/misaligned current-before-action V6 target')
    target = np.asarray(label['probabilities'], dtype=np.float32)
    if (target.shape != (19,) or not np.isfinite(target).all() or np.any(target < 0)
            or not np.isclose(target.sum(dtype=np.float64), 1., atol=2e-7, rtol=0)
            or int(target.argmax()) != label['action']):
        raise ValueError('Invalid normalized source19 demonstration distribution')
    return label, target


def collect_teacher_demonstrations(config, output, cfg, *, on_progress, check_stop,
                                   vector_class=None):
    """One finite prefix, no auto-reset or fabricated final gameover label."""
    from .thrl_training_mode import validate_training_mode
    validate_training_mode(config)
    from . import thrl_vector
    output = Path(output)
    schema=config.get('input_schema',SEPARATED_INPUT)
    fields=data_fields(schema,validation=cfg.validation_enabled)
    arrays = allocate_demonstrations(output / 'demonstrations', cfg,schema)
    vector = None
    previous_factory = thrl_vector.worker_factory
    thrl_vector.worker_factory = teacher_env_factory
    completed = frames = input_ticks = menu_ticks = samples = executed = 0
    external_cuts = 0
    infos, episodes = [], []
    final = None
    last_write = 0.
    failure = None

    def report(force=False):
        nonlocal last_write
        now = time.monotonic()
        if force or now - last_write >= 5.:
            on_progress({'stage': 'teacher_collection', 'teacher_samples': samples,
                'target_teacher_samples': cfg.samples, 'teacher_actions_executed': executed,
                'teacher_native_frames': frames, 'teacher_input_ticks': input_ticks,
                'teacher_menu_ticks': menu_ticks, 'teacher_pretrain_updates': 0,
                'teacher_complete_episodes': completed, 'teacher_external_cuts': external_cuts,
                'workers': cfg.workers, 'parameters': asdict(cfg),
                'current_environments': [{key: row.get(key) for key in
                    ('pid', 'native_episode', 'stage', 'stage_frame', 'lives', 'bombs',
                     'episode_frames', 'policy_steps')} | {'worker': i}
                    for i, row in enumerate(infos)]})
            last_write = now

    try:
        check_stop()
        cls = thrl_vector.THRLNativeVector if vector_class is None else vector_class
        vector = cls(_env_kwargs(config, output, cfg.workers), output / 'workers')
        observation, infos = vector.reset()
        episode_native_seeds=[row.get('full_story_native_seed',row.get('course_native_seed',-1)) for row in infos]
        menu_ticks += sum(int(row.get('menu_hook_ticks', 0)) for row in infos)
        starts = np.ones(cfg.workers, dtype=bool)
        report(True)
        for t in range(cfg.samples // cfg.workers):
            check_stop()
            labels_targets = [_validate_label(row,teacher_schema(config.get('teacher_revision'))) for row in infos]
            labels, targets = zip(*labels_targets)
            actions = np.asarray([row['action'] for row in labels], dtype=np.int64)
            for key in observation_shapes(schema):
                arrays[key][t] = observation[key]
            arrays['targets'][t] = np.stack(targets)
            arrays['actions'][t] = actions
            arrays['episode_starts'][t] = starts
            arrays['native_episodes'][t] = [row['native_episode'] for row in labels]
            if cfg.validation_enabled:
                # Keep every BC world in the existing training seed partition.
                # -1 is unknown, never interpreted as a unique held-out seed.
                arrays['native_seeds'][t] = episode_native_seeds
            arrays['before_frames'][t] = [row['frame'] for row in labels]
            before_episode_frames = [int(row['episode_frames']) for row in infos]
            before_action_counts = [int(row['teacher_actions_executed']) for row in infos]
            observation, _, terminated, truncated, infos = vector.step(actions)
            executed += cfg.workers
            done = np.asarray(terminated) | np.asarray(truncated)
            arrays['dones'][t] = done
            arrays['after_observation_frames'][t] = [row['observation_native_frame'] for row in infos]
            for i, (label, info) in enumerate(zip(labels, infos)):
                delta = int(info['episode_frames']) - before_episode_frames[i]
                if delta < 0:
                    raise ValueError('Native episode frame clock moved backwards')
                frames += delta
                input_ticks += int(info['elapsed_input_ticks'])
                if (info.get('executed_teacher_label_frame') != label['frame']
                        or info.get('executed_teacher_native_episode') != label['native_episode']
                        or info.get('policy_action') != int(actions[i])
                        or info.get('teacher_actions_executed') != before_action_counts[i] + 1):
                    raise ValueError('Executed teacher action differs from the saved before-state label')
                if done[i]:
                    row = dict(info['episode_summary'], worker=i)
                    episodes.append(row)
                    completed += int(bool(terminated[i]))
                    external_cuts += int(bool(truncated[i]))
                    with (output / 'demonstration_episodes.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            samples += cfg.workers
            starts = done
            if t % 64 == 0:
                for array in arrays.values():
                    array.flush()
            report()
            if done.any() and samples < cfg.samples:
                indices = np.flatnonzero(done).tolist()
                reset_infos = vector.reset_at(indices)
                for i, reset_info in zip(indices, reset_infos):
                    infos[i] = reset_info
                    episode_native_seeds[i]=reset_info.get('full_story_native_seed',reset_info.get('course_native_seed',-1))
                    menu_ticks += int(reset_info.get('menu_hook_ticks', 0))
                observation = vector.observation
        final = {'schema': teacher_schema(config.get('teacher_revision')), 'status': 'complete-budget-prefix',
            'samples': samples, 'teacher_actions_executed': executed,
            'native_frames': frames, 'native_input_ticks': input_ticks,
            'menu_hook_ticks': menu_ticks, 'complete_native_episodes': completed,
            'external_cuts': external_cuts,
            'world_endpoints': [{'worker': i, 'native_episode': info['native_episode'],
                'episode_frames': info['episode_frames'], 'stage': info.get('stage'),
                'stage_frame': info.get('stage_frame'),
                'outcome': (info['episode_summary']['outcome'] if starts[i]
                            else 'unknown-budget-prefix'),
                'terminated_by_budget': False, 'final_row_done': bool(starts[i]),
                'teacher_queries': info.get('teacher_calls'),
                'teacher_actions_executed': info.get('teacher_actions_executed')}
                for i, info in enumerate(infos)],
            'parameters': asdict(cfg), 'shape': [cfg.samples // cfg.workers, cfg.workers],
            'teacher': teacher_specification(config.get('teacher_revision')), 'fresh_original_interface': True,
            'observation': make_encoder(schema).specification()['schema'], 'input_schema':schema,
            'old_flat_data_used': False,
            'extra_terminal_or_budget_failure_labels': 0}
        report(True)
    except BaseException as error:
        failure = {'status': 'stopped-or-failed-prefix', 'error': repr(error),
            'acknowledged_samples': samples, 'acknowledged_native_frames': frames,
            'acknowledged_teacher_actions_executed': executed,
            'unknown_inflight_cost_possible': True}
        atomic_write_json(output / 'collection_prefix.json', failure)
        # Periodic progress can lag the last completely acknowledged batch.
        # Publish the exact paid prefix before the branch reports failure.
        report(True)
        raise
    finally:
        try:
            if vector is not None:
                vector.close()
                cleanup = {'close_errors': vector.close_errors,
                    'ownership': read_shared_json(vector.ownership_path)}
                atomic_write_json(output / 'native_cleanup.json', cleanup)
                if vector.close_errors and failure is None:
                    raise RuntimeError('Teacher demonstration native cleanup has unresolved errors')
        finally:
            thrl_vector.worker_factory = previous_factory
            _close_arrays(arrays)
    final['files'] = {name: {'path': str((output / 'demonstrations' / (name + '.npy')).resolve()),
                            'bytes': (output / 'demonstrations' / (name + '.npy')).stat().st_size,
                            'shape': [cfg.samples // cfg.workers, cfg.workers, *tail],
                            'dtype': np.dtype(dtype).name}
                      for name, (dtype, tail) in fields.items()}
    atomic_write_json(output / 'demonstrations.json', final)
    return final


def sequence_batches(cfg, *, rng):
    sequences = [(world, start) for world in range(cfg.workers)
                 for start in range(0, cfg.samples // cfg.workers, cfg.seq_len)]
    for begin in range(0, len(sequences), cfg.seqs_per_batch):
        if begin == 0:
            rng.shuffle(sequences)
        yield sequences[begin:begin + cfg.seqs_per_batch]


def _fit_teacher_legacy(torch, model, directory, cfg, *, on_progress, check_stop):
    """CPU or owning CUDA device; only one 16x4 FP32 map batch is copied at once."""
    arrays = {name: np.load(Path(directory) / (name + '.npy'), mmap_mode='r')
              for name in ('features', 'maps', 'targets', 'dones', 'episode_starts')}
    if is_separated_input(getattr(model,'input_schema','legacy')):
        arrays['objects']=np.load(Path(directory)/'objects.npy',mmap_mode='r')
    device = next(model.parameters()).device
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if not name.startswith(('critic_trunks.', 'critic_heads.'))]
    original_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(('critic_trunks.', 'critic_heads.')))
    optimizer = torch.optim.Adam([parameter for _, parameter in trainable], lr=cfg.learning_rate)
    updates = examples_seen = 0
    losses = []
    model.train()
    try:
        for epoch in range(cfg.epochs):
            epoch_loss = 0.
            batches = 0
            for descriptors in sequence_batches(cfg, rng=np.random):
                check_stop()

                def batch(name):
                    return torch.from_numpy(np.stack([arrays[name][start:start + cfg.seq_len, world]
                        for world, start in descriptors])).to(device)

                features, maps, targets = batch('features'), batch('maps'), batch('targets')
                dones, starts = batch('dones'), batch('episode_starts')
                if (not torch.isfinite(targets).all() or torch.any(targets < 0)
                        or not torch.allclose(targets.sum(-1), torch.ones_like(targets[..., 0]),
                                              atol=2e-7, rtol=0)):
                    raise ValueError('Non-normalized teacher targets before BC update')
                # Each sampled chunk begins with a zero hidden. Done resets
                # inside it are handled after action[t], before input[t+1].
                hidden = torch.zeros(1, len(descriptors), model.gru_hidden_size, device=device)
                if cfg.seq_len > 1 and not torch.equal(starts[:, 1:], dones[:, :-1]):
                    raise ValueError('Demonstration episode boundaries are not aligned')
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model.forward_sequence(features, maps, hidden, dones,
                    **({'objects':batch('objects')} if 'objects' in arrays else {}))
                from .thrl_policy_actions import teacher_cross_entropy
                loss = teacher_cross_entropy(logits,targets.reshape(-1,19))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite BC loss; stop instead of rejecting batches')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_([p for _, p in trainable],
                    cfg.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                examples_seen += len(descriptors) * cfg.seq_len
                epoch_loss += float(loss.detach())
                batches += 1
                if updates == 1 or updates % 8 == 0:
                    on_progress({'stage': 'teacher_pretraining', 'teacher_pretrain_updates': updates,
                        'teacher_samples': cfg.samples, 'target_teacher_samples': cfg.samples,
                        'teacher_examples_seen': examples_seen, 'teacher_pretrain_target_updates': cfg.updates,
                        'teacher_epoch': epoch + 1, 'teacher_epochs': cfg.epochs,
                        'metrics': {'teacher_soft_cross_entropy': float(loss.detach()),
                                    'teacher_grad_norm': float(grad)},
                        'workers': cfg.workers, 'parameters': asdict(cfg)})
            losses.append(epoch_loss / batches)
        if updates != cfg.updates or examples_seen != cfg.samples * cfg.epochs:
            raise ValueError('BC budget/accounting mismatch')
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_grad[name])
        model.zero_grad(set_to_none=True)
        for array in arrays.values():
            array._mmap.close()
    return optimizer, {'BC_adam_minibatch_steps': updates, 'examples_seen': examples_seen,
        'unique_teacher_samples': cfg.samples, 'epochs': cfg.epochs,
        'epoch_mean_soft_cross_entropy': losses,
        'optimizer_parameter_names': [name for name, _ in trainable],
        'critic_parameter_updates': 0, 'critic_targets': None,
        'chunk_initial_hidden': 'zero; reset at every chunk and after every done inside chunk',
        'loss': 'mean source19 soft cross entropy; no reward, advantage or label rejection'}


def grouped_teacher_split(native_seeds, native_episodes, cfg):
    """Disjoint native RNG groups, including repeats across worlds/episodes.

    Every seed and entire episode belongs to one side. A 16-step sequence that
    straddles different sides is excluded and accounted for, never split into
    near-identical train/holdout frames. No development/test seeds are consumed.
    """
    seeds=np.asarray(native_seeds);episodes=np.asarray(native_episodes)
    shape=(cfg.samples//cfg.workers,cfg.workers)
    if seeds.shape!=shape or episodes.shape!=shape:
        raise ValueError('Grouped BC requires aligned native seeds/episode matrices')
    base=dict(schema='teacher-native-seed-grouped-holdout-v1',seed=cfg.validation_seed,
              available=False,train=[],heldout=[],groups={},excluded_boundary_samples=0,
              native_development_or_test_seeds_used=False)
    if np.any(seeds<0) or np.any(seeds>65535):
        return {**base,'reason':'unknown-native-seed'}
    if np.any(seeds%10>=6):
        return {**base,'reason':'non-training-native-seed-partition'}
    for world in range(cfg.workers):
        for episode in np.unique(episodes[:,world]):
            if len(np.unique(seeds[episodes[:,world]==episode,world]))!=1:
                return {**base,'reason':'native-seed-changed-within-episode'}
    distinct=sorted(int(x) for x in np.unique(seeds))
    if len(distinct)<2:return {**base,'reason':'fewer-than-two-independent-native-seeds'}
    order=sorted(distinct,key=lambda s:hashlib.sha256(f'{cfg.validation_seed}:{s}'.encode()).digest())
    n_hold=max(1,min(len(order)-1,int(round(len(order)*cfg.validation_fraction))))
    hold=set(order[:n_hold]);train=set(order[n_hold:])
    for world in range(cfg.workers):
        for start in range(0,shape[0],cfg.seq_len):
            chunk=set(int(x) for x in seeds[start:start+cfg.seq_len,world])
            if chunk<=train:base['train'].append((world,start))
            elif chunk<=hold:base['heldout'].append((world,start))
            else:base['excluded_boundary_samples']+=cfg.seq_len
    base['groups']={kind:dict(native_seeds=sorted(values),
        episodes=[[world,int(ep)] for world in range(cfg.workers)
            for ep in np.unique(episodes[:,world])
            if int(seeds[np.flatnonzero(episodes[:,world]==ep)[0],world]) in values])
        for kind,values in [('train',train),('heldout',hold)]}
    base['available']=bool(base['train'] and base['heldout'])
    base['reason']='ready' if base['available'] else 'no-complete-sequences-in-one-partition'
    base['train_samples']=len(base['train'])*cfg.seq_len
    base['heldout_samples']=len(base['heldout'])*cfg.seq_len
    base['split_sha256']=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest()
    return base


def _teacher_fit_status(initial, best, cfg):
    improved=initial['teacher_KL']-best['teacher_KL']>=cfg.minimum_kl_improvement
    return ('fit-diagnostic-passed' if improved and best['teacher_KL']<=cfg.validation_max_kl
            else 'underfit-or-unhelpful')


def _fit_teacher_validated(torch, model, directory, cfg, *, on_progress, check_stop):
    from .thrl_policy_actions import teacher_cross_entropy, policy_logits
    directory=Path(directory);output=directory.parent
    arrays={name:np.load(directory/(name+'.npy'),mmap_mode='r')
        for name in ('features','maps','targets','dones','episode_starts','native_episodes','native_seeds')}
    if is_separated_input(getattr(model,'input_schema','legacy')):
        arrays['objects']=np.load(directory/'objects.npy',mmap_mode='r')
    device=next(model.parameters()).device
    trainable=[(name,p) for name,p in model.named_parameters()
               if not name.startswith(('critic_trunks.','critic_heads.'))]
    original_grad={name:p.requires_grad for name,p in model.named_parameters()}
    optimizer=torch.optim.Adam([p for _,p in trainable],lr=cfg.learning_rate)
    metadata=dict(BC_adam_minibatch_steps=0,examples_seen=0,unique_teacher_samples=cfg.samples,
        epochs=0,epoch_mean_soft_cross_entropy=[],optimizer_parameter_names=[n for n,_ in trainable],
        critic_parameter_updates=0,critic_targets=None,
        shared_encoder_changes_can_change_critic_values=True,PPO_optimizer_modified=False,
        chunk_initial_hidden=cfg.context_mode,
        loss='legal-action-conditioned soft cross entropy',native_performance_evaluated=False,
        validation_enabled=True,fit_status='validation-unavailable',fit_claimed=False)
    history=[]
    context_cache={}

    def refresh_context(mirrored=False):
        if cfg.context_mode!='episode-refresh' or not model.is_recurrent:return
        from .thrl_bc_context import recurrent_context_cache
        def fetch(world,start,stop):
            return {name:torch.from_numpy(arrays[name][start:stop,world].copy()).to(device)
                for name in ('features','maps','objects','targets','dones')}
        context_cache[mirrored]=recurrent_context_cache(model,fetch,cfg.workers,cfg.samples//cfg.workers,mirrored=mirrored)

    def batch(descriptors,name):
        return torch.from_numpy(np.stack([arrays[name][start:start+cfg.seq_len,world]
            for world,start in descriptors])).to(device)

    def forward(descriptors):
        features,maps,targets=batch(descriptors,'features'),batch(descriptors,'maps'),batch(descriptors,'targets')
        dones,starts=batch(descriptors,'dones'),batch(descriptors,'episode_starts')
        if (not torch.isfinite(targets).all() or torch.any(targets<0) or
            not torch.allclose(targets.sum(-1),torch.ones_like(targets[...,0]),atol=2e-7,rtol=0)):
            raise ValueError('Non-normalized teacher targets before validated BC')
        if cfg.seq_len>1 and not torch.equal(starts[:,1:],dones[:,:-1]):
            raise ValueError('Demonstration episode boundaries are not aligned')
        objects=batch(descriptors,'objects') if 'objects' in arrays else None
        mirror_selected=torch.zeros(len(descriptors),device=device,dtype=torch.bool)
        if model.training and cfg.mirror_probability:
            from .thrl_augmentation import horizontal_mirror
            mirror_selected=torch.rand(len(descriptors),device=device)<cfg.mirror_probability
            features,maps,objects,targets=horizontal_mirror(features,maps,objects,targets,
                mirror_selected)
        hidden=torch.zeros(1,len(descriptors),model.gru_hidden_size,device=device)
        if context_cache:
            ix=torch.tensor([world*(cfg.samples//cfg.workers)+start for world,start in descriptors],device=device)
            hidden=context_cache[False][ix].unsqueeze(0)
            if mirror_selected.any():hidden=torch.where(mirror_selected[None,:,None],context_cache[True][ix][None],hidden)
        logits,_=model.forward_sequence(features,maps,hidden,dones,
            **({'objects':objects} if objects is not None else {}))
        return logits,targets.reshape(-1,19)

    def evaluate(descriptors, constant=None):
        totals=dict(soft_CE=0.,teacher_KL=0.,target_entropy=0.,argmax_agreement=0.,
                    Bomb_probability=0.,teacher_Bomb_probability=0.,Bomb_argmax_rate=0.)
        n=0;model.eval()
        with torch.no_grad():
            for begin in range(0,len(descriptors),cfg.seqs_per_batch):
                check_stop();logits,target=forward(descriptors[begin:begin+cfg.seqs_per_batch])
                logits=policy_logits(logits);legal=torch.isfinite(logits)
                if constant is not None:
                    logits=constant.clamp_min(1e-20).log().expand_as(logits).masked_fill(~legal,-torch.inf)
                target=target*legal.to(target.dtype);mass=target.sum(-1,keepdim=True)
                if (mass<=0).any():raise ValueError('Teacher assigns no probability to a legal action')
                target=target/mass
                logs=torch.nn.functional.log_softmax(logits,dim=-1)
                probs=logs.exp();safe_logs=logs.masked_fill(~legal,0.)
                entropy=-(target*target.clamp_min(1e-20).log()).sum(-1)
                ce=-(target*safe_logs).sum(-1)
                values=dict(soft_CE=ce,teacher_KL=ce-entropy,target_entropy=entropy,
                    argmax_agreement=(probs.argmax(-1)==target.argmax(-1)).float(),
                    Bomb_probability=probs[:,18],teacher_Bomb_probability=target[:,18],
                    Bomb_argmax_rate=(probs.argmax(-1)==18).float())
                n+=len(target)
                for key,value in values.items():totals[key]+=float(value.sum())
        return {**{k:v/n for k,v in totals.items()},'samples':n}

    def publish():
        atomic_write_json(output/'teacher_fit_report.json',{**metadata,'epoch_history':history})
        on_progress(dict(stage='teacher_pretraining',teacher_pretrain_updates=metadata['BC_adam_minibatch_steps'],
            teacher_examples_seen=metadata['examples_seen'],teacher_epoch=metadata['epochs'],teacher_epochs=cfg.epochs,
            teacher_fit_status=metadata['fit_status'],teacher_fit_claimed=metadata['fit_claimed'],
            teacher_validation=metadata.get('best_heldout'),teacher_fit_report=str(output/'teacher_fit_report.json'),
            teacher_validation_reason=metadata['split']['reason'],
            teacher_pretrain_target_updates=metadata.get('maximum_BC_adam_minibatch_steps',0),
            metrics=history[-1] if history else {}))

    try:
        check_stop()
        split=grouped_teacher_split(arrays['native_seeds'],arrays['native_episodes'],cfg)
        metadata['split']=split
        atomic_write_json(output/'teacher_validation_split.json',split)
        if not split['available']:
            metadata['stopped_reason']='validation-unavailable-no-BC-updates'
            publish();return optimizer,metadata
        for name,p in model.named_parameters():p.requires_grad_(not name.startswith(('critic_trunks.','critic_heads.')))
        metadata['maximum_BC_adam_minibatch_steps']=math.ceil(len(split['train'])/cfg.seqs_per_batch)*cfg.epochs
        marginal=np.mean(np.concatenate([arrays['targets'][start:start+cfg.seq_len,world]
            for world,start in split['train']]),axis=0)
        constant=torch.as_tensor(marginal,device=device)
        refresh_context()
        metadata['constant_training_marginal_heldout']=evaluate(split['heldout'],constant)
        metadata['initial_train']=evaluate(split['train']);metadata['initial_heldout']=evaluate(split['heldout'])
        initial=metadata['initial_heldout'];best=initial
        best_state={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
        best_optimizer=deepcopy(optimizer.state_dict());best_epoch=0;stale=0
        significant_best_kl=best['teacher_KL']
        rng=np.random.default_rng(cfg.validation_seed)
        for epoch in range(1,cfg.epochs+1):
            refresh_context()
            if cfg.mirror_probability:refresh_context(True)
            descriptors=list(split['train']);rng.shuffle(descriptors);model.train();loss_sum=count=0
            for begin in range(0,len(descriptors),cfg.seqs_per_batch):
                check_stop();logits,target=forward(descriptors[begin:begin+cfg.seqs_per_batch])
                optimizer.zero_grad(set_to_none=True);loss=teacher_cross_entropy(logits,target)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite validated BC loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for _,p in trainable],cfg.max_grad_norm,error_if_nonfinite=True)
                optimizer.step();metadata['BC_adam_minibatch_steps']+=1;metadata['examples_seen']+=len(target)
                loss_sum+=float(loss.detach())*len(target);count+=len(target)
            refresh_context()
            train_metrics=evaluate(split['train']);heldout=evaluate(split['heldout'])
            metadata['epochs']=epoch;metadata['epoch_mean_soft_cross_entropy'].append(loss_sum/count)
            history.append(dict(epoch=epoch,train=train_metrics,heldout=heldout))
            if heldout['teacher_KL']<best['teacher_KL']:
                best=heldout;best_epoch=epoch
                best_state={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
                best_optimizer=deepcopy(optimizer.state_dict())
            if significant_best_kl-heldout['teacher_KL']>=cfg.minimum_kl_improvement:
                significant_best_kl=heldout['teacher_KL'];stale=0
            else:stale+=1
            metadata.update(best_heldout=best,best_epoch=best_epoch,
                fit_status=_teacher_fit_status(initial,best,cfg),fit_claimed=False,
                native_performance_evaluated=False)
            publish()
            if epoch>=cfg.minimum_epochs and stale>=cfg.patience:
                metadata['stopped_reason']='heldout-KL-patience';break
        else:metadata['stopped_reason']='maximum-epochs'
        model.load_state_dict(best_state);optimizer.load_state_dict(best_optimizer)
        refresh_context()
        metadata.update(context_mode=cfg.context_mode,restored_best_epoch=best_epoch,restored_best_BC_optimizer=True,
            final_train=evaluate(split['train']),final_heldout=evaluate(split['heldout']),
            counters_include_paid_updates_after_best=True,
            fit_claimed=False,fit_claim_scope='diagnostic only; no native performance claim')
        publish()
        return optimizer,metadata
    finally:
        for name,p in model.named_parameters():p.requires_grad_(original_grad[name])
        model.zero_grad(set_to_none=True)
        for array in arrays.values():array._mmap.close()


def fit_teacher_demonstrations(torch,model,directory,cfg,*,on_progress,check_stop):
    fit=_fit_teacher_validated if cfg.validation_enabled else _fit_teacher_legacy
    return fit(torch,model,directory,cfg,on_progress=on_progress,check_stop=check_stop)


def save_teacher_checkpoint(torch, path, model, optimizer, bc_optimizer, *, initialization,
                            config, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload = {'schema': 'th14-thrl-training-checkpoint-v1', 'created_utc': utc_now(),
        'model_config': model.model_config(), 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_parameter_names': list(dict(model.named_parameters())),
        'optimizer_bc': bc_optimizer.state_dict(),
        'optimizer_bc_parameter_names': metadata['pretraining']['optimizer_parameter_names'],
        'decisions': 0, 'rollout_updates': 0, 'adam_minibatch_steps': 0,
        'lifetime_decisions': initialization['inherited_decisions'],
        'lifetime_rollout_updates': initialization['inherited_rollout_updates'],
        'lifetime_adam_minibatch_steps': initialization['inherited_adam_minibatch_steps'],
        'config': {**config,
            'inherited_decisions': initialization['inherited_decisions'],
            'inherited_rollout_updates': initialization['inherited_rollout_updates'],
            'inherited_adam_minibatch_steps': initialization['inherited_adam_minibatch_steps']},
        'teacher_pretraining': metadata,
        'teacher_samples': metadata['pretraining']['unique_teacher_samples'],
        'teacher_pretrain_updates': metadata['pretraining']['BC_adam_minibatch_steps'],
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
        'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
        'native_resume': 'new native runs, zero recurrent state; no full-world snapshot'}
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'checkpoint': str(path.resolve()), 'checkpoint_sha256': _digest(path),
            'inherited_PPO_decisions': payload['lifetime_decisions'],
            'inherited_PPO_rollout_updates': payload['lifetime_rollout_updates'],
            'inherited_PPO_adam_minibatch_steps': payload['lifetime_adam_minibatch_steps']}


def validated_teacher_result_fields(config, cfg, pretraining, output):
    """Bind a completed diagnostic and split to its protocol for later resume."""
    output=Path(output)
    return dict(teacher_fit_status=pretraining['fit_status'],teacher_fit_claimed=False,
        teacher_validation=pretraining.get('final_heldout'),
        teacher_validation_reason=pretraining['split']['reason'],
        teacher_fit_report=str(output/'teacher_fit_report.json'),
        teacher_restored_best_epoch=pretraining.get('restored_best_epoch',0),
        teacher_validation_enabled=True,teacher_pretraining_parameters=asdict(cfg),
        teacher_protocol={key:config.get(key) for key in
            ('training_recipe','reward_schema','input_schema','curriculum_protocol','training_mode','teacher_revision')},
        teacher_fit_report_sha256=_digest(output/'teacher_fit_report.json'),
        teacher_validation_split=str(output/'teacher_validation_split.json'),
        teacher_validation_split_sha256=_digest(output/'teacher_validation_split.json'))


def pretrain_teacher_branch(torch, config, output, on_progress, check_stop):
    """Entry called exclusively by the owning C branch before autonomous PPO."""
    from .thrl_training_mode import validate_training_mode
    validate_training_mode(config)
    from .thrl_model import MOActorCritic
    from .thrl_ppo import make_optimizer
    from .thrl_branching import warm_start

    cfg = TeacherPretrainingConfig(**config['teacher_pretraining'])
    check_stop()
    if config['attention'] != 'both' or config['policy_action_count'] != 19:
        raise ValueError('C demonstration transfer requires both attention and source19')
    if not torch.cuda.is_available():
        raise RuntimeError('Teacher pretraining requires the owning CUDA branch; no CPU fallback')
    output = Path(output) / 'teacher_pretraining'
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    overrides = dict(config.get('model_overrides', {}))
    schema=config.get('input_schema',SEPARATED_INPUT)
    overrides.setdefault('feature_dim', observation_shapes(schema)['features'][0])
    overrides.setdefault('input_schema',schema)
    overrides.setdefault('object_embed_dim',256)
    overrides.setdefault('object_attention_heads',8)
    overrides.setdefault('object_attention_layers',4)
    overrides.setdefault('spatial_attention_layers',2)
    if overrides['feature_dim'] != observation_shapes(schema)['features'][0] or overrides['input_schema']!=schema:
        raise ValueError('Teacher observation and network input schemas differ')
    model = MOActorCritic(attention='both', action_dim=19, **overrides).cuda()
    optimizer = make_optimizer(model)
    initialization = warm_start(torch, model, optimizer, config['warm_start_checkpoint'],
                                config['warm_start_sha256'])
    atomic_write_json(output / 'initialization_audit.json', initialization)
    atomic_write_json(output / 'config.json', {**config, 'teacher': teacher_specification(config.get('teacher_revision')),
        'parameters': asdict(cfg), 'model_config': model.model_config(),
        'parameter_counts': model.parameter_counts(), 'initialization': initialization,
        'expected_BC_updates': None if cfg.validation_enabled else cfg.updates,
        'maximum_BC_updates_before_holdout':cfg.updates,
        'expected_map_payload_bytes': cfg.samples * 16 * 92 * 96 * 4})

    progress_state = {'stage': 'teacher_pretraining', 'teacher_samples': 0,
        'target_teacher_samples': cfg.samples, 'teacher_actions_executed': 0,
        'teacher_pretrain_updates': 0}

    def progress(row):
        # Stage/failure notifications contain only changed fields. Retain the
        # most recent confirmed collection and BC counters across them.
        progress_state.update(row)
        value = {'schema': 'thrl-C-teacher-pretraining-progress-v1', 'status': 'training',
            'updated_utc': utc_now(), 'wall_seconds': time.monotonic() - started,
            'attention': 'both', 'device': 'cuda:0', 'gpu': torch.cuda.get_device_name(0),
            'actual_decisions': 0, 'adam_minibatch_steps': 0, 'rollout_updates': 0,
            'inherited_decisions': initialization['inherited_decisions'],
            'inherited_adam_minibatch_steps': initialization['inherited_adam_minibatch_steps'],
            'parameter_counts': model.parameter_counts(), 'formal_test': False, **progress_state}
        atomic_write_json(output / 'progress.json', value)
        on_progress(value)

    try:
        check_stop()
        demonstrations = collect_teacher_demonstrations(config, output, cfg,
            on_progress=progress, check_stop=check_stop)
        check_stop()
        bc_optimizer, pretraining = fit_teacher_demonstrations(torch, model,
            output / 'demonstrations', cfg, on_progress=progress, check_stop=check_stop)
        if config.get('match_PPO_initial_RNG',False):
            restore_initial_sampling_rng(torch,config['warm_start_checkpoint'])
        metadata = {'demonstrations': demonstrations, 'pretraining': pretraining,
                    'initialization': initialization, 'PPO_teacher_actions_after_transfer': 0,
                    'PPO_optimizer_state_modified_by_BC': False,
                    'PPO_initial_sampling_RNG_matched':bool(config.get('match_PPO_initial_RNG',False))}
        result = save_teacher_checkpoint(torch, output / 'teacher_pretrained.pt', model,
            optimizer, bc_optimizer, initialization=initialization, config=config, metadata=metadata)
        result.update(teacher_samples=cfg.samples, teacher_actions_executed=cfg.samples,
            BC_adam_minibatch_steps=pretraining['BC_adam_minibatch_steps'],
            teacher_native_frames=demonstrations['native_frames'],
            teacher_input_ticks=demonstrations['native_input_ticks'],
            teacher_menu_ticks=demonstrations['menu_hook_ticks'],
            teacher_complete_native_episodes=demonstrations['complete_native_episodes'],
            extra_demonstration_cost_separate_from_PPO=True,
            result=str((output / 'result.json').resolve()))
        if cfg.validation_enabled:
            result.update(validated_teacher_result_fields(config,cfg,pretraining,output))
        atomic_write_json(output / 'result.json', result)
        progress({'status': 'finished', 'stage': 'teacher_pretraining',
            'teacher_samples': cfg.samples, 'teacher_actions_executed': cfg.samples,
            'teacher_pretrain_updates': pretraining['BC_adam_minibatch_steps'],
            'workers': cfg.workers, 'parameters': asdict(cfg), 'checkpoint': result['checkpoint']})
        return result
    except BaseException as error:
        progress({'status': 'stopped-or-failed',
                  'error': repr(error), 'parameters': asdict(cfg), 'workers': cfg.workers})
        raise
