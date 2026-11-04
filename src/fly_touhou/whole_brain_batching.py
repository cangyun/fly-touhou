"""Masked complete-episode batches for the unchanged full recurrent brain.

Rows are independent complete episodes. A row that reaches its end is frozen,
not recycled into another episode. Its padding contributes no loss, metric,
activity, state advance, delayed spike or neural noise consumption. This module
does not reduce the graph and does not construct an alternative action path.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import random
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .experiment import NeuralActivityAccumulator
from .whole_brain import WholeBrainState


def episode_batches(episodes: Sequence[Any], batch_size: int, *, seed: int | None = None,
                    max_length_ratio: float = 2.0) -> list[list[Any]]:
    """Length-neighbor groups; 450-step episodes never pad to a 4500-step one."""
    if (isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size < 1
            or not math.isfinite(max_length_ratio) or max_length_ratio < 1):
        raise ValueError("Invalid episode batch size/length ratio")
    ordered=list(episodes)
    if not ordered or any(len(episode)<1 for episode in ordered):
        raise ValueError("Episode batches require nonempty complete episodes")
    if len({episode.episode_id for episode in ordered})!=len(ordered):
        raise ValueError("Duplicate complete episode identity in batching")
    rng=random.Random(seed)
    if seed is not None:
        rng.shuffle(ordered)
    ordered.sort(key=len)  # Stable: seeded equal-length order is preserved.
    batches=[]
    for episode in ordered:
        if (not batches or len(batches[-1])>=batch_size
                or len(episode)>len(batches[-1][0])*max_length_ratio):
            batches.append([])
        batches[-1].append(episode)
    if seed is not None:
        rng.shuffle(batches)
    return batches


def stack_states(states: Sequence[WholeBrainState]) -> WholeBrainState:
    """Keep each serial state's explicit key; initial_state(B) changes row keys."""
    if not states or any(state.membrane.shape[0]!=1 for state in states):
        raise ValueError("Stack independent one-episode states")
    return WholeBrainState(**{field.name:(tuple(torch.cat([state.delay_queue[i] for state in states],dim=0)
        for i in range(len(states[0].delay_queue))) if field.name=="delay_queue" else
        torch.cat([getattr(state,field.name) for state in states],dim=0)) for field in fields(WholeBrainState)})


def select_state_rows(state: WholeBrainState, indices: torch.Tensor) -> WholeBrainState:
    return WholeBrainState(**{field.name:(tuple(value.index_select(0,indices) for value in state.delay_queue)
        if field.name=="delay_queue" else getattr(state,field.name).index_select(0,indices))
        for field in fields(WholeBrainState)})


def _retain_padding(before: WholeBrainState, after: WholeBrainState, valid: torch.Tensor) -> WholeBrainState:
    def choose(old,new):
        mask=valid.reshape((-1,)+(1,)*(old.ndim-1))
        return torch.where(mask,new,old)
    return WholeBrainState(**{field.name:(tuple(choose(old,new) for old,new in zip(before.delay_queue,after.delay_queue))
        if field.name=="delay_queue" else choose(getattr(before,field.name),getattr(after,field.name)))
        for field in fields(WholeBrainState)})


@dataclass
class EpisodeBatch:
    episode_ids: list[str]
    lengths: tuple[int,...]
    observations: torch.Tensor  # [time,episode,observation]
    probabilities: torch.Tensor
    actions: torch.Tensor
    valid: torch.Tensor
    difficulties: tuple[int,...]

    @property
    def effective_decisions(self):
        return sum(self.lengths)

    @property
    def padded_decisions(self):
        return max(self.lengths)*len(self.lengths)-self.effective_decisions


def pack_episode_batch(episodes: Sequence[Any], device: str | torch.device) -> EpisodeBatch:
    if not episodes or any(len(episode)<1 for episode in episodes):
        raise ValueError("Cannot pad an empty episode batch")
    width=episodes[0].observations.shape[1]
    actions=episodes[0].probabilities.shape[1]
    lengths=tuple(len(episode) for episode in episodes)
    n,t=len(episodes),max(lengths)
    observations=torch.zeros((t,n,width),dtype=torch.float32,device=device)
    probabilities=torch.zeros((t,n,actions),dtype=torch.float32,device=device)
    recorded=torch.zeros((t,n),dtype=torch.int64,device=device)
    valid=torch.zeros((t,n),dtype=torch.bool,device=device)
    for row,episode in enumerate(episodes):
        length=len(episode)
        if (episode.observations.shape!=(length,width) or episode.probabilities.shape!=(length,actions)
                or episode.actions.shape!=(length,) or not np.isfinite(episode.observations).all()
                or not np.isfinite(episode.probabilities).all()):
            raise ValueError("Complete episode dimensions or values differ in a batch")
        observations[:length,row]=torch.as_tensor(episode.observations,device=device)
        probabilities[:length,row]=torch.as_tensor(episode.probabilities,device=device)
        recorded[:length,row]=torch.as_tensor(episode.actions,device=device)
        valid[:length,row]=True
    return EpisodeBatch([episode.episode_id for episode in episodes],lengths,observations,probabilities,
        recorded,valid,tuple(int(episode.native_context["difficulty"]) for episode in episodes))


@torch.no_grad()
def prime_episode_batch(model: Any, batch: EpisodeBatch, warmup_ms: float) -> WholeBrainState:
    if not math.isfinite(warmup_ms) or warmup_ms<0:
        raise ValueError("warmup_ms must be nonnegative and finite")
    # Each row equals the corresponding serial initial_state(1), including its
    # configured RNG seed. Independent storage does not imply a new seed rule.
    state=stack_states([model.initial_state(1) for _ in batch.lengths])
    for _ in range(int(math.ceil(warmup_ms/model.neural_ms_per_decision))):
        _,state=model(batch.observations[0],state)
    return state.detach()


def masked_batch_step(model: Any, observation: torch.Tensor, state: WholeBrainState,
                      valid: torch.Tensor) -> tuple[torch.Tensor,WholeBrainState]:
    if valid.ndim!=1 or observation.ndim!=2 or observation.shape[0]!=len(valid):
        raise ValueError("Invalid batched observation/valid mask")
    if bool(valid.all()):
        return model(observation,state)
    if not bool(valid.any()):
        return observation.new_zeros((len(valid),model.n_actions)),state
    safe=torch.where(valid[:,None],observation,torch.zeros_like(observation))
    logits,after=model(safe,state)
    return torch.where(valid[:,None],logits,torch.zeros_like(logits)),_retain_padding(state,after,valid)


def masked_distillation_loss(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
                              hard_weight: float, bomb_target_weight: float = 1.0) -> torch.Tensor:
    if (not 0<=hard_weight<=1 or logits.shape!=targets.shape or logits.shape[:-1]!=valid.shape
            or not bool(valid.any())):
        raise ValueError("Invalid masked distillation shapes/weight or empty effective chunk")
    # Index before log-softmax: padding can contain arbitrary values without
    # contributing NaNs, normalizers or gradients to the actual objective.
    from .whole_brain_learning import mixed_distillation_loss
    return mixed_distillation_loss(logits[valid],targets[valid],hard_weight,bomb_target_weight)


def _sync(model):
    if model.device.type=="cuda":
        torch.cuda.synchronize(model.device)


def _group_record(batch: EpisodeBatch) -> dict:
    return {"episode_ids":batch.episode_ids,"lengths":list(batch.lengths),
        "effective_decisions":batch.effective_decisions,"padded_decisions":batch.padded_decisions,
        "padding_fraction":batch.padded_decisions/(batch.effective_decisions+batch.padded_decisions)}


def evaluate_episode_batches(model: Any, episodes: Sequence[Any], *, batch_size: int, warmup_ms: float,
                             metrics_factory: Callable[[],Any], difficulty_names: dict[int,str],
                             logger=None, role="evaluation") -> tuple[dict,dict]:
    model.eval()
    metrics,by_difficulty=metrics_factory(),{}
    activity=NeuralActivityAccumulator(model.output_indices.cpu().tolist(),neurons=model.n_neurons)
    groups=episode_batches(episodes,batch_size)
    started=time.monotonic()
    summaries=[]
    with torch.no_grad():
        for group in groups:
            batch=pack_episode_batch(group,model.device)
            state=prime_episode_batch(model,batch,warmup_ms)
            summaries.append(_group_record(batch))
            difficulties=torch.tensor(batch.difficulties,device=model.device)
            for step in range(max(batch.lengths)):
                valid=batch.valid[step]
                logits,state=masked_batch_step(model,batch.observations[step],state,valid)
                metrics.add(logits[valid],batch.probabilities[step,valid],batch.actions[step,valid])
                for difficulty in set(batch.difficulties):
                    selected=valid & (difficulties==difficulty)
                    if bool(selected.any()):
                        name=difficulty_names[difficulty]
                        by_difficulty.setdefault(name,metrics_factory()).add(
                            logits[selected],batch.probabilities[step,selected],batch.actions[step,selected])
                activity.add(state.window_spike_counts[valid],model.neural_ms_per_decision)
    _sync(model)
    elapsed=time.monotonic()-started
    if logger is not None:
        logger.event("episode_batch_evaluation",{"role":role,"episode_batch_size":batch_size,"groups":summaries,
            "effective_decisions":metrics.count,"elapsed_seconds":elapsed,
            "effective_decisions_per_second":metrics.count/max(elapsed,1e-12),
            "numerics":"row RNG keys/counters preserve serial semantics; batched floating point must be measured separately"})
    return {**metrics.summary(),"by_difficulty":{name:value.summary() for name,value in by_difficulty.items()}},activity.summary()


def train_episode_batches(model: Any, optimizer: Any, episodes: Sequence[Any], *, batch_size: int,
                          chunk_decisions: int, warmup_ms: float, hard_weight: float, gradient_clip: float,
                          metrics_factory: Callable[[],Any], clip_gradients: Callable,
                          seed: int, epoch: int, logger=None, progress_every_chunks: int=25,
                          bomb_target_weight: float=1.0) -> dict:
    if chunk_decisions<1 or progress_every_chunks<1:
        raise ValueError("Chunk/progress sizes must be positive")
    model.train()
    metrics=metrics_factory()
    activity=NeuralActivityAccumulator(model.output_indices.cpu().tolist(),neurons=model.n_neurons)
    groups=episode_batches(episodes,batch_size,seed=seed)
    updated={name:0 for name,_ in model.named_parameters()}
    chunks,gradient_norms,summaries=0,[],[]
    started=time.monotonic()
    for group in groups:
        batch=pack_episode_batch(group,model.device)
        state=prime_episode_batch(model,batch,warmup_ms)
        record=_group_record(batch)
        summaries.append(record)
        if logger is not None:
            logger.event("episode_batch_training_group",{"epoch":epoch,"group":len(summaries),**record,
                "initial_rng_keys":state.rng_keys.cpu().tolist(),"initial_noise_counters":state.noise_counter.cpu().tolist()})
        for first in range(0,max(batch.lengths),chunk_decisions):
            optimizer.zero_grad(set_to_none=True)
            outputs=[]
            end=min(first+chunk_decisions,max(batch.lengths))
            for step in range(first,end):
                valid=batch.valid[step]
                logits,state=masked_batch_step(model,batch.observations[step],state,valid)
                outputs.append(logits)
                metrics.add(logits[valid],batch.probabilities[step,valid],batch.actions[step,valid])
                activity.add(state.window_spike_counts[valid],model.neural_ms_per_decision)
            loss=masked_distillation_loss(torch.stack(outputs),batch.probabilities[first:end],
                                         batch.valid[first:end],hard_weight,bomb_target_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite masked whole-brain distillation loss")
            loss.backward()
            norm,per_parameter=clip_gradients(model,gradient_clip)
            gradient_norms.append(norm)
            previous={name:value.detach().clone() for name,value in model.named_parameters()}
            optimizer.step()
            for name,value in model.named_parameters():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Non-finite updated batched interface {name}")
                updated[name]+=int(bool(torch.any(value.detach()!=previous[name])))
            state=state.detach()  # Only explicit truncated-BPTT chunk boundaries.
            chunks+=1
            if logger is not None and chunks%progress_every_chunks==0:
                logger.event("training_chunk_progress",{"epoch":epoch,"chunks":chunks,"episode_ids":batch.episode_ids,
                    "last_episode_step":step,"loss":float(loss.detach()),"gradient_norm_fp64":norm,
                    "parameter_gradient_norms":per_parameter,"effective_chunk_decisions":int(batch.valid[first:end].sum()),
                    "effective_decisions_so_far":metrics.count})
        if logger is not None:
            logger.event("episode_batch_training_group_finished",{"epoch":epoch,"group":len(summaries),
                "episode_ids":batch.episode_ids,"final_rng_keys":state.rng_keys.cpu().tolist(),
                "final_noise_counters":state.noise_counter.cpu().tolist()})
    _sync(model)
    elapsed=time.monotonic()-started
    stats={"episode_batch_size":batch_size,"groups":summaries,"effective_decisions":metrics.count,
        "padded_decisions":sum(group["padded_decisions"] for group in summaries),"elapsed_seconds":elapsed,
        "effective_decisions_per_second":metrics.count/max(elapsed,1e-12),
        "max_length_ratio":2.,"rng_policy":"each row retains its configured serial initial_state(1) key",
        "optimizer_semantics":"one mean valid-decision loss per batched temporal chunk; differs from serial update order"}
    if logger is not None:
        logger.event("training_epoch_updates",{"epoch":epoch,"chunks":chunks,
            "changed_chunks_per_parameter":updated,"activity":activity.summary(),"batching":stats})
    return {"online":metrics.summary(),"gradient_norms":gradient_norms,"updated":updated,"batching":stats}
