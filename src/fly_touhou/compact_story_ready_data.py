"""Variable-length world histories, with real-row PPO and full-history GAE."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class IndexedSpikingStates:
    """Store only the worlds that actually have a recurrent block at this time."""
    worlds: tuple
    state: object

    def row(self,world):
        return self.state.row(self.worlds.index(int(world)))


def pack_world_records(records,states,bootstrap,metadata):
    from .spiking_engineered_control import SpikingState
    counts=[len(rows) for rows in records]
    if not counts or min(counts)<16 or any(n%16 for n in counts):
        raise ValueError('Every participating world must finish complete16-step blocks')
    length=max(counts);worlds=len(records)
    keys=set(records[0][0])
    if any(set(row)!=keys for history in records for row in history):
        raise ValueError('All real ready-driven decisions need the same recorded fields')
    data={}
    for key in keys:
        example=np.asarray(records[0][0][key])
        value=np.zeros((length,worlds,*example.shape),dtype=example.dtype)
        if key in ('outcomes','recorded_native_episode','recorded_episode_frames'):value.fill(-1)
        for world,history in enumerate(records):value[:len(history),world]=np.stack([r[key] for r in history])
        data[key]=torch.from_numpy(value)
    valid=torch.arange(length)[:,None]<torch.tensor(counts)[None]
    data['valid_mask']=valid
    # Padding is storage only. No native terminal/reset is emitted for it and
    # no padding row enters PPO, guidance, replay or observed-game statistics.
    data['terminated']|=~valid
    blocks={}
    for t in range(0,length,16):
        present=tuple(w for w,n in enumerate(counts) if t<n)
        blocks[t]=IndexedSpikingStates(present,SpikingState.pack([states[w][t] for w in present]))
    data.update(block_states=blocks,bootstrap_values=bootstrap,
        metadata=dict(metadata,collection_schema='ready-global-blocks-v1',world_decisions=counts,
            student_actions=sum(counts),padded_storage_rows=int((~valid).sum()),
            padding_is_training_data=False,brain_state_padding=False))
    return data


@torch.no_grad()
def prepare_ready_targets(data,cfg,device):
    """Keep GAE across all consecutive blocks of each world, then normalize once."""
    from .thrl_ppo import compute_gae,combine_advantages
    valid=data['valid_mask'].to(device);counts=valid.sum(0)
    if not bool((valid==(torch.arange(len(valid),device=device)[:,None]<counts[None])).all()):
        raise ValueError('Ready histories must be contiguous real prefixes')
    rewards=data['reward_vectors'].to(device);values=data['old_values'].to(device)
    bootstrap=data['bootstrap_values'].to(device);terminal=data['terminated'].to(device)
    # Different worlds have different rollout ends. These are value-bootstrap
    # boundaries only, never Game Over or an instruction to reset the game.
    boundary=torch.zeros_like(valid);world=torch.arange(valid.shape[1],device=device)
    boundary[counts-1,world]=True
    tail_values=torch.zeros_like(values);tail_values[counts-1,world]=bootstrap
    cuts=torch.zeros_like(valid)
    if cfg.gae_chunk_size:cuts[cfg.gae_chunk_size-1::cfg.gae_chunk_size]=True
    advantage,returns=compute_gae(rewards,values,bootstrap,terminal,data['episode_starts'].to(device),
        truncated=boundary,truncation_values=tail_values,gamma=cfg.gamma,gae_lambda=cfg.gae_lambda,
        trace_cut_after=cuts)
    combined=torch.zeros_like(valid,dtype=values.dtype)
    if 'objective_weights' in data:
        # Qualifying one card for speed must not activate speed on hard cards.
        weights=data['objective_weights'].to(device)
        raw=(advantage[valid]*weights[valid]).sum(-1)
        std=raw.std(unbiased=False)
        combined[valid]=(raw-raw.mean())/(std+1e-8) if std>1e-5 else torch.zeros_like(raw)
    else:
        combined[valid]=combine_advantages(advantage[valid],cfg.objective_weights)
    return advantage,returns,combined


def ready_sequence_groups(valid,length,rng):
    """Interleave real world histories without requiring equal worker progress."""
    valid=valid.cpu().numpy() if torch.is_tensor(valid) else np.asarray(valid,bool)
    queues=[]
    for world in range(valid.shape[1]):
        starts=[t for t in range(0,len(valid)-length+1,length) if valid[t:t+length,world].all()]
        queues.append(list(rng.permutation(starts)))
    order=list(rng.permutation(len(queues)));sequences=[]
    while any(queues):
        for world in order:
            if queues[world]:sequences.append((int(queues[world].pop()),int(world)))
    if len(sequences)*length!=int(valid.sum()) or len(sequences)%16:
        raise ValueError('The global budget must fill256 real PPO rows per minibatch')
    return [sequences[i:i+16] for i in range(0,len(sequences),16)]
