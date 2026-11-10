"""Successful NBNM histories for supervised retention, never PPO replay."""
from pathlib import Path
import json
import numpy as np

SCHEMA='nbnm-successful-history-rehearsal-v1'


def export_successes(rollout, episodes, output, *, source, split='train'):
    """Export only complete native-practice episodes with known initial state.

    Partial episodes at a segment boundary cannot reconstruct initial memory.
    Story prefixes need their explicit burn-in history and are not guessed.
    """
    from .thrl_training_mode import seed_split
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[];valid=rollout['valid_mask'].numpy()
    for row in episodes:
        if (not row.get('nbnm_success') or row.get('task_misses')!=0 or row.get('task_Bombs')!=0
            or row.get('nbnm_entry_mode')!='native_spell_practice' or seed_split(row['native_seed'])!=split):continue
        w=int(row['world']);episode=int(row['native_episode'])
        ids=np.flatnonzero(valid[:,w] & (rollout['recorded_native_episode'][:,w].numpy()==episode))
        if (len(ids)<16 or np.any(np.diff(ids)!=1) or not rollout['episode_starts'][ids[0],w]
            or not rollout['terminated'][ids[-1],w] or rollout['reward_vectors'][ids[-1],w,0]!=1):continue
        actions=rollout['actions'][ids,w].numpy()
        if (actions>=18).any() or rollout['recorded_observed_contacts'][ids,w].any():continue
        name=f'world{w}-episode{episode}.npz'
        np.savez_compressed(output/name,features=rollout['features'][ids,w].numpy(),actions=actions,
            starts=rollout['episode_starts'][ids,w].numpy())
        records.append(dict(file=name,task=row['task'],native_seed=row['native_seed'],split=split,
            decisions=len(ids),task_native_frames=row['task_native_frames'],nbnm_success=True,
            source_kind='successful_student_behavior',initial_memory='zero at native card entry'))
    manifest=dict(schema=SCHEMA,source=str(source),histories=records,PPO_rows=0,
        target='executed movements from successful NBNM episodes; not all-direction safety labels',
        native_risk_labels=False,source_policy_changing=True)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


class SuccessRehearsal:
    def __init__(self, manifest, *, sequence_length=16):
        self.path=Path(manifest);self.manifest=json.loads(self.path.read_text(encoding='utf-8'))
        if self.manifest.get('schema')!=SCHEMA:raise ValueError('NBNM history manifest required')
        self.rows=[r for r in self.manifest['histories'] if r['split']=='train' and r['nbnm_success']]
        if not self.rows:raise ValueError('No successful training-split NBNM histories available')
        self.length=int(sequence_length)
        if self.length!=16:raise ValueError('Use the existing16-step neural sequence')
        self.groups={}
        for i,r in enumerate(self.rows):self.groups.setdefault(json.dumps(r['task'],sort_keys=True),[]).append(i)

    def loss(self, model, *, index):
        import torch
        from .nbnm_prefix import burn_in
        from .compact_story_model import sequence
        # Equal task selection prevents the easy card filling the archive.
        rng=np.random.default_rng(index)
        groups=list(self.groups.values());pool=groups[index%len(groups)]
        row=self.rows[pool[int(rng.integers(len(pool)))]]
        with np.load(self.path.parent/row['file'],allow_pickle=False) as z:
            f=z['features'].copy();actions=z['actions'].copy()
        first=int(rng.integers(len(actions)-self.length+1));end=first+self.length
        # Recompute CURRENT-weight memory from ALL actual preceding actions.
        # No old hidden state, no zero state at an arbitrary middle window.
        state=burn_in(model,dict(policy_burn_in=dict(features=f[:first],actions=actions[:first])),model.initial_state(1))
        features=torch.as_tensor(f[first:end],device=model.device)[:,None]
        chosen=torch.as_tensor(actions[first:end],device=model.device)[:,None]
        starts=torch.zeros((self.length,1),device=model.device,dtype=torch.bool)
        if first==0:starts[0]=True
        out=sequence(model,features,starts,state,chosen,checkpointed=True)
        loss=torch.nn.functional.cross_entropy(out['logits'][...,:18].reshape(-1,18),chosen.reshape(-1))
        return loss,dict(rehearsal_movement_loss=float(loss.detach()),rehearsal_rows=self.length,
            rehearsal_PPO_rows=0,rehearsal_task=row['task'],rehearsal_burn_in=first,
            rehearsal_memory='current-weight complete committed-action history')

    def summary(self):
        return dict(manifest=str(self.path),histories=len(self.rows),tasks=len(self.groups),
            PPO_rows=0,brain_topology_changed=False,history='entire prefix, one candidate per action')
