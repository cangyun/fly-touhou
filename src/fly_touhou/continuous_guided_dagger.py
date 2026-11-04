"""Direct rule/geometry teaching through the complete continuous connectome.

Independent experimental schema. Old loaders cannot silently accept the new
port calibration or DN-normalized risk head. No game or learner starts here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .continuous_connectome import ContinuousConnectomeState
from .continuous_connectome_training import reset_activity
from .continuous_full_pair_geometry_model import build_full_pair_model
from .continuous_signed_clearance import signed_clearance4f_terms
from .thrl_durable_records import write_json

SCHEMA = 'continuous-calibrated-fullbrain-direct-teacher-DAgger-v1'
PARAMETER_SCHEMA = SCHEMA + '-parameters'
BURN_IN = 16
TRAIN_LENGTH = 16
SEQUENCE_LENGTH = BURN_IN + TRAIN_LENGTH


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class CalibratedPortDrive(nn.Module):
    """Three separately normalized branches, learned mapping, positive gains.

    No raw object/teacher shortcut to DN or heads exists. Fixed train-only
    calibration is followed by branch LN, so encoder learning can change the
    representation without relying on obsolete D95 activation scales.
    """
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.branch_norm = nn.LayerNorm(128, elementwise_affine=False)
        self.branch_log_gain = nn.Parameter(torch.zeros(3, device=linear.weight.device))
        self.port_log_gain = nn.Parameter(torch.zeros(linear.out_features, device=linear.weight.device))
        self.register_buffer('center', torch.zeros(384, device=linear.weight.device))
        self.register_buffer('scale', torch.ones(384, device=linear.weight.device))
        self.register_buffer('port_scale', torch.ones(linear.out_features, device=linear.weight.device))

    @property
    def weight(self):
        return self.linear.weight

    def normalized(self, x):
        z = ((x - self.center) / self.scale).reshape(*x.shape[:-1], 3, 128)
        z = self.branch_norm(z) * self.branch_log_gain.clamp(-1.3862944, 1.3862944).exp()[..., None]
        return z.flatten(-2)

    def forward(self, x):
        return self.linear(self.normalized(x)) * self.port_scale * self.port_log_gain.clamp(-1.3862944, 1.3862944).exp()

    @torch.no_grad()
    def calibrate(self, x, target_rms=.5):
        if x.ndim != 2 or x.shape[1] != 384 or len(x) < 32:
            raise ValueError('Training-mixture encoded observations required')
        self.center.copy_(x.mean(0))
        std = x.std(0, unbiased=False)
        self.scale.copy_(std.clamp_min(max(.01, float(std.median()) * .05)))
        self.linear.bias.zero_()
        z = self.normalized(x)
        contributions = [F.linear(z[:, i*128:(i+1)*128], self.weight[:, i*128:(i+1)*128]) for i in range(3)]
        rms = torch.stack([v.square().mean().sqrt() for v in contributions])
        # Balance branch stimulation before per-port calibration. This changes
        # the mapping deliberately; actor and risk are then relearned together.
        self.branch_log_gain.copy_((rms.mean() / rms.clamp_min(1e-6)).log().clamp(-1.3862944, 1.3862944))
        pre = self.linear(self.normalized(x))
        self.port_scale.copy_(target_rms / pre.square().mean(0).sqrt().clamp_min(1e-3))
        after = self(x)
        return dict(training_rows=len(x), target_drive_RMS=target_rms,
            drive_RMS=float(after.square().mean().sqrt()), drive_abs_gt3=float((after.abs()>3).float().mean()),
            branch_RMS_before_balancing=rms.cpu().tolist(), biological_constant=False,
            branch_normalization='train-mixture center/scale then per-observation LN within each128 branch',
            per_port_calibration=True, gains_trainable=True, all_8053_ports_retained=True)


class LayerNormalizedRisk(nn.Module):
    def __init__(self, width=1299):
        super().__init__()
        self.beta = 8.
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,128), nn.SiLU(),
            nn.Linear(128,128), nn.SiLU(), nn.Linear(128,128), nn.SiLU(), nn.Linear(128,18))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, DN):
        return self.scores_and_costs(DN)[1]

    def scores_and_costs(self,DN):
        scores=self.layers(DN)
        return scores,F.softplus(scores)


def install_guided_interfaces(model, *, reset_heads=True):
    model.core.input_encoder = CalibratedPortDrive(model.core.input_encoder)
    model.risk_head = LayerNormalizedRisk().to(model.device)
    if reset_heads:
        nn.init.normal_(model.actor[-1].weight, std=.001)
        nn.init.zeros_(model.actor[-1].bias)
    model.requires_grad_(True)
    return model


def parameter_payload(model, base_source, base_sha, calibration, counters):
    return dict(schema=PARAMETER_SCHEMA, base_source=str(base_source), base_sha256=base_sha,
        state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        calibration=calibration, counters=dict(counters), graph=model.core.graph_provenance,
        core_K=model.core.K, risk_input='following DN only;LayerNorm',
        teacher_at_inference=False, D95_runtime_queries=0, actor_and_risk_are_artificial_readouts=True)


def build_guided_model(payload, connectome, *, device='cuda:0'):
    if payload.get('schema') != PARAMETER_SCHEMA:
        raise ValueError('Independent calibrated DAgger model schema required')
    if sha(payload['base_source']) != payload['base_sha256']:
        raise ValueError('Preserved base checkpoint changed')
    with Path(payload['base_source']).open('rb') as stream:
        base = torch.load(stream, map_location='cpu', weights_only=True)
    model = build_full_pair_model(base, connectome, device=device)
    install_guided_interfaces(model, reset_heads=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model


def direction_labels(data):
    """All18 directions at4/12/24; actual chosen hit remains a separate fact."""
    from .thrl_dodge_supervision import prepare_dense_targets
    from .thrl_inputs import SEPARATED_LASER_INPUT
    n = len(data['features'])
    margins, trusted, evidence = prepare_dense_targets(data['features'][:,None], data['objects'][:,None],
        data['maps'][:,None], data['actions'][:,None], data['outcomes'][:,None],
        input_schema=SEPARATED_LASER_INPUT, max_labels=n, horizon_specific=True)
    data['margins'] = margins[:,0].numpy()
    data['trusted'] = trusted[:,0].numpy()
    from .thrl_movement_teacher import normal_unprotected
    data['unprotected_mask'] = normal_unprotected(data['features'])
    # No-hazard scenes have no geometric ranking. Do not invent counterfactual
    # safe labels where the original producer abstained because objects/maps
    # were missing or uncertain.
    return evidence


class SequenceReplay:
    """Persistent, actual raw observations; never cached DN as learning input."""
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root/'index.json'
        self.records = json.loads(self.index_path.read_text(encoding='utf8'))['records'] if self.index_path.exists() else []

    def add(self, data, *, origin, round_index, hard=False, metadata=None):
        if len(data['features']) != SEQUENCE_LENGTH:
            raise ValueError('16 burn-in +16 actual training observations required')
        number = len(self.records); path = self.root/f'sequence-{number:06}.npz'
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        row = dict(id=number,path=str(path),origin=origin,round=round_index,hard=bool(hard),
            sha256=sha(path), metadata=metadata or {})
        self.records.append(row)
        return row

    def commit(self):
        write_json(self.index_path, dict(schema=SCHEMA+'-raw-replay', records=self.records,
            raw_archive_used_for_training=True, cached_DN_used_for_training=False,
            native_rows_per_sequence=SEQUENCE_LENGTH, burn_in=BURN_IN, train_length=TRAIN_LENGTH))

    def sample(self, rng, *, current_round, warmup=False, batch=16):
        expert = [r for r in self.records if r['origin']=='expert']
        recent = [r for r in self.records if r['origin']=='student' and r['round']==current_round]
        hard = [r for r in self.records if r['origin']=='student' and r['hard']]
        pools = dict(expert=expert, recent=recent, historical_hard=hard)
        chosen=[]; counts={k:0 for k in pools}
        for _ in range(batch):
            key='expert' if warmup else rng.choice(['expert','recent','historical_hard'],p=[.2,.4,.4])
            if not pools[key]: key='recent' if recent else 'expert'
            record=pools[key][int(rng.integers(len(pools[key])))];chosen.append(record);counts[key]+=1
        arrays=[]
        for record in chosen:
            with np.load(record['path'], allow_pickle=False) as archive:
                arrays.append({k:archive[k] for k in archive.files})
        return {k:np.stack([a[k] for a in arrays],axis=1) for k in arrays[0]},counts


def guided_loss(raw, composed, predicted, features, task, valid, *, native_hit_actor=False,actual_policy_priority=False,
        raw_actor_coefficient=None):
    """Raw actor CE + safe-set supervision + physical regression + ordering.

    Targets are original rule distributions filtered by trustworthy immediate
    geometry. Many safe directions remain acceptable. Longer predictions only
    rank within the immediate feasible set. No teacher action is executed.
    """
    margin=task['margins']; trust=task['trusted'] & valid[...,None]
    normal=task['unprotected_mask'] & valid
    safe=margin[...,0,:]>1.; some=safe.any(-1)
    eligible=trust[...,0]&normal
    action=task['actions'].clamp(0,17)
    hit=normal&(task['outcomes']==1)
    observed_bad=F.one_hot(action,18).bool()&hit[...,None]
    p=task['rule_probabilities'][...,:18].clamp_min(0.)
    if native_hit_actor:
        # A measured hit falsifies the executed action, even when geometry
        # abstains. It says nothing about the unchosen actions' native safety.
        p=p.masked_fill(observed_bad,0.)
        safe=safe&~observed_bad;some=safe.any(-1)
    mass=p.sum(-1,keepdim=True)
    confidence=(task['rule_weights']>0)&(mass[...,0]>0)&valid
    p=p/mass.clamp_min(1e-9)
    future=(torch.stack([torch.sigmoid((margin[...,1,:]-1.)/4.),torch.sigmoid((margin[...,2,:]-1.)/6.)],-2)
        *trust[...,1:,None]).sum(-2)/trust[...,1:].sum(-1,keepdim=True).clamp_min(1)
    preference=(.25+future)*safe.to(raw.dtype)
    filtered=p*preference
    filtered=filtered/filtered.sum(-1,keepdim=True).clamp_min(1e-9)
    fallback=preference/preference.sum(-1,keepdim=True).clamp_min(1e-9)
    q=torch.where((filtered.sum(-1)>0)[...,None], filtered, fallback)
    target=torch.where((eligible&some)[...,None],q,p).detach()
    teacher_rows=valid & ((eligible&some)|confidence)
    # Never reinforce known immediately-dangerous teacher mass.
    teacher_rows &= ~(eligible & ~some)
    weights=teacher_rows.to(raw.dtype)*(1.+2.*(eligible&some&~safe.all(-1)))
    def ce(logits):
        per=-(target*logits[...,:18].log_softmax(-1)).sum(-1)
        return (per*weights).sum()/weights.sum().clamp_min(1.)
    actor_ce=ce(raw); composed_ce=ce(composed)
    mixed=eligible&some&~safe.all(-1)
    def set_ce(logits):
        x=logits[...,:18];keep=safe|(~some[...,None])
        row=x.logsumexp(-1)-x.masked_fill(~keep,-torch.inf).logsumexp(-1)
        return torch.where(mixed,row,0.).sum()/mixed.sum().clamp_min(1.)
    raw_weight,composed_weight=(.25,1.) if actual_policy_priority else (1.,.25)
    if raw_actor_coefficient is not None:
        import math
        if not math.isfinite(raw_actor_coefficient) or raw_actor_coefficient<0:
            raise ValueError('Optional raw-actor coefficient must be finite and nonnegative')
        raw_weight=float(raw_actor_coefficient)
    safety=raw_weight*set_ce(raw)+composed_weight*set_ce(composed)
    risk=signed_clearance4f_terms(predicted,task,features,valid)
    # Only a safe-vs-danger distinction is explicitly ordered. Equal-safe
    # options are not forced into arbitrary unique action labels.
    danger=margin[...,0,:]<=0.
    pairs=safe[..., :,None]&danger[...,None,:]&eligible[...,None,None]
    gaps=predicted[..., :,None]-predicted[...,None,:]
    rank=torch.where(pairs,F.softplus(1.-gaps),0.).sum()/pairs.sum().clamp_min(1.)
    chosen=predicted.gather(-1,action[...,None]).squeeze(-1)
    contact=torch.where(hit,F.softplus(chosen-.5),0.).sum()/hit.sum().clamp_min(1.)
    actor_hit=raw.sum()*0.
    if native_hit_actor:
        def negative_hit(logits):
            x=logits[...,:18].clamp(-80.,80.)
            row=x.logsumexp(-1)-x.masked_fill(F.one_hot(action,18).bool(),-torch.inf).logsumexp(-1)
            return torch.where(hit,row,0.).sum()/hit.sum().clamp_min(1.)
        actor_hit=negative_hit(raw)
        if actual_policy_priority:actor_hit=raw_weight*actor_hit+negative_hit(composed)
        elif raw_actor_coefficient is not None:actor_hit=raw_weight*actor_hit+composed_weight*negative_hit(composed)
    total=raw_weight*actor_ce+composed_weight*composed_ce+safety+risk['loss']+.2*rank+.5*contact+.1*actor_hit
    with torch.no_grad():
        def unsafe(logits):
            return ((margin[...,0,:].gather(-1,logits[...,:18].argmax(-1)[...,None]).squeeze(-1)<=0)&mixed).sum()
        minimum=predicted.argmax(-1)
        min_unsafe=((margin[...,0,:].gather(-1,minimum[...,None]).squeeze(-1)<=0)&mixed).sum()
    return total,dict(actor_CE=actor_ce,composed_CE=composed_ce,safe_set=safety,
        physical_risk=risk['loss'],risk_ordering=rank,actual_contact=contact,actor_native_hit=actor_hit,
        teacher_rows=teacher_rows.sum(),trusted4_rows=eligible.sum(),mixed_rows=mixed.sum(),
        raw_unsafe=unsafe(raw),composed_unsafe=unsafe(composed),risk_minimum_unsafe=min_unsafe,
        contact_rows=hit.sum(),contact_false_safe=((chosen>1)&hit).sum())


def learning_tail(tensors,learning_offsets=None):
    """A real game start can learn rows0:16; arbitrary windows still burn16.

    Selecting the opening half on some visits and the later half on others
    retains both halves without backpropagating through invented histories.
    """
    if learning_offsets is None:return {k:v[BURN_IN:] for k,v in tensors.items()}
    device=tensors['features'].device;b=tensors['features'].shape[1]
    offsets=torch.as_tensor(learning_offsets,device=device,dtype=torch.long)
    if offsets.shape!=(b,) or bool(((offsets!=0)&(offsets!=BURN_IN)).any()):
        raise ValueError('Per-sequence learning offset must be0 or16')
    if bool(((offsets==0)&~tensors['episode_starts'][0].bool()).any()):
        raise ValueError('Zero-history learning is only valid at a genuine native reset')
    times=torch.arange(TRAIN_LENGTH,device=device)[:,None]+offsets[None]
    worlds=torch.arange(b,device=device)[None].expand(TRAIN_LENGTH,b)
    return {k:v[times,worlds] for k,v in tensors.items()}


def supervised_sequence(model, data, *, stable_body=False, native_hit_actor=False,actual_policy_priority=False,
        attention_coefficient=0.,raw_actor_coefficient=None,learning_offsets=None,check_stop=lambda:None):
    """Current encoder/core burn-in, then T16 gradients; no archived activity."""
    device=model.device
    tensors={k:torch.as_tensor(v,device=device) for k,v in data.items()}
    before=model.initial_state(len(tensors['features'][0]))
    # Burn-in is an approximation from a 16-observation raw prefix, not a claim
    # of the exact state from the full native opening. Collection/evaluation
    # continue to carry the genuine full state across all native decisions.
    with torch.no_grad():
        z=model.encode_raw(tensors['features'][:BURN_IN],tensors['maps'][:BURN_IN],objects=tensors['objects'][:BURN_IN])
        for t in range(BURN_IN):
            check_stop(); before=reset_activity(before,tensors['episode_starts'][t])
            _,before=model.core(z[t],before)
    tail=learning_tail(tensors,learning_offsets)
    attention_logits=None
    if attention_coefficient>0:
        if stable_body:raise ValueError('Attention teaching requires joint sensory/body learning')
        encoded,attention_logits=model.encode_raw_with_geometry_attention(
            tail['features'],tail['maps'],objects=tail['objects'])
    elif stable_body:
        with torch.no_grad():
            encoded=model.encode_raw(tail['features'],tail['maps'],objects=tail['objects'])
    else: encoded=model.encode_raw(tail['features'],tail['maps'],objects=tail['objects'])
    raw_rows=[];combined_rows=[];margins=[];state=before.detach();DN_rows=[];VP=[]
    for t in range(TRAIN_LENGTH):
        check_stop();state=reset_activity(state,tail['episode_starts'][t])
        def core(z,h):
            DN,following=model.core(z,ContinuousConnectomeState(h));return DN,following.activity
        if stable_body:
            with torch.no_grad():DN,h=core(encoded[t],state.activity)
        else:DN,h=checkpoint(core,encoded[t],state.activity,use_reentrant=False)
        state=ContinuousConnectomeState(h)
        raw=model.actor(DN);score=model.risk_head.layers(DN)
        cost=F.softplus(score)
        # The physical loss trains the risk branch. Actor/composed guidance
        # cannot erase calibrated risk costs just to make imitation easier.
        combined=model.policy_logits(model.compose(raw,cost.detach(),tail['features'][t],check_finite=False),tail['features'][t])
        raw_rows.append(raw);combined_rows.append(combined);margins.append(1.-.5*score)
        DN_rows.append(DN.detach());VP.append(h[:,model.core.input_indices].detach())
    raw=torch.stack(raw_rows);combined=torch.stack(combined_rows);pred=torch.stack(margins)
    valid=torch.ones(raw.shape[:2],dtype=torch.bool,device=device)
    loss,metrics=guided_loss(raw,combined,pred,tail['features'],tail,valid,
        native_hit_actor=native_hit_actor,actual_policy_priority=actual_policy_priority,
        raw_actor_coefficient=raw_actor_coefficient)
    if attention_logits is not None:
        from .geometry_attention_teaching import attention_teaching_loss
        auxiliary,detail=attention_teaching_loss(attention_logits,tail['attention_margins'],tail['attention_valid_objects'])
        loss=loss+attention_coefficient*auxiliary
        metrics.update(attention_KL=auxiliary.detach(),attention_queries=detail['eligible_action_queries'],
            attention_coefficient=loss.new_tensor(attention_coefficient),
            attention_normalized_entropy=detail['normalized_entropy'],
            attention_critical_object_top1=detail['critical_object_top1'],
            attention_multi_object_queries=detail['multi_object_queries'])
    with torch.no_grad():
        vp=torch.stack(VP);drive=model.core.input_encoder(encoded.detach())
        metrics.update(VP_abs_gt095=(vp.abs()>.95).float().mean(),VP_abs_gt099=(vp.abs()>.99).float().mean(),
            port_drive_RMS=drive.square().mean().sqrt(),DN_RMS=torch.stack(DN_rows).square().mean().sqrt())
        metrics['native_opening_learning_sequences']=raw.new_tensor(0 if learning_offsets is None else int((np.asarray(learning_offsets)==0).sum()))
    return loss,metrics
