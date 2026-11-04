"""Frozen-DN signed4f risk fitting, followed by a same-size point-cost head.

No graph/model/native constructor is called here. Signed margins are geometric
proxies; directions not executed in the original game are not native outcomes.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .continuous_risk_calibration import train_statistics, _hash_tree

PROTOCOL = 'signed-clearance4f-GRU-v1'
EPOCHS, STEPS_PER_EPOCH, QUERY_BATCH = 40, 200, 4096
QUERY_COUNTS = (1024, 2048, 1024)
SEED = 71931
RISK_KEYS = ('layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_signed_cache(cache, *, check_stop=lambda:None):
    """One exact join of frozen final-theta DN and the original eight traces."""
    cache = Path(cache)
    report = json.loads((cache/'report.json').read_text('utf-8'))
    if report['status'] != 'completed_frozen_final_theta_replay' or report['costs']['real_rows'] != 61440:
        raise ValueError('Completed final-theta61440-row replay required')
    DN = np.load(report['all_DN_path'], mmap_mode='c')
    if DN.shape != (61440,1299) or DN.dtype != np.float32:
        raise ValueError('Original unquantized following-DN61440x1299 required')
    with np.load(report['label_path'],allow_pickle=False) as archive:
        labels = {k:archive[k].copy() for k in archive.files}
    fields = ('features','actions','episode_starts','game_ids','task_margins',
        'task_trusted','task_unprotected_mask','task_outcomes','d95_risk_costs',
        'd95_risk_queried_mask','d95_risk_valid_mask')
    blocks = []; sources = []
    for i,spec in enumerate(report['sources']):
        check_stop(); path = Path(spec['path']); digest = sha(path)
        sidecar = json.loads(path.with_suffix('.json').read_text('utf-8'))
        if digest != spec['sha256'] or digest != sidecar['trace_sha256'] or path.name != f'rollout-{i:03}.npz':
            raise ValueError('Original trace bytes/order differ from frozen DN provenance')
        with np.load(path,allow_pickle=False) as archive:
            block = {k:archive[k].copy() for k in fields}
        if block['actions'].shape != (256,30): raise ValueError('Actual256x30 source rows required')
        sl = slice(i*7680,(i+1)*7680)
        pairs = (('game_ids','game_id'),('episode_starts','episode_start'),
            ('d95_risk_costs','costs'),('d95_risk_queried_mask','queried'),('d95_risk_valid_mask','valid'))
        if (any(not np.array_equal(block[a].reshape(-1,*block[a].shape[2:]),labels[b][sl]) for a,b in pairs)
                or not np.array_equal(labels['source_rollout'][sl],np.full(7680,i))
                or not np.array_equal(labels['source_time'][sl],np.repeat(np.arange(256),30))
                or not np.array_equal(labels['world'][sl],np.tile(np.arange(30),256))):
            raise ValueError('DN/geometry source row,game,reset or label binding differs')
        blocks.append({k:v.reshape(-1,*v.shape[2:]) for k,v in block.items()})
        sources.append(dict(path=str(path),sha256=digest))
    if len(blocks) != 8: raise ValueError('Exactly the original eight contiguous traces required')
    data = {k:np.concatenate([b[k] for b in blocks]) for k in fields}
    f = data['features']; normal = (f[:,21]>.5)&(f[:,20]<=0)&(f[:,5]<=0)
    geometry = data['task_trusted'][:,0]&data['task_unprotected_mask']&normal
    margin = np.clip(data['task_margins'][:,0],-64.,64.).astype(np.float32)
    if not np.isfinite(margin).all(): raise ValueError('Finite clipped original signed4f targets required')
    held = data['game_ids']%5 == 0; train = geometry&~held
    if train.sum()<2 or not (geometry&held).any(): raise ValueError('Trusted train/calibration-held scenes required')
    metadata = dict(replay_report_sha256=sha(cache/'report.json'),DN_path=report['all_DN_path'],
        DN_sha256=sha(report['all_DN_path']),labels_sha256=sha(report['label_path']),sources=sources,
        geometry_rows=int(geometry.sum()),train_geometry_rows=int(train.sum()),
        held_geometry_rows=int((geometry&held).sum()),
        split='game_id%5==0 readout-fit-held only; PPO already saw these trajectories',
        normal='f21>.5 & f20<=0 & f5<=0',targets='task_margins[:,:,0,:],trusted[:,:,0],unprotected;clip[-64,64]',
        teacher_queries=0,old_DN_or_hidden_read=False,core_forwards=0,native_actions=0)
    return torch.from_numpy(DN),torch.from_numpy(margin),torch.from_numpy(train),torch.from_numpy(held),geometry,data,metadata


class SignedClearanceReadout(nn.Module):
    def __init__(self,width,*,seed=SEED):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.layers = nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,18))
    def forward(self,x): return 8.*self.layers(x)


def cosine_learning_rate(epoch_index, *, epochs=EPOCHS):
    """Original full-geometry D95 schedule, including its10% LR floor."""
    return .001*(.1+.9*.5*(1.+math.cos(math.pi*epoch_index/epochs)))


def signed_query_loss(prediction,target):
    known = (target<=0)|(target>1)
    regression = F.smooth_l1_loss(prediction/8.,target/8.,beta=.1)
    binary = F.binary_cross_entropy_with_logits(-(prediction[known]-.5),(target[known]<=0).float())
    return regression+.2*binary,regression,binary


def fit_signed_head(net,DN,target,train,mean,scale,*,seed=SEED,epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,check_stop=lambda:None,progress=lambda row:None):
    """Private risk4 Adam; pool percentages are scalar action-query counts."""
    if epochs<1 or steps_per_epoch<1: raise ValueError('Finite positive fixed fitting schedule required')
    goal = target.detach().cpu().clamp(-64.,64.)
    mask = train.detach().cpu().numpy()[:,None]
    y = goal.numpy()
    pools = [torch.from_numpy(np.flatnonzero((mask&condition).reshape(-1))) for condition in
        (y<=0,np.abs(y)<8,np.ones_like(y,bool))]
    if any(len(p)==0 for p in pools): raise ValueError('Train danger,boundary,and all-query pools required')
    device = next(net.parameters()).device; mean,scale = mean.to(device),scale.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0.)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    record = dict(protocol=PROTOCOL,optimizer_updates=0,optimizer_step_attempts=0,backward_passes=0,
        optimizer_transaction_complete=True,finite_optimizer_updates=0,scalar_training_queries=0,
        unique_scene_head_forward_rows=0,pool_sizes=[len(p) for p in pools],
        query_counts=list(QUERY_COUNTS),pool_overlap=True,history=[],
        core_forwards=0,PPO_updates=0,native_actions=0,teacher_queries=0)
    try:
        for epoch in range(epochs):
            lr = cosine_learning_rate(epoch,epochs=epochs)
            for group in optimizer.param_groups: group['lr'] = lr
            for step in range(steps_per_epoch):
                check_stop()
                ix = torch.cat([p[torch.randint(len(p),(n,),generator=generator)] for p,n in zip(pools,QUERY_COUNTS)])
                row,direction = ix//18,ix%18
                unique,inverse = torch.unique(row,return_inverse=True)
                x = (DN[unique].to(device)-mean)/scale
                prediction = net(x)[inverse.to(device),direction.to(device)]
                record['unique_scene_head_forward_rows'] += len(unique)
                t = goal[row,direction].to(device)
                loss,reg,binary = signed_query_loss(prediction,t)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite signed4f loss')
                optimizer.zero_grad(set_to_none=True); loss.backward(); record['backward_passes'] += 1
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in net.parameters()):
                    raise FloatingPointError('Missing/nonfinite signed-risk4 gradient')
                norm = nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
                record['optimizer_transaction_complete'] = False; record['optimizer_step_attempts'] += 1
                optimizer.step(); record['optimizer_updates'] += 1
                if any(not bool(torch.isfinite(p).all()) for p in net.parameters()):
                    raise FloatingPointError('Nonfinite signed-risk4 parameter')
                record['finite_optimizer_updates'] += 1; record['optimizer_transaction_complete'] = True
                record['scalar_training_queries'] += QUERY_BATCH
                if step+1 == steps_per_epoch:
                    record['history'].append(dict(epoch=epoch+1,updates=record['optimizer_updates'],
                        learning_rate=lr,loss=float(loss.detach()),Huber=float(reg.detach()),
                        BCE=float(binary.detach()),preclip_grad_norm=float(norm)))
                    progress(copy.deepcopy(record))
        return record
    except BaseException as error:
        error.signed_risk_fit_diagnostics = copy.deepcopy(record)
        error.signed_risk_work = dict(head={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
            optimizer=copy.deepcopy(optimizer.state_dict()),record=copy.deepcopy(record))
        raise


@torch.no_grad()
def predict_signed(net,DN,mean,scale,*,check_stop=lambda:None,paid=None):
    device = next(net.parameters()).device; mean,scale = mean.to(device),scale.to(device)
    result = []
    for start in range(0,len(DN),1024):
        check_stop(); x = (DN[start:start+1024].to(device)-mean)/scale
        pred = net(x)
        if not bool(torch.isfinite(pred).all()): raise FloatingPointError('Nonfinite signed-margin prediction')
        result.append(pred.cpu())
        if paid is not None: paid['head_inference_rows'] += len(x)
    return torch.cat(result)


def folded_cost_state(net,mean,scale):
    """m=8*linear -> softplus(2-2m), in original raw-DN coordinates."""
    state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    Wraw = state['layers.0.weight']/scale.cpu()
    braw = (state['layers.0.bias'].double()-Wraw.double()@mean.cpu().double()).float()
    state['layers.0.weight'] = Wraw; state['layers.0.bias'] = braw
    state['layers.2.weight'] *= -16.
    state['layers.2.bias'] = 2.-16.*state['layers.2.bias']
    return state


@torch.no_grad()
def predict_cost_state(state,DN,*,device='cpu',check_stop=lambda:None,paid=None):
    p = {k:v.to(device) for k,v in state.items()}; result = []
    for start in range(0,len(DN),1024):
        check_stop(); x = DN[start:start+1024].to(device)
        hidden = F.silu(F.linear(x,p['layers.0.weight'],p['layers.0.bias']))
        cost = F.softplus(F.linear(hidden,p['layers.2.weight'],p['layers.2.bias']))
        if not bool(torch.isfinite(cost).all()): raise FloatingPointError('Nonfinite folded point-cost prediction')
        result.append(cost.cpu())
        if paid is not None: paid['head_inference_rows'] += len(x)
    return torch.cat(result)


def _risk_optimizer_ids(payload):
    ids = []; names = []
    for signature,group in zip(payload['optimizer_parameter_signature'],payload['optimizer']['param_groups']):
        if len(signature)!=len(group['params']): raise ValueError('Dynamic Adam signature mismatch')
        for spec,identifier in zip(signature,group['params']):
            if spec['name'].startswith('risk_head.'):
                ids.append(identifier); names.append(spec['name'].removeprefix('risk_head.'))
    if set(names)!=set(RISK_KEYS) or len(ids)!=4 or len(set(ids))!=4:
        raise ValueError('Exactly the existing risk4 dynamic Adam coordinates required')
    return ids


def nonrisk_payload_identity(payload):
    risk_ids = set(_risk_optimizer_ids(payload)); digest = hashlib.sha256()
    _hash_tree({k:v for k,v in payload['model_payload'].items() if k!='risk_head'},digest)
    _hash_tree({k:v for k,v in payload['optimizer']['state'].items() if k not in risk_ids},digest)
    _hash_tree(payload['optimizer']['param_groups'],digest)
    _hash_tree(payload['optimizer_parameter_signature'],digest)
    return digest.hexdigest()


def install_cost_state(payload,state):
    """Mutate only risk4 weights/moments; retain the full original family payload."""
    before = nonrisk_payload_identity(payload)
    expected = payload['model_payload']['risk_head']
    if set(state)!=set(expected) or any(state[k].shape!=expected[k].shape or state[k].dtype!=expected[k].dtype
            or not bool(torch.isfinite(state[k]).all()) for k in state):
        raise ValueError('Same finite raw-DN risk4 tensors required')
    ids = _risk_optimizer_ids(payload)
    steps = [float(payload['optimizer']['state'].get(i,{}).get('step',0)) for i in ids]
    removed = sum(i in payload['optimizer']['state'] for i in ids)
    payload['model_payload']['risk_head'] = {k:v.detach().cpu().clone() for k,v in state.items()}
    for identifier in ids: payload['optimizer']['state'].pop(identifier,None)
    after = nonrisk_payload_identity(payload)
    if before!=after: raise RuntimeError('Non-risk model or Adam bytes changed')
    return dict(nonrisk_model_and_Adam_bitwise_preserved=True,identity_before=before,identity_after=after,
        risk_coordinates_reset=4,existing_risk_states_removed=removed,previous_risk_steps=steps,
        independent_signed_Adam_transferred=False,optimizer_groups_and_LRs_changed=False)
