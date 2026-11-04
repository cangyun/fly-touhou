"""CPU-only analysis of an already completed paired input-drive experiment."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='2'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,hashlib,io,json,math,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json

SCHEMA='fly-paired-input-drive-CPU-analysis-v1'
SOURCE_SCHEMA='fly-paired-hard-input-same-state-four-observation-probe-v1'
MODES=('poisson','stratified')


def distribution(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    return None if not x.size else dict(count=int(x.size),mean=float(x.mean()),std=float(x.std()),
        quantile_levels=[0,.01,.1,.5,.9,.99,1],quantiles=np.quantile(x,[0,.01,.1,.5,.9,.99,1]).tolist())


def softmax(x):
    x=np.asarray(x,dtype=np.float64);z=x-x.max(-1,keepdims=True);p=np.exp(z)
    return p/p.sum(-1,keepdims=True)


def geometry_masks(data):
    margins=data['margins'].reshape(-1,3,18)[:,0]
    geometry=data['trusted'].reshape(-1,3)[:,0]&data['unprotected'].reshape(-1)
    safe=margins>1.;danger=margins<=0;grey=(margins>0)&~safe
    critical=geometry&safe.any(-1)&(~safe).any(-1)
    return margins,geometry,safe,danger,grey,critical


def actor_statistics(data,mode):
    margins,geometry,safe,danger,grey,critical=geometry_masks(data)
    raw=data[mode+'_raw'].reshape(-1,19);legal=data['actual_legal_actions'].reshape(-1,19)
    clean=np.where(legal,np.clip(raw,-80,80),-np.inf)
    p=softmax(clean);move=softmax(clean[:,:18]);move_action=clean[:,:18].argmax(-1);full_action=clean.argmax(-1)
    index=np.arange(len(raw))
    def chosen(mask):
        return dict(rows=int(mask.sum()),danger=int(danger[index,move_action][mask].sum()),
            grey=int(grey[index,move_action][mask].sum()),safe=int(safe[index,move_action][mask].sum()),
            unsafe_including_grey=int((~safe[index,move_action])[mask].sum()))
    def mass(mask):
        return dict(rows=int(mask.sum()),conditional_danger=distribution((move*danger).sum(-1)[mask]),
            conditional_grey=distribution((move*grey).sum(-1)[mask]),
            conditional_unsafe_including_grey=distribution((move*~safe).sum(-1)[mask]))
    return dict(scope='same observations/state/weights,alternative hard-input replay;not native alternative outcomes',
        all_rows=len(raw),geometry_unavailable_rows=int((~geometry).sum()),movement_only_greedy=chosen(geometry),
        full_greedy_when_movement=chosen(geometry&(full_action<18)),critical_movement_only_greedy=chosen(critical),
        movement_mass_all_geometry=mass(geometry),movement_mass_critical=mass(critical),
        Bomb=dict(inventory_legal_rows=int(legal[:,18].sum()),mean_probability=float(p[:,18].mean()) if len(p) else None,
            probability_distribution=distribution(p[:,18]),greedy_proposals=int((full_action==18).sum()),
            probability_on_inventory_illegal_rows_max=float(p[~legal[:,18],18].max()) if (~legal[:,18]).any() else None),
        executed_native_actions_in_this_analysis=0)


def neural_statistics(data,mode):
    dn=data[mode+'_DN'].reshape(-1,1299);counts=data[mode+'_DN_window_counts'].reshape(-1,1299)
    voltage=data[mode+'_DN_voltage'].reshape(-1,1299)
    return dict(rows=len(dn),DN_cells=1299,window_counts=dict(distribution=distribution(counts),
        zero_cell_window_fraction=float((counts==0).mean()),per_row_silent_fraction=distribution((counts==0).mean(-1)),
        columns_never_spiking_fraction=float((counts==0).all(0).mean()),Hz_distribution=distribution(counts*50.)),
        EMA_rate=dict(reference_Hz=150.,Hz_distribution=distribution(dn*150.),zero_fraction=float((dn==0).mean()),
            columns_always_zero_fraction=float((dn==0).all(0).mean()),between_recorded_rows_std_distribution=distribution(dn.std(0))),
        membrane_mv=dict(distribution=distribution(voltage),interpretation='voltage alone has no silent/not-silent label'),
        variance_interpretation='between scenes/times/games variance;not repeated same-input noise variance')


def probe_metrics(scores,data,mask):
    from sklearn.metrics import roc_auc_score
    margins,geometry,safe,danger,grey,critical=geometry_masks(data)
    known=geometry[:,None]&(danger|safe);selected=scores.argmin(-1);index=np.arange(len(scores))
    critical=mask&critical
    pairs=known[:,:,None]&danger[:,:,None]&known[:,None,:]&safe[:,None,:]&mask[:,None,None]
    difference=scores[:,:,None]-scores[:,None,:];wins=(difference>0)+.5*(difference==0)
    n=pairs.sum((1,2));mixed=n>0;auc=[]
    for action in range(18):
        rows=mask&known[:,action];y=danger[rows,action]
        if len(np.unique(y))==2:auc.append(dict(action=action,rows=int(rows.sum()),positive=int(y.sum()),AUC=float(roc_auc_score(y,scores[rows,action]))))
    unsafe=int((~safe[index,selected])[critical].sum())
    return dict(rows=int(mask.sum()),geometry_rows=int((mask&geometry).sum()),critical_rows=int(critical.sum()),
        unsafe_chosen_count=unsafe,unsafe_chosen_rate=unsafe/int(critical.sum()) if critical.any() else None,
        strict_mixed_rows=int(mixed.sum()),within_scene_pair_order_macro=float(((wins*pairs).sum((1,2))[mixed]/n[mixed]).mean()) if mixed.any() else None,
        within_scene_pair_order_pooled=float(wins[pairs].mean()) if pairs.any() else None,
        same_action_macro_AUC=float(np.mean([a['AUC'] for a in auc])) if auc else None,same_action_AUC=auc,
        selected_directions=np.bincount(selected[critical],minlength=18).tolist())


def fit_probes(data,*,guard,qualified):
    margins,geometry,safe,danger,grey,critical=geometry_masks(data);games=data['actual_game_ids'].reshape(-1)
    held=games%5==4;train=~held;known=geometry[:,None]&(danger|safe)
    rows=np.flatnonzero(train&known.any(-1));train_games=sorted(set(games[train].tolist()));held_games=sorted(set(games[held].tolist()))
    constant=int((~safe[train&critical]).mean(0).argmin()) if (train&critical).any() else None
    report=dict(status='insufficient',split='actual_game_ids modulo5 equals4 held;no fallback split',train_games=train_games,held_games=held_games,
        training_labeled_rows=len(rows),held_critical_rows=int((held&critical).sum()),
        fixed_recipe=dict(epochs=40,batch_rows=256,Adam_lr=.001,weight_decay=1e-4,seed=71921,permutation_seed=21129,
            model='linear-with-bias18;train-only mean/std,std<1e-4 replaced1,clip50',objective='per-action class-balanced BCE on known proxy labels'),
        CPU_diagnostic_Adam_steps=0,production_Adam_steps=0,train_selected_constant_direction=constant,
        train_prior_held_unsafe_count=int((~safe[held&critical,constant]).sum()) if constant is not None else None,
        held_selected_constant_unsafe_count_optimistic=int((~safe[held&critical]).sum(0).min()) if (held&critical).any() else None)
    reasons=[]
    if not qualified:reasons.append('source full-brain/reference replay not qualified')
    if len(train_games)<2 or len(held_games)<2:reasons.append('need at least2 whole train and2 whole held games')
    if len(rows)<32 or int((held&critical).sum())<8:reasons.append('need32 labeled train rows and8 held critical rows')
    if reasons:report['reasons']=reasons;return report
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    positive=(known[train]&danger[train]).sum(0);negative=(known[train]&safe[train]).sum(0)
    weights=(np.where(danger,.5/np.maximum(positive,1),.5/np.maximum(negative,1))*known).astype(np.float32)*len(rows)
    y=torch.from_numpy(danger.astype(np.float32));w=torch.from_numpy(weights)
    report['train_actions_without_positive']=np.flatnonzero(positive==0).tolist()
    report['train_actions_without_negative']=np.flatnonzero(negative==0).tolist();report['representations']={}
    for name,key in (('visual384','actual_encoded_visual'),('poisson_DN','poisson_DN'),('stratified_DN','stratified_DN')):
        guard();features=data[key].reshape(len(games),-1)
        mean=features[rows].mean(0);scale=features[rows].std(0);scale[scale<1e-4]=1.
        x=torch.from_numpy(np.clip((features-mean)/scale,-50,50).astype(np.float32))
        torch.manual_seed(71921);model=nn.Linear(x.shape[1],18);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(21129);steps=0
        for epoch in range(40):
            order=torch.randperm(len(rows),generator=generator).numpy()
            for start in range(0,len(rows),256):
                guard();ix=rows[order[start:start+256]];prediction=model(x[ix]);weight=w[ix]
                loss=(F.binary_cross_entropy_with_logits(prediction,y[ix],reduction='none')*weight).sum()/weight.sum().clamp_min(1e-12)
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();steps+=1
        with torch.no_grad():scores=model(x).numpy()
        report['representations'][name]=dict(parameters=sum(p.numel() for p in model.parameters()),CPU_diagnostic_Adam_steps=steps,
            train=probe_metrics(scores,data,train),held=probe_metrics(scores,data,held),last_training_loss=float(loss.detach()))
        report['CPU_diagnostic_Adam_steps']+=steps
    report.update(status='completed',CUDA_initialized=torch.cuda.is_initialized(),candidate_exported=False)
    return report


def analyze(data,metadata,*,guard=lambda:None,run_fits=True):
    if metadata.get('schema')!=SOURCE_SCHEMA:raise ValueError('Explicit paired producer schema required')
    if metadata.get('selected_sequences')==0:
        return dict(schema=SCHEMA,status='insufficient',reason='producer selected zero eligible sequences',
            real_rows=0,sequences=0,actual_games=[],actor={},neural={},actual_reference_verified=False,
            native_counterfactual_claim=False,automatically_deployed=False,
            probes={'status':'insufficient','CPU_diagnostic_Adam_steps':0})
    shape=data['poisson_raw'].shape[:2]
    if len(shape)!=2 or shape[1]!=4:raise ValueError('Actual sequence-by4 layout required')
    required={'margins':(*shape,3,18),'trusted':(*shape,3),'unprotected':shape,'outcomes':shape,
        'actual_game_ids':shape,'actual_actions':shape,'actual_encoded_visual':(*shape,384),
        'actual_raw_logits':(*shape,19),'actual_action_probs':(*shape,19),'actual_legal_actions':(*shape,19),
        'selected_time_world':(shape[0],2)}
    for mode in MODES:
        for name,width in (('raw',19),('DN',1299),('DN_voltage',1299),('DN_window_counts',1299)):required[mode+'_'+name]=(*shape,width)
    for key,expected in required.items():
        if key not in data or data[key].shape!=expected:raise ValueError('Paired shape differs:'+key)
    if not np.all(data['actual_game_ids']==data['actual_game_ids'][:,:1]):raise ValueError('A paired sequence crossed natural games')
    if data['trusted'].dtype!=np.bool_ or data['unprotected'].dtype!=np.bool_ or data['actual_legal_actions'].dtype!=np.bool_:
        raise ValueError('Explicit boolean trust/normal/legality required')
    if not data['actual_legal_actions'][...,:18].all():raise ValueError('All18 movements must remain inventory legal')
    for key in ('actual_game_ids','actual_actions','selected_time_world'):
        if not np.issubdtype(data[key].dtype,np.integer):raise ValueError('Integer episode/action provenance required:'+key)
    actual=data['actual_actions']
    if ((actual<0)|(actual>=19)).any() or not np.take_along_axis(data['actual_legal_actions'],actual[...,None],-1).all():
        raise ValueError('Actual recorded actions must be inventory legal')
    for key in required:
        if key not in ('trusted','unprotected','actual_legal_actions') and not np.isfinite(data[key]).all():raise ValueError('Nonfinite paired input:'+key)
    qualified=metadata.get('full_brain') is True and metadata.get('actual_reference_verified') is True
    mode=metadata.get('actual_mode')
    replay_error=float(np.abs(data[mode+'_raw']-data['actual_raw_logits']).max()) if shape[0] and mode in MODES else None
    qualified=qualified and replay_error is not None and replay_error<=1e-5
    result=dict(schema=SCHEMA,status='completed' if qualified else 'unqualified_source',real_rows=shape[0]*4,
        sequences=shape[0],actual_games=sorted(set(data['actual_game_ids'].reshape(-1).tolist())),actual_reference_verified=qualified,
        independently_computed_actual_logit_error=replay_error,native_counterfactual_claim=False,automatically_deployed=False,
        probability_contract='T1 clamp80 then actual inventory support;18-move conditional probabilities and full19 Bomb marginal separately',
        label_definition=dict(danger='margin<=0',grey='0<margin<=1',safe='margin>1',known='trusted immediate and normal;danger or safe',
            critical='trusted normal with at least1 safe and1 not-safe candidate;grey counts not-safe'),
        actor={},neural={},limitations=['Counterfactual neural replays hold source observations fixed;no new native outcomes',
            'Within-scene choices/proxy labels do not establish native avoidance or clears',
            'Across-scene DN variance is not same-input noise variance','Held games are development diagnostics,not formal tests'])
    if not shape[0]:
        result.update(status='insufficient',reason='producer selected zero eligible sequences',probes={'status':'insufficient','CPU_diagnostic_Adam_steps':0});return result
    for mode in MODES:
        guard();result['neural'][mode]=neural_statistics(data,mode)
        result['actor'][mode]=actor_statistics(data,mode) if qualified else {'status':'unavailable_actual_reference_or_full_brain_unqualified'}
    result['probes']=fit_probes(data,guard=guard,qualified=qualified) if run_fits else {'status':'not_requested_fixture','CPU_diagnostic_Adam_steps':0}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh CPU report directory required')
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);process.cpu_affinity(process.cpu_affinity()[-2:])
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>120:raise TimeoutError('Finite120-second CPU analysis;no retry or deployment')
    report_path=args.input/'report.json';npz_path=args.input/'paired-features.npz'
    metadata=read_shared_json(report_path);guard();blob=npz_path.read_bytes()
    with np.load(io.BytesIO(blob),allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
    result=analyze(data,metadata,guard=guard)
    result.update(source_report=metadata,sources=[dict(path=str(path.resolve()),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
        for path,payload in ((report_path,report_path.read_bytes()),(npz_path,blob))],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        resources=dict(CPU_threads=2,CPU_affinity=process.cpu_affinity(),priority='BelowNormal',elapsed_seconds=time.monotonic()-started,
            maximum_seconds=120,CUDA_VISIBLE_DEVICES='-1',brain_forward_calls=0,native_actions=0,production_optimizer_steps=0))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Paired input-drive CPU analysis','',f"Status: {result['status']}; {result['real_rows']} rows, {len(result['actual_games'])} games.",
        'Geometry is a proxy; branch proposals are not native counterfactual outcomes. T1/clamp80/inventory legality.','',
        '|Branch|Geometry rows|Move danger/grey/safe|Full greedy Bomb|Mean pBomb|','|---|---:|---|---:|---:|']
    for mode in MODES:
        row=result['actor'].get(mode,{})
        move=row.get('movement_only_greedy');bomb=row.get('Bomb')
        lines.append(f"|{mode}|{move['rows']}|{move['danger']}/{move['grey']}/{move['safe']}|{bomb['greedy_proposals']}|{bomb['mean_probability']:.6g}|" if move else f'|{mode}|unavailable|unqualified source|—|—|')
    probe=result['probes'];lines+=['',f"CPU probes: {probe['status']}; diagnostic Adam={probe.get('CPU_diagnostic_Adam_steps',0)}, production Adam=0."]
    if probe.get('reasons'):lines.append('Insufficient: '+'; '.join(probe['reasons']))
    if 'train_games' in probe:lines.append(f"Whole train games={probe['train_games']}; held games={probe['held_games']}; fixed gameID%5 split.")
    if probe.get('representations'):
        lines+=['','|Representation/split|Unsafe/critical|Scene macro order|Same-action macro AUC|','|---|---:|---:|---:|']
        for name,values in probe['representations'].items():
            for split in ('train','held'):
                row=values[split];lines.append(f"|{name}/{split}|{row['unsafe_chosen_count']}/{row['critical_rows']}|{row['within_scene_pair_order_macro']}|{row['same_action_macro_AUC']}|")
        lines.append(f"Train-selected constant direction={probe['train_selected_constant_direction']}; held unsafe={probe['train_prior_held_unsafe_count']}/{probe['held_critical_rows']}.")
    lines+=['','DN distributions/zero-window statistics and all source hashes are in summary.json. Across-scene variance is not same-input noise. No candidate or encoding was deployed.']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],output=str(args.output),CPU_diagnostic_Adam_steps=result['probes'].get('CPU_diagnostic_Adam_steps',0),elapsed_seconds=result['resources']['elapsed_seconds'])))


if __name__=='__main__':main()
