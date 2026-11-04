"""Prepare a fixed30-world B1 correction; do not launch training or worlds."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
from fly_touhou.atomic_json_io import atomic_write_json as write


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(a):
    import torch
    from fly_touhou.thrl_route_credit import configuration,trace_weights
    from fly_touhou.thrl_pair_policy import model_from_config
    from fly_touhou.thrl_ppo import PPOConfig,make_optimizer
    from fly_touhou.thrl_branching import warm_start
    if sha(a.parent)!=a.parent_sha256:raise ValueError('Chosen parent pin differs')
    if a.training_output.exists():raise FileExistsError(a.training_output)
    manifest=read(a.runtime/'source_manifest.json');canonical=set()
    for row in manifest['files']:
        rel=row['path'].replace('\\','/')
        if rel.casefold() in canonical or sha(a.runtime/rel)!=row['sha256']:raise ValueError('Frozen source identity differs')
        canonical.add(rel.casefold())
    p=torch.load(a.parent,map_location='cpu',weights_only=False);torch.set_num_threads(1)
    c=configuration(read(a.base_config),p,checkpoint=a.parent,checkpoint_sha256=a.parent_sha256,
        runtime=a.runtime,manifest_sha256=sha(a.runtime/'source_manifest.json'),native_read_report=a.native_read_report,
        job_id=a.panel_id,control_root=a.panel_root)
    model=model_from_config(**p['model_config']);model.load_state_dict(p['model'],strict=True)
    optimizer=make_optimizer(model,PPOConfig(**c['parameters']))
    audit=warm_start(torch,model,optimizer,a.parent,a.parent_sha256,exact_optimizer=True,model_factory=model_from_config)
    if (not audit['common_weights_exact'] or audit['new_parameter_names'] or audit['additional_state_keys']
            or audit['common_Adam_states']!=len(p['optimizer']['state'])):raise ValueError('Policy/Adam transfer differs')
    if any(parameter.requires_grad for head in [model.risk,model.laser_risk] for parameter in head.parameters()):
        raise ValueError('The fixed correction retains both fitted frozen risk heads')
    a.output_control.mkdir(parents=True,exist_ok=False)
    write(a.output_control/'config.json',c)
    temp=a.training_output.parent/'native-temporary'/a.training_output.name;temp.mkdir(parents=True,exist_ok=False)
    write(a.output_control/'background_spec.json',dict(schema='fly-touhou-background-job-v1',
        python=sys.executable,script=str(a.runtime/'scripts/train_b1_entity_ppo.py'),
        arguments=['--config',str(a.output_control/'config.json'),'--output',str(a.training_output)],
        cwd=str(a.runtime),control_directory=str(a.output_control),environment=dict(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
            PYTHONPATH=str(a.runtime/'src'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            TEMP=str(temp),TMP=str(temp),TMPDIR=str(temp),FLY_TOUHOU_DESKTOP_NAME='fly-touhou-job-soft-route-credit')))
    receipt=dict(schema='B1-route-credit-CPU-preparation-v1',status='prepared-not-launched',utc=utc_now(),
        config=str(a.output_control/'config.json'),config_sha256=sha(a.output_control/'config.json'),
        source_runtime=str(a.runtime),source_manifest_sha256=c['source_manifest_sha256'],
        parent=str(a.parent),parent_sha256=a.parent_sha256,parent_candidate_optimized_decisions=p['candidate_optimized_decisions'],
        parent_lifetime_decisions=p['lifetime_decisions'],weights_and_Adam_transfer=audit,parameters=model.parameter_counts(),
        native_read_qualification_report=str(a.native_read_report),native_read_qualification_completed=a.native_read_report.exists(),
        parent_selection_provisional=True,independent_native_efficacy_not_yet_established=True,
        trace_weights_before=trace_weights(.999,.98,[500,1000]),trace_weights_after=trace_weights(.999,.995,[500,1000]),
        native_worlds_started=0,CUDA_initialized=torch.cuda.is_initialized(),PPO_updates=0,
        training_output=str(a.training_output),capacity=c['capacity'])
    write(a.output_control/'preparation_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','parent','parent_candidate_optimized_decisions',
        'parent_lifetime_decisions','parameters','capacity','CUDA_initialized']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for k in ['parent','base-config','runtime','output-control','training-output','native-read-report','panel-root']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--panel-id',required=True);run(p.parse_args())
