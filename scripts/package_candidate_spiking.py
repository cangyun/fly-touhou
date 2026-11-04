"""CPU-only packaging of existing learned brain and DN-only readout weights."""
from pathlib import Path
import argparse,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('brain','head','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    import torch
    from fly_touhou.thrl_durable_records import write_json
    torch.set_num_threads(2)
    brain=torch.load(a.brain,map_location='cpu',weights_only=False)
    head=torch.load(a.head,map_location='cpu',weights_only=False)
    if Path(head['brain_checkpoint']).resolve()!=a.brain.resolve():raise ValueError('Readout belongs to a different brain checkpoint')
    brain['state_dict']={k:v for k,v in brain['state_dict'].items() if not k.startswith('risk.')}
    brain['state_dict'].update({'risk.'+k:v for k,v in head['state_dict'].items()})
    brain.update(readout_kind=head['schema'],readout_width=head['width'],horizon_weights=(8.,4.,2.),stay_prior=.05,
        raw_danger_or_teacher_bypass=False,readout_provenance=dict(brain=str(a.brain),head=str(a.head),
        head_SHA256=hashlib.sha256(a.head.read_bytes()).hexdigest(),packaging_is_not_training=True))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:torch.save(brain,f)
    write_json(a.output.with_suffix('.json'),dict(checkpoint=str(a.output),SHA256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        full_brain_and_readout=True,packaging_only=True,new_optimizer_updates=0,native_result_not_inferred=True))


if __name__=='__main__':main()
