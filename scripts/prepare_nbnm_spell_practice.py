"""Prepare a direct native spell experiment, reusing model and dependencies."""
from pathlib import Path
import argparse,copy,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def prepare(spec, *, score, stage, spell_index, spell_id, attack, run_output):
    from fly_touhou.nbnm_spell_practice import validate_task
    result=copy.deepcopy(spec)
    task=dict(mode='attack',stage=stage,attack=attack,spell_index=spell_index,spell_id=spell_id)
    validate_task(task)
    if not Path(score).is_file():raise ValueError('Supply an existing scoreth14.dat; the original is never changed')
    result['tasks']={'card':task}
    result['world_groups']=['card']*result['workers'];result['group_weights']={'card':1.}
    result['nbnm'].update(task=task,entry_mode='native_spell_practice',score_template=str(Path(score).resolve()),
        practice_evaluation_seeds=dict(development=[4747,22687,35537,35846],test=[50018,50019,50028,50029]))
    result['nbnm'].pop('start_bank',None)
    result['output']=str(Path(run_output).resolve())
    result['tensorboard_directory']=str(Path(run_output).resolve()/'tensorboard')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True,help='Existing NBNM route/baseline configuration')
    p.add_argument('--score',type=Path,required=True,help='Native save with this card unlocked')
    p.add_argument('--stage',type=int,required=True)
    p.add_argument('--spell-index',type=int,required=True,help='Zero-based native menu row within the stage')
    p.add_argument('--spell-id',type=int,required=True,help='Measured native zero-based spell ID')
    p.add_argument('--attack',required=True,help='Measured native card ECL name')
    p.add_argument('--output',type=Path,required=True,help='New JSON configuration')
    p.add_argument('--run-output',type=Path,required=True,help='Separate training artifact directory')
    a=p.parse_args()
    spec=json.loads(a.config.read_text(encoding='utf-8-sig'))
    result=prepare(spec,score=a.score,stage=a.stage,spell_index=a.spell_index,
        spell_id=a.spell_id,attack=a.attack,run_output=a.run_output)
    from fly_touhou.nbnm_run import validate_spec
    validation=validate_spec(result)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Do not silently replace a live run's configuration.
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(validation,config=str(a.output),network_training=False),ensure_ascii=False))


if __name__=='__main__':main()
