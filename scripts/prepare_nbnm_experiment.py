"""Write runnable configurations and launchers; never start network training."""
from pathlib import Path
import argparse,copy,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/nbnm-preparation')
    p.add_argument('--start-bank',type=Path,default=ROOT/'outputs/nbnm-start-bank/manifest.json')
    p.add_argument('--game-exe',type=Path,required=True)
    p.add_argument('--environment-config',type=Path,default=ROOT/'configs/environment.json')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/cycle087-parameters.pt')
    p.add_argument('--connectome',type=Path,default=ROOT/'data/connectome')
    p.add_argument('--run-root',type=Path,default=ROOT/'outputs/nbnm-training')
    p.add_argument('--tensorboard-root',type=Path,default=ROOT/'outputs/tensorboard')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--decision-budget',type=int,default=32768)
    p.add_argument('--segment-decisions',type=int,default=8192)
    p.add_argument('--stage',type=int,choices=range(1,7),default=3)
    p.add_argument('--attack',default='BossCard1')
    a=p.parse_args()
    from fly_touhou.thrl_durable_records import write_json
    if not 4<=a.workers<=100 or a.workers%4:
        p.error('Use a multiple of four workers for the 75/25 task split (4..100).')
    if a.segment_decisions<=0 or a.decision_budget<=0 or a.segment_decisions%256 or a.segment_decisions<a.workers*16 or a.decision_budget%a.segment_decisions:
        p.error('Use full 256-row segments, at least16 rows/worker, and a whole number of segments.')
    a.output=a.output.expanduser().resolve()
    environment=json.loads(a.environment_config.read_text(encoding='utf-8-sig'))
    environment['game_exe']=str(a.game_exe.expanduser().resolve())
    environment['profile']=str((ROOT/environment['profile']).resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    write_json(a.output/'environment.json',environment)
    task=dict(mode='attack',stage=a.stage,attack=a.attack)
    card_workers=a.workers*3//4
    spec=dict(schema='shared-whole-connectome-NBNM-experiment-v1',seed=2026110911,
        initial_checkpoint=str(a.checkpoint.expanduser().resolve()),
        environment_config=str(a.output/'environment.json'),
        connectome=str(a.connectome.expanduser().resolve()),
        workers=a.workers,segment_decisions=a.segment_decisions,decision_budget=a.decision_budget,
        tasks=dict(card=task,full=dict(mode='full_story')),
        world_groups=['card']*card_workers+['full']*(a.workers-card_workers),group_weights=dict(card=.75,full=.25),
        movement_sampling_temperature=1.,evaluate_at_end=True,cuda_graph_inference=True,
        full_story_evaluation_seeds=[50016,50017,50026,50027],
        nbnm=dict(task=task,start_bank=str(a.start_bank.expanduser().resolve()),risk_mode='route',
            objective=dict(phase='completion',speed_weight=.05,time_scale_frames=3600)))
    a.output.mkdir(parents=True,exist_ok=True)
    for mode in ('route','baseline'):
        current=copy.deepcopy(spec);current['nbnm']['risk_mode']=mode
        current['output']=str(a.run_root.expanduser().resolve()/(f's{a.stage}-{a.attack.lower()}-'+mode+'-pilot-v1'))
        current['tensorboard_directory']=str(a.tensorboard_root.expanduser().resolve()/('NBNM-'+mode+'-pilot-v1'))
        write_json(a.output/(mode+'.json'),current)
    write_json(a.output/'preparation.json',dict(status='configured_not_trained',
        default_experiment='route',network_training=False,
        baseline='Same graph, budget, goals and online teacher; route input block zeroed',
        comparison='Use matching development starts and joint-mode decoder; compare successes first, time second',
        promotion='20 distinct frozen seeds with >=90% task NBNM before explicit speed phase; no automatic promotion',
        timing='No guaranteed speedup before measured whole-brain and native qualification',
        formal_training_command=[sys.executable,'-B','-X','utf8',
            str(ROOT/'scripts/train_nbnm.py'),'--config',str(a.output/'route.json'),'--execute']))
    print(a.output/'route.json')


if __name__=='__main__':main()
