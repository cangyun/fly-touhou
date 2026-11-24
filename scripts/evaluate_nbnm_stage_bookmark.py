"""Start an owned TH14 directly at a frozen student's archived sixth-stage entry."""
import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def run(book,output,limit,prepare_only=False,issun_lure=False,lure_scope='constrained',trace_story=False):
    import numpy as np
    import torch
    from capture_nbnm_human_replay import enter_replay
    from fly_touhou.background import action_mask
    from fly_touhou.compact_full_story import factory,story_kwargs
    from fly_touhou.nbnm_cuda import InferenceGraphs
    from fly_touhou.nbnm_run import read,runtime,load_model,environment,task_settings
    from fly_touhou.nbnm_stage_bookmark import rebuild_brain,restore_entry_encoder
    from fly_touhou.spiking_engineered_control import SpikingState
    from fly_touhou.th14_stage_replay import StageSixKeys,write_stage_replay
    from fly_touhou.thrl_durable_records import write_json,append_jsonl
    from fly_touhou.nbnm_sensation import current_attack
    from fly_touhou.thrl_job_paths import cleanup_native_temp
    from fly_touhou.thrl_native_env import SOURCE19_TO_NATIVE36
    output.mkdir(parents=True,exist_ok=False)
    temporary=runtime(output);env=None;keys=None;graphs=None
    try:
        future=(book/'bookmark.json').exists()
        entry=read(book/('bookmark.json' if future else 'entry.json'))
        frame=entry['stage_frame'];seed=entry['seed']
        spec=read(book/entry['configuration'])
        raw=entry.get('native_stage_record') or entry.get('first_native_sixth_stage')
        if not raw:raise ValueError('Missing original sixth-stage game record')
        record=read(book/raw);model,_=load_model(spec,book/entry['checkpoint'])
        brain=book/'brain.pt'
        reference=read(book/entry['reference']) if entry.get('reference') else []
        recorded=None
        if entry.get('recorded_inputs'):
            with np.load(book/entry['recorded_inputs']) as z:
                recorded=z['features'][entry['index']:entry['index']+len(reference)]
                if not brain.exists():
                    rebuild_brain(model,z['features'],z['actions'],entry['index'],seed,frame,brain)
        saved=torch.load(brain,map_location='cpu',weights_only=False)
        if (saved['opening_seed'],saved['stage_frame'])!=(seed,frame):
            raise ValueError('Recurrent state and native stage entry differ')
        state=SpikingState(*(t.to(model.device) for t in saved['state']))
        graphs=InferenceGraphs(model,sample=False,packed_binary_messages=spec.get('packed_binary_messages',False))
        def decide(features,memory):
            data=torch.as_tensor(features,dtype=torch.float32,device=model.device).reshape(1,-1)
            return graphs.frozen_batch(data,memory,1)
        if reference and recorded is not None:
            with torch.no_grad():
                probe=SpikingState(*(t.clone() for t in state.tensors()))
                for feature,row in zip(recorded[:8],reference[:8]):
                    out,probe,_=decide(feature,probe)
                    if int(out['actions'][0])!=row['action']:raise ValueError('Original frozen move differs after brain restoration')
                    if 'policy_logits' in row:
                        np.testing.assert_allclose(out['logits'][0,:18].cpu(),row['policy_logits'],atol=3e-5,rtol=0)
        if prepare_only:
            write_json(output/'report.json',dict(prepared=True,brain=str(brain),compared_decisions=min(8,len(reference))))
            return
        cfg=environment(spec,evaluation=True)
        task=spec['tasks']['s6_whole_stage' if entry.get('mode')=='practice' else 'full_story']
        cfg['nbnm']=task_settings(spec,task)
        if issun_lure:
            cfg['nbnm']['corridor_risk']=dict(cfg['nbnm']['corridor_risk'],issun_lure=True,issun_lure_scope=lure_scope)
        corridor=cfg['nbnm'].get('corridor_risk') or {}
        lure_enabled=bool(corridor.get('issun_lure'))
        env=factory(story_kwargs(cfg,output,1,'development',teacher=False)[0])
        env.native_seed=None;env.private_appdata=output/'userdata'
        write_stage_replay(env.private_appdata/'ShanghaiAlice/th14/replay/th14_01.rpy',record)
        c=env._create_controller();env.controller=c;c._checkpoint_performance_options['wait_sleep_ms']=1
        start=time.monotonic();c.start();s=enter_replay(c,first_frame=frame)
        if (s.stage,s.stage_frame,s.extra.get('replay_playback'))!=(6,frame,1):
            raise RuntimeError('The original replay did not open at the saved decision')
        keys=StageSixKeys(c,record)
        restore_entry_encoder(env.encoder,s,saved['features'],entry.get('attack_age_frames'))
        counters=s.extra['thrl_events'];initial_bombs=counters['bombs_used'];initial_misses=counters['misses']
        report=dict(seed=seed,native_stage_seed=record['stage_rng_seed'],stage=6,entry_frame=frame,
            source=entry['source'],checkpoint=str(book/entry['checkpoint']),brain_file=str(brain),
            native_start_seconds=time.monotonic()-start,previous_stages_simulated=0,training_updates=0,
            full_game_NBNM=False,full_story_prefix_played_by_current_policy=False,decisions=0,
            outcome='ongoing',stage_NBNM=False,matching_reference_decisions=0,issun_lure=lure_enabled,
            issun_lure_scope=corridor.get('issun_lure_scope','all') if lure_enabled else None,
            continues=record['continues'],difficulty=record['difficulty'],shot=record['shot'],subshot=record['subshot'],
            matching_reference_logits=0,maximum_reference_logit_difference=0.,
            attacks=[],lure_decisions=0,chosen_lure_first_moves=0)
        previous_dialogue=0
        with torch.no_grad():
            while True:
                event=s.extra['thrl_events']
                bombs=event['bombs_used']-initial_bombs;misses=event['misses']-initial_misses
                contact=bool(misses or s.player is not None and s.player.state==4)
                ending=bool(s.stage==6 and (s.extra.get('game_thread_flags',0)&0x4000 or
                    any(x.get('stage')==6 and x.get('ending') for x in s.extra.get('native_milestones',()))))
                report.update(stage_frame=s.stage_frame,bombs=bombs,misses=misses,native_contact=contact)
                if contact or bombs or ending:
                    report['outcome']='first-native-contact' if contact else 'Bomb-used' if bombs else 'native-stage-six-ending'
                    report['stage_NBNM']=ending and not contact and not bombs
                    break
                n=report['decisions']
                if limit and n>=limit:report['outcome']='requested-decision-limit';break
                if s.stage!=6 or not s.extra.get('replay_playback'):raise RuntimeError('Native branch ended without a game outcome')
                features=env.encoder.encode(s)['features']
                if n==0:np.testing.assert_allclose(features,saved['features'],atol=3e-5,rtol=0)
                result,state,_=decide(features,state);move=int(result['actions'][0])
                if n<len(reference):
                    row=reference[n]
                    np.testing.assert_allclose(features,recorded[n],atol=3e-5,rtol=0)
                    if (s.stage_frame,move)!=(row['stage_frame'],row['action']):raise ValueError('The live student diverged from the saved reference')
                    if 'policy_logits' in row:
                        logits=result['logits'][0,:18].cpu().numpy()
                        np.testing.assert_allclose(logits,row['policy_logits'],atol=3e-5,rtol=0)
                        report['matching_reference_logits']+=1
                        finite=np.isfinite(logits)
                        delta=np.abs(logits[finite]-np.asarray(row['policy_logits'])[finite])
                        report['maximum_reference_logit_difference']=max(report['maximum_reference_logit_difference'],
                            float(delta.max(initial=0.)))
                    report['matching_reference_decisions']+=1
                phase=current_attack(s) or 'road'
                if not report['attacks'] or report['attacks'][-1]['phase']!=phase:
                    report['attacks'].append(dict(phase=phase,frame=s.stage_frame))
                published=()
                if env.encoder.corridor_risk:
                    from fly_touhou.nbnm_corridor import carry_after
                    last=env.encoder.last_result
                    published=(last.get('corridor') or {}).get('issun_lure',{}).get('published',())
                    if published:
                        report['lure_decisions']+=1
                        report['chosen_lure_first_moves']+=any(c['action']==move for c in published)
                    env.encoder.corridor_carry=carry_after(last,last.get('corridor'),move,s.frame)
                if trace_story:
                    decision=dict(stage_frame=s.stage_frame,phase=phase,action=move,
                        position=[float(features[0])*368-184,float(features[1])*400+32])
                    if published:
                        decision['suggested_lure_moves']=[int(p['action']) for p in published]
                    append_jsonl(output/'original-stage-six-decisions.jsonl',decision)
                mask=action_mask(SOURCE19_TO_NATIVE36[move])
                dialogue=s.extra.get('native_dialogue_object') if s.dialogue else 0
                if dialogue and dialogue!=previous_dialogue:
                    keys.supply(mask&~1,1);s=c.advance(0,1)
                    keys.supply(mask,3);s=c.advance(0,3)
                else:
                    keys.supply(mask,4);s=c.advance(0,4)
                previous_dialogue=dialogue;report['decisions']=n+1
                if n%128==0:write_json(output/'progress.json',report)
        write_json(output/'report.json',report)
    except BaseException:
        write_json(output/'failure.json',dict(error=traceback.format_exc()));raise
    finally:
        if keys is not None:keys.close()
        if graphs is not None:graphs.clear()
        if env is not None:env.close()
        cleanup_native_temp(temporary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bookmark',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--decisions',type=int,help='Stop after this many autonomous decisions')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--issun-lure',action='store_true',help='Enable the dynamic orb-luring corridor on Stage 6 boss spell 5')
    parser.add_argument('--lure-scope',choices=('constrained','all'),default='constrained')
    parser.add_argument('--trace-story',action='store_true',help='Record native movements for comparison with the live Story before the patched spell')
    args=parser.parse_args()
    if args.decisions is not None and args.decisions<=0:parser.error('Decision limit must be positive')
    from fly_touhou.nbnm_job_phase import gpu_phase
    with gpu_phase('evaluation'):
        run(args.bookmark.resolve(),args.output.resolve(),args.decisions,args.prepare_only,args.issun_lure,args.lure_scope,args.trace_story)
