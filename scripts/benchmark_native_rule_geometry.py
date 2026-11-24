"""Paired encoder timings on captured original-game snapshots; no game/model."""
from pathlib import Path
import argparse,copy,json,pickle,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=('legacy','rules24','rules96'),default=['legacy','rules24','rules96'])
    args=p.parse_args();records=[]
    for file in args.captures.glob('*-encoding.pkl'):
        with file.open('rb') as stream:encoder,snapshot=pickle.load(stream)
        for mode in args.modes:
            rows=[]
            for repeat in range(4):
                e=copy.deepcopy(encoder);s=copy.deepcopy(snapshot)
                e.native_rule_geometry=None
                if mode=='legacy':
                    from fly_touhou.nbnm_geometry_revision import legacy_snapshot
                    s=legacy_snapshot(s)
                elif 'native_rule_geometry' in s.extra.get('hazard_metadata',{}):
                    s.extra['hazard_metadata']['native_rule_geometry']['horizon']=24 if mode=='rules24' else 96
                for raw in s.extra.get('hazard_metadata',{}).get('bullets',[]):
                    if mode=='legacy':
                        for key in list(raw):
                            if key.startswith('native_ex_program') or key in (
                                    'native_ex_full_program_start_index','native_ex_style_radii',
                                    'native_ex_reaction_selector','native_ex_bounce_global_bounds'):
                                raw.pop(key,None)
                    else:raw['native_ex_program_horizon']=24 if mode=='rules24' else 96
                began=time.perf_counter();e.encode(s);elapsed=time.perf_counter()-began
                if repeat:
                    hint=e.last_result.get('rule_long_hints') or {}
                    rows.append(dict(milliseconds=elapsed*1000,long_hint_ms=hint.get('seconds',0)*1000,
                        long_coverage=hint.get('coverage'),trusted=np.asarray(e.last_result['details']['trusted']).tolist()))
            records.append(dict(task=file.stem,mode=mode,median_ms=float(np.median([r['milliseconds'] for r in rows])),samples=rows))
    report=dict(records=records,warmup_calls_excluded=1,training_started=False)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([dict(task=r['task'],mode=r['mode'],median_ms=r['median_ms']) for r in records]))


if __name__=='__main__':main()
