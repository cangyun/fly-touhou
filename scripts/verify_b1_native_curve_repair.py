"""Pin actual curve birth/growth and compare the two frozen interpreters."""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess,sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(args):
    sys.path.insert(0,str(args.runtime/'src'))
    from fly_touhou.thrl_inputs import make_encoder,SEPARATED_LASER_INPUT
    from fly_touhou.state import Snapshot
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.atomic_json_io import atomic_write_json as write
    results=[]
    for row in read(args.dataset):
        raw=json.loads(gzip.decompress(Path(row['raw_snapshot']).read_bytes()))
        try:
            observation=make_encoder(SEPARATED_LASER_INPUT).encode(Snapshot.from_dict(raw['snapshot']))
            result=dict(accepted=True,hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in observation.items()})
        except ValueError as error:
            result=dict(accepted=False,error=str(error))
        results.append(dict(receipt=row,encoding=result))
    write(args.output,dict(runtime=str(args.runtime),manifest_sha256=sha(args.runtime/'source_manifest.json'),results=results))


def run(args):
    import psutil
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    from fly_touhou.thrl_study_io import read_shared_json as read,utc_now
    from fly_touhou.atomic_json_io import atomic_write_json as write
    report=read(args.probe/'report.json')
    if report['status']!='qualified' or read(args.probe/'cleanup.json')!=dict(closed=True,close_errors=[]):
        raise ValueError('Require qualified probe and clean native close')
    owned=read(args.probe/'workers/ownership.json')
    if owned['status']!='closed' or owned['close_errors']:raise ValueError('Native ownership not closed')
    for worker in owned['workers']:
        if worker['worker_alive_after_close']:raise ValueError('Worker still alive')
        for identity in [worker['worker_identity'],worker['native_identity'],*worker['native_history']]:
            try:
                if abs(psutil.Process(identity['pid']).create_time()-identity['create_time'])<.01:
                    raise ValueError('Owned original identity remains live')
            except psutil.NoSuchProcess:pass
    rows=[]
    for path in report['actual_native_receipts']:
        row=read(Path(path));snapshot=Path(row['raw_snapshot'])
        if (sha(snapshot)!=row['sha256'] or hashlib.sha256(gzip.decompress(snapshot.read_bytes())).hexdigest()!=row['uncompressed_sha256']):
            raise ValueError('Raw original snapshot identity changed')
        rows.append(dict(row,receipt_path=path))
    matched=[]
    for single in [row for row in rows if row['kind']=='single_node']:
        for grown in [row for row in rows if row['kind']=='grown_after_single']:
            common={r['id'] for r in single['curves'] if r['node_count']==1}&{r['id'] for r in grown['curves'] if r['node_count']>1}
            if (common and Path(single['raw_snapshot']).parent==Path(grown['raw_snapshot']).parent
                    and single['native_episode']==grown['native_episode'] and single['stage']==grown['stage']
                    and single['native_frame']<grown['native_frame']):
                matched.append(dict(ids=sorted(common),single=single['receipt_path'],grown=grown['receipt_path']))
    if not matched:raise ValueError('Raw evidence does not bind the same observed single-node object to its growth')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);dataset=args.output/'pinned_receipts.json';write(dataset,rows)
    for label,runtime in [('reference',args.reference_runtime),('repair',args.runtime)]:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child','--runtime',str(runtime),
            '--dataset',str(dataset),'--output',str(args.output/(label+'.json'))],check=True,capture_output=True,text=True)
    old=read(args.output/'reference.json');new=read(args.output/'repair.json');birth_failures=0;growth_exact=0
    for before,after in zip(old['results'],new['results']):
        if not after['encoding']['accepted']:raise ValueError('Repair still rejects actual native geometry')
        if before['receipt']['kind']=='single_node':
            if before['encoding']!=dict(accepted=False,error='Invalid current laser polyline'):
                raise ValueError('Real birth did not reproduce the previous source exception')
            birth_failures+=1
        elif before['encoding']['accepted']:
            if before['encoding']['hashes']!=after['encoding']['hashes']:raise ValueError('Accepted grown geometry drift')
            growth_exact+=1
    if not birth_failures or not growth_exact:raise ValueError('Missing both boundary and unchanged grown geometry')
    result=dict(schema='B1-actual-native-curve-birth-repair-verification-v1',status='qualified',utc=utc_now(),
        probe=str(args.probe),probe_report_sha256=sha(args.probe/'report.json'),raw_same_identity_growth=matched,
        actual_native_births_reproduce_frozen_V3_exception=birth_failures,accepted_grown_inputs_exact=growth_exact,
        repaired_encodes_all_pinned_native_snapshots=True,all_probe_workers_and_native_identities_closed=True,
        reference_manifest_sha256=old['manifest_sha256'],repair_manifest_sha256=new['manifest_sha256'],
        prior_failed_training_raw_snapshot_available=False,optimizer_updates=0,policy_skill_claimed=False)
    write(args.output/'report.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['probe','runtime','reference-runtime','dataset','output']:p.add_argument('--'+key,type=Path)
    p.add_argument('--child',action='store_true');a=p.parse_args()
    child(a) if a.child else run(a)
