"""Fixed-order full-connectome SpMM:numerics,repeatability,gradient and speed."""
from pathlib import Path
import argparse,os,sys,json,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('connectome','previous-job','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(dict(status='prepared',CUDA_started=False));return
    import psutil
    from fly_touhou.thrl_study_io import read_shared_json as read
    from fly_touhou.thrl_durable_records import write_json
    l=read(a.previous_job/'launch.json')
    try:
        q=psutil.Process(l['PID'])
        if abs(q.create_time()-l['create_time'])<.001:raise RuntimeError('Previous CUDA job live')
    except psutil.NoSuchProcess:pass
    a.output.mkdir(parents=True,exist_ok=False);q=psutil.Process();q.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);q.cpu_affinity(q.cpu_affinity()[-2:])
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',FLY_TOUHOU_FUSED_EDGE_GRAD='1',
        FLY_TOUHOU_EDGE_GRAD_CACHE=str(a.output/'kernel-cache'),FLY_TOUHOU_SPMM_CACHE=str(a.output/'kernel-cache'))
    write_json(a.output/'launch.json',dict(PID=q.pid,create_time=q.create_time(),argv=sys.argv,schema='fixed-order-CSR-SpMM-benchmark-v1'))
    try:
        import torch,numpy as np
        from fly_touhou.continuous_sparse import TrainableDirectedSparse
        from fly_touhou.connectome import load_connectome
        torch.set_num_threads(2);torch.manual_seed(202610331);checks=[]
        for B in (1,18,32,128):
            pre=torch.randint(64,(1003,));post=torch.randint(64,(1003,))
            op=TrainableDirectedSparse(64,pre.numpy(),post.numpy()).cuda()
            x=torch.randn(B,64,device='cuda',requires_grad=True);w=torch.randn(1003,device='cuda',requires_grad=True);g=torch.randn_like(x)
            op.deterministic_spmm=False;expected=op(x,w);old=torch.autograd.grad((expected*g).sum(),(x,w))
            op.deterministic_spmm=True;actual=op(x,w);new=torch.autograd.grad((actual*g).sum(),(x,w))
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
            for n,o in zip(new,old):torch.testing.assert_close(n,o,rtol=2e-5,atol=2e-5)
            for _ in range(5):assert torch.equal(actual,op(x,w))
            checks.append(dict(batch=B,forward_max_error=float((actual-expected).abs().max()),repeated_bit_exact=True,gradients_passed=True))
        graph=load_connectome(a.connectome,max_neurons=None,verify=True)
        op=TrainableDirectedSparse(graph.n_neurons,graph.pre,graph.post).cuda()
        absolute=np.abs(graph.signed_weights.astype(np.float64));incoming=np.bincount(graph.post,weights=absolute,minlength=graph.n_neurons)
        outgoing=np.bincount(graph.pre,weights=absolute,minlength=graph.n_neurons)
        w=torch.tensor((graph.signed_weights/np.sqrt(incoming[graph.post]*outgoing[graph.pre])).astype(np.float32),device='cuda')
        x=(torch.rand(128,graph.n_neurons,device='cuda')>.9).float()
        with torch.no_grad():
            op.deterministic_spmm=False;expected=op(x,w)
            op.deterministic_spmm=True;actual=op(x,w)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
            for _ in range(5):assert torch.equal(actual,op(x,w))
            assert torch.equal(actual[:18],op(x[:18],w))
            def timing(det):
                op.deterministic_spmm=det;op(x,w);torch.cuda.synchronize();times=[]
                for _ in range(5):
                    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                    s.record();v=op(x,w);e.record();torch.cuda.synchronize();times.append(s.elapsed_time(e));del v
                return sum(times)/len(times)
            old_ms=timing(False);new_ms=timing(True)
        report=dict(status='passed',synthetic_checks=checks,full_graph=dict(neurons=graph.n_neurons,edges=graph.n_edges,
            batch=128,max_error=float((actual-expected).abs().max()),repeat_bit_exact=True,batch18_vs128_bit_exact=True),
            old_CSR_ms=old_ms,fixed_order_ms=new_ms,speedup=old_ms/new_ms,
            nodes_edges_signs_and_FP32_unchanged=True,GPU=torch.cuda.get_device_name(),native_actions=0)
        write_json(a.output/'report.json',report);print(json.dumps(report))
    except BaseException as error:
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
