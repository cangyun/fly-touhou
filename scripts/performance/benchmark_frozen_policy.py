"""Time the existing frozen 407-column CUDA policy on the archived seed-60036 observations."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

def stats(samples):
    a = np.asarray(samples) * 1000
    return dict(mean_ms=float(a.mean()), p50_ms=float(np.quantile(a, .5)),
                p90_ms=float(np.quantile(a, .9)), p99_ms=float(np.quantile(a, .99)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--demo",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    opts=p.parse_args()
    opts.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(opts.repo/"src"),str(opts.repo/"scripts")]
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", FLY_TOUHOU_DETERMINISTIC_SPMM="1",
                      FLY_TOUHOU_SPMM_CACHE=str(opts.output/"cuda-cache"))
    import torch
    from fly_touhou.nbnm_run import load_model
    from fly_touhou.nbnm_cuda import InferenceGraphs
    from fly_touhou.nbnm_diagnostics import decision_record
    from fly_touhou.spiking_engineered_control import SpikingState
    from fly_touhou.thrl_durable_records import append_jsonl,write_json
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    weights=opts.demo/"model"
    spec=json.loads((weights/"archived-inference.json").read_text(encoding="utf-8-sig"))
    with np.load(weights/"original-decisions.npz",allow_pickle=False) as f:
        arrays={k:f[k] for k in ("features","frames","stages","stage_frames","actions","logits")}
    model,_=load_model(spec,weights/"cycle073-parameters.pt")
    model.frozen_safe_idle_bonus=spec["frozen_safe_idle_bonus"]
    gpu=dict(name=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    proof=[]
    idx=np.r_[0:16]
    # Actual opening history checks both graph kernels at batch size 1.
    with torch.no_grad():
        x=torch.as_tensor(arrays["features"][idx],device=model.device)
        a=InferenceGraphs(model,sample=False,packed_binary_messages=True)
        b=InferenceGraphs(model,sample=False,packed_binary_messages=False)
        sa,sb=model.initial_state(1),model.initial_state(1)
        for i in range(len(x)):
            pa,sa,_=a.frozen_batch(x[i:i+1],sa,1)
            pb,sb,_=b.frozen_batch(x[i:i+1],sb,1)
            score=pa["logits"][0,:18].cpu().numpy()
            proof.append(dict(index=int(i),four_state_tensors_equal=all(torch.equal(u,v) for u,v in zip(sa.tensors(),sb.tensors())),
                              logits_equal=bool(torch.equal(pa["logits"],pb["logits"])),
                              original_logits_max_abs_error=float(np.max(np.abs(score-arrays["logits"][i]))),
                              action=int(pa["actions"][0]),original_action=int(arrays["actions"][i])))
        a.clear();b.clear()
        del a,b,sa,sb,pa,pb,x
    gc.collect();torch.cuda.empty_cache()
    print("Opening check",all(x["four_state_tensors_equal"] and x["logits_equal"] for x in proof),flush=True)
    log=opts.output/"diagnostic-sample.jsonl"
    bands=[]
    for s in range(1,7):
        indices=np.flatnonzero(arrays["stages"]==s)
        bands.append([int(indices[int(len(indices)*r)]) for r in (.12,.33,.56,.81)])
    results=[]
    for packed,batch in ((True,1),(False,1),(True,2),(True,4),(True,8)):
        torch.cuda.reset_peak_memory_stats()
        graph=InferenceGraphs(model,sample=False,packed_binary_messages=packed)
        metrics={k:[] for k in ("predict","diagnostics_and_append","commit_and_action","total")}
        starts=np.asarray(bands).T.reshape(-1)
        groups=starts.reshape(-1,batch)
        # Warm each archived window for ten decisions. The opening check above
        # compares exact policy outputs against the original run.
        with torch.no_grad():
            iids=torch.arange(batch,device=model.device)
            for starts in groups:
                state=model.initial_state(batch)
                for j in range(26):
                    pos=starts+j
                    features=arrays["features"][pos]
                    span=perf_counter()
                    sub=SpikingState(*(v.index_select(0,iids) for v in state.tensors()))
                    tensor=torch.as_tensor(features,device=model.device)
                    out,nxt,_=graph.frozen_batch(tensor,sub,batch)
                    torch.cuda.synchronize()
                    tick=perf_counter()
                    if j>=10:
                        for row,n in enumerate(pos):
                            info=dict(native_seed=60036,native_episode=1,episode_frames=int(arrays["frames"][n]),
                                      stage=int(arrays["stages"][n]),stage_frame=int(arrays["stage_frames"][n]))
                            append_jsonl(log,decision_record(info,features[row],out,row,policy="benchmark"))
                    after_diag=perf_counter()
                    for target,value in zip(state.tensors(),nxt.tensors()):
                        target.index_copy_(0,iids,value)
                    cmds=out["actions"].cpu().tolist()
                    end=perf_counter()
                    if j>=10:
                        for key,value in zip(metrics,(tick-span,after_diag-tick,end-after_diag,end-span)):
                            metrics[key].append(value)
            allocated=torch.cuda.max_memory_allocated()/1024**3
            reserved=torch.cuda.max_memory_reserved()/1024**3
        row=dict(batch=batch,packed=packed,measured_batches=len(metrics["total"]),
                 wall_decisions_s=float(len(metrics["total"])*batch/sum(metrics["total"])),
                 predict_decisions_s=float(len(metrics["predict"])*batch/sum(metrics["predict"])),
                 peak_allocated_gib=allocated,peak_reserved_gib=reserved,**{k:stats(v) for k,v in metrics.items()})
        results.append(row)
        write_json(opts.output/"gpu-timing.json",dict(gpu=gpu,opening_kernel_equivalence=proof,batches=results,
                   diagnostics_scope="Synthetic minimal info with real score rows, GPU-to-CPU telemetry and file append. Native diagnostic payload may be larger."))
        print(batch,packed,round(row["predict"]["mean_ms"],2),round(row["wall_decisions_s"],2),flush=True)
        graph.clear()
        del graph,state,sub,nxt,out,tensor
        gc.collect();torch.cuda.empty_cache()

if __name__=="__main__":
    main()
