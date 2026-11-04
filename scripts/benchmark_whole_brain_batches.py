"""Compute-only CUDA batch throughput/numerics on real resource observations.

Every size/repetition starts from the same immutable complete-brain parameters.
Benchmark updates are discarded; no deployable checkpoint or game is created.
Numerical differences are measured without pretending CUDA is bitwise serial.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fly_touhou.connectome import EXPECTED_NEURONS,load_connectome  # noqa: E402
from fly_touhou.experiment import ExperimentLogger,file_sha256  # noqa: E402
from fly_touhou.whole_brain import EXPECTED_EDGE_RECORDS,WholeBrainState  # noqa: E402
from fly_touhou.whole_brain_batching import (masked_batch_step,masked_distillation_loss,  # noqa: E402
    pack_episode_batch,prime_episode_batch,stack_states)
from fly_touhou.whole_brain_learning import (CHECKPOINT_SCHEMA,_clip_gradients_fp64,  # noqa: E402
    _prime_state,_restore_trainable,graph_fingerprint,load_campaign_demonstrations,load_whole_brain_checkpoint)

SCHEMA="whole-brain-batch-compute-benchmark-v1"
DECISIONS=16
BPTT_DECISIONS=4


def write_new_json(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def tensor_difference(actual,reference):
    if actual.shape!=reference.shape or actual.dtype!=reference.dtype:
        raise ValueError("Serial/batch state dimensions or dtypes differ")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(reference).all()):
        raise FloatingPointError("Non-finite serial/batch state or logits")
    delta=(actual.double()-reference.double()).abs()
    return {"max_abs":float(delta.max()) if delta.numel() else 0.,
        "mean_abs":float(delta.mean()) if delta.numel() else 0.,
        "different_elements":int((actual!=reference).sum()),"elements":actual.numel(),
        "exact_equal":bool(torch.equal(actual,reference))}


def state_differences(actual:WholeBrainState,reference:WholeBrainState):
    result={}
    for field in fields(WholeBrainState):
        value,expected=getattr(actual,field.name),getattr(reference,field.name)
        if field.name=="delay_queue":
            if len(value)!=len(expected):
                raise ValueError("Serial/batch conduction-delay queue lengths differ")
            result[field.name]=[tensor_difference(a,b) for a,b in zip(value,expected)]
        else:
            result[field.name]=tensor_difference(value,expected)
    return result


def validate_state(state,model,rows,expected_counter):
    if state.membrane.shape!=(rows,model.n_neurons) or len(state.delay_queue)!=model.delay_steps:
        raise ValueError("Whole-brain benchmark state lost neurons/rows/delay steps")
    for field in fields(WholeBrainState):
        value=getattr(state,field.name)
        tensors=value if field.name=="delay_queue" else (value,)
        if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise FloatingPointError("Non-finite benchmark state: "+field.name)
    if (not bool((state.rng_keys==(int(model.rng_seed)&0x7fffffff)).all())
            or not bool((state.noise_counter==expected_counter).all())):
        raise ValueError("Batch row changed serial neural RNG key/counter semantics")
    counts=state.window_spike_counts
    if counts.shape!=(rows,model.n_neurons) or bool((counts<0).any()) or bool((counts>model.substeps).any()):
        raise ValueError("Invalid whole-brain decision spike counts")
    if not bool(((state.spikes==0)|(state.spikes==1)).all()):
        raise ValueError("Forward spikes stopped being binary")


def memory_snapshot(device):
    free,total=torch.cuda.mem_get_info(device)
    return {"allocated_bytes":torch.cuda.memory_allocated(device),
        "reserved_bytes":torch.cuda.memory_reserved(device),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
        "device_free_bytes":free,"device_total_bytes":total,
        "scope":"this PyTorch process allocator; device_free includes all processes"}


def source_rows(episodes,size):
    candidates=[episode for episode in episodes if len(episode)>=DECISIONS]
    if not candidates:
        raise ValueError("Need at least 16 real recorded decisions in a resource episode")
    rows,provenance=[],[]
    for row in range(size):
        source=candidates[row%len(candidates)]
        rows.append(replace(source,episode_id=f"benchmark-row-{row}",
            observations=source.observations[:DECISIONS].copy(),
            probabilities=source.probabilities[:DECISIONS].copy(),actions=source.actions[:DECISIONS].copy(),
            native_frames=0))
        provenance.append({"row":row,"source_episode_id":source.episode_id,"source_path":source.source_path,
            "source_decision_indices":list(range(DECISIONS)),"row_reused":row>=len(candidates)})
    return rows,provenance


@torch.no_grad()
def compare_to_serial(model,batch,rows,warmup_ms):
    """Same fixed weights/input/key/counter; record every difference, not a tolerance verdict."""
    batched=prime_episode_batch(model,batch,warmup_ms)
    serial=[_prime_state(model,torch.as_tensor(row.observations[0],device=model.device),warmup_ms) for row in rows]
    prime_counter=int(math.ceil(warmup_ms/model.neural_ms_per_decision))*model.substeps
    validate_state(batched,model,len(rows),prime_counter)
    reference=stack_states(serial)
    validate_state(reference,model,len(rows),prime_counter)
    comparison=[{"phase":"after_independent_first_frame_warmup","state":state_differences(batched,reference),
        "batch_noise_counters":batched.noise_counter.cpu().tolist(),
        "serial_noise_counters":reference.noise_counter.cpu().tolist()}]
    for step in range(DECISIONS):
        logits,batched=masked_batch_step(model,batch.observations[step],batched,batch.valid[step])
        serial_logits=[]
        for row in range(len(rows)):
            value,serial[row]=model(batch.observations[step,row],serial[row])
            serial_logits.append(value)
        reference=stack_states(serial)
        counter=prime_counter+(step+1)*model.substeps
        validate_state(batched,model,len(rows),counter)
        validate_state(reference,model,len(rows),counter)
        comparison.append({"phase":"decision","step":step,"logits":tensor_difference(logits,torch.stack(serial_logits)),
            "state":state_differences(batched,reference),
            "batch_rng_keys":batched.rng_keys.cpu().tolist(),"serial_rng_keys":reference.rng_keys.cpu().tolist(),
            "batch_noise_counters":batched.noise_counter.cpu().tolist(),
            "serial_noise_counters":reference.noise_counter.cpu().tolist()})
    return {"steps":comparison,"noise_semantics_match":True,
        "numerical_policy":"differences are reported without a tolerance rejection; nonlinear spike differences require interpretation"}


@torch.no_grad()
def forward_measurement(model,batch,warmup_ms):
    model.eval()
    torch.cuda.synchronize(model.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    allocated_before=torch.cuda.memory_allocated(model.device)
    state=prime_episode_batch(model,batch,warmup_ms)
    torch.cuda.synchronize(model.device)
    started=time.perf_counter()
    for step in range(DECISIONS):
        logits,state=masked_batch_step(model,batch.observations[step],state,batch.valid[step])
    torch.cuda.synchronize(model.device)
    elapsed=time.perf_counter()-started
    prime_counter=int(math.ceil(warmup_ms/model.neural_ms_per_decision))*model.substeps
    validate_state(state,model,len(batch.lengths),prime_counter+DECISIONS*model.substeps)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Non-finite forward logits")
    return {"elapsed_seconds":elapsed,"effective_decisions":batch.effective_decisions,
        "effective_decisions_per_second":batch.effective_decisions/elapsed,
        "allocated_before_bytes":allocated_before,"memory":memory_snapshot(model.device),
        "final_noise_counters":state.noise_counter.cpu().tolist(),
        "timing_scope":"16 forward decisions/row; excludes first-frame warmup and validation; CUDA synchronized"}


def backward_measurement(model,batch,warmup_ms,hard_weight,learning_rate,gradient_clip):
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    previous={name:value.detach().clone() for name,value in model.named_parameters()}
    torch.cuda.synchronize(model.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    allocated_before=torch.cuda.memory_allocated(model.device)
    state=prime_episode_batch(model,batch,warmup_ms)
    chunks=[]
    torch.cuda.synchronize(model.device)
    started=time.perf_counter()
    for first in range(0,DECISIONS,BPTT_DECISIONS):
        optimizer.zero_grad(set_to_none=True)
        outputs=[]
        for step in range(first,first+BPTT_DECISIONS):
            logits,state=masked_batch_step(model,batch.observations[step],state,batch.valid[step])
            outputs.append(logits)
        loss=masked_distillation_loss(torch.stack(outputs),batch.probabilities[first:first+BPTT_DECISIONS],
            batch.valid[first:first+BPTT_DECISIONS],hard_weight)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite batched compute-benchmark loss")
        loss.backward()
        norm,individual=_clip_gradients_fp64(model,gradient_clip)
        optimizer.step()
        if not all(bool(torch.isfinite(value).all()) for value in model.parameters()):
            raise FloatingPointError("Non-finite benchmark parameter update")
        chunks.append({"first_step":first,"valid_decisions":BPTT_DECISIONS*len(batch.lengths),
            "loss":float(loss.detach()),"gradient_norm_fp64":norm,"parameter_gradient_norms_fp64":individual})
        state=state.detach()
    torch.cuda.synchronize(model.device)
    elapsed=time.perf_counter()-started
    changes={name:float((value.detach().double()-previous[name].double()).square().sum().sqrt())
             for name,value in model.named_parameters()}
    prime_counter=int(math.ceil(warmup_ms/model.neural_ms_per_decision))*model.substeps
    validate_state(state,model,len(batch.lengths),prime_counter+DECISIONS*model.substeps)
    return {"elapsed_seconds":elapsed,"effective_decisions":batch.effective_decisions,
        "effective_decisions_per_second":batch.effective_decisions/elapsed,"optimizer_steps":len(chunks),
        "bptt_decisions":BPTT_DECISIONS,"chunks":chunks,"parameter_change_l2":changes,
        "input_interface_updated":changes["input_encoder.weight"]>0,
        "descending_readout_updated":changes["action_readout.weight"]>0,
        "allocated_before_bytes":allocated_before,"memory":memory_snapshot(model.device),
        "final_noise_counters":state.noise_counter.cpu().tolist(),
        "timing_scope":"forward+4-step BPTT+FP64 gradient clipping+Adam; excludes warmup and final validation"}


def run(args):
    if args.threads<1 or args.repeats<1 or not math.isfinite(args.learning_rate) or args.learning_rate<=0:
        raise ValueError("Positive threads/repeats/learning rate required")
    if not math.isfinite(args.gradient_clip) or args.gradient_clip<=0:
        raise ValueError("Positive finite gradient clipping required")
    sizes=sorted(set(args.sizes))
    if not sizes or any(size not in (1,2,4,8) for size in sizes):
        raise ValueError("Benchmark sizes are 1,2,4,8; never silently reduce a failed size")
    checkpoint=args.checkpoint.resolve(strict=True)
    checkpoint_hash=file_sha256(checkpoint)
    parent=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (file_sha256(checkpoint)!=checkpoint_hash or parent.get("schema")!=CHECKPOINT_SCHEMA
            or parent.get("neurons")!=EXPECTED_NEURONS or parent.get("edge_records")!=EXPECTED_EDGE_RECORDS
            or parent.get("fixed_weights_stored") is not False
            or parent.get("encoder",{}).get("schema")!="native-resource-observation-v2"):
        raise ValueError("Use an immutable complete-brain resource-observation wholebrain2 checkpoint")
    source=args.source_dir.resolve(strict=True)
    config={"schema":SCHEMA,"checkpoint_sha256":checkpoint_hash,"sizes":sizes,"repeats":args.repeats,
        "threads":args.threads,"decisions_per_row":DECISIONS,"bptt_decisions":BPTT_DECISIONS,
        "learning_rate":args.learning_rate,"gradient_clip":args.gradient_clip,
        "warmup_ms":parent["training_config"]["warmup_ms"],
        "hard_weight":parent["training_config"]["hard_weight"],"encoder":parent["encoder"],
        "device":args.device,"parameter_policy":"reset same immutable parent before every size/phase/repetition",
        "rng_policy":"independent serial initial_state(1) rows; same configured keys and counters",
        "numerics":"report serial/batch differences; reject count/nonfinite/noise-semantic errors, not tiny FP differences",
        "control_only":True,"intervention":"compute-only capacity benchmark; updates discarded",
        "evidence_scope":"computation/precision only; no learning-success or original-game ability claim"}
    files={"checkpoint":checkpoint,"ports":args.ports.resolve(),"benchmark_code":Path(__file__).resolve(),
        "completeness":source/"Completeness_783.csv","connectivity":source/"Connectivity_783.parquet",
        "upstream_model":source/"model.py"}
    files.update({f"trace_{i}":path.resolve(strict=True) for i,path in enumerate(args.traces)})
    for name in ("whole_brain","whole_brain_batching","whole_brain_learning","connectome","experiment",
                 "resources","resource_observation","observation","planner","state"):
        files[f"code_{name}"]=ROOT/"src"/"fly_touhou"/(name+".py")
    logger=ExperimentLogger(args.output,config,files=files)
    model=None
    with logger.phase("complete_brain_cuda_compute_benchmark"):
        torch.set_num_threads(args.threads)
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("This capacity measurement requires CUDA; no CPU or smaller-graph substitute")
        logger.record_runtime()
        dataset=load_campaign_demonstrations(args.traces)
        if dataset.encoder.specification()!=parent["encoder"]:
            raise ValueError("Checkpoint and measured resource observations differ")
        graph=load_connectome(source,max_neurons=None)
        policy=load_whole_brain_checkpoint(checkpoint,graph,args.device)
        model=policy.student
        if (model.n_neurons!=EXPECTED_NEURONS or graph.n_edges!=EXPECTED_EDGE_RECORDS
                or model.recurrent._nnz()!=EXPECTED_EDGE_RECORDS or graph_fingerprint(graph)!=parent["graph_fingerprint"]):
            raise ValueError("Full graph counts/fingerprint changed")
        coverage=logger.bind_whole_brain(graph,model,source,ports_manifest=args.ports)
        results=[]
        try:
            for size in sizes:
                row_episodes,provenance=source_rows(dataset.episodes,size)
                batch=None
                try:
                    with logger.phase(f"cuda_batch_size_{size}"):
                        _restore_trainable(model,parent)
                        model.zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                        batch=pack_episode_batch(row_episodes,model.device)
                        model.eval()
                        comparison=compare_to_serial(model,batch,row_episodes,config["warmup_ms"])
                        comparison_file=logger.directory/f"size_{size}.serial_comparison.json"
                        write_new_json(comparison_file,comparison)
                        forwards,backwards=[],[]
                        for repetition in range(args.repeats):
                            _restore_trainable(model,parent)
                            model.zero_grad(set_to_none=True)
                            forwards.append(forward_measurement(model,batch,config["warmup_ms"]))
                            _restore_trainable(model,parent)
                            model.zero_grad(set_to_none=True)
                            backwards.append(backward_measurement(model,batch,config["warmup_ms"],
                                config["hard_weight"],args.learning_rate,args.gradient_clip))
                        measured={"size":size,"status":"measured","source_rows":provenance,
                            "full_neurons":model.n_neurons,"edge_records":graph.n_edges,
                            "recurrent_nnz":model.recurrent._nnz(),"graph_fingerprint":parent["graph_fingerprint"],
                            "forward":forwards,"backward":backwards,
                            "forward_decisions_per_second_median":statistics.median(r["effective_decisions_per_second"] for r in forwards),
                            "backward_decisions_per_second_median":statistics.median(r["effective_decisions_per_second"] for r in backwards),
                            "comparison":{"path":str(comparison_file),"sha256":file_sha256(comparison_file)},
                            "all_input_updates":all(r["input_interface_updated"] for r in backwards),
                            "all_readout_updates":all(r["descending_readout_updated"] for r in backwards),
                            "deployable_checkpoint_saved":False,"learning_success_claimed":False}
                        write_new_json(logger.directory/f"size_{size}.measurements.json",measured)
                        logger.event("cuda_batch_size_measured",measured)
                        results.append(measured)
                        print("BATCH BENCHMARK",size,"forward decisions/s",measured["forward_decisions_per_second_median"],
                            "backward decisions/s",measured["backward_decisions_per_second_median"],flush=True)
                except torch.cuda.OutOfMemoryError as error:
                    failed={"size":size,"status":"out_of_memory","error":str(error),
                        "full_neurons":model.n_neurons,"edge_records":graph.n_edges,"no_size_fallback":True}
                    write_new_json(logger.directory/f"size_{size}.out_of_memory.json",failed)
                    logger.event("cuda_batch_size_capacity_failure",failed)
                    results.append(failed)
                finally:
                    batch=None
                    model.zero_grad(set_to_none=True)
                    _restore_trainable(model,parent)
                    gc.collect()
                    torch.cuda.empty_cache()
            result={"schema":SCHEMA,"config":config,"coverage":coverage,"sizes":results,
                "parent_parameters_restored":True,"deployable_checkpoint_saved":False,
                "scope":"CUDA compute/numerical evidence only; repeated rows are not independent training data"}
            write_new_json(logger.directory/"benchmark_summary.json",result)
            logger.event("cuda_batch_benchmark_finished",{"summary_sha256":file_sha256(logger.directory/"benchmark_summary.json"),
                "measured_sizes":[r["size"] for r in results if r["status"]=="measured"],
                "capacity_failed_sizes":[r["size"] for r in results if r["status"]!="measured"],
                "deployable_checkpoint_saved":False})
            logger.verify_inputs()
            return result
        finally:
            if model is not None:
                _restore_trainable(model,parent)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--traces",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-dir",type=Path,default=ROOT/"data"/"connectome")
    parser.add_argument("--ports",type=Path,default=ROOT/"data"/"connectome"/"annotations"/"selected_ports_1024_v783.json")
    parser.add_argument("--sizes",type=int,nargs="+",choices=(1,2,4,8),default=[1,2,4,8])
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--learning-rate",type=float,default=1e-5)
    parser.add_argument("--gradient-clip",type=float,default=1.)
    parser.add_argument("--device",default="cuda")
    run(parser.parse_args())


if __name__=="__main__":
    main()
