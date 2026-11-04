"""CPU-only MLP diagnostic of saved observation/teacher-label learnability.

This is a nonbiological reference, never an original-game controller or a
replacement for the complete brain. It preserves the original whole-episode
split; causal history never crosses an episode or reads future observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

# Local to this diagnostic process; existing game/GPU jobs are untouched.
os.environ["CUDA_VISIBLE_DEVICES"]=""
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from fly_touhou.condition_audit import (find_training_manifest,held_out_episodes,  # noqa: E402
                                        verify_training_trace_hashes)
from fly_touhou.experiment import ExperimentLogger,file_sha256  # noqa: E402
from fly_touhou.whole_brain_learning import (CHECKPOINT_SCHEMA,constant_mixed_baseline,  # noqa: E402
    load_campaign_demonstrations,mixed_distillation_loss)

SCHEMA="diagnostic-observation-mlp-v1"


def causal_stack(observations,history=3):
    values=np.asarray(observations,dtype=np.float32)
    if values.ndim!=2 or not len(values) or history<0:
        raise ValueError("Causal stacking needs a nonempty [time,features] episode")
    indices=np.arange(len(values))
    return np.concatenate([values[np.maximum(indices-lag,0)] for lag in range(history+1)],axis=1)


def episode_arrays(episodes,history):
    observations=np.concatenate([causal_stack(episode.observations,history) for episode in episodes])
    targets=np.concatenate([episode.probabilities for episode in episodes]).astype(np.float32)
    return torch.from_numpy(observations),torch.from_numpy(targets)


def probability_metrics(probabilities,targets,hard_weight=.5):
    predictions=np.asarray(probabilities,dtype=np.float64)
    targets=np.asarray(targets,dtype=np.float64)
    if predictions.shape!=targets.shape or not np.isfinite(predictions).all():
        raise ValueError("Invalid diagnostic probabilities/targets")
    lp=np.log(np.maximum(predictions,1e-30))
    hard=targets.argmax(-1)
    action=predictions.argmax(-1)
    soft=float(-(targets*lp).sum(-1).mean())
    hard_ce=float(-lp[np.arange(len(hard)),hard].mean())
    entropy=float(-(targets*np.log(np.maximum(targets,1e-30))).sum(-1).mean())
    classes=np.unique(hard)
    return {"decisions":len(hard),"mixed_cross_entropy":(1-hard_weight)*soft+hard_weight*hard_ce,
        "soft_cross_entropy":soft,"argmax_cross_entropy":hard_ce,"teacher_kl":soft-entropy,
        "teacher_entropy":entropy,"teacher_argmax_agreement":float((action==hard).mean()),
        "balanced_agreement_present_classes":float(np.mean([(action[hard==label]==label).mean() for label in classes])),
        "predicted_action_counts":np.bincount(action,minlength=36).tolist(),
        "teacher_action_counts":np.bincount(hard,minlength=36).tolist()}


def fixed_time_baseline(train,evaluate,hard_weight):
    mixed=[]
    for episode in train:
        labels=episode.probabilities.astype(np.float64)
        mixed.append((1-hard_weight)*labels+hard_weight*np.eye(36)[labels.argmax(-1)])
    constant=np.concatenate(mixed).mean(0)
    maximum=max(map(len,mixed))
    profile=np.stack([np.mean([values[step] for values in mixed if step<len(values)],axis=0)
                      for step in range(maximum)])
    # Fixed before this run: 21-step window, 10% shrinkage to the training
    # constant. Uses training labels only; no validation tuning or future input.
    smoothed=np.stack([profile[max(0,step-10):min(maximum,step+11)].mean(0) for step in range(maximum)])
    smoothed=.9*smoothed+.1*constant[None,:]
    output=[]
    for episode in evaluate:
        values=np.repeat(constant[None,:],len(episode),axis=0)
        values[:min(len(episode),maximum)]=smoothed[:min(len(episode),maximum)]
        output.append(values)
    return {"metrics":probability_metrics(np.concatenate(output),
                np.concatenate([episode.probabilities for episode in evaluate]),hard_weight),
        "training_time_profile_probabilities":smoothed.tolist(),"window_decisions":21,
        "constant_shrinkage":.1,"estimated_from":"training labels only; fixed control, no validation tuning",
        "boundary_policy":"validation steps beyond longest training episode use the training constant"}


class DiagnosticMLP(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,36))

    def forward(self,observations):
        return self.network(observations)


@torch.no_grad()
def score_model(model,observations,targets,hard_weight):
    model.eval()
    probabilities=F.softmax(model(observations).double(),dim=-1).cpu().numpy()
    return probability_metrics(probabilities,targets.numpy(),hard_weight),probabilities


def write_new_json(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def save_diagnostic(path,model,*,config,group,epoch,metrics):
    with Path(path).open("xb") as stream:
        torch.save({"schema":SCHEMA,"diagnostic_only":True,"not_for_game_control":True,
            "biological_model":False,"architecture":[model.network[0].in_features,128,64,36],
            "parameters":{name:value.detach().cpu().clone() for name,value in model.named_parameters()},
            "group":group,"epoch":epoch,"metrics":metrics,"config":config},stream)
    return {"path":str(Path(path).resolve()),"sha256":file_sha256(path),"bytes":Path(path).stat().st_size}


def run(args):
    if (args.epochs!=200 or args.seed!=42 or args.threads!=4 or args.batch_size<1
            or not math.isfinite(args.learning_rate) or args.learning_rate<=0):
        raise ValueError("This fixed diagnostic uses 200 epochs, seed42, threads4 and positive optimizer settings")
    if torch.cuda.is_initialized():
        raise RuntimeError("Run this diagnostic in a fresh CPU-only process")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    checkpoint=args.checkpoint.resolve(strict=True)
    digest=file_sha256(checkpoint)
    parent=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (file_sha256(checkpoint)!=digest or parent.get("schema")!=CHECKPOINT_SCHEMA
            or parent.get("encoder",{}).get("schema")!="native-resource-observation-v2"
            or parent["encoder"]["size"]!=464 or parent.get("fixed_weights_stored") is not False
            or parent["training_config"]["hard_weight"]!=.5):
        raise ValueError("Use the immutable 464-input wholebrain2 reference with mixed weight .5")
    manifest_path=find_training_manifest(checkpoint,args.training_manifest)
    training_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_training_trace_hashes(training_manifest,args.traces)
    config={"schema":SCHEMA,"purpose":"CPU observation/teacher-label learnability control only",
        "checkpoint_sha256":digest,"reference_checkpoint_epoch":parent["epoch"],"hidden_sizes":[128,64],
        "activation":"ReLU","optimizer":"Adam","learning_rate":args.learning_rate,"batch_size":args.batch_size,
        "epochs":200,"seed":42,"threads":4,"device":"cpu","hard_weight":.5,
        "groups":{"raw464":{"history":0,"input_dim":464},"causal4x464":{"history":3,"input_dim":1856}},
        "causal_order":"current,t-1,t-2,t-3; repeat first frame before episode start; never cross episodes",
        "selection":"minimum mixed validation CE; same saved whole-episode split; no independent final test",
        "control_only":True,"intervention":"ordinary MLP diagnostic, prohibited as native-game actor",
        "teacher_action_calls":0,"game_process_started":False,"deployable_brain_checkpoint_saved":False}
    files={"reference_checkpoint":checkpoint,"training_manifest":manifest_path,"diagnostic_code":Path(__file__).resolve()}
    files.update({f"trace_{i}":Path(path).resolve(strict=True) for i,path in enumerate(args.traces)})
    for name in ("whole_brain_learning","condition_audit","resource_observation","resources","observation",
                 "state","planner","experiment"):
        files[f"code_{name}"]=ROOT/"src"/"fly_touhou"/(name+".py")
    logger=ExperimentLogger(args.output,config,files=files)
    with logger.phase("cpu_observation_learnability"):
        logger.event("cpu_runtime",{"python":sys.version,"platform":platform.platform(),"torch":str(torch.__version__),
            "numpy":np.__version__,"threads":torch.get_num_threads(),"device":"cpu",
            "cuda_initialized":torch.cuda.is_initialized(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")})
        data=load_campaign_demonstrations(args.traces)
        if data.encoder.specification()!=parent["encoder"]:
            raise ValueError("Reference observations differ from trace encoder")
        train,validation=held_out_episodes(parent,data.episodes)
        logger.record_split(parent["train_episode_ids"],parent["validation_episode_ids"],name="original-full-episodes")
        baseline={"optimal_mixed_constant":constant_mixed_baseline(train,validation,.5),
                  "time_only":fixed_time_baseline(train,validation,.5),"reference_brain":parent["metrics"]["validation"]}
        write_new_json(logger.directory/"baselines.json",baseline)
        logger.event("diagnostic_baselines",{k:v if k!="time_only" else {"metrics":v["metrics"],
            "window_decisions":v["window_decisions"],"constant_shrinkage":v["constant_shrinkage"]} for k,v in baseline.items()})
        group_results={}
        for name,options in config["groups"].items():
            directory=logger.directory/name
            directory.mkdir()
            train_x,train_y=episode_arrays(train,options["history"])
            val_x,val_y=episode_arrays(validation,options["history"])
            torch.manual_seed(42)  # The same initialization seed in each fixed architecture.
            model=DiagnosticMLP(train_x.shape[1]).cpu()
            optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
            sampler=torch.Generator(device="cpu").manual_seed(42)
            best_score=float("inf")
            best=None
            history=[]
            started=time.monotonic()
            with (directory/"epochs.jsonl").open("x",encoding="utf-8") as epoch_stream:
                for epoch in range(201):
                    epoch_start=time.monotonic()
                    if epoch:
                        model.train()
                        order=torch.randperm(len(train_x),generator=sampler)
                        for first in range(0,len(order),args.batch_size):
                            index=order[first:first+args.batch_size]
                            optimizer.zero_grad(set_to_none=True)
                            loss=mixed_distillation_loss(model(train_x[index]),train_y[index],.5)
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("Non-finite diagnostic MLP loss")
                            loss.backward()
                            if not all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in model.parameters()):
                                raise FloatingPointError("Non-finite diagnostic MLP gradient")
                            optimizer.step()
                    train_metrics,_=score_model(model,train_x,train_y,.5)
                    val_metrics,val_probabilities=score_model(model,val_x,val_y,.5)
                    record={"group":name,"epoch":epoch,"training":train_metrics,"validation":val_metrics,
                        "epoch_seconds":time.monotonic()-epoch_start,"elapsed_seconds":time.monotonic()-started}
                    history.append(record)
                    epoch_stream.write(json.dumps(record,allow_nan=False)+"\n")
                    epoch_stream.flush()
                    logger.event("diagnostic_epoch",record)
                    if val_metrics["mixed_cross_entropy"]<best_score:
                        best_score=val_metrics["mixed_cross_entropy"]
                        artifact=save_diagnostic(directory/f"best_epoch_{epoch:04d}.pt",model,config=config,
                            group=name,epoch=epoch,metrics=record)
                        best={"epoch":epoch,"training":train_metrics,"validation":val_metrics,"artifact":artifact}
                        prediction=directory/f"best_epoch_{epoch:04d}.validation.json"
                        write_new_json(prediction,{"validation_episode_ids":parent["validation_episode_ids"],
                            "lengths":[len(episode) for episode in validation],"probabilities":val_probabilities.tolist(),
                            "teacher_argmax":val_y.argmax(-1).tolist()})
                        logger.event("diagnostic_best_checkpoint",{"group":name,**best,
                            "predictions":{"path":str(prediction),"sha256":file_sha256(prediction)}})
                    if epoch%20==0:
                        print("CPU MLP DIAGNOSTIC",name,epoch,"val mixed CE",val_metrics["mixed_cross_entropy"],
                            "agreement",val_metrics["teacher_argmax_agreement"],flush=True)
                last=save_diagnostic(directory/"last_epoch_0200.pt",model,config=config,group=name,epoch=200,metrics=history[-1])
            group_results[name]={"best_by_validation_mixed_ce":best,"last":history[-1],"last_artifact":last,
                "maximum_validation_agreement_observed":max(r["validation"]["teacher_argmax_agreement"] for r in history),
                "epochs_metrics":{"path":str(directory/"epochs.jsonl"),"sha256":file_sha256(directory/"epochs.jsonl")},
                "wall_seconds":time.monotonic()-started}
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU diagnostic unexpectedly initialized CUDA")
        result={"schema":SCHEMA,"config":config,"baseline":baseline,"groups":group_results,
            "train_episode_ids":parent["train_episode_ids"],"validation_episode_ids":parent["validation_episode_ids"],
            "interpretation_boundary":["An MLP advantage diagnoses available predictive information/adapter limitations, not biological superiority.",
                "Weak held-out MLP performance can reflect limited data, distribution shift, teacher memory or representation; it does not isolate one cause.",
                "Causal history excludes future observations but can expose motion and past policy/context.",
                "Repeated model selection on one held-out episode is not an independent gameplay test."],
            "not_for_original_game_control":True,"final_game_acceptance_achieved":False}
        write_new_json(logger.directory/"diagnostic_summary.json",result)
        logger.event("cpu_diagnostic_finished",{"summary_sha256":file_sha256(logger.directory/"diagnostic_summary.json"),
            "groups":group_results,"cuda_initialized":False,"not_for_original_game_control":True})
        logger.verify_inputs()
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--traces",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--training-manifest",type=Path)
    parser.add_argument("--epochs",type=int,default=200,choices=[200])
    parser.add_argument("--seed",type=int,default=42,choices=[42])
    parser.add_argument("--threads",type=int,default=4,choices=[4])
    parser.add_argument("--learning-rate",type=float,default=1e-3)
    parser.add_argument("--batch-size",type=int,default=128)
    run(parser.parse_args())


if __name__=="__main__":
    main()
