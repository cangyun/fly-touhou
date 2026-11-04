"""Ready-driven recurrent rollout, same per-world PPO/GAE sequence contract.

One frozen behavior-policy version per complete T*N rollout. Reset waits do
not stop other worlds. No environment runs while PPO updates its parameters.
Torch is passed in by the learner; worker imports stay free of CUDA/Torch.
"""
import numpy as np
from .thrl_policy_actions import policy_logits, validated_action_temperature


def collect_ready(torch,model,vector,data,hidden,starts,infos,*,logit_clamp,check_stop,on_step,
                  reset_options=None,on_reset=None,history=None,movement_tracker=None,
                  shared_observations=False,action_temperature=1.):
    from torch.distributions import Categorical
    from .thrl_inputs import forward_observation
    action_temperature=validated_action_temperature(action_temperature)
    if type(shared_observations) is not bool:
        raise ValueError('Explicit boolean shared-observation mode required')
    steps,count=data['actions'].shape
    counts=np.zeros(count,np.int64)
    hidden=hidden.clone();starts=starts.clone()
    current=(vector.observation if shared_observations else
        {key:value.copy() for key,value in vector.observation.items()})
    if shared_observations and any(data[key].device.type!='cpu' or data[key].dtype!=torch.float32 for key in current):
        raise ValueError('Borrowed observations require original FP32 CPU rollout storage')
    batch_counts={};received_rows=0

    def dispatch(indices):
        if not indices:return
        if shared_observations and any(i in vector._inflight for i in indices):
            raise RuntimeError('Cannot read a borrowed observation while its world is in flight')
        batch_counts[len(indices)]=batch_counts.get(len(indices),0)+1
        # One device transfer per ready batch instead of one per worker.
        before_hidden=hidden[0,indices].detach().cpu()
        for j,i in enumerate(indices):
            t=int(counts[i])
            for key in current:
                if shared_observations:
                    np.copyto(data[key][t,i].numpy(),current[key][i],casting='no')
                else:data[key][t,i].copy_(torch.from_numpy(current[key][i]))
            data['hidden_states'][t,i].copy_(before_hidden[j])
            data['episode_starts'][t,i].copy_(starts[i])
            if 'teacher_targets' in data:
                from .thrl_online_reference import current_targets
                data['teacher_targets'][t,i].copy_(torch.from_numpy(current_targets(infos[i],
                    input_schema=getattr(model,'input_schema',None))))
                if 'teacher_weights' in data:
                    from .thrl_online_reference import current_target_weight
                    data['teacher_weights'][t,i]=current_target_weight(infos[i],input_schema=model.input_schema)
            if history is not None:
                history.append(i,{key:value[i] for key,value in current.items()})
        with torch.no_grad():
            logits,values,next_hidden=forward_observation(torch,model,
                {key:value[indices] for key,value in current.items()},hidden[:,indices])
            distribution=Categorical(logits=policy_logits(logits,logit_clamp,
                action_temperature=action_temperature))
            actions=distribution.sample();log_probs=distribution.log_prob(actions)
        cpu_actions=actions.cpu();cpu_log_probs=log_probs.cpu();cpu_values=values.cpu()
        for j,i in enumerate(indices):
            t=int(counts[i]);data['actions'][t,i].copy_(cpu_actions[j])
            data['old_log_probs'][t,i].copy_(cpu_log_probs[j]);data['values'][t,i].copy_(cpu_values[j])
        hidden[:,indices]=next_hidden
        vector.submit_at(indices,'step',cpu_actions.tolist())

    dispatch(list(range(count)))
    while vector._inflight:
        check_stop()
        ready=(vector.receive_ready(copy_observations=False) if shared_observations else vector.receive_ready())
        received_rows+=len(ready);next_indices=[]
        for message in ready:
            i=message['index'];observation=message['observation'];before_info=infos[i];infos[i]=message['info']
            for key,value in observation.items():
                if shared_observations:
                    if not np.shares_memory(current[key][i],value):
                        raise RuntimeError('Borrowed acknowledgement differs from the shared observation row')
                else:current[key][i]=value
            if message['operation']=='reset':
                starts[i]=True
                if history is not None:
                    history.reset(i,message.get('burn_in'))
                    hidden[:,i:i+1]=history.refresh(torch,model,[i])
                if on_reset is not None:on_reset(i,message['info'])
                if counts[i]<steps:next_indices.append(i)
                continue
            t=int(counts[i]);terminated,truncated=bool(message['terminated']),bool(message['truncated'])
            if 'movement_outcomes' in data:
                from .thrl_movement_learning import transition_outcome
                outcome=transition_outcome(data['features'][t,i],before_info,
                    int(data['actions'][t,i]),observation['features'],message['info'])
                data['movement_outcomes'][t,i]=outcome
                if movement_tracker is not None:
                    movement_tracker.record(i,data['features'][t,i],before_info,int(data['actions'][t,i]),
                        observation['features'],message['info'],outcome)
            data['rewards'][t,i].copy_(torch.as_tensor(message['info']['reward_vector']))
            data['terminated'][t,i]=terminated;data['truncated'][t,i]=truncated
            data['truncation_values'][t,i].zero_()
            if truncated and not terminated:
                with torch.no_grad():
                    _,values,_=forward_observation(torch,model,
                        {key:(value[None].copy() if shared_observations else value[None])
                            for key,value in observation.items()},hidden[:,i:i+1])
                data['truncation_values'][t,i].copy_(values[0].cpu())
            counts[i]+=1
            on_step(i,message['info'],terminated,truncated)
            starts[i]=terminated or truncated
            if terminated or truncated:
                hidden[:,i]=0
                # Finish the reset even at a rollout boundary, so the old-policy
                # bootstrap/current observation cannot refer to a retired world.
                if reset_options is None:vector.submit_at([i],'reset')
                else:vector.submit_at([i],'reset',options=[reset_options(i)])
            elif counts[i]<steps:next_indices.append(i)
        dispatch(next_indices)
    if not np.all(counts==steps):raise RuntimeError('Incomplete ready-driven on-policy rollout')
    vector.last_collection_transport=dict(shared_observations=shared_observations,
        action_temperature=action_temperature,action_selection='categorical_sample',
        policy_batch_size_histogram=batch_counts,policy_batches=sum(batch_counts.values()),
        received_rows=received_rows,actual_policy_decisions=int(counts.sum()),
        mandatory_rollout_and_history_copies_preserved=True)
    # The outer learner receives an owned frozen boundary snapshot, including
    # completed resets. No shared row can escape to PPO or bootstrap consumers.
    if shared_observations:current={key:value.copy() for key,value in current.items()}
    return current,hidden,starts,infos
