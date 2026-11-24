"""Static whole-brain inference graphs; no dynamics or topology changes.

Graphs exist only within one collection segment. Drop them before optimizer
updates, then capture new graphs lazily for the next frozen policy version.
"""
import time
from contextlib import nullcontext
import torch
from ..spiking_engineered_control import SpikingState
from ..compact_story_behavior import story_behavior_logits


class InferenceGraphs:
    def __init__(self,model,*,sample=True,movement_temperature=1.,bomb_temperature=1.,return_all_DN=False,
                 packed_binary_messages=False):
        self.model=model;self.sample=sample;self.graphs={};self.capture_seconds=0.
        self.return_all_DN=return_all_DN
        self.movement_temperature=movement_temperature;self.bomb_temperature=bomb_temperature
        self.packed_binary_messages=packed_binary_messages
        self._message_parameters=None

    @torch.no_grad()
    def _build(self,features,state,previous_actions=None,corridor_exit_bias=None):
        from ..compact_story_behavior import story_behavior_logits
        started=time.perf_counter();model=self.model
        x=features.clone();s=SpikingState(*(v.clone() for v in state.tensors()))
        def choose(logits):
            if not self.sample:return logits.argmax(-1)
            logits=story_behavior_logits(logits,self.movement_temperature,self.bomb_temperature)
            return torch.distributions.Categorical(logits=logits,validate_args=False).sample()
        extra={'return_all_DN':True} if self.return_all_DN else {}
        history=previous_actions.clone() if previous_actions is not None else None
        if history is not None:extra['previous_actions']=history
        bias=corridor_exit_bias.clone() if corridor_exit_bias is not None else None
        if bias is not None:extra['corridor_exit_bias']=bias
        cached=model.core.cache_inference_parameters
        model.core.cache_inference_parameters=False
        try:
            if self.packed_binary_messages and self._message_parameters is None:
                from ..frozen_binary_messages import FrozenBinaryMessages
                self._message_parameters=FrozenBinaryMessages(model.core)
            with self._message_parameters.capture() if self._message_parameters is not None else nullcontext():
                stream=torch.cuda.Stream(device=features.device)
                stream.wait_stream(torch.cuda.current_stream(features.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):model.step(x,s,action_selector=choose,**extra)
                torch.cuda.current_stream(features.device).wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):output=model.step(x,s,action_selector=choose,**extra)
        finally:model.core.cache_inference_parameters=cached
        self.capture_seconds+=time.perf_counter()-started
        self.graphs[len(features),history is not None,bias is not None]=(graph,x,s,history,bias,output)

    @torch.no_grad()
    def __call__(self,features,state,previous_actions=None,corridor_exit_bias=None):
        if self.model.training:raise ValueError('Inference graph cannot run while updating the policy')
        key=(len(features),previous_actions is not None,corridor_exit_bias is not None)
        if key not in self.graphs:self._build(features,state,previous_actions,corridor_exit_bias)
        graph,x,s,history,bias,output=self.graphs[key]
        x.copy_(features)
        if history is not None:history.copy_(previous_actions)
        if bias is not None:bias.copy_(corridor_exit_bias)
        for target,value in zip(s.tensors(),state.tensors()):target.copy_(value)
        graph.replay()
        # Consumers copy committed state/telemetry before the next invocation.
        return output

    def clear(self):
        self.graphs.clear()
        self._message_parameters=None

    @torch.no_grad()
    def frozen_batch(self, features, state, batch_size, previous_actions=None, corridor_exit_bias=None):
        """Reuse one CUDA graph for differently sized batches of independent frozen worlds."""
        if self.sample or batch_size<1:
            raise ValueError('Fixed capture batches require frozen greedy inference')
        if len(features)>batch_size:
            predictions=[self.frozen_batch(features[t:t+batch_size],
                SpikingState(*(v[t:t+batch_size] for v in state.tensors())),batch_size,
                previous_actions[t:t+batch_size] if previous_actions is not None else None,
                corridor_exit_bias[t:t+batch_size] if corridor_exit_bias is not None else None)
                for t in range(0,len(features),batch_size)]
            return ({k:torch.cat([p[0][k] for p in predictions]) for k in predictions[0][0]},
                SpikingState.pack([p[1] for p in predictions]),torch.cat([p[2] for p in predictions]))
        count=len(features)
        if count<batch_size:
            features=torch.cat((features,features[-1:].expand(batch_size-count,-1)))
            state=SpikingState(*(torch.cat((v,v[-1:].expand(batch_size-count,-1))) for v in state.tensors()))
            if previous_actions is not None:
                previous_actions=torch.cat((previous_actions,previous_actions.new_full((batch_size-count,),-1)))
            if corridor_exit_bias is not None:
                corridor_exit_bias=torch.cat((corridor_exit_bias,corridor_exit_bias.new_zeros((batch_size-count,18))))
        options = {}
        if previous_actions is not None:options['previous_actions']=previous_actions
        if corridor_exit_bias is not None:options['corridor_exit_bias']=corridor_exit_bias
        out,following,dn=self(features,state,**options)
        # Graph outputs are static scratch buffers. Own all real rows before
        # the next chunk or ready batch replays the same graph.
        return ({k:v[:count].clone() for k,v in out.items()},
            SpikingState(*(v[:count].clone() for v in following.tensors())),dn[:count].clone())
