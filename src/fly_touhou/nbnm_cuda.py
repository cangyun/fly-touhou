"""Static whole-brain inference graphs; no dynamics or topology changes.

Graphs exist only within one collection segment. Drop them before optimizer
updates, then capture new graphs lazily for the next frozen policy version.
"""
import time
import torch
from .spiking_engineered_control import SpikingState
from .compact_story_behavior import story_behavior_logits


class InferenceGraphs:
    def __init__(self,model,*,sample=True,movement_temperature=1.,bomb_temperature=1.):
        self.model=model;self.sample=sample;self.graphs={};self.capture_seconds=0.
        self.movement_temperature=movement_temperature;self.bomb_temperature=bomb_temperature

    @torch.no_grad()
    def _build(self,features,state):
        from .compact_story_behavior import story_behavior_logits
        started=time.perf_counter();model=self.model
        x=features.clone();s=SpikingState(*(v.clone() for v in state.tensors()))
        def choose(logits):
            if not self.sample:return logits.argmax(-1)
            logits=story_behavior_logits(logits,self.movement_temperature,self.bomb_temperature)
            return torch.distributions.Categorical(logits=logits,validate_args=False).sample()
        cached=model.core.cache_inference_parameters
        model.core.cache_inference_parameters=False
        try:
            stream=torch.cuda.Stream(device=features.device)
            stream.wait_stream(torch.cuda.current_stream(features.device))
            with torch.cuda.stream(stream):
                for _ in range(2):model.step(x,s,action_selector=choose)
            torch.cuda.current_stream(features.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=model.step(x,s,action_selector=choose)
        finally:model.core.cache_inference_parameters=cached
        self.capture_seconds+=time.perf_counter()-started
        self.graphs[len(features)]=(graph,x,s,output)

    @torch.no_grad()
    def __call__(self,features,state):
        if self.model.training:raise ValueError('Inference graph cannot run while updating the policy')
        if len(features) not in self.graphs:self._build(features,state)
        graph,x,s,output=self.graphs[len(features)]
        x.copy_(features)
        for target,value in zip(s.tensors(),state.tensors()):target.copy_(value)
        graph.replay()
        # Consumers copy committed state/telemetry before the next invocation.
        return output

    def clear(self):
        self.graphs.clear()
