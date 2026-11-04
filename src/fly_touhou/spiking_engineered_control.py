"""Trainable signed full-connectome LIF controller with engineered sensation.

This is an engineered discrete-time SNN, not a reconstruction of Shiu's
millisecond parameters. Messages contain hard binary spikes. Candidate
geometry is an observation-only sensory input, never an action/risk shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .continuous_connectome import ContinuousConnectomeCore

SCHEMA = 'full-connectome-engineered-sensation-trainable-LIF-v2'
ENGINEERING_FIELDS = ('clearance4_div32', 'clearance12_div32', 'clearance24_div32',
    'known4', 'known12', 'known24', 'three_continuation_safe_fraction', 'endpoint_wall_margin_div64')


def engineering_features(features, objects, maps):
    """Current observation only: no action, outcome, teacher choice or RNG.

Escape is explicitly a three-continuation kinematic proxy, not measured free
area or an oracle about the future game script. Unknown forecasts have masks.
"""
    from .thrl_movement_teacher import MovementSafetyTeacher, normal_unprotected
    f, o, m = map(np.asarray, (features, objects, maps))
    shape = f.shape[:-1]
    f, o, m = f.reshape(-1, 25), o.reshape(-1, 96, 29), m.reshape(-1, 16, 92, 96)
    result = np.zeros((len(f), 18, 8), np.float32)
    teacher = MovementSafetyTeacher(); velocity = teacher.velocities
    # Nearby velocities include slow/fast alternatives; source19 mapping stays intact.
    neighbours = np.argsort(((velocity[:, None]-velocity[None])**2).sum(-1), axis=1, kind='stable')[:, :3]
    for i in range(len(f)):
        if not normal_unprotected(f[i]):
            continue
        known = [teacher._uncertainty(f[i], o[i], m[i], horizon=h) is None for h in (4, 12, 24)]
        if not known[0]:
            continue
        origin = np.array([f[i, 0]*368-184, f[i, 1]*400+32])
        starts = np.broadcast_to(origin, (18, 2)).copy()
        for j, h in enumerate((4, 12, 24)):
            if known[j]:
                margins = teacher.clearances(f[i], o[i], starts, velocity, duration=h)
                result[i, :, j] = np.clip(margins, -64, 64)/32
            result[i, :, 3+j] = known[j]
        ends = np.clip(starts+velocity*4, [-184, 32], [184, 432])
        wall = np.minimum(ends-[-184, 32], [184, 432]-ends).min(-1)
        result[i, :, 7] = np.clip(wall, 0, 64)/64
        if known[2]:
            continued = teacher.clearances(f[i], o[i], np.repeat(ends, 3, axis=0),
                velocity[neighbours].reshape(-1, 2), duration=12, initial_time=4).reshape(18, 3)
            first = result[i, :, 0]*32
            result[i, :, 6] = (np.minimum(first[:, None], continued) > 1).mean(-1)
    return result.reshape(*shape, 144)


class HardSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return .1*gradient/(1.+x.abs()).square()


@dataclass(frozen=True)
class SpikingState:
    membrane: torch.Tensor
    synaptic: torch.Tensor
    spikes: torch.Tensor
    rate: torch.Tensor

    def tensors(self): return self.membrane, self.synaptic, self.spikes, self.rate
    def detach(self): return type(self)(*(x.detach() for x in self.tensors()))
    def to(self, device): return type(self)(*(x.to(device) for x in self.tensors()))
    def reset(self, mask):
        return type(self)(*(torch.where(mask[:, None], torch.zeros_like(x), x) for x in self.tensors()))
    def row(self, i): return type(self)(*(x[i:i+1].clone() for x in self.tensors()))
    @classmethod
    def pack(cls, states): return cls(*(torch.cat([s.tensors()[j] for s in states]) for j in range(4)))


class SpikingConnectomeCore(ContinuousConnectomeCore):
    """Reuse verified graph/CSR mechanics, replacing all continuous dynamics."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = self.n_neurons
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.initialization_seed+79)
            self.geometry_encoder = nn.Sequential(nn.LayerNorm(144), nn.Linear(144, len(self.input_indices), bias=False))
            self.visual_norm = nn.LayerNorm(384)
        self.synaptic_logits = nn.Parameter(torch.zeros(n))
        self.threshold_logits = nn.Parameter(torch.full((n,), math.log(math.expm1(.25))))
        with torch.no_grad():
            self.alpha_logits.fill_(math.log(.2/.8))
            self.neuron_bias.fill_(.20)
            self.message_gain_logit.fill_(math.log(math.expm1(4.-self.epsilon)))

    def internal_parameters(self):
        return (*super().internal_parameters(), *(getattr(self, k) for k in ('synaptic_logits', 'threshold_logits') if hasattr(self, k)))

    def initial_state(self, batch, *, device=None):
        device = self.device if device is None else device
        return SpikingState(*(torch.zeros(batch, self.n_neurons, device=device) for _ in range(4)))

    def forward(self, encoded, state):
        if not isinstance(state, SpikingState) or encoded.shape != (len(state.membrane), 528):
            raise ValueError('Explicit LIF state and separately encoded visual384+candidate144 required')
        drive_ports = F.softplus(self.input_encoder(self.visual_norm(encoded[:, :384]))
                                 + self.geometry_encoder(encoded[:, 384:]))
        drive = torch.zeros_like(state.membrane).index_add(1, self.input_indices, drive_ports)
        values = self.effective_edge_values(); alpha = self.alpha()
        syn_alpha = .05+.9*self.synaptic_logits.sigmoid()
        threshold = .05+F.softplus(self.threshold_logits)
        membrane, synaptic, spikes, rate = state.tensors()
        for _ in range(self.K):
            # A one-tick conduction delay and one-tick refractory interval.
            message = self.sparse(spikes, values)
            synaptic = (1-syn_alpha)*synaptic+syn_alpha*message
            candidate = (1-alpha)*membrane+alpha*(drive+self.neuron_bias+self.message_gain()*synaptic)
            available = (spikes.detach() == 0).to(candidate.dtype)
            spikes = HardSpike.apply((candidate-threshold)/threshold)*available
            membrane = candidate-spikes.detach()*threshold
            rate = .75*rate+.25*spikes
        following = SpikingState(membrane, synaptic, spikes, rate)
        return self.read_DN(following), following

    def read_DN(self, state):
        # Both channels are actual DN states; no raw observation enters any head.
        return torch.cat((state.rate.index_select(1, self.output_indices),
                          state.membrane.index_select(1, self.output_indices)), -1)

    def config(self):
        return dict(schema=SCHEMA, n_neurons=self.n_neurons, n_edges=self.n_edges, K=self.K,
            graph=self.graph_provenance, input_ports=len(self.input_indices), output_ports=len(self.output_indices),
            dynamics='discrete LIF;hard binary messages;one-tick delay/refractory;soft voltage reset',
            timing='engineered neural ticks,not calibrated biological milliseconds',
            surrogate='0.1/(1+abs(normalized_voltage_gap))^2;forward remains binary',
            DN_readout='actual filtered spikes and subthreshold membrane,2x1299',
            trainable='all signed-preserving positive edge gains,membrane/synaptic leaks,thresholds,biases,inputs',
            seed=self.initialization_seed, sparse=self.sparse.config(), source_Shiu_dynamics_claim=False)


class SpikingEngineeredPolicy(nn.Module):
    def __init__(self, core, sensory, *, seed=71931):
        super().__init__()
        from .continuous_guided_dagger import LayerNormalizedRisk
        from .fly_gru_aligned_model import ThreeDNValueCritics
        self.core, self.sensory_adapter = core, sensory
        width = 2*len(core.output_indices)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actor = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,256), nn.SiLU(), nn.Linear(256,19))
            nn.init.normal_(self.actor[-1].weight, std=.001); nn.init.zeros_(self.actor[-1].bias)
            self.risk_head = LayerNormalizedRisk(width)
            self.critic = ThreeDNValueCritics(width)
        self.seed = seed

    @property
    def device(self): return self.core.device
    def initial_state(self, batch): return self.core.initial_state(batch)
    def encode(self, f, m, o, engineering):
        return torch.cat((self.sensory_adapter.encode_raw(f,m,o), engineering), -1)
    def compose(self, raw, cost, f):
        from .fly_dn_cost_native_policy import compose_from_active_mask, normal_unprotected_from_features
        return compose_from_active_mask(raw,cost,normal_unprotected_from_features(f),beta=8.,check_finite=False)

    def parameter_payload(self, base_path, base_sha, costs):
        return dict(schema=SCHEMA, base_source=str(base_path), base_sha256=base_sha,
            seed=self.seed, core=self.core.config(), engineering_fields=ENGINEERING_FIELDS,
            state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()}, costs=dict(costs),
            raw_risk_or_teacher_action_bypass=False, actual_binary_spikes=True)


def build_spiking_model(source, connectome, *, seed=71931, device='cuda:0'):
    from .continuous_guided_dagger import build_guided_model
    base = build_guided_model(source, connectome, device='cpu')
    c = base.core
    core = SpikingConnectomeCore(c.graph, c.input_indices.cpu().tolist(), c.output_indices.cpu().tolist(),
        K=8, trainable_brain=True, port_provenance=c.graph_provenance['port_provenance'],
        edge_chunk_size=262144, seed=seed, require_full_graph=True)
    sensory = base.sensory_adapter
    del base
    model = SpikingEngineeredPolicy(core,sensory,seed=seed).to(device)
    return model.train().requires_grad_(True)


def forward_sequence(model, data, engineering, state, *, use_checkpoint=True):
    from .continuous_object_avoidance import no_bomb_policy_logits
    z = model.encode(data['features'], data['maps'], data['objects'], engineering)
    raw_rows=[]; margin_rows=[]; logits_rows=[]; DN_rows=[]
    for t in range(len(z)):
        state = state.reset(data['episode_starts'][t])
        def step(encoded, *tensors):
            DN, nxt = model.core(encoded, SpikingState(*tensors))
            return DN, *nxt.tensors()
        outputs = checkpoint(step,z[t],*state.tensors(),use_reentrant=False) if use_checkpoint else step(z[t],*state.tensors())
        DN, *tensors = outputs; state = SpikingState(*tensors)
        raw = model.actor(DN); scores = model.risk_head.layers(DN)
        logits = no_bomb_policy_logits(model.compose(raw,F.softplus(scores),data['features'][t]), data['features'][t])
        raw_rows.append(raw); margin_rows.append(1.-.5*scores); logits_rows.append(logits); DN_rows.append(DN)
    return dict(raw_logits=torch.stack(raw_rows), predicted_margins=torch.stack(margin_rows),
        composed_logits=torch.stack(logits_rows), actual_logits=torch.stack(logits_rows),
        DN=torch.stack(DN_rows), following_state=state)


class SpikingNativePolicy(nn.Module):
    """Native ready-vector adapter; exact complete LIF state owned per game."""
    family='deep'; movement_readout='composed'; gru_hidden_size=1
    def __init__(self, model):
        super().__init__(); self.model=model; self.workers={}; self.context=None; self.last_policy_rows=None
    def set_native_episode_context(self, context): self.context=tuple(context)
    def actual_population_rows(self):
        ix=self.model.core.input_indices
        return [self.workers[w][1].rate.index_select(1,ix)[0].cpu().numpy() for w,g in self.context]
    def actual_complete_activity_rows(self):
        return [self.workers[w][1].rate[0].cpu().numpy() for w,g in self.context]
    def forward_step(self, features, maps, cursor, *, objects):
        from .continuous_object_avoidance import no_bomb_policy_logits
        f,m,o=(torch.as_tensor(x,device=self.model.device,dtype=torch.float32) for x in (features,maps,objects))
        eng=torch.from_numpy(engineering_features(f.cpu().numpy(),o.cpu().numpy(),m.cpu().numpy())).to(f.device)
        states=[]
        for (w,g),token in zip(self.context,cursor.reshape(-1).tolist()):
            if token==0:
                if w in self.workers and self.workers[w][0]==g: raise ValueError('Cannot reset an existing LIF game')
                states.append(self.model.initial_state(1))
            elif token==g+1 and w in self.workers and self.workers[w][0]==g: states.append(self.workers[w][1])
            else: raise ValueError('Stale native game/cursor')
        z=self.model.encode(f,m,o,eng); DN,state=self.model.core(z,SpikingState.pack(states))
        raw=self.model.actor(DN); scores=self.model.risk_head.layers(DN); cost=F.softplus(scores)
        combined=self.model.compose(raw,cost,f); actual=no_bomb_policy_logits(combined,f)
        for j,(w,g) in enumerate(self.context): self.workers[w]=(g,state.row(j))
        self.last_policy_rows=dict(raw_logits=raw.detach(),composed_logits=combined.detach(),predicted_costs=cost.detach(),
            noBomb_policy_logits=actual.detach(),DN_activation=DN.detach(),risk_latent_scores=scores.detach(),
            DN_membrane=state.membrane[:,self.model.core.output_indices].detach(),
            DN_spikes=state.spikes[:,self.model.core.output_indices].detach(),
            VP_membrane=state.membrane[:,self.model.core.input_indices].detach(),
            VP_spikes=state.spikes[:,self.model.core.input_indices].detach())
        return actual,None,cursor.new_tensor([g+1 for w,g in self.context]).reshape(1,len(states),1)
