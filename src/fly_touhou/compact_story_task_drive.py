"""Action-conditioned task progress sent through existing visual VP ports.

These are sensory cues for the shared connectome, not policy scores or an
action selector. The129 observation and47 per-candidate input remain intact.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA='compact-story-task-progress-opponent-drive-v1'
TASK_PROGRESS_CUE_NAMES=('attack_horizontal_progress','power_item_progress','valuable_item_progress')


def task_progress_cues(neural47):
    """Three signed four-native-frame progress cues from the existing input.

    Globals18 occupy8:26, TASK_NAMES18 occupy26:44 and proposed vx/4.5,
    vy/4.5,focus occupy44:47. Focus is already reflected in velocity. Target
    positions are held at their observed values; no future/teacher labels
    enter the code. Positive means this candidate closes the observed gap.
    """
    x=neural47
    if x.shape[-1]!=47:raise ValueError('Existing47 per-candidate neural features are required')
    globals=x[...,8:26];task=x[...,26:44]
    position=torch.stack((globals[...,0]*368-184,globals[...,1]*400+32),-1)
    proposed=position+18*x[...,44:46]
    end=torch.stack((proposed[...,0].clamp(-184,184),proposed[...,1].clamp(32,432)),-1)
    displacement=end-position
    target_dx=task[...,3]*368
    attack=(target_dx.abs()-(target_dx-displacement[...,0]).abs())/18
    attack=attack*(task[...,2]>.5).to(x.dtype)
    power=torch.stack((task[...,8]*368,task[...,9]*400),-1)
    power_progress=(torch.linalg.vector_norm(power,dim=-1)
                    -torch.linalg.vector_norm(power-displacement,dim=-1))/18
    power_progress=power_progress*(1-globals[...,2]).clamp(0,1)
    power_present=(task[...,1]>.5)&(task[...,7]>0)
    power_progress=power_progress*power_present.to(x.dtype)
    valuable=torch.stack((task[...,13]*368,task[...,14]*400),-1)
    valuable_progress=(torch.linalg.vector_norm(valuable,dim=-1)
                       -torch.linalg.vector_norm(valuable-displacement,dim=-1))/18
    valuable_progress=valuable_progress*(task[...,12]>.5).to(x.dtype)
    cues=torch.stack((attack,power_progress,valuable_progress),-1).clamp(-1,1)
    live=(globals[...,14]>.5)&(globals[...,17]<.5)
    return torch.where(live[...,None],cues,torch.zeros_like(cues))


class TaskProgressPortDrive(nn.Module):
    """Three opponent populations on existing feature6/7 VP ports only.

    Every selected port keeps its original opponent polarity and gets one
    task cue. Positive/negative populations are allocated evenly over the
    three cues. Per-port positive gains are trainable; reserved risk0..2 and
    confidence3..5 ports receive exactly zero task current, even after updates.
    No original encoder parameter, port identity or connectome edge changes.
    """
    def __init__(self,feature,polarity,*,initial_amplitude=.025):
        super().__init__()
        feature=torch.as_tensor(feature,dtype=torch.long).detach()
        polarity=torch.as_tensor(polarity,dtype=torch.float32,device=feature.device).detach()
        if feature.ndim!=1 or polarity.shape!=feature.shape:
            raise ValueError('Paired existing VP feature and polarity vectors are required')
        if not math.isfinite(initial_amplitude) or initial_amplitude<=0:
            raise ValueError('Positive initial task current amplitude is required')
        self.n_ports=len(feature);self.initial_amplitude=float(initial_amplitude)
        eligible=(feature==6)|(feature==7)
        indices=eligible.nonzero().flatten()
        selected_polarity=polarity.index_select(0,indices).clone()
        assignment=torch.empty(len(indices),dtype=torch.long,device=feature.device)
        for sign in (1.,-1.):
            population=(selected_polarity==sign).nonzero().flatten()
            assignment[population]=torch.arange(len(population),device=feature.device)%3
        if bool(((selected_polarity!=1)&(selected_polarity!=-1)).any()):
            raise ValueError('Existing opponent ports must have polarity+1 or-1')
        self.register_buffer('port_indices',indices.clone())
        self.register_buffer('task_index',assignment)
        self.register_buffer('opponent_polarity',selected_polarity)
        self.gain_logits=nn.Parameter(torch.full((len(indices),),math.log(math.expm1(initial_amplitude)),
            dtype=torch.float32,device=feature.device))

    def drive_from_cues(self,cues):
        if cues.shape[-1]!=3:raise ValueError('Three task progress cues are required')
        source=cues.index_select(-1,self.task_index)*self.opponent_polarity
        current=source.relu()*F.softplus(self.gain_logits)
        return cues.new_zeros((*cues.shape[:-1],self.n_ports)).index_copy(-1,self.port_indices,current)

    def forward(self,neural47):
        return self.drive_from_cues(task_progress_cues(neural47))

    def specification(self):
        return dict(schema=SCHEMA,cues=TASK_PROGRESS_CUE_NAMES,selected_original_features=[6,7],
            selected_VP_ports=len(self.port_indices),total_VP_ports=self.n_ports,
            initial_amplitude=self.initial_amplitude,trainable_per_port_positive_gain=True,
            phase='state1 only,including protection;state4 and ignored phases zero',
            policy_action_override=False,brain_edges_changed=False)
