"""NoTorch declaration of the prepared26-tensor joint PPO/cost objective."""
from dataclasses import asdict, dataclass
import math

SCHEMA='fly-GRU-aligned-composed-DNcost26-joint-v1'


@dataclass(frozen=True)
class ContinuousCostConfig:
    coefficient: float = 1.
    scale: float = 8.
    huber_beta: float = .1

    def __post_init__(self):
        for name in ('coefficient','scale','huber_beta'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Finite explicit continuous-cost configuration required')
        if self.coefficient<0 or self.scale<=0 or self.huber_beta<=0:
            raise ValueError('Nonnegative cost coefficient and positive conditioning parameters required')


def risk26_plan(cost_config=ContinuousCostConfig()):
    return dict(schema=SCHEMA,status='prepared_not_executed',brain_BPTT=4,neural_width=16,
        real_worlds=30,rollout_steps=256,real_rollout_rows=7680,real_rows_per_Adam=256,
        microblock_rows=64,microblocks_per_Adam=4,epochs_max=3,Adam_max_per_rollout=90,
        parameter_order='brain4+critic18+risk_head4',trainable_tensors=26,
        actor='canonical same-following-DN cost composition;T1;original stock mask',
        raw_logits='bare brain output',composed_logits='actual policy before source clamp/legal',
        values='same following DN,three channels,shared gradient.1',
        cost=asdict(cost_config),cost_scope='continuous relativeD95 cost18;notBCE/rank',
        cost_normalizer='whole256-real-row minibatch queried&postfact-valid scene count',
        teacher_targets_enter_action_readout=False,stored_DN_features_used_by_PPO=False,
        fixed_head_statistics_fitted_before_capture=True,
        default_CUDA_graph_enabled=False,CUDA_initialized=False,native_worlds_started=0,
        optimizer_updates=0,qualified_GPU=False)
