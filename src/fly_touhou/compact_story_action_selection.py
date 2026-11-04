"""Explicit frozen-policy decoders; no game geometry or teacher override.

PPO samples the joint source19 distribution. Joint mode and family mode are
two different deterministic deployment rules and must be reported separately.
"""
import torch


MODES=('joint_mode','family_mode','sample')


class StoryActionSelector:
    def __init__(self,mode,worlds,*,seed,movement_temperature=1.,bomb_temperature=1.):
        if mode not in MODES:raise ValueError(f'Unknown action selection: {mode}')
        from .compact_story_behavior import sampling_temperature
        self.mode=mode
        self.movement_temperature=sampling_temperature(movement_temperature)
        self.bomb_temperature=sampling_temperature(bomb_temperature)
        # Each original game owns its policy RNG stream. Receive/batch order
        # must not change which random draw that game receives.
        self.generators=[torch.Generator(device='cpu').manual_seed(
            (int(seed)+0x9E3779B97F4A7C15*(i+1))%((1<<63)-1)) for i in range(worlds)]

    def __call__(self,logits,worlds):
        if self.mode=='joint_mode':return logits.argmax(-1)
        if self.mode=='family_mode':
            # Compare the total probability of moving with that of Bomb,
            # then choose the most probable movement conditional on moving.
            # This is not the joint distribution's mode.
            move=logits[:,:18]
            return torch.where(logits[:,18]>torch.logsumexp(move,-1),18,move.argmax(-1))
        from .compact_story_behavior import story_behavior_logits
        behavior=story_behavior_logits(logits.detach().float(),self.movement_temperature,self.bomb_temperature)
        probability=behavior.softmax(-1).cpu()
        chosen=[torch.multinomial(row,1,generator=self.generators[i]).item()
                for row,i in zip(probability,worlds)]
        return torch.tensor(chosen,device=logits.device,dtype=torch.long)
