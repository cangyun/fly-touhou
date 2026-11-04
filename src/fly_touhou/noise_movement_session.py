"""Training-only input-noise randomization at complete original-game resets."""
import math

import torch

from .movement_lesson_session import MovementLessonSession
from .whole_brain_ppo import _native_observation


class NoiseMovementLessonSession(MovementLessonSession):
    def _reset(self):
        self.policy.reset()
        value,info=self.env.reset(seed=self.episodes_started)
        self.snapshot=info.get('snapshot',self.env.snapshot)
        self.observation=_native_observation(value,self.policy.encoder,self.snapshot,info)
        if self.snapshot.character!=0 or self.snapshot.subshot!=0:raise ValueError('Original ReimuA required')
        model=self.policy.student;key=(10001+self.episodes_started*9973)&0x7fffffff
        state=model.initial_state(1,rng_seed=key)
        with torch.no_grad():
            tensor=torch.as_tensor(self.observation,device=model.device).unsqueeze(0)
            for _ in range(math.ceil(self.policy.warmup_ms/model.neural_ms_per_decision)):
                _,state=model(tensor,state)
        self.policy.state=state.detach()
        if self.feedback:self.feedback.reset()
        self.episode_id=f'{self.run_id}:{self.episodes_started}';self.episodes_started+=1
        self.episode_frames=self.vulnerable_frames=self.episode_steps=self.episode_teacher_actions=0
        self.logger.record_native_context(self.snapshot,info)
        self.logger.event('movement_lesson_episode_started',{'episode_id':self.episode_id,'pid':self.env.pid,
            'difficulty':self.snapshot.difficulty,'action_domain':list(range(18)),'first_hit':True,
            'training_action_source':'student','initial_neural_key':key,'warmup_counter':int(state.noise_counter[0]),
            'desktop_input':False,'game_rng_seed_applied':False,'actual_game_rng':self.snapshot.extra.get('rng'),
            'neural_state_reset':'new original game only','final_acceptance':False})
