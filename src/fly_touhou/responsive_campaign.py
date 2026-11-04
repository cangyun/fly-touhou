"""Independent original campaigns with a responsive native checkpoint wait."""
from .parallel_campaign import ParallelCampaignEnv, ParallelResourceCampaignEnv
from .responsive_controller import (ResponsiveParallelGameController,
    ResponsiveParallelResourceGameController, ResponsiveParallelDialogueGameController,
    ResponsiveParallelDialogueResourceGameController)


class _ResponsiveCampaignMixin:
    responsive_checkpoint_wait = True

    def set_runtime_phase(self, phase, details):
        """Label this worker only when its training phase actually changes."""
        if self.controller is None or not self.window_title:
            return
        labels = {'collecting': '正在采集游戏', 'updating': '正在GPU更新 · 游戏停帧',
                  'checkpointed': '已保存进度', 'finished': '本轮训练完成', 'failed': '训练异常 · 已留档'}
        if phase not in labels or getattr(self, '_last_runtime_label_phase', None) == phase:
            return
        from .dialogue_controller import label_owned_game_windows
        self._last_runtime_label_phase = phase
        label_owned_game_windows(self.controller,
            f'{self.window_title} · {labels[phase]} · {details.get("native_steps", 0)}步')

    def _create_controller(self):
        if self._resource_controller:
            kind = ResponsiveParallelDialogueResourceGameController if self.dialogue_skip else ResponsiveParallelResourceGameController
        else:
            kind = ResponsiveParallelDialogueGameController if self.dialogue_skip else ResponsiveParallelGameController
        return kind(self.executable, self.profile_path, timeout=self.timeout,
            private_appdata=self.private_appdata, worker_name=self.window_title or self.private_appdata.name)


class ResponsiveParallelCampaignEnv(_ResponsiveCampaignMixin, ParallelCampaignEnv):
    pass


class ResponsiveParallelResourceCampaignEnv(_ResponsiveCampaignMixin, ParallelResourceCampaignEnv):
    pass
