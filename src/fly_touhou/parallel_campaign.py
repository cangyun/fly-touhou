"""Isolated campaign wrappers for the verified startup-only mutex sidecar.

Environment construction is lazy. The frozen controllers still own only their
own child process, process-local input/frame hooks and private save directory.
No controller, dialogue logic or gameplay action semantics are duplicated here.
"""

from .isolated_campaign import IsolatedCampaignEnv, IsolatedResourceCampaignEnv
from .parallel_controller import (
    ParallelDialogueGameController,
    ParallelDialogueResourceGameController,
    ParallelGameController,
    ParallelResourceGameController,
)


class _ParallelCampaignMixin:
    parallel_startup = True

    def _create_controller(self):
        if self._resource_controller:
            controller_type = ParallelDialogueResourceGameController if self.dialogue_skip else ParallelResourceGameController
        else:
            controller_type = ParallelDialogueGameController if self.dialogue_skip else ParallelGameController
        return controller_type(self.executable, self.profile_path, timeout=self.timeout,
                               private_appdata=self.private_appdata,
                               worker_name=self.window_title or self.private_appdata.name)

    def _info(self, elapsed=0):
        info = super()._info(elapsed)
        audit = getattr(self.controller, 'parallel_startup_audit', None)
        info.update({'parallel_startup': True,
                     'parallel_startup_isolation': audit() if callable(audit) else None})
        return info


class ParallelCampaignEnv(_ParallelCampaignMixin, IsolatedCampaignEnv):
    """Legacy raw observations with independent startup mutex and APPDATA."""


class ParallelResourceCampaignEnv(_ParallelCampaignMixin, IsolatedResourceCampaignEnv):
    """Measured resource observations with independent startup and APPDATA."""
