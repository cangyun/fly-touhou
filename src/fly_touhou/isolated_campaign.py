"""Explicit per-process save roots for independent original-game campaigns.

Construction and controller factories never start or attach a process. Each
worker must supply its own private APPDATA directory; the existing controller
continues to own process-local input, native frames, and resource inventories.
"""
from __future__ import annotations

from pathlib import Path

from .campaign import CampaignTouhouEnv
from .dialogue_controller import (
    DialogueGameController,
    DialogueResourceGameController,
    label_owned_game_windows,
)
from .frida_controller import OriginalGameController
from .resource_environment import ResourceCampaignEnv, ResourceGameController


class _IsolatedCampaignMixin:
    _resource_controller = False

    def __init__(self, *args, private_appdata, dialogue_skip=False,
                 window_title=None, hide_window=False, **kwargs):
        if private_appdata is None or not str(private_appdata).strip():
            raise ValueError('An explicit private_appdata directory is required for each worker')
        private_path = Path(private_appdata).resolve()
        if len(str(private_path)) > 210 or not str(private_path).isascii():
            raise ValueError('private_appdata must resolve to an ASCII path shorter than 211 characters')
        if not isinstance(dialogue_skip, bool):
            raise TypeError('dialogue_skip must be a bool')
        self.private_appdata = private_path
        self.dialogue_skip = dialogue_skip
        self.window_title = str(window_title) if window_title else None
        if not isinstance(hide_window, bool):
            raise TypeError('hide_window must be an explicit bool')
        self.hide_window = hide_window
        self._hidden_window_pid = None
        self.window_label = None
        self._window_label_pid = None
        super().__init__(*args, **kwargs)

    def _create_controller(self):
        if self._resource_controller:
            controller_type = DialogueResourceGameController if self.dialogue_skip else ResourceGameController
        else:
            controller_type = DialogueGameController if self.dialogue_skip else OriginalGameController
        return controller_type(self.executable, self.profile_path,
                               timeout=self.timeout, private_appdata=self.private_appdata)

    def _info(self, elapsed=0):
        info = super()._info(elapsed)
        if self.hide_window and self.pid is not None and self.pid != self._hidden_window_pid:
            from .owned_window_visibility import hide_owned_windows
            self.hidden_window_audit = hide_owned_windows(self.controller)
            self._hidden_window_pid = self.pid
        # The helper only targets this controller's PID and never requests focus
        # or keyboard input. Empty titles retain the existing window behavior.
        if self.window_title and self.pid is not None and self.pid != self._window_label_pid:
            self.window_label = label_owned_game_windows(self.controller, self.window_title)
            self._window_label_pid = self.pid
        info.update({
            'private_appdata': str(self.private_appdata),
            'dialogue_skip': self.dialogue_skip,
            'window_title': self.window_title,
            'window_label': self.window_label,
            'window_label_status': getattr(self.controller, 'window_label_status', None),
            'dialogue_release_events': len(getattr(self.controller, 'dialogue_input_events', ())),
        })
        if self.hide_window:
            info['hidden_window_audit'] = getattr(self, 'hidden_window_audit', None)
        return info


class IsolatedCampaignEnv(_IsolatedCampaignMixin, CampaignTouhouEnv):
    """Legacy observations/actions with an explicit worker save directory."""


class IsolatedResourceCampaignEnv(_IsolatedCampaignMixin, ResourceCampaignEnv):
    """Resource observations/actions with an explicit worker save directory."""
    _resource_controller = True
