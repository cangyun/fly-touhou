"""One recurrent connectome state and direction-specific DN readout."""
import torch
from torch import nn
from torch.nn import functional as F


REVISION = 'single-lif-step-direction-readout-v1'


class SharedDirections(nn.Module):
    def __init__(self, cue_width, ports, dn_width, *, device):
        super().__init__()
        # Five short-range cues, eight route cues, and the configured
        # per-direction extension beyond the repeated scene context.
        self.cue_width = cue_width
        self.sensory = nn.Linear(18 * (13 + cue_width - 63), ports, bias=False, device=device)
        self.readout = nn.Sequential(nn.LayerNorm(dn_width, device=device),
            nn.Linear(dn_width, 128, device=device), nn.SiLU(),
            nn.Linear(128, 18 * 32, device=device))
        self.direction_projection = nn.Linear(32, dn_width, bias=False, device=device)
        nn.init.normal_(self.sensory.weight, std=.025)
        nn.init.normal_(self.readout[-1].weight, std=.02)
        nn.init.zeros_(self.readout[-1].bias)

    def inputs(self, cues):
        shared = cues.mean(dim=1)
        # Collection cues are common to all actions and enter through the shared sensory drive.
        directional = torch.cat((cues[..., :3], cues[..., 6:8], cues[..., 47:55], cues[..., 63:self.cue_width]), -1)
        # Keep the direction-specific differences in action order.
        distinctive = directional - directional.mean(dim=1, keepdim=True)
        return shared, distinctive.flatten(start_dim=1)

    def port_delta(self, directional):
        # Fixed small GEMM width keeps the membrane threshold decisions
        # independent of the number of worlds ready at once.
        rows = []
        for part in directional.split(16):
            count = len(part)
            rows.append(self.sensory(F.pad(part, (0, 0, 0, 16 - count)))[:count])
        return .25 * torch.cat(rows).tanh()

    def directions(self, dn):
        latent = self.readout(dn).reshape(len(dn), 18, 32)
        latent = latent - latent.mean(dim=1, keepdim=True)
        return dn[:, None, :] + self.direction_projection(latent)
