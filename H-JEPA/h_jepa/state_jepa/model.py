"""The MLP-JEPA on the state: encoder, predictor and B1's losses (prediction, SIGReg, one-step IDM)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from loss import DiscreteInverseDynamicsLoss, InverseDynamicsModel, SIGReg
from probing import targets as tg

NUM_ACTIONS = len(tg.ACTION_NAMES)


@dataclass
class Config:
    latent_dim: int = 192  # B1's latent
    hidden: int = 512
    history: int = 3  # B1: history 3, rollout_n 1, so a clip holds 4 frames
    sigreg: float = 0.09  # B1's weight
    idm: float = 0.1  # B1's weight
    idm_hidden: int = 256  # B1's IDM width
    stop_grad: bool = False  # A4: the prediction target is detached

    def to_dict(self) -> dict:
        return asdict(self)


def mlp(sizes: list[int]) -> nn.Sequential:
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


class StateJEPA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.latent_dim, cfg.hidden
        # the encoder ends in BatchNorm, as B1's projector does
        self.encoder = nn.Sequential(mlp([tg.STATE_DIM, h, h, d]), nn.BatchNorm1d(d))
        # the predictor reads `history` latents and their actions, the empty slots at the start of a clip zero with a flag
        self.predictor = mlp([cfg.history * (d + NUM_ACTIONS + 1), h, h, d])
        self.idm = DiscreteInverseDynamicsLoss(InverseDynamicsModel(d, cfg.idm_hidden, NUM_ACTIONS), 1, NUM_ACTIONS)
        self.sigreg = SIGReg()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latents (..., D) of states (..., 1345)."""
        return self.encoder(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], -1)

    def contexts(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """The predictor's input for every frame t = 1 .. T-1 of a clip: the latents and actions of frames t - history
        .. t - 1 (zero and flagged where the clip has none), as B1's causal predictor scores every position.
        z: (B, T, D), action: (B, T) with -1 = none. Returns (B, T - 1, history * (D + A + 1))."""
        B, T, _ = z.shape
        a = F.one_hot(action.clamp(min=0), NUM_ACTIONS).float() * (action >= 0).unsqueeze(-1)
        frame = torch.cat([z, a, torch.ones(B, T, 1, device=z.device)], -1)  # the flag: a real frame
        pad = frame.new_zeros(B, self.cfg.history - 1, frame.shape[-1])
        padded = torch.cat([pad, frame[:, :-1]], 1)  # position t of `padded[:, t : t + history]` ends at frame t - 1
        windows = padded.unfold(1, self.cfg.history, 1)  # (B, T - 1, F, history)
        return windows.permute(0, 1, 3, 2).flatten(2)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predictions (B, T - 1, D) of the latents of frames 1 .. T - 1."""
        return self.predictor(self.contexts(z, action))

    def losses(self, x: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """B1's losses on a batch of clips: x (B, T, 1345), action (B, T)."""
        z = self.encode(x)
        target = z[:, 1:].detach() if self.cfg.stop_grad else z[:, 1:]
        pred = F.mse_loss(self.predict(z, action), target)
        sigreg = self.sigreg(z.transpose(0, 1))  # (T, B, D): one statistic per position, as in B1
        idm = self.idm(z, action[:, :, None])
        total = pred + self.cfg.sigreg * sigreg + self.cfg.idm * idm
        return {"loss": total, "pred": pred, "sigreg": sigreg, "idm": idm, "idm_accuracy": self.idm.last_accuracy}
