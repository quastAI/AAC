"""Vector-quantized bottleneck for the latent actions of the upper levels."""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class ActionSpec(NamedTuple):
    """What a planner needs to build upper-level action candidates.

    codebook: (groups, num_codes, group_dim) code vectors.
    durations: allowed step durations k, or None when the duration is not part
        of the action token.
    default_duration: the duration of a fixed-stride step (one of durations).
    """

    codebook: torch.Tensor
    durations: tuple[int, ...] | None
    default_duration: int | None = None


class VQActionEncoder(nn.Module):
    """Wrap a sequence action encoder with an EMA vector-quantized bottleneck.

    The wrapped encoder pools a chunk of lower-level actions into one latent
    vector. The vector is split into `groups` parts, each snapped to the nearest
    code of its own codebook of `num_codes` codes (a factorized codebook: an
    action is a tuple of `groups` indices, `num_codes ** groups` combinations).
    An upper-level action is therefore discrete and a planner can search over
    indices. Gradients pass through the quantizer straight-through. The
    codebooks are not trained by gradient: they follow the encoder outputs with
    an exponential moving average, and codes that stop being used are re-seeded
    from the current batch.

    With `durations`, the action token also carries the duration k of the step
    (how many lower-level steps the chunk spans) as a one-hot vector, so the
    code only has to say what happens and not how long it takes. This is what a
    band of horizons needs: a step may span any k in `durations`. k is passed
    explicitly (`duration`, shape (B, T)); without it every step has
    `default_duration` (the fixed stride).

    Args:
        encoder: Module with the SequenceEncoder interface and an `output_dim`.
        num_codes: Codes per group.
        groups: Number of independent codebooks; must divide the encoder's
            output dimension.
        decay: EMA decay of the code statistics.
        commitment: Weight of the commitment loss pulling encoder outputs to
            their codes.
        dead_threshold: A code whose EMA count falls below this is re-seeded.
            The count is per batch (all-reduced over ranks), so it scales with
            the batch size.
        eps: Laplace smoothing of the EMA counts.
        durations: Allowed step durations, or None to leave k out of the token.
        default_duration: Duration used when none is passed (one of durations).
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_codes: int,
        groups: int = 1,
        decay: float = 0.99,
        commitment: float = 0.25,
        dead_threshold: float = 1.0,
        eps: float = 1e-5,
        durations=None,
        default_duration: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.code_dim = int(encoder.output_dim)
        self.num_codes = int(num_codes)
        self.groups = int(groups)
        if self.code_dim % self.groups:
            raise ValueError(
                f"groups={groups} must divide the encoder output dim {self.code_dim}"
            )
        self.group_dim = self.code_dim // self.groups
        self.decay = float(decay)
        self.commitment = float(commitment)
        self.dead_threshold = float(dead_threshold)
        self.eps = float(eps)

        self.durations = None if durations is None else tuple(int(k) for k in durations)
        self.default_duration = None if default_duration is None else int(default_duration)
        if self.durations is not None:
            if len(set(self.durations)) != len(self.durations):
                raise ValueError(f"durations must be distinct, got {self.durations}")
            if self.default_duration not in self.durations:
                raise ValueError(
                    f"default_duration={default_duration} must be one of {self.durations}"
                )
        # size of the action token the predictor's action embedder receives
        self.token_dim = self.code_dim + (0 if self.durations is None else len(self.durations))

        shape = (self.groups, self.num_codes, self.group_dim)
        self.register_buffer("codebook", torch.zeros(shape))
        self.register_buffer("ema_sum", torch.zeros(shape))
        self.register_buffer("ema_count", torch.ones(self.groups, self.num_codes))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        # moving average of the share of codes re-seeded per training step (monitoring only)
        self.register_buffer("reseed_rate", torch.zeros(()), persistent=False)

        self.last_indices = None  # (B, T, groups) code indices of the last forward
        self.last_duration_index = None  # (B, T, 1) index into `durations`, or None
        self._aux = None

    def action_spec(self) -> ActionSpec:
        return ActionSpec(self.codebook, self.durations, self.default_duration)

    def forward(self, inputs, padding_mask=None, duration=None):
        z = self.encoder(inputs, padding_mask)  # (B, T, code_dim)
        flat = z.reshape(-1, self.groups, self.group_dim).float()  # (N, G, d)
        if padding_mask is None:
            counted = torch.ones(flat.size(0), dtype=torch.bool, device=z.device)
        else:
            # A chunk cut by the end of the clip is zero-padded. It still gets a code (the
            # token is part of the sequence), but it is not a sample of what the codes should
            # represent: it stays out of the codebook statistics, the re-seeding and the
            # commitment loss.
            counted = ~padding_mask.any(-1).reshape(-1)  # (N,)

        if self.training and not bool(self.initialized) and counted.any():
            self._init_codebook(flat[counted])

        indices = self._nearest(flat)  # (N, G)
        code = self.codebook.float()
        quantized = code[torch.arange(self.groups, device=z.device), indices]  # (N, G, d)
        quantized = quantized.reshape(z.shape).to(z.dtype)

        offset = indices + torch.arange(self.groups, device=z.device) * self.num_codes
        counts = torch.bincount(
            offset[counted].flatten(), minlength=self.groups * self.num_codes
        ).float()
        counts = counts.reshape(self.groups, self.num_codes)
        if self.training:
            self._update_codebook(flat[counted], offset[counted], counts.clone())

        probs = counts / counts.sum(1, keepdim=True).clamp_min(1.0)
        entropy = -(probs * probs.clamp_min(1e-10).log()).sum(1)
        z_rows, q_rows = z.reshape(-1, self.code_dim), quantized.reshape(-1, self.code_dim)
        commit = F.mse_loss(z_rows[counted], q_rows[counted].detach()) if counted.any() else z.sum() * 0.0
        self._aux = {
            "loss": self.commitment * commit,
            "perplexity": entropy.exp().mean().detach(),
            "usage": (counts > 0).float().mean().detach(),
        }
        self.last_indices = indices.reshape(*z.shape[:-1], self.groups)

        # straight-through: the value of the code, the gradient of the encoder output
        token = z + (quantized - z).detach()
        if self.durations is None:
            self.last_duration_index = None
            return token
        index = self._duration_index(duration, z)
        self.last_duration_index = index.unsqueeze(-1)
        return torch.cat([token, F.one_hot(index, len(self.durations)).to(token.dtype)], dim=-1)

    def pop_aux(self):
        """Commitment loss and code-usage statistics of the last forward, once."""
        aux, self._aux = self._aux, None
        return aux

    def _duration_index(self, duration, z) -> torch.Tensor:
        """(B, T) index into `durations` of the step durations."""
        if duration is None:
            duration = torch.full(z.shape[:-1], self.default_duration, device=z.device)
        allowed = torch.tensor(self.durations, device=z.device)
        match = duration.long().unsqueeze(-1) == allowed
        if not match.any(-1).all():
            raise ValueError(f"duration outside the allowed set {self.durations}")
        return match.float().argmax(-1)

    def _nearest(self, flat: torch.Tensor) -> torch.Tensor:
        """(N, G, d) -> (N, G) index of the nearest code in each group."""
        code = self.codebook.float()
        # float32 also under autocast: bf16 dot products are off by about 0.02 for unit-scale
        # groups, enough to flip near-ties, and planning (float32) would assign other codes
        with torch.autocast(device_type=flat.device.type, enabled=False):
            distance = (
                flat.pow(2).sum(-1, keepdim=True)
                - 2 * torch.einsum("ngd,gkd->ngk", flat, code)
                + code.pow(2).sum(-1)
            )
        return distance.argmin(dim=-1)

    @torch.no_grad()
    def _init_codebook(self, flat: torch.Tensor) -> None:
        """Start every codebook at random encoder outputs of the first batch."""
        pick = torch.randint(0, flat.size(0), (self.num_codes,), device=flat.device)
        start = flat[pick].transpose(0, 1)  # (G, K, d)
        self.codebook.copy_(start + 0.01 * torch.randn_like(self.codebook))
        self.ema_sum.copy_(self.codebook)
        self.ema_count.fill_(1.0)
        self._sync_codebook()
        self.initialized.fill_(True)

    @torch.no_grad()
    def _update_codebook(self, flat, offset, counts) -> None:
        """EMA update; `offset` indexes the codebooks stacked into (G * K) entries."""
        sums = torch.zeros(self.groups * self.num_codes, self.group_dim, device=flat.device)
        sums.index_add_(0, offset.flatten(), flat.reshape(-1, self.group_dim))
        sums = sums.reshape_as(self.ema_sum)
        if _distributed():
            torch.distributed.all_reduce(counts)
            torch.distributed.all_reduce(sums)

        self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        total = self.ema_count.sum(1, keepdim=True)
        count = (self.ema_count + self.eps) / (total + self.num_codes * self.eps) * total
        self.codebook.copy_(self.ema_sum / count.unsqueeze(-1))

        dead = self.ema_count < self.dead_threshold  # (G, K)
        self.reseed_rate.mul_(self.decay).add_(dead.float().mean(), alpha=1 - self.decay)
        if dead.any():  # synced counts: every rank takes this branch together
            if flat.size(0) > 0:
                group, code = dead.nonzero(as_tuple=True)
                rows = torch.randint(0, flat.size(0), (group.numel(),), device=flat.device)
                seed = flat[rows, group]
                self.codebook[group, code] = seed
                self.ema_sum[group, code] = seed
                self.ema_count[group, code] = 1.0
            self._sync_codebook()

    @torch.no_grad()
    def _sync_codebook(self) -> None:
        """Make every rank use rank 0's codebook after a random (re-)initialisation."""
        if _distributed():
            for buffer in (self.codebook, self.ema_sum, self.ema_count):
                torch.distributed.broadcast(buffer, src=0)
