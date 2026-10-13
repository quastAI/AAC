"""Vector-quantized bottleneck for the latent actions of the upper levels."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class VQActionEncoder(nn.Module):
    """Wrap a sequence action encoder with an EMA vector-quantized bottleneck.

    The wrapped encoder pools a chunk of lower-level actions into one latent
    vector; this module snaps it to the nearest of `num_codes` code vectors, so
    an upper-level action is a discrete code and a planner can search over
    indices. Gradients pass through the quantizer straight-through. The
    codebook is not trained by gradient: it follows the encoder outputs with an
    exponential moving average, and codes that stop being used are re-seeded
    from the current batch.

    Args:
        encoder: Module with the SequenceEncoder interface and an `output_dim`.
        num_codes: Size of the codebook.
        decay: EMA decay of the code statistics.
        commitment: Weight of the commitment loss pulling encoder outputs to
            their codes.
        dead_threshold: A code whose EMA count falls below this is re-seeded.
        eps: Laplace smoothing of the EMA counts.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_codes: int,
        decay: float = 0.99,
        commitment: float = 0.25,
        dead_threshold: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = int(encoder.output_dim)
        self.num_codes = int(num_codes)
        self.decay = float(decay)
        self.commitment = float(commitment)
        self.dead_threshold = float(dead_threshold)
        self.eps = float(eps)

        self.register_buffer("codebook", torch.zeros(self.num_codes, self.output_dim))
        self.register_buffer("ema_sum", torch.zeros(self.num_codes, self.output_dim))
        self.register_buffer("ema_count", torch.ones(self.num_codes))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

        self.last_indices = None  # (B, T, 1) code index of the last forward
        self._aux = None

    def forward(self, inputs, padding_mask=None):
        z = self.encoder(inputs, padding_mask)
        flat = z.reshape(-1, self.output_dim).float()

        if self.training and not bool(self.initialized):
            self._init_codebook(flat)

        indices = self._nearest(flat)
        quantized = self.codebook[indices].reshape(z.shape).to(z.dtype)

        counts = torch.bincount(indices, minlength=self.num_codes).float()
        if self.training:
            self._update_codebook(flat, indices, counts.clone())

        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs * probs.clamp_min(1e-10).log()).sum()
        self._aux = {
            "loss": self.commitment * F.mse_loss(z, quantized.detach()),
            "perplexity": entropy.exp().detach(),
            "usage": (counts > 0).float().mean().detach(),
        }
        self.last_indices = indices.reshape(*z.shape[:-1], 1)

        # straight-through: the value of the code, the gradient of the encoder output
        return z + (quantized - z).detach()

    def pop_aux(self):
        """Commitment loss and code-usage statistics of the last forward, once."""
        aux, self._aux = self._aux, None
        return aux

    def _nearest(self, flat: torch.Tensor) -> torch.Tensor:
        code = self.codebook.float()
        distance = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ code.T + code.pow(2).sum(1)
        return distance.argmin(dim=1)

    @torch.no_grad()
    def _init_codebook(self, flat: torch.Tensor) -> None:
        """Start the codebook at random encoder outputs of the first batch."""
        pick = torch.randint(0, flat.size(0), (self.num_codes,), device=flat.device)
        self.codebook.copy_(flat[pick] + 0.01 * torch.randn_like(self.codebook))
        self.ema_sum.copy_(self.codebook)
        self.ema_count.fill_(1.0)
        self._sync_codebook()
        self.initialized.fill_(True)

    @torch.no_grad()
    def _update_codebook(self, flat, indices, counts) -> None:
        sums = torch.zeros_like(self.ema_sum).index_add_(0, indices, flat)
        if _distributed():
            torch.distributed.all_reduce(counts)
            torch.distributed.all_reduce(sums)

        self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        total = self.ema_count.sum()
        count = (self.ema_count + self.eps) / (total + self.num_codes * self.eps) * total
        self.codebook.copy_(self.ema_sum / count.unsqueeze(1))

        dead = self.ema_count < self.dead_threshold
        if dead.any():
            pick = torch.randint(0, flat.size(0), (int(dead.sum()),), device=flat.device)
            self.codebook[dead] = flat[pick]
            self.ema_sum[dead] = flat[pick]
            self.ema_count[dead] = 1.0
            self._sync_codebook()

    @torch.no_grad()
    def _sync_codebook(self) -> None:
        """Make every rank use rank 0's codebook after a random (re-)initialisation."""
        if _distributed():
            for buffer in (self.codebook, self.ema_sum, self.ema_count):
                torch.distributed.broadcast(buffer, src=0)
