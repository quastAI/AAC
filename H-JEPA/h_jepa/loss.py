import torch
import torch.nn as nn
import torch.nn.functional as F


def init_module_weights(m, std=0.02):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (ECF means averaged across DDP ranks)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        cos, sin = x_t.cos().mean(-3), x_t.sin().mean(-3)
        n = proj.size(-2)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # untracked by autograd: each rank backprops its local ECF, DDP's grad average
            # then yields the gradient of the global statistic
            world_size = torch.distributed.get_world_size()
            for ecf in (cos, sin):
                torch.distributed.all_reduce(ecf)
                ecf.data.div_(world_size)
            n = n * world_size
        err = (cos - self.phi).square() + sin.square()
        statistic = (err @ self.weights) * n
        return statistic.mean()


class InverseDynamicsModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, extra_dim=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2 + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_t_plus_1, extra=None):
        parts = [state_t, state_t_plus_1] + ([] if extra is None else [extra])
        return self.model(torch.cat(parts, dim=1))


class InverseDynamicsLoss(nn.Module):
    """MSE between the IDM action predicted from (z_t, z_t+1) and the raw action a_t."""

    def __init__(self, idm):
        super().__init__()
        self.idm = idm

    def forward(self, emb, action):
        """
        emb: (B, T, D), action: (B, T, A) or (B, T-1, A)
        """
        pred = self.idm(emb[:, :-1].flatten(0, 1), emb[:, 1:].flatten(0, 1))  # [B*(T-1), A]
        target = action[:, : emb.size(1) - 1].flatten(0, 1)  # [B*(T-1), A]
        return F.mse_loss(pred, target.detach())


class DiscreteInverseDynamicsLoss(nn.Module):
    """Cross-entropy of the IDM prediction of the action taken between z_t and z_t+1.

    The target holds one class index per slot (frameskip actions at level 1, one
    code at the upper levels); -1 marks "no action" and is ignored. The accuracy
    of the last call is kept in `last_accuracy` for logging.
    """

    def __init__(self, idm, num_slots: int, num_classes: int, num_durations: int = 0):
        super().__init__()
        self.idm = idm
        self.num_slots, self.num_classes = int(num_slots), int(num_classes)
        self.num_durations = int(num_durations)
        self.last_accuracy = None

    def forward(self, emb, target, duration=None):
        """
        emb: (B, T, D), target: (B, T or T-1, num_slots) long, -1 = ignore,
        duration: (B, T or T-1, 1) index of the step duration, when the model has durations
        """
        extra = None
        if self.num_durations:
            duration = duration[:, : emb.size(1) - 1, 0].flatten(0, 1)
            extra = F.one_hot(duration, self.num_durations).float()
        logits = self.idm(emb[:, :-1].flatten(0, 1), emb[:, 1:].flatten(0, 1), extra)
        logits = logits.reshape(-1, self.num_slots, self.num_classes)
        target = target[:, : emb.size(1) - 1].flatten(0, 1)  # [B*(T-1), slots]
        valid = target >= 0
        if not valid.any():
            self.last_accuracy = emb.new_zeros(())
            return logits.sum() * 0.0
        loss = F.cross_entropy(logits[valid], target[valid])
        with torch.no_grad():
            self.last_accuracy = (logits[valid].argmax(-1) == target[valid]).float().mean()
        return loss


class ViewMapLoss(nn.Module):
    """Auxiliary decoder of the view map from the latent (the ablation A2).

    A small MLP reads each frame's latent and predicts the block type of every view tile (cross-entropy)
    and which mobs stand on it (binary cross-entropy). The gradient reaches the latent, the projector and
    the encoder: this is no probe (a probe would not change the latent), the latent has to hold the map for
    the loss to fall. The head is used in training only and takes no part in planning.

    The tiles are very unequal (grass 41%, coal 1%, diamond 0.05%): a plain loss would let the rare ones go.
    Each class gets the weight count^-power of the batch (the weights of the classes in the batch,
    normalised: the loss keeps the scale of a plain cross-entropy), at most `max_ratio` times the weight of
    the most common class. `class_power` 0 is the plain loss. `tile_mask` leaves tiles out (the player's own
    tile is covered by its sprite). The metrics of the last call are kept in `last_metrics`.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_tiles: int,
        num_blocks: int,
        num_mobs: int,
        class_power: float = 0.5,
        max_ratio: float = 32.0,
        tile_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_tiles, self.num_blocks, self.num_mobs = int(num_tiles), int(num_blocks), int(num_mobs)
        self.class_power, self.max_ratio = float(class_power), float(max_ratio)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_tiles * (self.num_blocks + self.num_mobs)),
        )
        self.apply(init_module_weights)
        if tile_mask is None:
            tile_mask = torch.ones(self.num_tiles, dtype=torch.bool)
        self.register_buffer("tile_mask", tile_mask.bool(), persistent=False)
        self.last_metrics = {}

    def _weights(self, counts: torch.Tensor) -> torch.Tensor:
        """Weight per class from its count in the batch (classes without tiles: 1, they are never used)."""
        if self.class_power == 0:
            return torch.ones_like(counts, dtype=torch.float32)
        weight = counts.clamp(min=1).float().pow(-self.class_power)
        present = counts > 0
        floor = weight[present].min()  # the most common class
        return torch.where(present, weight.clamp(max=floor * self.max_ratio), torch.ones_like(weight))

    def forward(self, emb: torch.Tensor, blocks: torch.Tensor, mobs: torch.Tensor) -> torch.Tensor:
        """emb: (B, T, D); blocks: (B, T, tiles) long class indices; mobs: (B, T, tiles, num_mobs) 0/1."""
        logits = self.decoder(emb.flatten(0, 1)).reshape(-1, self.num_tiles, self.num_blocks + self.num_mobs)
        keep = self.tile_mask
        block_logits = logits[:, keep, : self.num_blocks].reshape(-1, self.num_blocks)
        mob_logits = logits[:, keep, self.num_blocks :].reshape(-1, self.num_mobs)
        block_target = blocks.flatten(0, 1)[:, keep].reshape(-1).long()
        mob_target = mobs.flatten(0, 1)[:, keep].reshape(-1, self.num_mobs).float()

        with torch.no_grad():
            counts = torch.bincount(block_target, minlength=self.num_blocks)
            class_weight = self._weights(counts)
            positives = mob_target.sum(0)
            negatives = mob_target.size(0) - positives
            if self.class_power == 0:
                pos_weight = torch.ones_like(positives)
            else:
                pos_weight = ((negatives + 1) / (positives + 1)).pow(self.class_power).clamp(max=self.max_ratio)
        tile_weight = class_weight[block_target]
        block_loss = (F.cross_entropy(block_logits, block_target, reduction="none") * tile_weight).sum() / tile_weight.sum()
        element_weight = torch.where(mob_target > 0, pos_weight.expand_as(mob_target), torch.ones_like(mob_target))
        mob_loss = (
            F.binary_cross_entropy_with_logits(mob_logits, mob_target, reduction="none") * element_weight
        ).sum() / element_weight.sum()

        with torch.no_grad():
            hit = block_logits.argmax(-1) == block_target
            per_class = torch.zeros(self.num_blocks, device=hit.device).index_add_(0, block_target, hit.float())
            present = counts > 0
            found = mob_logits > 0
            self.last_metrics = {
                "block": block_loss.detach(),
                "mob": mob_loss.detach(),
                "accuracy": hit.float().mean(),
                "balanced_accuracy": (per_class[present] / counts[present]).mean(),
                "mob_recall": (found & (mob_target > 0)).sum() / positives.sum().clamp(min=1),
            }
        return block_loss + mob_loss
