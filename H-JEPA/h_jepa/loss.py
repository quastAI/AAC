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
