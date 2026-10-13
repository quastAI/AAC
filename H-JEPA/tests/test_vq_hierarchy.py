"""VQ action encoder, discrete IDM and the hierarchical models on Craftax."""

import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn


pytest.importorskip('transformers')
pytest.importorskip('stable_pretraining')
hydra = pytest.importorskip('hydra')
pytest.importorskip('sklearn')

H_JEPA = Path(__file__).parents[1] / 'h_jepa'
sys.path.insert(0, str(H_JEPA))

from loss import (  # noqa: E402
    DiscreteInverseDynamicsLoss,
    InverseDynamicsModel,
)
from models.vq import VQActionEncoder  # noqa: E402


class MeanEncoder(nn.Module):
    """Stands in for the SequenceEncoder: (B, T, L, C) -> (B, T, D)."""

    def __init__(self, input_dim=8, output_dim=4):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs, padding_mask=None):
        return self.linear(inputs.mean(2))


def make_vq(num_codes=8, **kwargs):
    return VQActionEncoder(MeanEncoder(), num_codes=num_codes, **kwargs)


def test_vq_quantizes_to_codes_with_straight_through_gradient():
    vq = make_vq()
    inputs = torch.randn(3, 5, 4, 8)
    vq(inputs)  # training mode: initialises the codebook
    vq.pop_aux()

    vq.eval()  # frozen codebook: outputs can be compared with it
    out = vq(inputs)
    assert out.shape == (3, 5, 4)
    # the value is a code vector, the gradient reaches the encoder
    gap = (out[..., None, :] - vq.codebook).abs().sum(-1).min(-1).values
    assert (gap < 1e-5).all()
    out.sum().backward()
    assert vq.encoder.linear.weight.grad.abs().sum() > 0
    assert vq.last_indices.shape == (3, 5, 1)
    aux = vq.pop_aux()
    assert aux['loss'] > 0 and 0 < aux['usage'] <= 1
    assert vq.pop_aux() is None  # consumed


def test_vq_codebook_follows_the_data_and_eval_mode_is_frozen():
    torch.manual_seed(0)
    vq = make_vq(num_codes=4, decay=0.9)
    centers = torch.randn(4, 8) * 3

    def batch():
        idx = torch.randint(0, 4, (64,))
        return (centers[idx] + 0.05 * torch.randn(64, 8)).reshape(8, 8, 1, 8)

    vq(batch())
    first = vq.pop_aux()['loss'].item()
    for _ in range(80):
        vq(batch())
    last = vq.pop_aux()['loss'].item()
    assert last < first  # the codes moved onto the clusters
    assert vq.pop_aux() is None

    vq.eval()
    before = vq.codebook.clone()
    vq(batch())
    assert torch.equal(before, vq.codebook)


def test_vq_reseeds_dead_codes():
    torch.manual_seed(0)
    vq = make_vq(num_codes=16, decay=0.5, dead_threshold=0.5)
    cluster = torch.randn(1, 8) * 3
    for _ in range(40):  # one tight cluster: most codes receive nothing
        vq((cluster + 0.01 * torch.randn(32, 8)).reshape(4, 8, 1, 8))
    assert (vq.ema_count >= 0.5).all()  # none stays dead
    assert vq.pop_aux()['perplexity'] >= 1.0


def test_discrete_idm_learns_and_ignores_missing_actions():
    torch.manual_seed(0)
    classes, dim = 5, 16
    loss_fn = DiscreteInverseDynamicsLoss(
        InverseDynamicsModel(dim, 64, 2 * classes), 2, classes
    )
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=1e-2)
    target = torch.randint(0, classes, (32, 6, 2))
    # the state change encodes the actions of the step
    emb = torch.zeros(32, 7, dim)
    for t in range(6):
        emb[:, t + 1] = (
            emb[:, t]
            + torch.nn.functional.one_hot(target[:, t, 0], dim).float()
        )
    first = loss_fn(emb, target).item()
    for _ in range(150):
        optimizer.zero_grad()
        loss_fn(emb, target).backward()
        optimizer.step()
    assert loss_fn(emb, target).item() < first
    assert loss_fn.last_accuracy > 0.4

    ignored = torch.full_like(target, -1)
    assert loss_fn(emb, ignored) == 0


# --- whole hierarchical models ---------------------------------------------

CONFIGS = [
    ('craftax_hjepa_l2', 2),
    ('craftax_hwm_l2', 2),
    ('craftax_hjepa_l3', 3),
    ('craftax_hwm_l3', 3),
]


def compose_train(name):
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    with hydra.initialize_config_dir(
        config_dir=str(H_JEPA / 'config' / 'train'), version_base=None
    ):
        return hydra.compose(config_name=name)


class FakeModule(nn.Module):
    def __init__(self, model, losses):
        super().__init__()
        self.model = model
        for name, loss in losses.items():
            setattr(self, name, loss)

    def log_dict(self, *args, **kwargs):
        pass


def level1_frames(cfg):
    """Frames of the level-1 stream that the top level needs."""
    frames = 1
    for level in range(int(cfg.num_levels), 1, -1):
        frames = (
            (4 - 1) * cfg[f'level{level}'].stride + 1
            if level == int(cfg.num_levels)
            else (frames - 1) * cfg[f'level{level}'].stride + 1
        )
    return frames


@pytest.mark.parametrize('name,levels', CONFIGS)
def test_hierarchical_training_step(name, levels):
    from hjepa_utils import create_world_model, hjepa_forward

    cfg = compose_train(name)
    model, losses = create_world_model(cfg)
    assert model.num_levels == levels
    for level in range(2, levels + 1):
        assert model.get_level(level).get_action_codebook().shape == (128, 16)

    frames = level1_frames(cfg)
    index = torch.randint(0, 17, (2, frames))
    batch = {
        'pixels_level1': torch.randn(2, frames, 3, 63, 63),
        'action_level1': torch.nn.functional.one_hot(index, 17).float(),
    }
    out = hjepa_forward(
        FakeModule(model, losses),
        batch,
        'train',
        cfg,
        normalize_batch=lambda batch: None,
    )
    assert torch.isfinite(out['loss'])
    for level in range(1, levels + 1):
        assert torch.isfinite(out[f'idm_loss_level{level}'])
        assert 0 <= out[f'idm_accuracy_loss_level{level}'] <= 1
    for level in range(2, levels + 1):
        assert torch.isfinite(out[f'vq_loss_level{level}'])
        assert out[f'vq_perplexity_loss_level{level}'] >= 1
        assert out[f'action_index_{level}'].shape[-1] == 1

    out['loss'].backward()
    top = model.get_level(2)
    assert top.action_encoder.encoder.head[0].weight.grad is not None
    assert model.get_level(1).idm_loss_fn.idm.model[0].weight.grad is not None


def test_hierarchical_cem_plans_with_codes():
    import planning_eval
    from eval_config_utils import _build_policy_plan_config
    from hierarchical_solver import build_hierarchical_solver
    from hjepa_utils import create_world_model

    model, _ = create_world_model(compose_train('craftax_hjepa_l2'))
    model.eval()
    cfg = planning_eval.load_eval_config(config_name='craftax_l2')
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for level in (1, 2):
        solver_cfg = cfg.solver.solvers[f'level{level}']
        solver_cfg.update(
            device='cpu', num_samples=6, n_iters=2, topk=2, batch_size=2
        )

    n_envs = 2
    solver = build_hierarchical_solver(cfg, model)
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17] * n_envs),
        n_envs=n_envs,
        config=_build_policy_plan_config(cfg),
    )
    info = {
        'pixels': torch.randn(n_envs, 1, 3, 63, 63),
        'goal': torch.randn(n_envs, 1, 3, 63, 63),
        'action': torch.zeros(n_envs, 1, 17),
    }
    out = solver.solve(info, steps_taken=0, eval_budget=20)

    actions = out['actions']
    assert actions.shape[0] == n_envs and actions.dtype == torch.long
    assert ((0 <= actions) & (actions < 17)).all()
    records = [
        r for s in solver.level_solvers.values() for r in s.solve_records
    ]
    assert sorted(r['level'] for r in records) == [1, 2]
