"""Discrete-action path: one-hot data, normalizer artifact, categorical CEM."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


H_JEPA = Path(__file__).parents[1] / 'h_jepa'
sys.path.insert(0, str(H_JEPA))

import data as hjepa_data  # noqa: E402
from stable_worldmodel import PlanConfig  # noqa: E402
from stable_worldmodel.solver import CategoricalCEMSolver  # noqa: E402


def test_one_hot_column():
    action = torch.tensor([[[3.0], [float('nan')]], [[0.0], [16.0]]])
    sample = hjepa_data.OneHotColumn(17, 'action', 'action')(
        {'action': action}
    )
    onehot = sample['action']
    assert onehot.shape == (2, 2, 17)
    assert onehot.sum(-1).tolist() == [
        [1.0, 0.0],
        [1.0, 1.0],
    ]  # NaN: no action
    assert onehot.argmax(-1)[0, 0] == 3 and onehot.argmax(-1)[1, 1] == 16


def test_one_hot_column_rejects_out_of_range():
    transform = hjepa_data.OneHotColumn(17, 'action', 'action')
    with pytest.raises(ValueError, match='outside'):
        transform({'action': torch.tensor([[[17.0]]])})


def _artifact_cfg(num_actions):
    return OmegaConf.create(
        {
            'level1': {'wm': {'num_actions': num_actions}},
            'data': {
                'dataset': {
                    'name': 'craftax',
                    'keys_to_load': ['pixels', 'action', 'achievements'],
                    'keys_to_cache': ['action'],
                }
            },
        }
    )


def test_normalizer_artifact_skips_discrete_action(dataset):
    discrete = hjepa_data.build_normalizer_artifact(_artifact_cfg(17), dataset)
    assert 'action' not in discrete['stats']
    assert 'achievements' in discrete['stats']
    assert discrete['metadata']['num_actions'] == 17

    continuous = hjepa_data.build_normalizer_artifact(
        _artifact_cfg(None), dataset
    )
    assert 'action' in continuous['stats']
    assert continuous['metadata']['num_actions'] is None


class ToyCost:
    """Cost = number of planned actions that differ from a target sequence."""

    level = 1

    def __init__(self, num_actions, block, codebook=False):
        self.num_actions, self.block = num_actions, block
        self.codebook = torch.eye(num_actions) if codebook else None

    def get_action_codebook(self):
        return self.codebook

    def get_cost(self, info, candidates):
        b, s, h, _ = candidates.shape
        shape = (b, s, h, self.block, self.num_actions)
        chosen = candidates.reshape(shape).argmax(-1)
        info['embed_0'] = info['goal_embed_0'] = torch.zeros(1)
        return (chosen != info['target'][:, 0, None]).float().sum((-2, -1))

    def rollout(self, info, candidates):
        b, s, h, _ = candidates.shape
        info['predicted_embed_0'] = torch.zeros(b, s, h, 3)
        return info


def make_solver(num_actions, block, n_envs, **kwargs):
    solver = CategoricalCEMSolver(
        ToyCost(num_actions, block), **{'seed': 0, **kwargs}
    )
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([num_actions] * n_envs),
        n_envs=n_envs,
        config=PlanConfig(horizon=6, receding_horizon=3, action_block=block),
    )
    return solver


@pytest.mark.parametrize('block', [1, 2])
def test_cem_finds_the_optimal_sequence(block):
    generator = torch.Generator().manual_seed(1)
    target = torch.randint(0, 5, (3, 6, block), generator=generator)
    solver = make_solver(
        5, block, 3, num_samples=300, n_iters=15, topk=30, batch_size=2
    )
    out = solver.solve({'target': target}, planning_horizon=6)

    assert out['actions'].shape == (3, 6, block)
    assert out['actions'].dtype == torch.long
    assert torch.equal(out['actions'], target)
    assert out['predictions']['predicted_embed_0'].shape == (3, 6, 3)
    assert solver.action_dim == 5 * block

    (record,) = solver.solve_records
    assert record['solver'] == 'categorical_cem' and not record['backward']
    assert record['n_iters'] == [15] * 3 and record['num_samples'] == 300


def test_expand_info_dict_for_samples():
    from stable_worldmodel.solver.solver import expand_info_dict_for_samples

    info = {
        'tensor': torch.arange(6).reshape(3, 2),
        'array': np.arange(3),
        'other': 'text',
    }
    out = expand_info_dict_for_samples(info, 1, 3, 4, 'cpu')
    assert out['tensor'].shape == (2, 4, 2)
    assert torch.equal(out['tensor'][:, 0], info['tensor'][1:3])
    assert out['array'].shape == (2, 4) and out['other'] == 'text'


def test_cem_plans_codes_above_level_1():
    n_envs, horizon, num_codes = 2, 4, 6
    model = ToyCost(num_codes, 1, codebook=True)  # one-hot codes
    solver = CategoricalCEMSolver(
        model, num_samples=200, n_iters=12, topk=20, seed=0
    )
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17] * n_envs),  # not used
        n_envs=n_envs,
        config=PlanConfig(horizon=horizon, receding_horizon=1),
        level=2,
    )
    target = torch.randint(0, num_codes, (n_envs, horizon, 1))
    out = solver.solve({'target': target}, planning_horizon=horizon)

    assert torch.equal(out['action_indices'], target)
    # `actions` are the code vectors the upper-level model consumes
    assert out['actions'].shape == (n_envs, horizon, num_codes)
    assert torch.equal(out['actions'].argmax(-1), target[..., 0])
    assert solver.action_dim == num_codes
    assert solver.solve_records[0]['level'] == 2


def test_cem_rejects_unsupported_spaces():
    solver = CategoricalCEMSolver(ToyCost(5, 1))
    config = PlanConfig(horizon=2, receding_horizon=1)
    with pytest.raises(ValueError, match='codebook'):
        solver.configure(
            action_space=gym.spaces.MultiDiscrete([5]),
            n_envs=1,
            config=config,
            level=2,
        )
    with pytest.raises(TypeError):
        solver.configure(
            action_space=gym.spaces.Box(-1, 1, (1, 2)),
            n_envs=1,
            config=config,
        )


def test_cem_topk_must_not_exceed_samples():
    with pytest.raises(ValueError):
        CategoricalCEMSolver(ToyCost(5, 1), num_samples=4, topk=5)


# --- with the real model (needs the training dependencies) -----------------

transformers = pytest.importorskip('transformers')
pytest.importorskip('stable_pretraining')
hydra = pytest.importorskip('hydra')
pytest.importorskip('sklearn')


@pytest.fixture(scope='module')
def train_cfg():
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    with hydra.initialize_config_dir(
        config_dir=str(H_JEPA / 'config' / 'train'), version_base=None
    ):
        return hydra.compose(config_name='craftax_lewm')


@pytest.fixture(scope='module')
def world_model(train_cfg):
    from hjepa_utils import create_world_model

    model, losses = create_world_model(train_cfg)
    return model, losses


class FakeModule(torch.nn.Module):
    """Stands in for the Lightning module that hjepa_forward expects."""

    def __init__(self, model, losses):
        super().__init__()
        self.model = model
        for name, loss in losses.items():
            setattr(self, name, loss)

    def log_dict(self, *args, **kwargs):
        pass


def test_training_step_with_one_hot_actions(train_cfg, world_model):
    from hjepa_utils import hjepa_forward

    model, losses = world_model
    embedder = model.get_level(1).action_embed
    assert embedder.patch_embed.in_channels == 17

    steps = 4  # history 3 + rollout 1
    index = torch.randint(0, 17, (2, steps))
    batch = {
        'pixels_level1': torch.randn(2, steps, 3, 63, 63),
        'action_level1': torch.nn.functional.one_hot(index, 17).float(),
    }
    out = hjepa_forward(
        FakeModule(model, losses),
        batch,
        'train',
        train_cfg,
        normalize_batch=lambda batch: None,
    )
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert embedder.patch_embed.weight.grad is not None


def test_cem_plans_with_the_real_model(world_model):
    model, _ = world_model
    jepa = model.get_level(1).eval()
    n_envs = 2
    info = {
        'pixels': torch.randn(n_envs, 1, 3, 63, 63),
        'goal': torch.randn(n_envs, 1, 3, 63, 63),
        'action': torch.zeros(n_envs, 1, 17),
    }
    solver = CategoricalCEMSolver(
        jepa, num_samples=6, n_iters=2, topk=2, device='cpu', seed=0
    )
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17] * n_envs),
        n_envs=n_envs,
        config=PlanConfig(horizon=3, receding_horizon=2, action_block=1),
    )
    out = solver.solve(info, planning_horizon=3)
    assert out['actions'].shape == (n_envs, 3, 1)
    assert ((0 <= out['actions']) & (out['actions'] < 17)).all()
    assert 'predicted_embed_0' in out['predictions']


def test_eval_process_skips_discrete_action():
    import planning_eval

    discrete = {'stats': {}, 'metadata': {'num_actions': 17}}
    assert (
        planning_eval._normalizer_artifact_to_process(discrete, ['action'])
        == {}
    )
    continuous = {'stats': {}, 'metadata': {'num_actions': None}}
    with pytest.raises(KeyError):
        planning_eval._normalizer_artifact_to_process(continuous, ['action'])
