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
from models.vq import ActionSpec  # noqa: E402
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

    def __init__(self, num_actions, block):
        self.num_actions, self.block = num_actions, block

    def get_action_spec(self):
        return None

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


class ToySlotCost:
    """Cost = number of slots that differ from a target (codes, then duration).

    The codebooks are one-hot, so a token vector can be decoded back to its
    indices.
    """

    level = 2

    def __init__(self, groups, num_codes, durations=None):
        self.groups, self.num_codes = groups, num_codes
        book = torch.eye(num_codes).expand(groups, -1, -1).clone()
        default = None if durations is None else durations[0]
        self.spec = ActionSpec(book, durations, default)

    def get_action_spec(self):
        return self.spec

    def get_cost(self, info, candidates):
        groups, num_codes = self.groups, self.num_codes
        b, s, h, _ = candidates.shape
        codes = candidates[..., : groups * num_codes]
        chosen = [codes.reshape(b, s, h, groups, num_codes).argmax(-1)]
        if self.spec.durations is not None:
            chosen.append(
                candidates[..., groups * num_codes :].argmax(-1, keepdim=True)
            )
        chosen = torch.cat(chosen, dim=-1)
        info['embed_0'] = info['goal_embed_0'] = torch.zeros(1)
        return (chosen != info['target'][:, :1]).float().sum((-2, -1))

    def rollout(self, info, candidates):
        b, s, h, _ = candidates.shape
        info['predicted_embed_0'] = torch.zeros(b, s, h, 3)
        return info


def solve_slots(model, target, horizon, **kwargs):
    solver = CategoricalCEMSolver(
        model,
        **{'num_samples': 300, 'n_iters': 15, 'topk': 30, 'seed': 0, **kwargs},
    )
    n_envs = target.shape[0]
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17] * n_envs),  # not used
        n_envs=n_envs,
        config=PlanConfig(horizon=horizon, receding_horizon=1),
        level=2,
    )
    return solver, solver.solve({'target': target}, planning_horizon=horizon)


def test_cem_plans_codes_above_level_1():
    num_codes = 6
    target = torch.randint(0, num_codes, (2, 4, 1))
    solver, out = solve_slots(ToySlotCost(1, num_codes), target, 4)

    assert torch.equal(out['action_indices'], target)
    # `actions` are the token vectors the upper-level model consumes
    assert out['actions'].shape == (2, 4, num_codes)
    assert torch.equal(out['actions'].argmax(-1), target[..., 0])
    assert 'durations' not in out
    assert solver.action_dim == num_codes
    assert solver.solve_records[0]['level'] == 2


def test_cem_plans_factorized_codes_with_durations():
    groups, num_codes, durations = 2, 5, (2, 3, 5)
    generator = torch.Generator().manual_seed(2)
    target = torch.cat(
        [
            torch.randint(0, num_codes, (3, 4, groups), generator=generator),
            torch.randint(0, len(durations), (3, 4, 1), generator=generator),
        ],
        dim=-1,
    )
    model = ToySlotCost(groups, num_codes, durations)
    solver, out = solve_slots(
        model, target, 4, num_samples=400, n_iters=20, allowed_durations='all'
    )

    # one slot per group, plus one for the duration with its own size
    assert out['action_indices'].shape == (3, 4, groups + 1)
    assert torch.equal(out['action_indices'], target)
    assert torch.equal(
        out['durations'], torch.tensor(durations)[target[..., -1]]
    )
    assert out['actions'].shape == (3, 4, groups * num_codes + len(durations))
    assert solver.action_dim == groups * num_codes + len(durations)


def duration_target(durations, n_envs=2, horizon=3):
    """Target indices (1 code group) with a random duration class per step."""
    generator = torch.Generator().manual_seed(4)
    return torch.cat(
        [
            torch.randint(0, 5, (n_envs, horizon, 1), generator=generator),
            torch.randint(
                0, len(durations), (n_envs, horizon, 1), generator=generator
            ),
        ],
        dim=-1,
    )


def test_cem_pins_the_duration_by_default():
    durations = (2, 3, 5)
    target = duration_target(durations)
    assert (target[..., -1] != 0).any()  # the target asks for other durations
    solver, out = solve_slots(
        ToySlotCost(1, 5, durations), target, 3, num_samples=100
    )
    # safe default: every step keeps the default duration, whatever the cost
    assert (out['durations'] == durations[0]).all()
    assert (out['action_indices'][..., -1] == 0).all()
    assert solver.solve_records[0]['allowed_durations'] == 'default'


def test_cem_restricts_durations_to_a_subset():
    durations = (2, 3, 5)
    target = duration_target(durations)
    target[..., -1] = 2  # asks for the longest jump, which is not allowed
    _, out = solve_slots(
        ToySlotCost(1, 5, durations),
        target,
        3,
        num_samples=100,
        allowed_durations=[2, 3],
    )
    assert set(out['durations'].flatten().tolist()) <= {2, 3}


def test_cem_proposals_bring_data_sequences_into_the_search():
    # 2 groups of 40 codes over 4 steps: 1600 ** 4 plans, far too many for a
    # few hundred uniform samples, but the plan we want is a data sequence
    groups, num_codes, horizon = 2, 40, 4
    generator = torch.Generator().manual_seed(0)
    bank = torch.randint(0, num_codes, (25, horizon, groups), generator=generator)
    target = bank[7].unsqueeze(0)  # (1, H, groups)
    model = ToySlotCost(groups, num_codes)

    _, plain = solve_slots(model, target, horizon, n_iters=3)
    assert not torch.equal(plain['action_indices'], target)  # uniform CEM does not find it

    solver = CategoricalCEMSolver(
        model, num_samples=300, n_iters=3, topk=30, seed=0,
        proposal_fraction=0.5,
    )  # fmt: skip
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17]),
        n_envs=1,
        config=PlanConfig(horizon=horizon, receding_horizon=1),
        level=2,
    )
    solver.set_proposal(bank)
    out = solver.solve({'target': target}, planning_horizon=horizon)
    assert torch.equal(out['action_indices'], target)
    assert solver.solve_records[-1]['proposal_fraction'] == 0.5

    solver.set_proposal(None)  # removed: back to the distribution alone
    assert solver._bank is None


def test_cem_proposals_longer_horizon_keeps_the_distribution_beyond_the_bank():
    bank = torch.zeros(5, 2, 1, dtype=torch.long)  # sequences of 2 steps
    model = ToySlotCost(1, 6)
    solver = CategoricalCEMSolver(
        model, num_samples=40, n_iters=2, topk=4, seed=0, proposal_fraction=0.9
    )
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17]),
        n_envs=1,
        config=PlanConfig(horizon=4, receding_horizon=1),
        level=2,
    )
    solver.set_proposal(bank)
    seen, cost = [], model.get_cost

    def spy(info, candidates):
        seen.append(candidates.argmax(-1))
        return cost(info, candidates)

    model.get_cost = spy
    solver.solve({'target': torch.zeros(1, 4, 1, dtype=torch.long)}, planning_horizon=4)
    first = seen[0][0]  # (S, H, groups) of the first iteration
    assert (first[1:37, :2] == 0).all()  # 36 proposals: the first two steps
    assert first[1:37, 2:].unique().numel() > 1  # the rest is still random


def test_cem_proposals_must_match_the_model_and_the_level():
    with pytest.raises(ValueError, match='proposal_fraction'):
        CategoricalCEMSolver(ToyCost(5, 1), proposal_fraction=1.0)
    solver = CategoricalCEMSolver(ToySlotCost(2, 5), proposal_fraction=0.5)
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17]),
        n_envs=1,
        config=PlanConfig(horizon=3, receding_horizon=1),
        level=2,
    )
    with pytest.raises(ValueError, match='M > 0|must be'):
        solver.set_proposal(torch.zeros(0, 3, 2))
    solver.set_proposal(torch.zeros(4, 3, 1, dtype=torch.long))  # 1 group, 2 expected
    with pytest.raises(ValueError, match='code slots'):
        solver.solve({'target': torch.zeros(1, 3, 2, dtype=torch.long)}, planning_horizon=3)


def test_cem_keeps_a_configured_duration_list_plain():
    # a list from a Hydra config is a ListConfig; it is written to JSON records
    import json

    solver = CategoricalCEMSolver(
        ToyCost(5, 1), allowed_durations=OmegaConf.create([2, 4])
    )
    assert solver.allowed_durations == [2, 4]
    assert type(solver.allowed_durations) is list
    json.dumps(solver.allowed_durations)
    assert CategoricalCEMSolver(ToyCost(5, 1)).allowed_durations == 'default'


def test_cem_rejects_bad_allowed_durations():
    for bad in ([7], []):
        with pytest.raises(ValueError, match='allowed_durations'):
            solve_slots(
                ToySlotCost(1, 5, (2, 3)),
                torch.zeros(1, 2, 2, dtype=torch.long),
                2,
                allowed_durations=bad,
            )


def test_cem_never_samples_a_missing_class():
    # the duration slot has 3 classes next to 8-class code slots
    model = ToySlotCost(1, 8, (1, 2, 3))
    solver = CategoricalCEMSolver(
        model, num_samples=50, n_iters=2, topk=5, allowed_durations='all'
    )
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17]),
        n_envs=1,
        config=PlanConfig(horizon=3, receding_horizon=1),
        level=2,
    )
    seen, cost = [], model.get_cost

    def spy(info, candidates):
        seen.append(candidates)
        return cost(info, candidates)

    model.get_cost = spy
    target = torch.zeros(1, 3, 2, dtype=torch.long)
    solver.solve({'target': target}, planning_horizon=3)
    # every duration part is a valid one-hot over the 3 durations
    assert all((c[..., 8:].sum(-1) == 1).all() for c in seen)


@pytest.mark.parametrize('classes', [17, 64, 128])
def test_cem_converges_whatever_the_number_of_classes(classes):
    # a per-class probability floor capped big slots (0.01 per class: 0.55 at
    # 64 classes), so the elite plan was almost never sampled again
    solver = CategoricalCEMSolver(ToyCost(5, 1), num_samples=30, topk=10)
    slots = 3
    valid = torch.ones(slots, classes, dtype=torch.bool)
    valid[2, classes // 2 :] = False  # a slot with fewer classes
    probs = (valid.float() / valid.sum(-1, keepdim=True)).expand(1, 2, -1, -1)
    samples = torch.full((1, 30, 2, slots), 5)
    costs = torch.rand(1, 30)
    for _ in range(40):
        probs = solver._update_probs(probs, samples, costs, valid)
    assert torch.allclose(probs.sum(-1), torch.ones(1, 2, slots))
    assert probs[..., 5].min() > 0.88
    assert (probs[:, :, 2, classes // 2 :] == 0).all()  # absent classes


def test_cem_vq_levels_need_action_block_1():
    solver = CategoricalCEMSolver(ToySlotCost(1, 4))
    with pytest.raises(ValueError, match='action_block'):
        solver.configure(
            action_space=gym.spaces.MultiDiscrete([17]),
            n_envs=1,
            config=PlanConfig(horizon=2, receding_horizon=1, action_block=2),
            level=2,
        )


def test_cem_rejects_unsupported_spaces():
    solver = CategoricalCEMSolver(ToyCost(5, 1))
    config = PlanConfig(horizon=2, receding_horizon=1)
    with pytest.raises(ValueError, match='VQ action encoder'):
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
        return hydra.compose(
            config_name='craftax_hjepa', overrides=['num_levels=1']
        )


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
    assert embedder.patch_embed is None
    assert embedder.embed[0].in_features == 17

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
    assert embedder.embed[0].weight.grad is not None


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
