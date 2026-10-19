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


@pytest.mark.parametrize('groups', [1, 2])
def test_vq_quantizes_to_codes_with_straight_through_gradient(groups):
    vq = make_vq(groups=groups)
    inputs = torch.randn(3, 5, 4, 8)
    vq(inputs)  # training mode: initialises the codebook
    vq.pop_aux()

    vq.eval()  # frozen codebook: outputs can be compared with it
    out = vq(inputs)
    assert out.shape == (3, 5, 4)
    assert vq.codebook.shape == (groups, 8, 4 // groups)
    # every group of the output is one of that group's codes
    parts = out.reshape(3, 5, groups, 1, 4 // groups)
    gap = (parts - vq.codebook).abs().sum(-1).min(-1).values
    assert (gap < 1e-5).all()
    out.sum().backward()
    assert vq.encoder.linear.weight.grad.abs().sum() > 0
    assert vq.last_indices.shape == (3, 5, groups)
    assert vq.last_duration_index is None
    aux = vq.pop_aux()
    assert aux['loss'] > 0 and 0 < aux['usage'] <= 1
    assert vq.pop_aux() is None  # consumed


def test_vq_rejects_groups_that_do_not_divide_the_dimension():
    with pytest.raises(ValueError, match='divide'):
        make_vq(groups=3)


@pytest.mark.parametrize('groups', [1, 2])
def test_vq_codebook_follows_the_data_and_eval_mode_is_frozen(groups):
    torch.manual_seed(0)
    vq = make_vq(num_codes=4, groups=groups, decay=0.9)
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


def test_vq_reseeds_dead_codes_in_every_group():
    torch.manual_seed(0)
    vq = make_vq(num_codes=16, groups=2, decay=0.5, dead_threshold=0.5)
    cluster = torch.randn(1, 8) * 3
    for _ in range(40):  # one tight cluster: most codes receive nothing
        vq((cluster + 0.01 * torch.randn(32, 8)).reshape(4, 8, 1, 8))
    assert (vq.ema_count >= 0.5).all()  # none stays dead, in any group
    assert vq.pop_aux()['perplexity'] >= 1.0


def padded_batch(garbage):
    """(2, 3, 4, 8) actions; the last chunk of each clip is cut by the clip end."""
    inputs = torch.randn(2, 3, 4, 8, generator=torch.Generator().manual_seed(0))
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    mask[:, 2, 1:] = True  # one real action, then padding
    inputs[:, 2] = garbage
    return inputs, mask


def test_vq_leaves_cut_chunks_out_of_the_statistics():
    # a chunk cut by the end of the clip has a code (the token is part of the
    # sequence) but must not count as a sample for the codebook or the loss
    vq = make_vq(dead_threshold=0.0)
    vq(*padded_batch(1000.0))  # initialises the codebook
    assert vq.codebook.abs().max() < 100  # not from the cut chunks
    before = vq.ema_count.sum(-1).clone()
    inputs, mask = padded_batch(-1000.0)
    vq(inputs, mask)
    n_counted = 4  # 6 chunks, 2 of them cut
    expected = 0.99 * before + 0.01 * n_counted
    assert torch.allclose(vq.ema_count.sum(-1), expected, atol=1e-4)

    vq.eval()
    vq(*padded_batch(1000.0))
    loss_a = vq.pop_aux()['loss']
    vq(*padded_batch(-1000.0))
    loss_b = vq.pop_aux()['loss']
    assert torch.allclose(loss_a, loss_b)


def test_vq_with_only_cut_chunks_is_harmless():
    vq = make_vq()
    inputs = torch.randn(2, 3, 4, 8)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    out = vq(inputs, mask)
    assert out.shape == (2, 3, 4)
    assert vq.pop_aux()['loss'] == 0
    assert not bool(vq.initialized)  # nothing to initialise from


def test_vq_reseed_rate_shows_codes_that_keep_dying():
    torch.manual_seed(0)
    vq = make_vq(num_codes=16, decay=0.5, dead_threshold=0.5)
    cluster = torch.randn(1, 8) * 3
    for _ in range(10):  # one tight cluster: most codes receive nothing
        vq((cluster + 0.01 * torch.randn(32, 8)).reshape(4, 8, 1, 8))
    assert vq.reseed_rate > 0  # re-seeding happened, which dead_frac could not show
    healthy = make_vq(num_codes=4, dead_threshold=0.0)
    for _ in range(10):
        healthy(torch.randn(4, 8, 4, 8))
    assert healthy.reseed_rate == 0


class ChunkLimitedEncoder(MeanEncoder):
    max_chunk = 5


def test_durations_beyond_the_encoder_chunk_length_are_refused():
    # the sequence encoder has one position embedding per chunk step
    VQActionEncoder(
        ChunkLimitedEncoder(),
        num_codes=8,
        durations=(2, 5),
        default_duration=2,
    )
    with pytest.raises(ValueError, match='max_chunk'):
        VQActionEncoder(
            ChunkLimitedEncoder(),
            num_codes=8,
            durations=(2, 8),
            default_duration=2,
        )


def test_vq_duration_token():
    vq = make_vq(groups=2, durations=(2, 4, 8), default_duration=4)
    assert vq.token_dim == 4 + 3
    inputs = torch.randn(2, 3, 4, 8)

    default = vq(inputs)  # no duration given: the fixed stride
    assert default.shape == (2, 3, 7)
    assert (default[..., 4:] == torch.tensor([0.0, 1.0, 0.0])).all()
    assert (vq.last_duration_index == 1).all()

    duration = torch.tensor([[2, 4, 8], [8, 8, 2]])
    token = vq(inputs, duration=duration)
    assert token[..., 4:].argmax(-1).tolist() == [[0, 1, 2], [2, 2, 0]]
    assert vq.last_duration_index.shape == (2, 3, 1)

    with pytest.raises(ValueError, match='allowed'):
        vq(inputs, duration=torch.full((2, 3), 5))


def test_vq_duration_config_is_validated():
    with pytest.raises(ValueError, match='default_duration'):
        make_vq(durations=(2, 4), default_duration=3)
    with pytest.raises(ValueError, match='distinct'):
        make_vq(durations=(2, 2), default_duration=2)


def test_vq_action_spec():
    spec = make_vq(
        groups=2, durations=(2, 4), default_duration=2
    ).action_spec()
    assert spec.codebook.shape == (2, 8, 2) and spec.durations == (2, 4)
    assert make_vq().action_spec().durations is None


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


def test_discrete_idm_uses_the_duration_input():
    torch.manual_seed(0)
    dim, num_durations = 8, 3
    loss_fn = DiscreteInverseDynamicsLoss(
        InverseDynamicsModel(dim, 64, num_durations, extra_dim=num_durations),
        1,
        num_durations,
        num_durations=num_durations,
    )
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=1e-2)
    emb = torch.randn(64, 5, dim)  # the states say nothing about the action
    duration = torch.randint(0, num_durations, (64, 4, 1))
    for _ in range(150):  # only the duration input can predict the target
        optimizer.zero_grad()
        loss_fn(emb, duration.clone(), duration).backward()
        optimizer.step()
    loss_fn(emb, duration.clone(), duration)
    assert loss_fn.last_accuracy > 0.95


# --- whole hierarchical models ---------------------------------------------

CONFIGS = [
    ('craftax_hjepa', 2),
    ('craftax_hwm', 2),
    ('craftax_hjepa', 3),
    ('craftax_hwm', 3),
    ('craftax_hjepa', 4),
]


def compose_train(name, overrides=()):
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    with hydra.initialize_config_dir(
        config_dir=str(H_JEPA / 'config' / 'train'), version_base=None
    ):
        return hydra.compose(config_name=name, overrides=list(overrides))


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


@pytest.mark.parametrize('levels', [1, 2, 3, 4])
def test_the_dataset_gets_clip_blocks_of_the_used_levels_only(levels):
    # the depth is a setting: the config defines levels 2 to 4 and the
    # dataset must not build clips for the ones above num_levels
    from data import dataset_kwargs

    cfg = compose_train('craftax_hjepa', [f'num_levels={levels}'])
    assert 'level4' in cfg.data.dataset  # defined, not used
    kwargs = dataset_kwargs(cfg)
    assert {k for k in kwargs if k.startswith('level')} == {
        f'level{n}' for n in range(1, levels + 1)
    }
    assert 'val_name' not in kwargs and 'name' in kwargs


@pytest.mark.parametrize('name,levels', CONFIGS)
def test_hierarchical_training_step(name, levels):
    from hjepa_utils import create_world_model, hjepa_forward

    cfg = compose_train(name, [f'num_levels={levels}'])
    model, losses = create_world_model(cfg)
    assert model.num_levels == levels
    for level in range(2, levels + 1):
        jepa = model.get_level(level)
        assert jepa.get_action_spec().codebook.shape == (2, 64, 8)
        # no dropout before the quantiser: the code of a chunk must not change
        # from step to step
        transformer = jepa.action_encoder.encoder.encoder
        assert all(layer.dropout.p == 0 for layer in transformer.layers)

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
        assert out[f'action_index_{level}'].shape[-1] == 2  # two code groups
        assert f'duration_index_{level}' not in out

    out['loss'].backward()
    top = model.get_level(2)
    assert top.action_encoder.encoder.head[0].weight.grad is not None
    assert model.get_level(1).idm_loss_fn.idm.model[0].weight.grad is not None


def test_hierarchical_cem_plans_with_codes():
    import planning_eval
    from eval_config_utils import _build_policy_plan_config
    from hierarchical_solver import build_hierarchical_solver
    from hjepa_utils import create_world_model

    model, _ = create_world_model(compose_train('craftax_hjepa'))
    model.eval()
    cfg = planning_eval.load_eval_config(config_name='craftax_plan')
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


BAND = [
    '+level2.action_encoder.vq.durations=[2,4,8]',
    '+level2.action_encoder.vq.default_duration=4',
    'level2.action_encoder.max_chunk=8',  # chunks of up to max(durations) steps
]


def test_training_step_with_a_duration_band():
    from hjepa_utils import create_world_model, hjepa_forward

    cfg = compose_train('craftax_hjepa', BAND)
    model, losses = create_world_model(cfg)
    top = model.get_level(2)
    assert top.get_action_spec().durations == (2, 4, 8)
    # the action embedder takes the code vectors and the one-hot duration
    assert top.action_embed.patch_embed is None
    assert top.action_embed.embed[0].in_features == 16 + 3

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
    # fixed stride: every step has the default duration (index 1 of [2, 4, 8])
    assert (out['duration_index_2'] == 1).all()
    assert torch.isfinite(out['idm_loss_level2'])
    out['loss'].backward()
    idm = top.idm_loss_fn.idm.model[0]
    assert idm.in_features == 2 * 192 + 3  # the IDM also sees the duration


def test_hierarchical_cem_plans_codes_and_durations():
    import planning_eval
    from eval_config_utils import _build_policy_plan_config
    from hierarchical_solver import build_hierarchical_solver
    from hjepa_utils import create_world_model

    model, _ = create_world_model(compose_train('craftax_hjepa', BAND))
    model.eval()
    cfg = planning_eval.load_eval_config(config_name='craftax_plan')
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for level in (1, 2):
        cfg.solver.solvers[f'level{level}'].update(
            device='cpu', num_samples=6, n_iters=2, topk=2, batch_size=2
        )
    solver = build_hierarchical_solver(cfg, model)
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17] * 2),
        n_envs=2,
        config=_build_policy_plan_config(cfg),
    )
    info = {
        'pixels': torch.randn(2, 1, 3, 63, 63),
        'goal': torch.randn(2, 1, 3, 63, 63),
        'action': torch.zeros(2, 1, 17),
    }
    out = solver.solve(info, steps_taken=0, eval_budget=20)
    assert out['actions'].dtype == torch.long and out['actions'].shape[0] == 2
    assert solver.level_solvers[2].action_dim == 16 + 3


def test_projected_cost_planning_runs_with_vq_levels():
    """Cost only: level 1 plans in the level-2 latent; level 2 is skipped."""
    import planning_eval
    from eval_config_utils import _build_policy_plan_config
    from hierarchical_solver import build_hierarchical_solver
    from hjepa_utils import create_world_model

    model, _ = create_world_model(compose_train('craftax_hjepa'))
    model.eval()
    cfg = planning_eval.load_eval_config(config_name='craftax_l2_project')
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for level in (1, 2):
        cfg.solver.solvers[f'level{level}'].update(
            device='cpu', num_samples=6, n_iters=2, topk=2, batch_size=2
        )
    solver = build_hierarchical_solver(cfg, model)
    solver.configure(
        action_space=gym.spaces.MultiDiscrete([17] * 2),
        n_envs=2,
        config=_build_policy_plan_config(cfg),
    )
    info = {
        'pixels': torch.randn(2, 1, 3, 63, 63),
        'goal': torch.randn(2, 1, 3, 63, 63),
        'action': torch.zeros(2, 1, 17),
    }
    out = solver.solve(info, steps_taken=0, eval_budget=20)
    assert out['actions'].shape[1] == 8  # the level-1 horizon
    assert solver.level_solvers[1].solve_records  # level 1 planned
    assert not solver.level_solvers[2].solve_records  # level 2 was skipped


def source_batch(cfg, sources):
    frames = level1_frames(cfg)
    n = len(sources)
    index = torch.randint(0, 17, (n, frames))
    return {
        'pixels_level1': torch.randn(n, frames, 3, 63, 63),
        'action_level1': torch.nn.functional.one_hot(index, 17).float(),
        'source_level1': torch.tensor(sources).reshape(n, 1).expand(n, frames),
    }


def forward(model, losses, cfg, batch):
    from hjepa_utils import hjepa_forward

    return hjepa_forward(
        FakeModule(model, losses),
        batch,
        'train',
        cfg,
        normalize_batch=lambda b: None,
    )


def test_random_episodes_train_level_1_only():
    from hjepa_utils import create_world_model, upper_clip_mask

    cfg = compose_train('craftax_hjepa', ['num_levels=3'])
    assert list(cfg.upper_loss_exclude_sources) == [2]
    sources = [0, 2, 1, 2, 0, 0]
    batch = source_batch(cfg, sources)
    assert upper_clip_mask(batch, cfg).tolist() == [
        True, False, True, False, True, True
    ]  # fmt: skip

    torch.manual_seed(0)
    model, losses = create_world_model(cfg)
    model.eval()  # no dropout, no codebook update: only the batch differs
    torch.manual_seed(1)  # the clips are cropped at random
    out = forward(model, losses, cfg, batch)
    assert out['embed_1'].shape[0] == 6  # level 1 sees every clip
    for level in (2, 3):
        assert out[f'embed_{level}'].shape[0] == 4
        assert out[f'action_index_{level}'].shape[0] == 4
    assert float(out['upper_clip_share_loss']) == pytest.approx(4 / 6)

    # the level-1 loss does not depend on which clips the upper levels use
    unmasked = {k: v for k, v in batch.items() if k != 'source_level1'}
    torch.manual_seed(1)
    plain = forward(model, losses, cfg, unmasked)
    # (SIGReg draws new random projections on every call, so compare the
    # deterministic terms)
    for key in ('pred_loss_level1', 'idm_loss_level1'):
        assert torch.allclose(out[key], plain[key]), key


def test_clips_of_other_sources_stay_out_of_the_codebook():
    import copy

    from hjepa_utils import create_world_model

    cfg = compose_train('craftax_hjepa')
    torch.manual_seed(0)
    model, losses = create_world_model(cfg)
    model.train()
    vq = model.get_level(2).action_encoder
    vq.dead_threshold = 0.0  # no re-seeding: the counts only grow with the data
    forward(model, losses, cfg, source_batch(cfg, [0, 0, 0, 0]))  # init
    state = copy.deepcopy(model.state_dict())

    def growth(sources):
        model.load_state_dict(state)
        before = vq.ema_count.sum(-1).clone()
        forward(model, losses, cfg, source_batch(cfg, sources) | {})
        return (vq.ema_count.sum(-1) - 0.99 * before) / 0.01

    all_clips = growth([0, 0, 0, 0])
    half = growth([0, 2, 0, 2])
    assert torch.allclose(half, all_clips / 2, atol=1e-3)  # 2 clips of 4 count
    assert vq.clip_mask is None  # the mask lives for one forward only


def test_too_few_kept_clips_fall_back_to_all_clips():
    from hjepa_utils import upper_clip_mask

    cfg = compose_train('craftax_hjepa')
    only_one = source_batch(cfg, [0, 2, 2, 2])
    assert upper_clip_mask(only_one, cfg) is None  # one clip: no statistics
    everyone = source_batch(cfg, [0, 0, 1, 1])
    assert upper_clip_mask(everyone, cfg).all()
    cfg.upper_loss_exclude_sources = []
    assert upper_clip_mask(source_batch(cfg, [2, 2, 2]), cfg) is None
    no_source = {'pixels_level1': torch.zeros(2, 5, 3, 63, 63)}
    assert upper_clip_mask(no_source, compose_train('craftax_hjepa')) is None


def test_the_source_column_reaches_the_batch_unchanged(dataset):
    import data as hjepa_data
    from omegaconf import OmegaConf

    cfg = compose_train(
        'craftax_hjepa', ['num_levels=1', 'data.dataset.name=craftax']
    )
    assert 'source' in cfg.data.dataset.keys_to_load
    kwargs = hjepa_data.dataset_kwargs(cfg)
    kwargs['level1'] = {'frameskip': 1, 'num_steps': 4}
    ds = hjepa_data.build_hdf5_dataset(
        kwargs, cache_dir=dataset.h5_path.parent
    )
    assert OmegaConf.is_config(cfg)
    item = ds[0]
    assert item['source_level1'].tolist() == [2, 2, 2, 2]  # not z-scored
    # no normalizer statistics are built for it either
    artifact = hjepa_data.build_normalizer_artifact(cfg, ds)
    assert 'source' not in artifact['stats']
