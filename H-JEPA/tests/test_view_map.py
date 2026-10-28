"""Auxiliary view-map decoder (the ablation A2): targets from the state
column, class weights, the loss itself, the config and a training step."""

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

H_JEPA = Path(__file__).parents[1] / 'h_jepa'
sys.path.insert(0, str(H_JEPA))

from hjepa_utils import view_map_targets  # noqa: E402
from loss import ViewMapLoss  # noqa: E402
from probing import targets as tg  # noqa: E402

GRASS, COAL = (tg.BLOCK_NAMES.index(n) for n in ('grass', 'coal'))
ZOMBIE = tg.MOB_NAMES.index('zombie')
COAL_TILE = 10
TILES, BLOCKS, MOBS = tg.VIEW_TILES, tg.NUM_BLOCKS, tg.NUM_MOBS


def planted_state(batch=2, frames=4):
    """Grass everywhere, a coal tile at COAL_TILE in the first clip and a
    zombie on the tile next to it in the second."""
    state = torch.zeros(batch, frames, tg.STATE_DIM, dtype=torch.uint8)
    tiles = state[..., : TILES * tg.VIEW_CHANNELS].reshape(
        batch, frames, TILES, tg.VIEW_CHANNELS
    )
    tiles[..., GRASS] = 1
    tiles[0, :, COAL_TILE, GRASS] = 0
    tiles[0, :, COAL_TILE, COAL] = 1
    tiles[1, :, COAL_TILE + 1, BLOCKS + ZOMBIE] = 1
    return state


def small_loss(**kwargs):
    kwargs = (
        dict(
            state_dim=16,
            hidden_dim=64,
            num_tiles=6,
            num_blocks=4,
            num_mobs=2,
            class_power=0.5,
            max_ratio=32.0,
        )
        | kwargs
    )
    return ViewMapLoss(**kwargs)


def decodable(batch=64, frames=3, dim=16, tiles=6, classes=4, mobs=2, seed=0):
    """A latent that holds the map: block classes are a linear function of it,
    the mob flags are its signs."""
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(batch, frames, dim, generator=g)
    w = torch.randn(dim, tiles * classes, generator=g)
    blocks = (emb @ w).reshape(batch, frames, tiles, classes).argmax(-1)
    flags = (emb[..., :mobs] > 0).float()
    return emb, blocks, flags[:, :, None, :].expand(-1, -1, tiles, -1)


# --- targets ---------------------------------------------------------------


def test_targets_follow_the_state_column():
    state = planted_state()
    emb = torch.zeros(2, 4, 8)
    blocks, mobs = view_map_targets({'state_level1': state}, 1, emb)
    assert blocks.shape == (2, 4, TILES) and mobs.shape == (2, 4, TILES, MOBS)
    assert blocks[0, 0, COAL_TILE] == COAL and blocks[1, 0, COAL_TILE] == GRASS
    assert mobs[1, 2, COAL_TILE + 1, ZOMBIE] == 1
    assert mobs.sum() == 4  # one tile in each of the 4 frames
    assert mobs.dtype == torch.float32


def test_targets_refuse_a_missing_column_a_wrong_level_and_other_frames():
    emb = torch.zeros(2, 4, 8)
    with pytest.raises(KeyError, match='state'):
        view_map_targets({}, 1, emb)
    with pytest.raises(ValueError, match='level-1'):
        view_map_targets({'state_level1': planted_state()}, 2, emb)
    with pytest.raises(ValueError, match='frames'):
        view_map_targets({'state_level1': planted_state(frames=3)}, 1, emb)


# --- the loss --------------------------------------------------------------


def test_class_weights_lift_the_rare_classes_up_to_the_cap():
    loss = small_loss(class_power=0.5, max_ratio=20.0)
    counts = torch.tensor([10000, 100, 1, 0])
    w = loss._weights(counts)
    # count^-0.5: 100x rarer is 10x heavier, 10000x rarer would be 100x
    assert w[1] / w[0] == pytest.approx(10.0)
    assert w[2] / w[0] == pytest.approx(20.0)  # capped at max_ratio
    assert w[3] == 1.0  # absent in the batch: never used
    assert (small_loss(class_power=0)._weights(counts) == 1).all()


def test_a_single_class_gives_the_plain_cross_entropy():
    # all tiles one class: the weights cancel in the weighted mean
    emb = torch.randn(4, 3, 16)
    blocks = torch.full((4, 3, 6), 2)
    mobs = torch.zeros(4, 3, 6, 2)
    torch.manual_seed(0)
    weighted = small_loss(class_power=0.5)
    torch.manual_seed(0)
    plain = small_loss(class_power=0)
    plain.load_state_dict(weighted.state_dict())
    a = weighted(emb, blocks, mobs)
    b = plain(emb, blocks, mobs)
    assert weighted.last_metrics['block'] == pytest.approx(
        plain.last_metrics['block'].item(), rel=1e-5
    )
    assert torch.isfinite(a) and torch.isfinite(b)


def test_the_players_tile_is_left_out():
    mask = torch.ones(6, dtype=torch.bool)
    mask[3] = False
    loss = small_loss(tile_mask=mask)
    emb, blocks, mobs = decodable()
    base = loss(emb, blocks, mobs).item()
    other = blocks.clone()
    other[:, :, 3] = (other[:, :, 3] + 1) % 4  # another label under the player
    assert loss(emb, other, mobs).item() == pytest.approx(base)
    other = blocks.clone()
    other[:, :, 2] = (other[:, :, 2] + 1) % 4
    assert loss(emb, other, mobs).item() != pytest.approx(base)


def test_the_decoder_reads_a_latent_that_holds_the_map():
    torch.manual_seed(0)
    loss = small_loss(hidden_dim=128)
    emb, blocks, mobs = decodable()
    opt = torch.optim.Adam(loss.parameters(), lr=3e-3)
    first = loss(emb, blocks, mobs).item()
    for _ in range(500):
        opt.zero_grad()
        value = loss(emb, blocks, mobs)
        value.backward()
        opt.step()
    assert value.item() < 0.5 * first
    m = loss.last_metrics
    assert m['accuracy'] > 0.7 and m['balanced_accuracy'] > 0.6
    assert m['mob_recall'] > 0.8


def test_the_gradient_reaches_the_latent():
    # this is no probe: a latent that does not hold the map is pushed to
    emb, blocks, mobs = decodable()
    emb.requires_grad_(True)
    small_loss()(emb, blocks, mobs).backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0


# --- config and training step ---------------------------------------------

hydra = pytest.importorskip('hydra')
pytest.importorskip('transformers')
pytest.importorskip('stable_pretraining')
pytest.importorskip('sklearn')


def compose_train(overrides=()):
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    with hydra.initialize_config_dir(
        config_dir=str(H_JEPA / 'config' / 'train'), version_base=None
    ):
        return hydra.compose(
            config_name='craftax_hjepa', overrides=['num_levels=1', *overrides]
        )


class FakeModule(nn.Module):
    def __init__(self, model, losses):
        super().__init__()
        self.model = model
        for name, loss in losses.items():
            setattr(self, name, loss)

    def log_dict(self, *args, **kwargs):
        pass


def step(cfg, with_state=True):
    from hjepa_utils import create_world_model, hjepa_forward

    model, losses = create_world_model(cfg)
    frames = 4  # history 3 + rollout 1
    batch = {
        'pixels_level1': torch.randn(2, frames, 3, 63, 63),
        'action_level1': nn.functional.one_hot(
            torch.randint(0, 17, (2, frames)), 17
        ).float(),
    }
    if with_state:
        batch['state_level1'] = planted_state(2, frames)
    out = hjepa_forward(
        FakeModule(model, losses),
        batch,
        'train',
        cfg,
        normalize_batch=lambda b: None,
    )
    return model, out


def test_the_map_decoder_is_off_by_default_and_loads_no_state():
    import data

    cfg = compose_train()
    assert cfg.level1.map_coeff == 0
    assert 'state' not in data.dataset_kwargs(cfg)['keys_to_load']
    model, out = step(cfg, with_state=False)  # no state needed when it is off
    assert not hasattr(model.get_level(1), 'map_loss_fn')
    assert 'map_loss_level1' not in out
    assert 'state' in data.RAW_COLUMNS  # never z-scored
    assert 'state' not in data.normalizer_columns_from_dataset_cfg(
        OmegaConf.create({'keys_to_load': ['pixels', 'action', 'state']})
    )


def test_the_map_decoder_loads_the_state_column_once():
    from data import dataset_kwargs

    cfg = compose_train(['level1.map_coeff=0.1'])
    keys = list(dataset_kwargs(cfg)['keys_to_load'])
    assert keys[-1] == 'state' and keys.count('state') == 1
    others = list(cfg.data.dataset.keys_to_load)
    assert keys[:-1] == others  # the others as before
    cfg.data.dataset.keys_to_load.append('state')
    assert list(dataset_kwargs(cfg)['keys_to_load']).count('state') == 1


def test_a_training_step_with_the_map_decoder():
    cfg = compose_train(['level1.map_coeff=0.1'])
    model, out = step(cfg)
    jepa = model.get_level(1)
    assert isinstance(jepa.map_loss_fn, ViewMapLoss)
    assert jepa.map_loss_fn.tile_mask.sum() == TILES - 1  # without the player
    assert not jepa.map_loss_fn.tile_mask[tg.PLAYER_INDEX]
    for key in ('map_loss', 'map_block_loss', 'map_mob_loss'):
        assert torch.isfinite(out[f'{key}_level1'])
    assert 0 <= out['map_accuracy_loss_level1'] <= 1
    assert 0 <= out['map_balanced_accuracy_loss_level1'] <= 1
    assert torch.isfinite(out['loss'])

    # the map loss alone moves the encoder and the projector, not only the head
    out['map_loss_level1'].backward(retain_graph=True)
    assert jepa.map_loss_fn.decoder[0].weight.grad.abs().sum() > 0
    grads = [p.grad for p in jepa.encoder.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_the_step_names_the_missing_state_column():
    cfg = compose_train(['level1.map_coeff=0.1'])
    with pytest.raises(KeyError, match='state'):
        step(cfg, with_state=False)
