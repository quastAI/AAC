"""Balanced clip sampling (the ablation A1): the faced block and the events per
frame, the items per clip, scenes, the capped weights and the sampler."""

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

import balance  # noqa: E402
from data import dataset_kwargs  # noqa: E402
from probing import targets as tg  # noqa: E402

COAL, DIAMOND, GRASS = (
    tg.BLOCK_NAMES.index(n) for n in ('coal', 'diamond', 'grass')
)
LEFT = 1  # facing 1 looks at the tile to the left of the player
FACED_TILE = tg.PLAYER_TILE[0] * tg.VIEW_COLS + tg.PLAYER_TILE[1] - 1
SPAN = 4
EPISODES, LENGTH = 6, 200
COLLECT_COAL = tg.COLLECT['coal'][0]


def planted_file(path, coal_at=(20, 21, 22, 23), diamond_in=(0,)):
    """Six episodes of 200 frames facing left. In each, the faced tile is coal
    in `coal_at` and the player collects it at frame 21 (`do`, coal + 1 at
    frame 22); a diamond is faced at frames 30-31 in the episodes `diamond_in`.
    Episode 1 also crafts a wooden pickaxe at frame 40 and the player of
    episode 2 presses `do` on coal at frame 21 without getting it."""
    n = EPISODES * LENGTH
    state = np.zeros((n, tg.STATE_DIM), np.uint8)
    tiles = state[:, : tg.VIEW_TILES * tg.VIEW_CHANNELS].reshape(
        n, tg.VIEW_TILES, tg.VIEW_CHANNELS
    )
    tiles[:, :, GRASS] = 1
    action = np.zeros((n, 1), np.float32)
    inventory = np.zeros((n, 12), np.int16)
    vitals = np.full((n, 4), 9, np.int16)
    for e in range(EPISODES):
        o = e * LENGTH
        for t in coal_at:
            tiles[o + t, FACED_TILE, GRASS] = 0
            tiles[o + t, FACED_TILE, COAL] = 1
        action[o + 21] = tg.DO
        if e != 2:
            inventory[o + 22 : o + LENGTH, 2] = 1  # coal + 1 after the do
        if e in diamond_in:
            for t in (30, 31):
                tiles[o + t, FACED_TILE, GRASS] = 0
                tiles[o + t, FACED_TILE, DIAMOND] = 1
        if e == 1:
            action[o + 40] = tg.ACTION_NAMES.index('make_wood_pickaxe')
            inventory[o + 41 : o + LENGTH, 6] = 1
        action[o + LENGTH - 1] = np.nan  # no action at the last frame
    with h5py.File(path, 'w') as f:
        f['ep_len'] = np.full(EPISODES, LENGTH)
        f['ep_offset'] = np.arange(EPISODES) * LENGTH
        f['state'] = state
        f['facing'] = np.full((n, 1), LEFT, np.int64)
        f['action'] = action
        f['inventory'] = inventory
        f['vitals'] = vitals


def fake_dataset(path, pad=0):
    lengths = np.full(EPISODES, LENGTH)
    clips = [
        (e, s) for e in range(EPISODES) for s in range(LENGTH + pad - SPAN + 1)
    ]
    return SimpleNamespace(
        h5_path=Path(path),
        lengths=lengths,
        offsets=np.arange(EPISODES) * LENGTH,
        clip_indices=clips,
        last_level={'span': SPAN},
    )


@pytest.fixture()
def data(tmp_path):
    path = tmp_path / 'train.h5'
    planted_file(path)
    return fake_dataset(path)


def planted(path):
    with h5py.File(path, 'r') as f:
        total = EPISODES * LENGTH
        faced = balance.faced_blocks(f, total)
        event = balance.frame_events(
            f['action'][:, 0].astype(np.float64),
            f['inventory'][:],
            f['vitals'][:],
            faced,
            np.full(EPISODES, LENGTH),
            np.arange(EPISODES) * LENGTH,
        )
    return faced, event


def test_the_faced_block_follows_state_and_facing(data):
    faced, _ = planted(data.h5_path)
    assert faced[20] == COAL and faced[23] == COAL and faced[24] == GRASS
    assert faced[30] == DIAMOND  # episode 0
    assert faced[LENGTH + 30] == GRASS  # not in the others


def test_events_follow_the_success_rule_of_the_probe_suite(data):
    faced, event = planted(data.h5_path)
    names = balance.EVENTS
    assert event[21] == names.index(COLLECT_COAL)
    # a `do` on coal that gave nothing is no event
    assert event[2 * LENGTH + 21] == -1
    assert event[LENGTH + 40] == names.index('make_wood_pickaxe')
    # the last frame of an episode has no transition
    assert (event[LENGTH - 1 :: LENGTH] == -1).all()
    # a coal collect in five episodes (not the second one) and the craft
    assert (event >= 0).sum() == EPISODES - 1 + 1


def test_clips_hold_the_items_of_their_frames_and_transitions(data):
    faced, event = planted(data.h5_path)
    has, ep, start = balance.clip_table(
        faced, event, data.clip_indices, data.offsets, data.lengths, SPAN
    )
    item = balance.ITEMS.index
    coal, collect = item('faced:coal'), item(f'event:{COLLECT_COAL}')
    first = [i for i, (e, s) in enumerate(data.clip_indices) if (e, s)[0] == 0]
    starts = {data.clip_indices[i][1] for i in first if has[i, coal]}
    assert starts == set(range(17, 24))  # clips that overlap frames 20-23
    # the collect is the transition 21 -> 22: clips of frames s..s+3, s <= 21
    starts = {data.clip_indices[i][1] for i in first if has[i, collect]}
    assert starts == set(range(19, 22))  # transitions s..s+2 hold frame 21
    stats = balance.item_stats(has, ep, start)
    assert stats['scenes'][coal] == EPISODES  # one run of clips per episode
    assert stats['episodes'][coal] == EPISODES
    assert stats['scenes'][item('faced:diamond')] == 1


def test_a_padded_death_frame_adds_no_event_or_block(tmp_path):
    path = tmp_path / 'train.h5'
    planted_file(path)
    padded = fake_dataset(path, pad=3)
    faced, event = planted(path)
    has, *_ = balance.clip_table(
        faced, event, padded.clip_indices, padded.offsets, padded.lengths, SPAN
    )
    assert has.shape[0] == EPISODES * (LENGTH + 3 - SPAN + 1)
    # the pad repeats the last real frame: its block, never an event
    last = [i for i, (e, s) in enumerate(padded.clip_indices) if s == LENGTH]
    assert not has[last][:, len(tg.BLOCK_NAMES) :].any()


def test_rare_items_are_lifted_up_to_the_cap(data):
    weights, summary = balance.compute(
        data, share=0.2, max_weight=4.0, min_scenes=3
    )
    coal = summary['items']['faced:coal']
    assert coal['lifted'] and coal['ratio'] == 4.0  # share 0.2 / ~0.03 > 4
    assert coal['share_balanced'] > 2 * coal['share']
    assert weights.max() == 4.0 and weights.min() == 1.0
    assert summary['clips'] == len(data.clip_indices)
    assert 0 < summary['effective_clips'] < summary['clips']
    grass = summary['items']['faced:grass']
    assert not grass['lifted']  # more than the target share: weight 1


def test_few_scenes_are_not_lifted_however_rare(data):
    _, summary = balance.compute(data, share=0.2, max_weight=4.0, min_scenes=3)
    diamond = summary['items']['faced:diamond']
    assert diamond['scenes'] == 1 and not diamond['lifted']
    assert diamond['share_balanced'] <= diamond['share'] * 1.01  # not lifted
    # allow the single scene and the same item is lifted
    _, summary = balance.compute(data, share=0.2, max_weight=4.0, min_scenes=1)
    assert summary['items']['faced:diamond']['lifted']


def test_the_cap_bounds_the_weight_of_the_rarest_items(data):
    cap2, _ = balance.compute(data, share=0.9, max_weight=2.0, min_scenes=1)
    cap8, _ = balance.compute(data, share=0.9, max_weight=8.0, min_scenes=1)
    assert cap2.max() == 2.0 and cap8.max() == 8.0
    # a clip with several lifted items takes the largest weight, not the sum
    assert cap8.max() <= 8.0


def test_the_sampler_draws_lifted_clips_more_often(data):
    cfg = OmegaConf.create({'share': 0.2, 'max_weight': 4.0, 'min_scenes': 3})
    gen = torch.Generator().manual_seed(0)
    sampler, summary = balance.build_sampler(data, cfg, generator=gen)
    assert len(sampler) == len(data.clip_indices)
    weights, _ = balance.compute(data, 0.2, 4.0, 3)
    drawn = np.array(
        list(
            torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                50_000,
                replacement=True,
                generator=torch.Generator().manual_seed(1),
            )
        )
    )
    lifted = weights > 1
    expected = weights[lifted].sum() / weights.sum()
    assert abs(lifted[drawn].mean() - expected) < 0.01
    assert lifted[drawn].mean() > lifted.mean()  # more than uniform draws


def test_the_weights_are_cached_by_data_and_settings(data, monkeypatch):
    cfg = OmegaConf.create({'share': 0.2, 'max_weight': 4.0, 'min_scenes': 3})
    first, summary = balance.clip_weights_for(data, cfg)
    assert list(data.h5_path.parent.glob('balance_*.npz'))

    def fail(*a, **k):
        raise AssertionError('recomputed')

    monkeypatch.setattr(balance, 'compute', fail)
    again, same = balance.clip_weights_for(data, cfg)
    assert np.array_equal(first, again) and same == summary
    other = OmegaConf.create(
        {'share': 0.2, 'max_weight': 2.0, 'min_scenes': 3}
    )
    with pytest.raises(AssertionError, match='recomputed'):
        balance.clip_weights_for(data, other)  # another cap: a new key


def test_the_training_config_has_it_off_and_apart_from_the_dataset():
    path = Path(__file__).parents[1] / 'h_jepa/config/train/base/craftax.yaml'
    cfg = OmegaConf.load(path)
    assert cfg.data.balance.enabled is False
    assert cfg.data.balance.max_weight > 1 and cfg.data.balance.min_scenes > 1
    cfg.num_levels = 1
    assert 'balance' not in dataset_kwargs(cfg)  # not a dataset argument
