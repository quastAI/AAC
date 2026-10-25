"""The pieces of the probe suite on planted signals: targets, readouts, the
tile sample, swaps, the change test, events, mobs, monotonicity, loss
attribution, verdicts and findings."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

import health  # noqa: E402
from probing import collect as co  # noqa: E402
from probing import encoder as enc  # noqa: E402
from probing import predictor as pr  # noqa: E402
from probing import readout as ro  # noqa: E402
from probing import targets as tg  # noqa: E402
from probing import verdict as vd  # noqa: E402

IRON, PATH, GRASS = (
    tg.BLOCK_NAMES.index(n) for n in ('iron', 'path', 'grass')
)


def make_state(blocks, mobs=()):
    """A `state` vector with the given 63 block ids and (tile, mob) flags."""
    state = torch.zeros(tg.STATE_DIM)
    tiles = state[: 63 * 21].view(63, 21)
    tiles[torch.arange(63), blocks.flatten()] = 1
    for tile, kind in mobs:
        tiles[tile, 17 + kind] = 1
    return state


# --- targets ----------------------------------------------------------------


def test_tile_maps_faced_tile_rings_and_strata():
    blocks = torch.full((7, 9), GRASS)
    blocks[3, 3] = IRON  # left of the player, who stands on (3, 4)
    # a zombie on tile 0, an arrow and a skeleton on tile 1
    b, m = tg.tile_maps(make_state(blocks, [(0, 0), (1, 3), (1, 2)]))
    assert b[30] == IRON and m[0, 0] and m[1, 3] and m[1, 2]
    faced = tg.faced_index(torch.tensor([1, 2, 3, 4]))  # left right up down
    assert faced.tolist() == [30, 32, 22, 40]
    rings = tg.tile_rings()
    assert rings[tg.PLAYER_INDEX] == 0 and rings[30] == 1 and rings[0] == 4
    strata = tg.tile_strata(b, m)
    assert strata[0] == tg.NUM_BLOCKS + 0  # the zombie's stratum
    assert strata[1] == tg.NUM_BLOCKS + 3  # the arrow beats the skeleton
    assert strata[30] == IRON


def test_frame_targets_read_the_state_column():
    blocks = torch.full((7, 9), GRASS)
    blocks[2, 4] = tg.BLOCK_NAMES.index('crafting_table')  # above
    state = torch.stack([make_state(blocks, [(5, 1)]), make_state(blocks)])
    state[1, -1] = 1  # asleep
    zeros = {
        k: torch.zeros(1, 2, d)
        for k, d in (
            ('achievements', 22),
            ('inventory', 12),
            ('vitals', 4),
            ('daylight', 1),
            ('position', 2),
        )
    }
    batch = {f'{k}_level1': v for k, v in zeros.items()}
    batch['state_level1'] = state[None]
    batch['facing_level1'] = torch.tensor([[[3], [1]]])  # up, left
    y = tg.frame_targets(batch)
    assert y['view_blocks'][0, 0, GRASS] == 62
    assert y['facing_block'][0, :, 0].tolist() == [11, GRASS]
    assert y['facing'][0, :, 0].tolist() == [2, 0]  # classes 0..3
    assert y['mobs_present'][0, 0].tolist() == [0, 1, 0, 0]  # a cow
    assert y['sleeping'][0, :, 0].tolist() == [0, 1]
    assert y['blocks'].shape == (1, 2, 63) and y['mobs'].shape == (1, 2, 63, 4)


# --- readouts ---------------------------------------------------------------


def test_ridge_recovers_a_linear_map_and_flags_the_grid_edge():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(400, 8, generator=g)
    y = z @ torch.randn(8, 3, generator=g) + 2.0
    predict, lam, edge = ro.fit_with_validation(z[:300], y[:300])
    assert min(ro.r2_columns(y[300:], predict(z[300:]))) > 0.98
    assert lam == ro.LAMBDAS[0] and edge  # no noise: the weakest strength
    noise = torch.randn(400, 3, generator=g)
    predict, _, _ = ro.fit_with_validation(z[:300], noise[:300])
    assert max(ro.r2_columns(noise[300:], predict(z[300:]))) < 0.15


def test_weighted_auroc_counts_a_sample_as_often_as_its_weight():
    rng = np.random.default_rng(0)
    y = rng.random(60) > 0.5
    s = rng.random(60) + 0.3 * y
    w = rng.integers(1, 4, 60)
    dup = ro.auroc(np.repeat(y, w), np.repeat(s, w))
    assert ro.auroc(y, s, w) == pytest.approx(dup)
    ties = np.array([0] * 5 + [1] * 5)
    assert ro.auroc(ties, np.zeros(10)) == pytest.approx(0.5)
    assert ro.auroc(np.array([1] * 3 + [0] * 7), np.zeros(10)) is None


def test_weighted_recall_balanced_accuracy_and_confusion():
    y = torch.tensor([0, 0, 1, 1, 2])
    p = torch.tensor([0, 1, 1, 1, 0])
    w = torch.tensor([1.0, 3.0, 1.0, 1.0, 1.0])
    n, recall = ro.class_recalls(y, p, 4, w)
    assert n.tolist() == [2, 2, 1, 0]
    assert recall[0] == pytest.approx(0.25) and recall[1] == 1
    assert recall[2] == 0 and np.isnan(recall[3])
    assert ro.balanced_accuracy(n, recall, min_n=2) == pytest.approx(0.625)
    assert ro.confusion(y, p, 4, w)[0, 1] == 3


def test_episode_bootstrap_resamples_whole_episodes():
    groups = np.repeat(np.arange(10), 5)
    x = np.repeat(np.arange(10.0), 5)
    lo, hi = ro.episode_bootstrap(lambda i: x[i].mean(), groups, n=200)
    assert lo < 4.5 < hi and hi - lo < 4.5
    assert ro.episode_bootstrap(lambda i: 1.0, groups, n=10) == (1.0, 1.0)
    lo, hi = ro.episode_bootstrap(
        lambda i: np.array([x[i].mean(), np.nan]), groups, n=20
    )
    assert np.isnan(lo[1]) and not np.isnan(lo[0])
    assert ro.episode_bootstrap(lambda i: 1.0, groups, n=0) is None


def test_spearman_handles_ties_and_constants():
    assert ro.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert ro.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert ro.spearman([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1)
    assert ro.spearman([1, 2, 3], [5, 5, 5]) is None
    assert ro.spearman([1, 2], [1, 2]) is None


def test_heads_read_classes_flags_and_values_at_once():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(3000, 6, generator=g)
    targets = {
        'c': (z[:, :1] > 0).long(),
        'f': (z[:, 1:3] > 0).float(),
        'v': z[:, 3:4] * 5 + 2,
    }
    targets['c'][:100] = -1  # not scored
    heads = [
        ro.Head('c', 'cls', 1, 2),
        ro.Head('f', 'bin', 2),
        ro.Head('v', 'reg', 1),
    ]
    train = {k: v[:2000] for k, v in targets.items()}
    # 2000 samples are 4 steps per epoch: 400 epochs to converge
    out = ro.train_heads('linear', z[:2000], train, heads, epochs=400)(
        z[2000:]
    )
    assert out['c'].shape == (1000, 1, 2)
    right = out['c'][:, 0].argmax(-1) == targets['c'][2000:, 0]
    assert right.float().mean() > 0.95
    flag = ro.auroc(targets['f'][2000:, 0].numpy(), out['f'][:, 0].numpy())
    assert flag > 0.97
    # values come back in their own units (mean 2, std 5)
    assert ro.r2_columns(targets['v'][2000:], out['v'])[0] > 0.9
    assert float(out['v'].mean()) == pytest.approx(2, abs=0.5)


def test_a_step_budget_counts_whole_epochs():
    z, y = torch.randn(2000, 4), torch.randint(0, 2, (2000, 1))
    head = [ro.Head('y', 'cls', 1, 2)]
    by_steps = ro.train_heads('linear', z, {'y': y}, head, steps=10)(z)['y']
    by_epochs = ro.train_heads('linear', z, {'y': y}, head, epochs=3)(z)['y']
    assert torch.equal(by_steps, by_epochs)  # 4 batches per epoch: 3 epochs


def xor_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 8, generator=g)
    return z, ((z[:, 0] > 0) ^ (z[:, 1] > 0)).long()[:, None]


def test_the_mlp_reads_what_the_linear_probe_cannot():
    (ztr, ytr), (zva, yva) = xor_data(4000, 0), xor_data(1000, 1)
    head = [ro.Head('y', 'cls', 1, 2)]
    acc = {}
    for kind in ('linear', 'mlp'):
        out = ro.train_heads(kind, ztr, {'y': ytr}, head, epochs=10)(zva)
        acc[kind] = (out['y'][:, 0].argmax(-1) == yva[:, 0]).float().mean()
    assert acc['linear'] < 0.65 and acc['mlp'] > 0.9


def test_the_balanced_loss_finds_a_rare_class_the_plain_loss_misses():
    g = torch.Generator().manual_seed(0)
    n = 8000
    y = (torch.rand(n, generator=g) < 0.04).long()
    z = torch.randn(n, 8, generator=g)
    z[:, 0] += 1.2 * y  # overlapping: the common class wins when unweighted
    recall = {}
    for balanced in (False, True):
        out = ro.train_heads(
            'linear',
            z[:4000],
            {'y': y[:4000, None]},
            [ro.Head('y', 'cls', 1, 2)],
            epochs=30,
            balanced=balanced,
        )(z[4000:])
        rare = y[4000:] == 1
        recall[balanced] = (out['y'][rare, 0].argmax(-1) == 1).float().mean()
    assert recall[True] > recall[False] + 0.2


# --- the collection pass ----------------------------------------------------


def test_tile_crops_paste_and_crop_agree():
    raw = torch.arange(3 * 63 * 63, dtype=torch.int32).reshape(1, 3, 63, 63)
    tile = 3 * 9 + 5
    expected = raw[0, :, 21:28, 35:42].reshape(-1)
    assert torch.equal(co.tile_crops(raw)[0, tile], expected)
    assert torch.equal(co.crop_of(raw[0], tile).reshape(-1), expected)
    patch = torch.zeros(3, 7, 7, dtype=raw.dtype)
    edited = co.paste(raw[0], tile, patch)
    assert torch.equal(co.crop_of(edited, tile), patch)
    assert torch.equal(edited[:, :21], raw[0, :, :21])  # the rest untouched


def test_the_reservoir_caps_each_stratum_and_weights_it_back():
    res = co._Reservoir(cap=5, seed=0)
    offset = 0
    for _ in range(4):
        for stratum, n in ((0, 10), (1, 2)):
            ids = torch.arange(n) + offset
            offset += n
            keep_old, keep_new = res.offer(stratum, n)
            res.store(stratum, keep_old, {'id': ids[keep_new]})
    out = res.result()
    assert len(out['id']) == 10 and len(torch.unique(out['id'])) == 10
    assert sorted(set(out['weight'].tolist())) == pytest.approx([1.6, 8])
    assert out['seen'] == {0: 40, 1: 8}


class FakeLayer(nn.Module):
    def forward(self, x):
        return (x + 1,)


class FakeViT(nn.Module):
    """Transformers-4 (`encoder.layer`) or -5 (`layers`) layout."""

    def __init__(self, v5):
        super().__init__()
        blocks = nn.ModuleList([FakeLayer() for _ in range(4)])
        if v5:
            self.layers = blocks
        else:
            self.encoder = nn.Module()
            self.encoder.layer = blocks
        self.blocks = blocks

    def forward(self, x):
        h = torch.zeros(len(x), 82, 4)
        for block in self.blocks:
            h = block(h)[0]
        return SimpleNamespace(last_hidden_state=h * 10)


@pytest.mark.parametrize('v5', [False, True])
def test_the_tap_reads_layers_in_both_layouts(v5):
    vit = FakeViT(v5)
    with co.EncoderTap(vit, (2, 9)) as tap:  # layer 9 does not exist
        vit(torch.zeros(3, 3, 63, 63))
    final, layers = tap.take()
    assert final.shape == (3, 82, 4) and final[0, 0, 0] == 40
    assert set(layers) == {2} and layers[2][0, 0, 0] == 2
    assert not vit._forward_hooks  # removed


def test_sprites_come_from_free_tiles_only():
    g = torch.Generator().manual_seed(0)
    raw = torch.randint(0, 255, (2, 3, 63, 63), generator=g, dtype=torch.uint8)
    blocks = torch.full((2, 63), GRASS)
    blocks[:, 0] = IRON
    blocks[:, tg.PLAYER_INDEX] = tg.BLOCK_NAMES.index('lava')
    mobs = torch.zeros(2, 63, 4, dtype=torch.bool)
    mobs[:, 5, 0] = True  # a zombie on grass
    sprites = {'blocks': {}, 'mobs': {}}
    co._add_sprites(sprites, raw, blocks, mobs, torch.tensor([0, 1]))
    assert torch.equal(sprites['blocks'][IRON][0], co.crop_of(raw[0], 0))
    assert len(sprites['blocks'][GRASS]) == co.SPRITES_PER_CLASS
    assert tg.BLOCK_NAMES.index('lava') not in sprites['blocks']  # player
    assert torch.equal(sprites['mobs'][(0, GRASS)][0], co.crop_of(raw[0], 5))


# --- encoder ----------------------------------------------------------------


def test_a_perfect_map_prediction_scores_one():
    g = torch.Generator().manual_seed(0)
    blocks = torch.randint(2, 6, (40, 63), generator=g)
    mobs = torch.rand(40, 63, 4, generator=g) < 0.2
    split = {
        'y': {
            'blocks': blocks,
            'mobs': mobs,
            'facing_raw': torch.randint(1, 5, (40,), generator=g),
        }
    }
    e = enc.evaluate_tiles(blocks.clone(), mobs.float() * 2 - 1, split, 1)
    assert e['accuracy'] == 1 and e['balanced'] == 1
    assert e['faced']['accuracy'] == 1
    assert e['mob_auroc'] == [1.0] * 4
    assert set(e['rings']) == set(tg.RINGS)


def test_causes_follow_the_planted_relations():
    rank = np.arange(17.0)
    res = {
        'mlp balanced': {
            'n': [50] * 17,
            'recall': list(np.linspace(0.1, 0.9, 17)),
            'confusion': (np.eye(17) * 10 + 1).tolist(),
        }
    }
    shares = np.linspace(0.01, 0.2, 17)
    distances = np.abs(np.subtract.outer(rank, rank)) / 17
    c = enc.causes({'latent': res}, {}, shares, distances, min_n=20)
    stage = c['stages']['latent']
    assert stage['recall_vs_log_share'] == pytest.approx(1)
    assert stage['classes'] == 17 and stage['pairs'] == 17 * 16
    assert c['nearest_to'][0] == 'out_of_bounds'


def test_sprite_distances_compare_the_first_crops():
    sprites = {
        'blocks': {
            2: [torch.zeros(3, 7, 7, dtype=torch.uint8)],
            9: [torch.full((3, 7, 7), 255, dtype=torch.uint8)],
        }
    }
    d = enc.sprite_distances(sprites)
    assert d[2, 9] == pytest.approx(1) and d[2, 2] == 0
    assert np.isnan(d[3, 2])


def test_distance_sensitivity_finds_what_moves_the_latent():
    g = torch.Generator().manual_seed(0)
    B, T = 4, 20
    facing = torch.randint(1, 5, (B * T,), generator=g)
    z = F.one_hot(facing - 1, 4).float() * 3  # a turn moves it by 18
    y = {
        k: torch.zeros(B * T, d)
        for k, d in (
            ('inventory', 12),
            ('vitals', 4),
            ('daylight', 1),
            ('sleeping', 1),
            ('achievements', 22),
        )
    }
    y.update(
        position=torch.zeros(B * T, 2),
        facing_raw=facing,
        mobs=torch.zeros(B * T, 63, 4, dtype=torch.bool),
        blocks=torch.zeros(B * T, 63, dtype=torch.long),
    )
    split = {'clips': B, 'T': T, 'frames': {'latent': z}, 'y': y}
    out = enc.distance_sensitivity(split, scale=1.0, gaps=(1,))[1]
    assert set(out['features']) == {'turned'}
    assert out['features']['turned']['coef'] == pytest.approx(18, abs=1e-4)
    assert out['r2'] == pytest.approx(1)


class TileMeanViT(nn.Module):
    """CLS token = the mean of each tile's pixels (63 dims)."""

    def forward(self, x, interpolate_pos_encoding=None):
        h = torch.zeros(len(x), 82, 63)
        h[:, 0] = co.tile_crops(x).mean(-1)
        return SimpleNamespace(last_hidden_state=h)


class PixelModel(nn.Module):
    num_levels = 1

    def __init__(self):
        super().__init__()
        from models.jepa import ProjectedEncoder

        self.jepa = nn.Module()
        self.jepa.encoder = ProjectedEncoder(TileMeanViT(), nn.Identity())

    def get_level(self, level):
        return self.jepa


def test_a_swap_moves_the_latent_and_a_null_swap_does_not():
    B, T = 2, 5
    pixels = torch.zeros(3, 3, 63, 63, dtype=torch.uint8)  # all grass
    steps = torch.zeros(B * T, 63)
    steps[:, 0] = torch.arange(T).repeat(B).float()  # a step of 1
    split = {
        'clips': B,
        'T': T,
        'frames': {'latent': steps, 'cls': steps},
        'episode': torch.tensor([0] * T + [1] * T),
        'y': {
            'blocks': torch.full((B * T, 63), GRASS),
            'mobs': torch.zeros(B * T, 63, 4, dtype=torch.bool),
            'sleeping': torch.zeros(B * T, 1),
        },
        'swap': {'pixels': pixels, 'frame': torch.tensor([0, 1, 5])},
    }
    sprites = {
        'blocks': {
            GRASS: [torch.zeros(3, 7, 7, dtype=torch.uint8)],
            IRON: [torch.full((3, 7, 7), 255, dtype=torch.uint8)],
        },
        'mobs': {},
    }

    def to_unit(batch):
        batch['pixels'] = batch['pixels'].float() / 255

    out = enc.swap_sensitivity(
        PixelModel(),
        split,
        sprites,
        {},
        normalize=to_unit,
        device='cpu',
        boot=0,
    )['targets']
    assert out['null']['latent_shift'] == 0
    assert out['null']['projector_ratio'] is None  # the CLS did not move
    assert 'block:grass' not in out  # every tile is grass already
    assert out['block:iron']['latent_shift'] == pytest.approx(1)
    assert out['block:iron']['projector_ratio'] == pytest.approx(1)


# --- predictor --------------------------------------------------------------


class LookupJepa:
    """Returns the next latent of a table (exact), or copies the context."""

    def __init__(self, table, exact=True):
        self.table, self.exact = table, exact
        self.predictor = nn.Identity()

    def predict(self, ctx, act):
        last = ctx[:, -1]
        if not self.exact:
            return ctx
        b, t = last[:, 0].long(), last[:, 1].long()
        nxt = self.table[b, (t + 1).clamp(max=self.table.shape[1] - 1)]
        return torch.cat([ctx[:, 1:], nxt[:, None]], dim=1)


class ShiftJepa(LookupJepa):
    """Predicts the context's last latent plus one: a constant error."""

    def predict(self, ctx, act):
        return ctx + 1


def index_latents(B, T, *extra):
    """Latents whose first two entries are the clip and the frame index."""
    b = torch.arange(B)[:, None].expand(B, T).float()
    t = torch.arange(T)[None].expand(B, T).float()
    return torch.cat([b[..., None], t[..., None], *extra], dim=-1)


def flat_split(emb, y, action=None, **extra):
    B, T = emb.shape[:2]
    split = {
        'clips': B,
        'T': T,
        'frames': {'latent': emb.flatten(0, 1)},
        'act': torch.zeros(B * T, 1),
        'action': (
            torch.full((B * T,), -1) if action is None else action.flatten()
        ),
        'clip_episode': torch.arange(B),
        'source': None,
        'y': {k: v.flatten(0, 1) for k, v in y.items()},
    }
    split['y'].setdefault('sleeping', torch.zeros(B * T, 1))
    split.update(extra)
    return split


def test_open_loop_aligns_predictions_with_their_frames():
    emb = index_latents(3, 20)
    pred = pr.open_loop(
        LookupJepa(emb),
        emb[:, :8],
        torch.zeros(3, 8, 1),
        history=2,
        steps=6,
        device='cpu',
    )
    assert pred.shape == (3, 6, 2)
    assert pred[:, :, 1].tolist() == [[2, 3, 4, 5, 6, 7]] * 3


def coal_clips(B=4, T=24):
    """Coal rises every 5 frames; the latent carries it exactly."""
    coal = (torch.arange(T) // 5).float()[None, :, None].expand(B, T, 1)
    inventory = torch.zeros(B, T, 12)
    inventory[..., 2:3] = coal
    return index_latents(B, T, inventory), inventory


def test_the_change_test_tells_a_predictor_from_a_copy():
    emb, inventory = coal_clips()
    split = flat_split(emb, {'inventory': inventory})
    spec = {'inventory': tg.TARGETS['inventory']}
    fitted = ro.fit_targets(
        emb.flatten(0, 1), {'inventory': inventory.flatten(0, 1)}, spec
    )
    kw = dict(history=1, horizons=(1, 4, 8), every=2, device='cpu', boot=5)
    exact = pr.change_test(fitted, split, LookupJepa(emb), **kw)
    coal = exact['by_k'][4]['inventory']['columns']['coal']
    assert coal['changed'] > 0 and coal['pred_changed'] == 1
    assert coal['copy_changed'] == 0 and coal['recovered'] == 1
    assert coal['shown'] == pytest.approx(1, abs=1e-3)
    assert exact['by_k'][4]['inventory']['columns']['wood']['changed'] == 0
    assert exact['by_k'][4]['inventory']['pred_changed_interval'] is not None
    lazy = pr.change_test(fitted, split, LookupJepa(emb, exact=False), **kw)
    coal = lazy['by_k'][4]['inventory']['columns']['coal']
    assert coal['recovered'] == 0 and coal['pred_same'] == 1


def iron_clips(B=4, T=30):
    """`do` facing left every step. Iron is faced except right after a
    successful press (every 4th step), when the tile has become path."""
    t = torch.arange(T)
    mined = t % 4 == 1  # the frame after a success
    iron = torch.cumsum(mined.float(), 0)
    inventory = torch.zeros(B, T, 12)
    inventory[..., 3] = iron
    blocks = torch.full((B, T, 63), GRASS)
    blocks[:, :, 30] = torch.where(mined, PATH, IRON)
    flag = mined.float()[None, :, None].expand(B, T, 1)
    emb = index_latents(B, T, inventory[..., 3:4], flag)
    action = torch.full((B, T), tg.DO)
    action[:, -1] = -1
    y = {
        'inventory': inventory,
        'vitals': torch.zeros(B, T, 4),
        'blocks': blocks,
        'facing_raw': torch.ones(B, T, dtype=torch.long),
    }
    return emb, y, action


def path_decoder(z):
    """Reads path at every tile when the latent's flag is set, else iron."""
    logits = torch.zeros(len(z), 63, 17)
    logits[:, :, PATH] = (z[:, 3:4] > 0.5).float() * 10
    logits[:, :, IRON] = (z[:, 3:4] <= 0.5).float() * 10
    return {'blocks': logits}


def test_collect_events_show_the_item_and_the_mined_tile():
    emb, y, action = iron_clips()
    split = flat_split(emb, y, action)
    spec = {k: tg.TARGETS[k] for k in ('inventory', 'vitals')}
    fitted = ro.fit_targets(
        emb.flatten(0, 1),
        {k: v.flatten(0, 1) for k, v in y.items() if k in spec},
        spec,
    )
    exact = pr.event_probes(
        fitted, split, LookupJepa(emb), path_decoder, history=1
    )
    e = exact['collect_iron']
    assert e['succeeded'] == 4 * 8 and e['failed'] == 4 * 14
    assert e['shown'] == pytest.approx(1, abs=0.05)
    assert e['tile_changes'] == 32
    assert (e['tile_pred'], e['tile_enc'], e['tile_copy']) == (1, 1, 0)
    assert exact['collect (all)']['succeeded'] == 32
    lazy = pr.event_probes(
        fitted, split, LookupJepa(emb, exact=False), path_decoder, history=1
    )
    assert lazy['collect_iron']['shown'] == pytest.approx(0, abs=0.05)
    assert lazy['collect_iron']['tile_pred'] == 0


def test_a_predicted_arrow_flies_on_and_a_copy_stays_behind():
    B, T = 2, 15
    arrows = torch.zeros(B, T, 63, 4, dtype=torch.bool)
    t = torch.arange(T)
    col = 1 + t % 7  # one tile right per step along row 1, then a new one
    arrows[:, t, 9 + col, 3] = True
    emb = index_latents(B, T, arrows[..., 3].float())

    def decoder(z):
        mobs = torch.zeros(len(z), 63, 4)
        mobs[..., 3] = z[:, 2:] * 10 - 5
        return {'mobs': mobs.reshape(len(z), -1)}

    split = flat_split(
        emb,
        {'mobs': arrows, 'position': torch.zeros(B, T, 2, dtype=torch.long)},
    )
    out = pr.mob_dynamics(split, LookupJepa(emb), decoder, history=1)
    arrow = out['arrow']
    assert arrow['continuing'] == B * 12 and arrow['new'] == B * 2
    assert arrow['continuing_pred'] == 1 and arrow['continuing_enc'] == 1
    assert arrow['continuing_copy'] < 0.5
    assert out['zombie']['continuing'] == 0


def test_monotonicity_of_an_exact_predictor_and_of_a_copy():
    B, T = 3, 30
    emb = index_latents(B, T)
    split = flat_split(emb, {'inventory': torch.zeros(B, T, 12)})
    kw = dict(history=1, offsets=(4, 8), every=4, device='cpu')
    out = pr.monotonicity(split, LookupJepa(emb), **kw)
    assert not out['expert_only']
    e = out['offsets'][8]['all']
    assert e['encoded']['rho_median'] == pytest.approx(1)
    assert e['encoded']['falls'] == 1 and e['predicted']['falls'] == 1
    assert e['predicted']['final_over_start'] == 0
    copy = pr.monotonicity(split, LookupJepa(emb, exact=False), **kw)
    p = copy['offsets'][8]['all']['predicted']
    assert p['falls'] == 0 and p['final_over_start'] == 1


def test_loss_attribution_splits_the_loss_by_transition():
    emb, y, action = iron_clips(B=2, T=12)
    action[:, :3] = tg.ACTION_NAMES.index('left')
    sleeping = torch.zeros(2, 12, 1)
    sleeping[:, 3] = 1
    y['sleeping'] = sleeping
    y['mobs'] = torch.zeros(2, 12, 63, 4, dtype=torch.bool)
    split = flat_split(emb, y, action)
    out = pr.loss_attribution(split, ShiftJepa(emb), history=1)
    rows = out['rows']
    assert {
        'move',
        'asleep',
        'collect (succeeded)',
        'press without effect',
    } <= set(rows)
    assert sum(r['share'] for r in rows.values()) == pytest.approx(1)
    assert sum(r['loss_share'] for r in rows.values()) == pytest.approx(1)
    assert rows['move']['transitions'] == 2 * 3
    assert out['overlapping'] == {}  # no mobs


def test_idm_accuracy_per_action_counts_awake_steps_and_outcomes():
    B, T = 2, 10
    stone = torch.zeros(B, T)
    for t in range(1, T):
        stone[:, t] = stone[:, t - 1] - (1 if t % 2 == 0 else 0)
    inventory = torch.zeros(B, T, 12)
    inventory[..., 1] = stone + 100
    action = torch.full((B, T), 7)  # place_stone
    action[:, :4] = 0  # noop first
    action[:, -1] = -1
    sleeping = torch.zeros(B, T, 1)
    sleeping[:, 1] = 1

    class AlwaysNoop(nn.Module):
        num_slots, num_classes = 1, 17

        def idm(self, a, b, extra):
            logits = torch.zeros(len(a), 17)
            logits[:, 0] = 1
            return logits

    split = flat_split(
        index_latents(B, T),
        {'inventory': inventory, 'sleeping': sleeping},
        action,
    )
    out = pr.idm_per_action(
        split, SimpleNamespace(idm_loss_fn=AlwaysNoop()), 'cpu'
    )
    assert out['asleep']['steps'] == 2 and out['awake']['steps'] == 16
    assert out['actions']['noop'] == {'steps': 6, 'accuracy': 1.0}
    placed = out['actions']['place_stone']
    assert placed['steps'] == 10 and placed['accuracy'] == 0.0
    assert placed['succeeded']['steps'] + placed['failed']['steps'] == 10


# --- verdicts and findings --------------------------------------------------


def test_the_chain_stops_at_the_first_link_that_fails():
    assert (
        vd.chain(
            [('raw', 0.9), ('final', 0.85), ('cls', 0.6), ('latent', 0.5)]
        )
        == 'lost at cls'
    )
    assert vd.chain([('raw', 0.1), ('latent', 0.9)]) == 'not readable from raw'
    assert vd.chain([('raw', None), ('final', 0.9), ('latent', 0.8)]) == 'kept'
    assert vd.chain([('raw', None)]) == 'no data'
    links = [('swap', 0.9), ('one step', 0.2), ('rollout', 0.1)]
    assert vd._later('kept', links) == 'lost at one step'
    assert vd._later('lost at cls', links) == 'lost at cls'


def tile_result(recall, n=100):
    r = [0.0] * 17
    r[IRON] = recall
    counts = [0] * 17
    counts[IRON] = n
    return {
        'mlp balanced': {
            'n': counts,
            'recall': r,
            'mob_auroc': [0.9, None, 0.5, 0.6],
        }
    }


def test_entity_verdicts_walk_tiles_mobs_and_items():
    summary = {
        'tokens': {'raw': tile_result(1.0), 'final': tile_result(0.98)},
        'maps': {'cls': tile_result(0.7), 'latent': tile_result(0.3)},
        'swaps': {
            'targets': {
                'block:iron': {
                    'latent_reads_after': 0.4,
                    'latent_reads_before': 0.1,
                }
            }
        },
        'events': {'collect_iron': {'shown': 0.9, 'succeeded': 30}},
        'strip': {
            s: {'inventory': {'columns': {'iron': v}}}
            for s, v in (
                ('raw', 0.9),
                ('tokens', 0.9),
                ('cls', 0.85),
                ('latent', 0.8),
            )
        },
        'change_plan': {
            'by_k': {
                8: {'inventory': {'columns': {'iron': {'recovered': 0.2}}}}
            }
        },
    }
    v = vd.entity_verdicts(summary, min_n=20)
    iron = next(r for r in v['blocks'] if r['entity'] == 'iron')
    assert iron['verdict'] == 'lost at cls'
    assert dict(iron['links'])['swap'] == pytest.approx(0.3)
    zombie = next(r for r in v['mobs'] if r['entity'] == 'zombie')
    assert zombie['stages'][0][1] == pytest.approx(0.8)  # 2 * 0.9 - 1
    item = next(r for r in v['items'] if r['entity'] == 'inventory iron')
    assert item['verdict'] == 'lost at rollout'
    assert item['event'] == 'collect_iron'


def test_findings_of_the_change_test_events_and_verdicts():
    change = {
        'by_k': {
            1: {
                'facing': {
                    'recovered': 0.9,
                    'pred_same': 0.9,
                    'copy_same': 0.95,
                },
                'inventory': {
                    'recovered': 0.2,
                    'pred_same': 0.7,
                    'copy_same': 0.99,
                },
            },
            2: {'facing': {'recovered': 0.6}, 'inventory': {'recovered': 0.9}},
            4: {'facing': {'recovered': 0.3}, 'inventory': {}},
        }
    }
    summary = {
        'change_plan': change,
        'events': {
            'collect (all)': {'shown': 0.1, 'succeeded': 40, 'failed': 60}
        },
        'state': {'lambda_at_edge': ['inventory']},
        'verdicts': {
            'blocks': [
                {'entity': 'iron', 'verdict': 'lost at cls'},
                {'entity': 'grass', 'verdict': 'kept'},
            ]
        },
    }
    found = {f.code: f for f in health.diagnose_probing(summary)}
    assert (
        '2 (facing), 0 (inventory)' in found['PROBE_ROLLOUT_HORIZON'].message
    )
    assert 'inventory' in found['PROBE_INVENTS_CHANGES'].message
    assert 'collect' in found['PROBE_EVENT_IGNORED'].message
    assert found['PROBE_VERDICT'].message == 'lost at cls: iron'
    assert (
        'PROBE_LAMBDA_EDGE' in found and 'PROBE_ROLLOUT_NO_GAIN' not in found
    )
    none = {'change_plan': {'by_k': {1: {'facing': {'recovered': 0.1}}}}}
    codes = {f.code for f in health.diagnose_probing(none)}
    assert 'PROBE_ROLLOUT_NO_GAIN' in codes
