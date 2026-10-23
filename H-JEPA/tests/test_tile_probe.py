"""The faced-block probes on synthetic latents and tokens."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

import probe_eval as pe  # noqa: E402
import tile_probe as tp  # noqa: E402


def xor_data(n, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, dim, generator=g)
    y = ((z[:, 0] > 0) ^ (z[:, 1] > 0)).long()  # not linearly separable
    return z, y


def accuracy(pred, y):
    return float((pred == y).float().mean())


def test_the_mlp_reads_what_the_linear_probe_cannot():
    ztr, ytr = xor_data(4000)
    zva, yva = xor_data(1000, seed=1)
    _, lin = tp.train_probe('linear', ztr, ytr, zva, classes=2, epochs=10)
    _, mlp = tp.train_probe('mlp', ztr, ytr, zva, classes=2, epochs=10)
    assert accuracy(lin, yva) < 0.65
    assert accuracy(mlp, yva) > 0.9


def test_both_read_a_linear_signal():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(3000, 8, generator=g)
    y = (z[:, 2] > 0).long()
    for kind in ('linear', 'mlp'):
        _, pva = tp.train_probe(
            kind, z[:2000], y[:2000], z[2000:], classes=2, epochs=60
        )
        assert accuracy(pva, y[2000:]) > 0.95


def test_per_class_recall_mistake_and_the_classes_left_out():
    names = ['a', 'b', 'c', 'd']
    yva = torch.tensor([0] * 60 + [1] * 60 + [2] * 5)
    pred = torch.tensor([0] * 60 + [1] * 40 + [0] * 20 + [2] * 5)
    rows, skipped = tp.per_class(yva, {'mlp': pred}, names, min_n=50)
    assert [r['block'] for r in rows] == ['a', 'b']
    assert skipped == {'c': 5}  # 'd' has no samples at all: not listed
    b = rows[1]
    assert abs(b['mlp'] - 40 / 60) < 1e-6
    assert b['mistake'] == 'a' and abs(b['mistake_share'] - 20 / 60) < 1e-6
    assert rows[0]['mistake'] is None


def test_the_faced_patch_is_the_token_of_the_faced_tile():
    # the player's tile is (3, 4) of the 7 x 9 map; direction 1 left,
    # 2 right, 3 up, 4 down; tokens after the CLS token, row by row
    got = tp.faced_patch_index(torch.tensor([1, 2, 3, 4]))
    expected = [1 + 3 * 9 + 3, 1 + 3 * 9 + 5, 1 + 2 * 9 + 4, 1 + 4 * 9 + 4]
    assert got.tolist() == expected


class FakeViT(nn.Module):
    """Tokens that tell their own position: the value is the token index."""

    def forward(self, x):
        n = x.shape[0]
        index = torch.arange(82).float()[None, :, None].expand(n, 82, 4)
        return SimpleNamespace(last_hidden_state=index.clone())


class FakeModel(nn.Module):
    num_levels = 1

    def __init__(self):
        super().__init__()
        self.vit = FakeViT()
        self.jepa = SimpleNamespace(
            encoder=SimpleNamespace(encoder=self.vit), target_length=3
        )

    def get_level(self, level):
        return self.jepa

    def encode_hierarchical(self, batch):
        pixels = batch['pixels_level1']
        b, t = pixels.shape[:2]
        self.vit(pixels.flatten(0, 1))
        return {'embed_1': torch.zeros(b, t, 4)}


def test_collect_features_reads_the_token_of_the_faced_tile():
    # one clip of 2 frames facing right, the tile to the right is stone (4)
    state = torch.zeros(1, 2, pe.STATE_DIM)
    tiles = state[..., : 7 * 9 * 21].reshape(1, 2, 7, 9, 21)
    tiles[..., 2] = 1  # grass everywhere
    tiles[..., 3, 5, 2] = 0
    tiles[..., 3, 5, 4] = 1  # stone to the right of the player
    batch = {
        'pixels_level1': torch.zeros(1, 2, 3, 63, 63),
        'state_level1': state,
        'facing_level1': torch.full((1, 2, 1), 2),
    }
    feats, block = tp.collect_features(
        FakeModel(), [batch], None, 'cpu', lambda b: None
    )
    assert block.tolist() == [4, 4]
    assert feats['faced_patch'][:, 0].tolist() == [1 + 3 * 9 + 5] * 2
    assert feats['cls'][:, 0].tolist() == [0, 0]
    mean_of_map = sum(range(1, 64)) / 63
    assert abs(float(feats['mean_patches'][0, 0]) - mean_of_map) < 1e-4
    assert feats['latent'].shape == (2, 4)


def test_the_report_has_a_row_per_class_and_names_the_left_out():
    accuracy = {
        f: {
            'linear': {'train': 0.7, 'val': 0.6},
            'mlp': {'train': 0.9, 'val': 0.8},
        }
        for f in tp.FEATURES
    }
    summary = {
        'n_train': 10,
        'n_val': 5,
        'majority': 0.5,
        'accuracy': accuracy,
    }
    row = {
        'block': 'grass',
        'n': 30,
        'mistake': 'sand',
        'mistake_share': 0.05,
        **{f: 0.5 for f in tp.FEATURES},
    }
    text = tp.render(summary, [row], {'lava': 35}, list(tp.FEATURES))
    assert '| grass | 30 | 0.50 | 0.50 | 0.50 | 0.50 | sand (0.05) |' in text
    assert 'lava (35)' in text
    row = '| patch token at the faced tile | 0.70 | 0.60 | 0.90 | 0.80 |'
    assert row in text
