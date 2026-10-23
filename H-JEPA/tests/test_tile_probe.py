"""The faced-block probes on synthetic latents."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

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


def test_per_class_gives_recall_and_the_most_frequent_mistake():
    names = ['a', 'b', 'c']
    yva = torch.tensor([0] * 30 + [1] * 30 + [2] * 5)
    pred = torch.tensor([0] * 30 + [1] * 20 + [0] * 10 + [2] * 5)
    rows = tp.per_class(yva, {'mlp': pred}, names, min_n=20)
    assert [r['block'] for r in rows] == ['a', 'b']  # c is too rare
    b = rows[1]
    assert abs(b['mlp'] - 20 / 30) < 1e-6
    assert b['mistake'] == 'a' and abs(b['mistake_share'] - 10 / 30) < 1e-6
    assert rows[0]['mistake'] is None


def test_the_report_has_a_row_per_class():
    accuracy = {
        'linear': {'train': 0.7, 'val': 0.6},
        'mlp': {'train': 0.9, 'val': 0.8},
    }
    summary = {
        'n_train': 10,
        'n_val': 5,
        'majority': 0.5,
        'accuracy': accuracy,
    }
    rows = [
        {
            'block': 'grass',
            'n': 30,
            'linear': 0.5,
            'mlp': 0.9,
            'mistake': 'sand',
            'mistake_share': 0.05,
        }
    ]
    text = tp.render(summary, rows)
    assert '| grass | 30 | 0.50 | 0.90 | sand (0.05) |' in text
    assert '| mlp | 0.90 | 0.80 |' in text
