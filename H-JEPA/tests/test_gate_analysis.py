"""What the Sieve's gate would prune (sieve/gate_analysis.py): the patch
labels, the group numbers and the whole pass."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))
sys.path.insert(0, str(Path(__file__).parent))

from probing import targets as tg  # noqa: E402
from sieve import gate_analysis as ga  # noqa: E402

GRASS = tg.BLOCK_NAMES.index('grass')
TREE = tg.BLOCK_NAMES.index('tree')
DIAMOND = tg.BLOCK_NAMES.index('diamond')


def planted(frames):
    """A sample part from block maps (frames, 63) and optional mobs
    {frame: (tile, mob)}: only the view channels of `state` are filled."""
    n = len(frames)
    state = np.zeros((n, tg.STATE_DIM), np.float32)
    view = np.zeros((n, tg.VIEW_TILES, tg.VIEW_CHANNELS), np.float32)
    for i, (blocks, mobs) in enumerate(frames):
        view[i, np.arange(tg.VIEW_TILES), blocks] = 1.0
        for tile, mob in mobs:
            view[i, tile, tg.NUM_BLOCKS + mob] = 1.0
    state[:, : tg.VIEW_TILES * tg.VIEW_CHANNELS] = view.reshape(n, -1)
    return types.SimpleNamespace(
        state=state,
        action=np.zeros(n, np.float32),
        inventory=np.zeros((n, 12), np.float32),
        vitals=np.zeros((n, 4), np.float32),
    )


def grass_with(**tiles):
    blocks = np.full(tg.VIEW_TILES, GRASS)
    for index, block in tiles.items():
        blocks[int(index[1:])] = block
    return blocks


def test_a_move_makes_the_far_column_enter_and_keeps_what_was_in_view():
    # the player steps right: what was at tile 11 (row 1, column 2) is now
    # at tile 10; column 8 is new
    part = planted(
        [(grass_with(t11=DIAMOND), []), (grass_with(t10=DIAMOND), [])]
    )
    part.action[0] = 2  # right
    part.inventory[1, 4] = 1  # diamond collected: the strip changes
    lab = ga.labels(part, np.array([0]), np.array([1]))
    stratum, kind = lab['stratum'][0], lab['kind'][0]
    assert lab['moved'][0] and lab['shift_agreement'] == 1.0
    assert (
        stratum[10] == DIAMOND and kind[10] == 2
    )  # in view before, unchanged
    entering = np.flatnonzero(kind[:63] == 0)
    assert len(entering) == 7 and set(entering % 9) == {
        8
    }  # only the far column entered
    player = tg.PLAYER_TILE[0] * tg.VIEW_COLS + tg.PLAYER_TILE[1]
    assert stratum[player] == ga.PLAYER and kind[player] == 2
    assert (stratum[63:] == ga.STRIP).all() and (
        kind[63:] == 1
    ).all()  # items changed


def test_a_blocked_move_shifts_nothing_and_a_new_mob_is_a_change():
    left_of_player = tg.PLAYER_TILE[0] * tg.VIEW_COLS + tg.PLAYER_TILE[1] - 1
    part = planted(
        [
            (grass_with(**{f't{left_of_player}': TREE}), []),
            (
                grass_with(**{f't{left_of_player}': TREE}),
                [(20, tg.MOB_NAMES.index('cow'))],
            ),
        ]
    )
    part.action[0] = 1  # left, into a tree: the player does not move
    lab = ga.labels(part, np.array([0]), np.array([1]))
    kind, stratum = lab['kind'][0], lab['stratum'][0]
    assert not lab['moved'][0]
    assert (kind[:63] != 0).all()  # nothing entered
    assert (
        stratum[20] == tg.NUM_BLOCKS + tg.MOB_NAMES.index('cow')
        and kind[20] == 1
    )
    assert kind[10] == 2 and (kind[63:] == 2).all()


def test_a_group_reports_the_share_under_tau_and_the_gate_at_wider_taus():
    tau = 0.01
    s = ga.summarise(
        np.array([0.5, 2.0, 50.0]) * tau, np.full(3, 0.5 * tau), tau
    )
    assert (
        s['n'] == 3 and s['own_ok'] == 1.0 and abs(s['sharp'] - 1 / 3) < 1e-9
    )
    assert (
        abs(s['gate@1'] - (np.exp(-0.5) + np.exp(-2) + np.exp(-50)) / 3) < 1e-9
    )
    assert s['gate@16'] > s['gate@4'] > s['gate@1']
    assert abs(s['median_e_over_tau'] - 2.0) < 1e-9
    assert ga.summarise(np.zeros(0), np.zeros(0), tau) == {'n': 0}


def test_the_one_stage_gate_is_floored_and_relative_to_the_sample():
    tau = 0.01
    e_diff = np.array([0.0, 1.0, 1.0, 10.0]) * tau
    s = ga.summarise(
        e_diff,
        e_diff,
        tau,
        e_diff,
        joint_scale=float(e_diff.mean()),
        floor=0.1,
        beta=1.0,
    )
    expect = 0.1 + 0.9 * np.exp(-e_diff / e_diff.mean())
    assert abs(s['joint_gate'] - expect.mean()) < 1e-9
    assert s['joint_shut'] == 0.25  # the patch at ten tau sits near the floor
    assert 'joint_gate' not in ga.summarise(e_diff, e_diff, tau, e_diff)


def test_the_rank_correlation():
    assert abs(ga.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
    assert abs(ga.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9
    assert np.isnan(ga.spearman([1, 1, 1], [1, 2, 3]))


def test_the_comparison_pairs_the_parents_gate_with_the_probe_change():
    parent = {
        'by_stratum': {
            'zombie': dict(n=100, sharp=0.1, **{'gate@1': 0.05}, own_ok=0.9),
            'cow': dict(n=100, sharp=0.3, **{'gate@1': 0.2}, own_ok=0.9),
            'coal': dict(n=100, sharp=0.5, **{'gate@1': 0.4}, own_ok=0.9),
            'water': dict(n=100, sharp=0.9, **{'gate@1': 0.8}, own_ok=0.9),
        }
    }
    loss = {
        'mob_zombie': (0.9, 0.5, 3),
        'mob_cow': (0.7, 0.6, 3),
        'map_coal': (0.3, 0.25, 3),
        'map_water': (0.9, 0.9, 3),
        'map_iron': (0.1, 0.05, 3),
    }
    out = ga.compare(parent, loss)
    assert set(out['rows']) == {
        'mob_zombie',
        'mob_cow',
        'map_coal',
        'map_water',
    }  # iron has no patches
    assert (
        out['rho']['entities'] == 4 and out['rho']['sharp'] > 0.9
    )  # the shut gate lost most


def test_the_whole_pass_on_a_random_model_writes_the_report(tmp_path):
    from omegaconf import OmegaConf

    from hjepa_utils import create_world_model
    from test_sieve import compose, write_files

    write_files(tmp_path)
    cfg = compose('craftax_sieve', ['level1.sieve.stage=1'])
    model = create_world_model(cfg)[0]
    model.get_level(1).decoder.set_pixel_variance(0.05)
    parent = tmp_path / 'V4_epoch_1_object.ckpt'
    torch.save(model, parent)
    run2 = tmp_path / 'stage2'
    (run2 / 'sieve').mkdir(parents=True)
    torch.save(model, run2 / 'V4_stage2_from1_epoch_2_object.ckpt')
    for epoch, shift in ((0, 0.0), (2, -0.1)):
        probes = {
            f'holds_{name}': 0.6 + shift + 0.01 * i
            for i, name in enumerate(ga.PROBE_OF)
        }
        (run2 / 'sieve' / f'epoch_{epoch}.json').write_text(json.dumps(probes))
    out = tmp_path / 'out'
    ga.main(
        [
            '--parent',
            str(parent),
            '--stage2-run',
            str(run2),
            '--train-file',
            str(tmp_path / 'train.h5'),
            '--val-file',
            str(tmp_path / 'val.h5'),
            '--out',
            str(out),
            '--size',
            'quick',
            '--device',
            'cpu',
        ]
    )
    text = (out / 'gate_analysis.md').read_text()
    for part in (
        '# What the gate prunes',
        'By what the patch shows',
        'grass',
        'inventory_strip',
        'entering',
        'By daylight',
        '(end of stage 2)',
    ):
        assert part in text, part
    data = json.loads((out / 'gate_analysis.json').read_text())
    assert len(data['results']) == 2
    o = data['results'][0]['overall']
    assert 0.0 <= o['sharp'] <= 1.0 and o['gate@16'] >= o['gate@1']
    assert OmegaConf is not None
