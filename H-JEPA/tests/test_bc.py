"""Chunked behaviour cloning on the state (bc.py): the compact table, the
windows, the entities, and the importance of an entity on a planted rule (the
action is the facing)."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

import bc  # noqa: E402
from probing import targets as tg  # noqa: E402

LENGTHS = (100,) * 20


def write_file(path, seed, lengths=LENGTHS, sources=None):
    """Episodes with random observations; the action is the facing (1 to 4,
    the one-hot of the state), NaN in the last frame of an episode."""
    rng = np.random.default_rng(seed)
    n = sum(lengths)
    state = np.zeros((n, tg.STATE_DIM), np.float32)
    tiles = state[:, : bc.VIEW_DIM].reshape(n, bc.TILES, bc.CHANNELS)
    blocks = rng.integers(1, bc.NUM_BLOCKS, (n, bc.TILES, 1))
    np.put_along_axis(tiles[:, :, : bc.NUM_BLOCKS], blocks, 1, axis=2)
    tiles[:, :, bc.NUM_BLOCKS :] = (
        rng.random((n, bc.TILES, bc.NUM_MOBS)) < 0.05
    )
    state[:, bc.INVENTORY : bc.FACING] = rng.integers(0, 10, (n, 16)) / 10
    facing = rng.integers(0, 4, n)
    state[np.arange(n), bc.FACING + facing] = 1
    state[:, bc.DAYLIGHT] = rng.random(n)
    action = (facing + 1).astype(np.float32)[:, None]
    action[np.cumsum(lengths) - 1] = np.nan
    source = np.repeat(
        np.zeros(len(lengths), np.int8) if sources is None else sources,
        lengths,
    )
    with h5py.File(path, 'w') as f:
        f['ep_len'] = np.array(lengths, np.int32)
        f['source'] = source
        f['state'] = state
        f['action'] = action
        f['action_policy'] = action.copy()
    return state, action


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp('bc')
    write_file(folder / 'train.h5', 0, lengths=(100,) * 40)
    write_file(folder / 'val.h5', 1, lengths=(100,) * 8)
    return folder


def test_the_table_rebuilds_the_observation_of_the_kept_episodes(tmp_path):
    state, action = write_file(
        tmp_path / 'f.h5', 0, (10, 7, 5), np.array([0, 3, 0], np.int8)
    )
    table = bc.load_table(tmp_path / 'f.h5', sources=(0,))
    kept = np.r_[0:10, 17:22]
    rows = torch.arange(len(kept))
    assert np.array_equal(table.features(rows).numpy(), state[kept])
    assert table.action.tolist() == [
        -1 if np.isnan(a) else int(a) for a in action[kept, 0]
    ]
    assert table.episode.unique().tolist() == [0, 2]  # the file's episodes
    first = bc.load_table(tmp_path / 'f.h5', sources=(0, 3), max_episodes=1)
    assert len(first.action) == 10
    with pytest.raises(SystemExit, match='no frame of source'):
        bc.load_table(tmp_path / 'f.h5', sources=(2,))


def test_a_window_never_crosses_an_episode_or_the_missing_last_action(
    tmp_path,
):
    write_file(tmp_path / 'f.h5', 0, (10, 6))
    table = bc.load_table(tmp_path / 'f.h5')
    assert bc.chunk_starts(table, 4).tolist() == [*range(6), 10, 11]
    assert bc.chunk_starts(table, 1).tolist() == [*range(9), *range(10, 15)]
    assert len(bc.chunk_starts(table, 12)) == 0
    rows = torch.tensor([0, 10])
    y = bc.chunk_targets(table, rows, 4)
    assert y.shape == (2, 4)
    assert y[0].tolist() == table.action[:4].tolist()


def test_the_entities_cut_the_state_without_gaps_or_overlaps():
    cols = bc.entities()

    def covers(prefix, columns):
        got = np.concatenate(
            [v for n, v in cols.items() if n.startswith(prefix)]
        )
        assert sorted(got) == sorted(columns)

    view = np.arange(bc.VIEW_DIM)
    covers('tile:', view)
    covers('ring:', view)
    covers('group:', np.arange(tg.STATE_DIM))
    assert len(cols['ring:0']) == bc.CHANNELS  # the player's own tile
    assert len(cols['ring:4']) == 14 * bc.CHANNELS  # the two outer columns
    assert 'block:invalid' not in cols and len(cols['block:stone']) == 63
    assert len(cols['mob:arrow']) == 63
    values = np.concatenate(
        [
            v
            for n, v in cols.items()
            if n.split(':')[0] in ('inventory', 'vitals', 'state')
        ]
    )
    assert sorted(values) == list(range(bc.VIEW_DIM, tg.STATE_DIM))
    keep = bc.column_mask(['state:facing', 'inventory:wood'], cols)
    assert keep.sum() == tg.STATE_DIM - 5


@pytest.fixture(scope='module')
def trained(files):
    train = bc.load_table(files / 'train.h5')
    val = bc.load_table(files / 'val.h5')
    model = bc.fit(
        train, val, 4, epochs=20, batch_size=128, hidden=32, depth=2, lr=3e-3
    )
    return train, val, model


def test_the_model_learns_the_action_it_can_read_from_the_state(trained):
    train, val, model = trained
    rows = bc.chunk_starts(val, 4)
    ce, acc = bc.evaluate(model, val, rows, 4)
    assert acc[0] > 0.95 and ce[0] < 0.2  # a_t is the facing of frame t
    assert acc[1:].max() < 0.5 and ce[1:].min() > 1.0  # later ones are not


@pytest.fixture(scope='module')
def importance(trained):
    train, val, model = trained
    rows = bc.chunk_starts(val, 4)
    marginal = bc.marginal_log_probs(train, bc.chunk_starts(train, 4), 4)
    return bc.entity_importance(
        model, val, rows, 4, bc.entities(), marginal, repeats=2, boot=40
    )


def test_the_facing_is_the_entity_the_actions_need(importance):
    ent = importance['entities']
    ranked = sorted(ent, key=lambda n: -ent[n]['skill_lost'])
    assert set(ranked[:2]) == {'state:facing', 'group:facing'}
    assert ent['state:facing']['skill_lost'] > 0.2
    low, high = ent['state:facing']['skill_lost_ci']
    assert low > 0.1 and low <= ent['state:facing']['skill_lost'] <= high
    steps = ent['state:facing']['delta_ce_steps']  # only step 1 reads it
    assert steps[0] > 0.5 and max(steps[1:]) < 0.1
    assert ent['state:facing']['delta_accuracy'] < -0.1
    for n in ('group:view', 'block:stone', 'ring:1', 'inventory:wood'):
        assert abs(ent[n]['skill_lost']) < 0.05, n
    assert not ent['state:facing']['constant']
    assert ent['state:sleeping']['constant']  # never asleep in the file
    assert importance['ce'] < importance['marginal_ce']


def test_an_entity_left_out_of_training_costs_the_model_its_skill(
    trained, importance
):
    train, val, _ = trained
    out = bc.retrain_without(
        train,
        val,
        4,
        bc.entities(),
        ['group:facing', 'group:view'],
        importance,
        epochs=20,
        batch_size=128,
        hidden=32,
        depth=2,
        lr=3e-3,
    )
    assert out['group:facing']['skill_lost'] > 0.2
    assert out['group:view']['skill_lost'] < 0.1  # noise: no cost


def test_the_report_has_every_family_and_the_json_round_trips(importance):
    importance = importance | {
        'retrain': {
            'group:facing': {'ce': 1.0, 'accuracy': 0.3, 'skill_lost': 0.9}
        }
    }
    text = bc.report(importance, 4)
    for heading in (
        'Coarse parts',
        'Block types',
        'Mobs in the view',
        'Distance from the player',
        'Inventory',
        'Vitals',
        'Facing, daylight, sleeping',
        'Skill lost by place',
        'Trained again without',
    ):
        assert f'## {heading}' in text, heading
    assert '| facing |' in text and '(constant)' in text
    assert json.loads(json.dumps(importance))['entities']


def test_the_command_line_writes_a_report_per_chunk_length(
    files, tmp_path, monkeypatch
):
    argv = [
        'bc.py',
        '--train-file', str(files / 'train.h5'),
        '--val-file', str(files / 'val.h5'),
        '--out', str(tmp_path / 'out'),
        '--chunk', '2', '3',
        '--epochs', '1',
        '--hidden', '16',
        '--boot', '5',
        '--repeats', '1',
        '--retrain-top', '1',
    ]  # fmt: skip
    monkeypatch.setattr(sys, 'argv', argv)
    bc.main()
    for chunk in (2, 3):
        result = json.loads(
            (tmp_path / 'out' / f'bc_entities_chunk{chunk}.json').read_text()
        )
        assert len(result['ce_steps']) == chunk
        assert 'group:view' in result['retrain']
        assert len(result['retrain']) == 6 + 1  # the groups and the top one
        assert (tmp_path / 'out' / f'bc_entities_chunk{chunk}.md').exists()
