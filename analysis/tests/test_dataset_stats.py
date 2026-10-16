"""dataset_stats on a synthetic file with known answers."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import dataset_stats as ds  # noqa: E402

LENGTHS = [100, 200, 50]


def write(path, lengths=LENGTHS, seed_base=0, derived=True, motion=True):
    rng = np.random.default_rng(0)
    cols = {k: [] for k in ds.Data.COLUMNS}
    for e, n in enumerate(lengths):
        action = np.where(np.arange(n) % 2 == 0, 5, 0).astype(
            np.float32
        )  # do, noop
        action[-1] = np.nan
        inventory = np.zeros((n, 12), np.int16)
        inventory[1:, 0] = np.cumsum(
            action[:-1] == 5
        )  # a `do` gives wood next frame
        flags = np.zeros((n, 22), np.uint8)
        flags[10 + e :, 0] = 1  # wood
        flags[30 + e :, 1] = 1  # table
        position = np.cumsum(rng.integers(-1, 2, (n, 2)), axis=0).astype(
            np.int16
        )
        cols['action'].append(action[:, None])
        cols['achievements'].append(flags)
        cols['inventory'].append(inventory)
        cols['vitals'].append(np.full((n, 4), 9, np.int8))
        facing = rng.integers(1, 5, (n, 1)).astype(np.int8)
        if (
            not motion
        ):  # nothing moves: only `do` and the unlocks change the state
            facing, position = np.ones_like(facing), np.zeros_like(position)
        cols['facing'].append(facing)
        cols['daylight'].append(np.full((n, 1), e / 10, np.float32))
        cols['position'].append(position)
        cols['mobs_in_view'].append(np.zeros((n, 4), np.int8))
        cols['source'].append(np.zeros(n, np.int8))
        cols['seed'].append(np.full(n, seed_base + e, np.int64))
    with h5py.File(path, 'w') as f:
        for name, parts in cols.items():
            if derived or name in ('action', 'achievements', 'seed', 'source'):
                f.create_dataset(name, data=np.concatenate(parts))
        f.create_dataset('ep_len', data=np.array(lengths, np.int32))
        f.create_dataset(
            'ep_offset', data=np.concatenate([[0], np.cumsum(lengths)[:-1]])
        )
    return path


@pytest.fixture()
def data(tmp_path):
    return ds.Data(write(tmp_path / 'a.h5'))


@pytest.fixture()
def still(tmp_path):
    return ds.Data(write(tmp_path / 's.h5', motion=False))


def test_composition(data):
    c = ds.composition(data)
    assert c['episodes'] == 3 and c['frames'] == sum(LENGTHS)
    assert c['length']['min'] == 50 and c['length']['max'] == 200
    assert c['by_source'] == {'expert': 3}
    assert c['diamond_episodes'] == 0
    assert c['achievements_per_episode_mean'] == pytest.approx(2.0)


def test_clip_counts_follow_the_span_formula(data):
    rows = {(r['depth'], r['stride']): r for r in ds.clip_counts(data)}
    one = rows[(1, 1)]
    assert one['span'] == 4 and one['clips'] == sum(n - 3 for n in LENGTHS)
    assert one['independent_clips'] == sum(n // 4 for n in LENGTHS)
    two = rows[(2, 4)]
    assert two['span'] == (4 - 1) * 4 + 1 == 13
    assert two['clips'] == sum(n - 12 for n in LENGTHS)
    three = rows[(3, 4)]
    assert three['span'] == (13 - 1) * 4 + 1 == 49
    assert three['episodes_long_enough'] == 3  # 50 >= 49
    # a longer span than the shortest episode leaves it out
    assert rows[(3, 8)]['span'] == (25 - 1) * 8 + 1 == 193
    assert rows[(3, 8)]['episodes_long_enough'] == 1


def test_event_durations(data):
    ev = ds.event_durations(data)
    assert ev['unlock_frames']['collect_wood']['median'] == 11
    assert ev['unlock_frames']['place_table']['median'] == 31
    assert list(ev['unlock_frames']) == [
        'collect_wood',
        'place_table',
    ]  # in order
    assert (
        ev['gap_between_unlocks']['p50'] == 20
    )  # table - wood in every episode
    assert ev['stride_hint']['2_levels'] == 20
    assert ev['stride_hint']['3_levels'] == round(20**0.5)


def test_time_scales_separate_fast_from_slow(data):
    ts = ds.time_scales(data, min_pairs=1)
    assert ts['facing']['speed'] == 'fast'
    assert ts['facing']['ratio'][1] == pytest.approx(1.0, abs=0.2)  # iid
    assert ts['facing']['saturation_lag'] == 1
    assert ts['vitals']['ratio'][8] == 0 or ts['vitals']['ratio'][8] < 0.01
    assert ts['daylight']['ratio'][64] < 0.05  # constant within an episode
    assert (
        256 not in ts['daylight']['ratio']
    )  # no pair that long inside an episode
    walk = ts['position']['ratio']
    assert walk[1] < walk[16] < walk[64]  # a random walk decorrelates slowly
    assert 0 < ts['position']['changed'][1] <= 1


def test_chunk_diversity(still):
    ch = ds.chunk_diversity(still)
    assert ch[1]['distinct_chunks'] == 2  # do or noop
    assert ch[2]['distinct_chunks'] == 1  # always (do, noop)
    assert ch[2]['chunk_entropy_bits'] == pytest.approx(0)
    assert ch[1]['chunk_entropy_bits'] == pytest.approx(1, abs=0.05)
    # one `do` per two steps: wood changes, the second step changes nothing
    assert ch[1]['distinct_interaction_effects'] >= 2
    assert ch[1]['no_effect_share'] > 0.3  # every second step is a no-op


def test_action_effects(still):
    rows = {r['action']: r for r in ds.action_effects(still)}
    assert rows['do']['changes_state'] > 0.9  # every do gives wood
    assert (
        rows['noop']['changes_state'] < 0.2
    )  # only the unlock steps change anything
    assert rows['do']['share'] == pytest.approx(0.5, abs=0.02)
    assert rows['left']['changes_state'] is None  # never taken
    assert sum(r['share'] for r in rows.values()) == pytest.approx(1)


def test_seed_overlap_is_reported(tmp_path):
    a = ds.Data(write(tmp_path / 'a.h5', seed_base=0))
    b = ds.Data(
        write(tmp_path / 'b.h5', seed_base=2)
    )  # seed 2 appears in both
    c = ds.Data(write(tmp_path / 'c.h5', seed_base=100))
    assert ds.seed_overlap({'train': a, 'val': c}) == []
    notes = ds.seed_overlap({'train': a, 'val': b})
    assert len(notes) == 1 and 'not held out' in notes[0]


def test_old_files_skip_what_they_lack_and_report_renders(tmp_path):
    old = ds.Data(write(tmp_path / 'old.h5', derived=False))
    result = ds.analyse({'old': old})
    assert result['files']['old']['time_scales'] is None
    text = ds.render(result)
    assert 'skipped' in text and '# Dataset statistics' in text

    new = ds.Data(write(tmp_path / 'new.h5'))
    text = ds.render(ds.analyse({'train': new}))
    for heading in (
        'Clips per depth',
        'When achievements unlock',
        'Time scales',
        'Action chunks',
        'What the actions do',
    ):
        assert heading in text
    assert ds.to_jsonable(ds.analyse({'train': new}))  # serialisable


def test_max_frames_keeps_whole_episodes(tmp_path):
    d = ds.Data(write(tmp_path / 'a.h5'), max_frames=150)
    assert d.n_episodes == 1 and d.total == 100


def test_noisy_lags_are_left_out_by_default(data):
    ts = ds.time_scales(data)  # 350 frames: no lag has 1000 pairs
    assert all(not v['ratio'] for v in ts.values())


def test_a_frame_budget_samples_over_the_whole_file(tmp_path):
    # D1 appends expert, then noised, then random episodes to one file: a
    # prefix of the file would be the expert episodes alone
    lengths = [100] * 6 + [20] * 6
    path = write(tmp_path / 'mixed.h5', lengths=lengths)
    source = np.repeat(np.array([0] * 6 + [2] * 6, np.int8), lengths)
    with h5py.File(path, 'r+') as f:
        del f['source']
        f.create_dataset('source', data=source)

    d = ds.Data(path, max_frames=300)
    assert d.sampled is not None and d.sampled[1] == 12
    assert d.total <= 300
    comp = ds.composition(d)
    assert comp['episodes'] == 12 and comp['frames'] == sum(lengths)
    assert comp['by_source'] == {'expert': 6, 'random': 6}
    assert comp['frame_share_by_source']['expert'] == pytest.approx(
        600 / 720
    )
    assert set(d.cols['source'].tolist()) == {0, 2}  # both sources are in
    # clip counts use every episode, not the sampled ones
    rows = ds.clip_counts(d, strides=(2,), depths=(1,))
    assert rows[0]['episodes_long_enough'] == 12
    assert 'every k-th episode' in ds.render(ds.analyse({'mixed': d}))

    full = ds.Data(path)
    assert full.sampled is None and full.total == sum(lengths)


def test_death_statistics_by_source_and_cause(tmp_path):
    lengths = [50, 50, 50, 50, 30]
    path = write(tmp_path / 'deaths.h5', lengths=lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    last = offsets + np.array(lengths) - 1
    dead = np.zeros(sum(lengths), np.uint8)
    vitals = np.full((sum(lengths), 4), 9, np.int8)
    # expert: lives; noised: damage; random: lava, exhaustion, damage
    for episode, health, food in ((1, 0, 5), (2, 4, 5), (3, 0, 0), (4, 0, 5)):
        dead[last[episode]] = 1
        vitals[last[episode], 0] = health
        vitals[last[episode], 1] = food
    source = np.repeat(np.array([0, 1, 2, 2, 2], np.int8), lengths)
    with h5py.File(path, 'r+') as f:
        f.create_dataset('dead', data=dead)
        del f['vitals'], f['source']
        f.create_dataset('vitals', data=vitals)
        f.create_dataset('source', data=source)

    d = ds.Data(path)
    out = ds.composition(d)['deaths']
    assert out['expert'] == {'episodes': 1, 'died': 0, 'died_share': 0.0}
    assert out['noised_expert']['died'] == 1
    assert out['noised_expert']['causes'] == {
        'lava': 0,
        'exhaustion': 0,
        'damage': 1,
    }
    assert out['random']['died'] == 3
    assert out['random']['causes'] == {
        'lava': 1,
        'exhaustion': 1,
        'damage': 1,
    }
    assert out['random']['steps_to_death']['median'] == 49  # 50 frames
    assert 'Deaths, random: 3 of 3 episodes' in ds.render(
        ds.analyse({'deaths': d})
    )
    # files collected before the column existed have no death statistics
    assert 'deaths' not in ds.composition(ds.Data(write(tmp_path / 'old.h5')))


def test_coverage_of_the_top_level_against_the_time_to_diamond(tmp_path):
    lengths = [100, 200, 150, 60]
    path = write(tmp_path / 'cov.h5', lengths=lengths)
    diamond = {0: 60, 1: 120, 2: 40}  # the last episode never gets one
    flags = np.zeros((sum(lengths), 22), np.uint8)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    for e, frame in diamond.items():
        flags[offsets[e] + frame : offsets[e] + lengths[e], 19] = 1
    source = np.repeat(np.array([0, 0, 0, 2], np.int8), lengths)
    with h5py.File(path, 'r+') as f:
        del f['achievements'], f['source']
        f.create_dataset('achievements', data=flags)
        f.create_dataset('source', data=source)

    cov = ds.coverage(ds.Data(path), horizons=(4,), strides=(2, 4))
    assert cov['episodes'] == 3
    assert cov['steps_to_diamond'] == {'p50': 60.0, 'p75': 90.0, 'p90': 108.0}
    rows = {(r['depth'], r['stride']): r for r in cov['rows']}
    assert rows[(1, 1)]['reach'] == {4: 4} and rows[(1, 1)]['covers'] is None
    assert rows[(2, 4)]['reach'] == {4: 16}
    assert rows[(3, 4)]['reach'] == {4: 64}
    assert rows[(3, 4)]['covers'] == 'p50'  # 64 >= 60, not >= 90
    assert rows[(3, 4)]['top_step'] == 16
    assert rows[(4, 4)]['reach'] == {4: 256} and rows[(4, 4)]['covers'] == 'p90'
    # smallest equal stride: horizon x stride^(depth - 1) >= quantile
    assert cov['smallest_stride']['depth2_horizon4'] == {
        'p50': 15,
        'p75': 23,
        'p90': 27,
    }
    assert cov['smallest_stride']['depth3_horizon4']['p75'] == 5  # 4 * 25 = 100
    text = ds.render(ds.analyse({'cov': ds.Data(path)}))
    assert 'Reach of the top level' in text and 'p90' in text
    # a file without diamond episodes has no coverage
    assert ds.coverage(ds.Data(write(tmp_path / 'none.h5'))) is None


def test_provenance_of_the_collector_is_reported(tmp_path):
    import json

    path = tmp_path / 'p.h5'
    write(path)
    runs = [
        {
            'started': '2026-10-16T08:00:00+00:00',
            'git_commit': 'abcdef0123456789',
            'git_dirty': True,
            'versions': {'craftax': '1.6.1'},
            'args': {'eps': 0.0, 'seed_start': 0, 'seed_end': 1000000},
            'checkpoint': {'sha256': 'f' * 64, 'training': {'n_updates': 381}},
            'kept': 12,
            'finished': False,
        }
    ]
    with h5py.File(path, 'a') as f:
        f.attrs['runs'] = json.dumps(runs)
    result = ds.analyse({'a': ds.Data(path)})
    record = result['files']['a']['provenance'][0]
    assert record['git'] == 'abcdef01+dirty' and record['checkpoint'] == 'ffffffff'
    assert record['checkpoint_updates'] == 381 and not record['finished']
    text = ds.render(result)
    assert 'commit abcdef01+dirty' in text and 'NOT FINISHED' in text


def test_a_file_without_provenance_says_so(tmp_path):
    path = tmp_path / 'q.h5'
    write(path)
    assert 'No provenance recorded' in ds.render(ds.analyse({'a': ds.Data(path)}))


def test_early_checkpoints_and_action_coverage_per_source(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, 'MIN_ACTION_STEPS', 50)  # the toy file has a few hundred frames
    lengths = [100, 100, 60, 60, 40]
    path = write(tmp_path / 'early.h5', lengths=lengths)
    source = np.repeat(np.array([0, 0, 3, 3, 3], np.int8), lengths)
    updates = np.repeat(np.array([-1, -1, 381, 381, 1143], np.int32), lengths)
    with h5py.File(path, 'r+') as f:
        del f['source']
        f.create_dataset('source', data=source)
        f.create_dataset('policy_updates', data=updates)

    d = ds.Data(path)
    comp = ds.composition(d)
    assert comp['by_source'] == {'expert': 2, 'early': 3}
    assert comp['early_checkpoints'] == {
        '381': {'episodes': 2, 'frames': 120},
        '1143': {'episodes': 1, 'frames': 40},
    }
    cov = ds.action_coverage(d)
    assert set(cov['by_source']) == {'expert', 'early'}
    # the toy policy only takes noop and do, every second step
    expert = cov['by_source']['expert']['actions']
    assert expert['left']['steps'] == 0 and expert['do']['steps'] > 0
    assert cov['by_source']['early']['actions']['do']['steps'] > 0
    assert 'left' in cov['thin'] and 'do' not in cov['thin']
    text = ds.render(ds.analyse({'e': d}))
    assert 'Early checkpoints (training updates' in text
    assert '### Action coverage per source' in text and 'Thin actions' in text


def test_a_file_without_early_checkpoints_has_no_checkpoint_section(data):
    comp = ds.composition(data)
    assert 'early_checkpoints' not in comp
