"""dataset_stats on a synthetic file with known answers."""

import json
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
    rows = {r['T']: r for r in ds.clip_counts(data)}
    one = rows[1]
    assert one['span'] == 4 and one['clips'] == sum(n - 3 for n in LENGTHS)
    assert one['independent_clips'] == sum(n // 4 for n in LENGTHS)
    assert one['clips_upper'] is None  # no level above 1
    # the span depends on T only: (4 - 1) T + 1, whatever the depth or the strides
    assert rows[4]['span'] == 13 and rows[4]['clips'] == sum(n - 12 for n in LENGTHS)
    assert rows[16]['span'] == 49 and rows[16]['episodes_long_enough'] == 3  # 50 >= 49
    assert rows[64]['span'] == 193 and rows[64]['episodes_long_enough'] == 1  # only the 200-frame episode
    assert rows[64]['clips'] == 200 - 193 + 1 and rows[64]['independent_clips'] == 1


def test_episodes_that_ended_in_death_are_padded_by_T_minus_one(tmp_path):
    path = write(tmp_path / 'dead.h5', lengths=[100, 50])
    with h5py.File(path, 'r+') as f:
        dead = np.zeros(150, np.uint8)
        dead[149] = 1  # the 50-frame episode ended in death
        f.create_dataset('dead', data=dead)
    rows = {r['T']: r for r in ds.clip_counts(ds.Data(path))}
    # T = 16: span 49, the dead episode gets 15 copies: 65 frames -> 17 clips, 50 alone would give 2
    assert rows[16]['clips'] == (100 - 48) + (50 + 15 - 48)
    # T = 64: span 193, 50 + 63 = 113 is still too short
    assert rows[64]['episodes_long_enough'] == 0
    # a living episode is never padded
    assert rows[8]['clips'] == (100 - 24) + (50 + 7 - 24)


def test_clips_above_level_one_leave_out_the_random_episodes(tmp_path):
    lengths = [100, 100, 100]
    path = write(tmp_path / 'src.h5', lengths=lengths)
    with h5py.File(path, 'r+') as f:
        del f['source']
        f.create_dataset('source', data=np.repeat(np.array([0, 2, 3], np.int8), lengths))
    row = {r['T']: r for r in ds.clip_counts(ds.Data(path))}[4]
    assert row['clips'] == 3 * 88 and row['clips_upper'] == 2 * 88
    assert row['by_source'] == {'expert': 88, 'random': 88, 'early': 88}


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
    assert 'stride_hint' not in ev  # D0 assumes no strides


def test_distance_horizon_separates_fast_from_slow(data):
    ts = ds.distance_horizon(data, min_pairs=1)
    assert ts['facing']['probe_group'] == 'fast'
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
    assert result['files']['old']['spectral'] is None
    text = ds.render(result)
    assert 'skipped' in text and '# Dataset statistics' in text

    new = ds.Data(write(tmp_path / 'new.h5'))
    text = ds.render(ds.analyse({'train': new}))
    for heading in (
        'Clips per top-level step T',
        'When achievements unlock',
        'Distance horizon',
        'Is a hierarchy useful',
        'Task feasibility',
        'Achievement ladder',
        'Action chunks',
        'What the actions do',
    ):
        assert heading in text
    assert ds.to_jsonable(ds.analyse({'train': new}))  # serialisable


def test_max_frames_keeps_whole_episodes(tmp_path):
    d = ds.Data(write(tmp_path / 'a.h5'), max_frames=150)
    assert d.n_episodes == 1 and d.total == 100


def test_noisy_lags_are_left_out_by_default(data):
    ts = ds.distance_horizon(data)  # 350 frames: no lag has 1000 pairs
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
    rows = ds.clip_counts(d, totals=(1,))
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


def test_required_top_step_from_the_time_to_diamond_and_the_offsets(tmp_path):
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

    req = ds.required_top_step(ds.Data(path), horizons=(2, 4), offsets=(25, 200))
    assert req['expert_episodes_with_diamond'] == 3
    assert req['steps_to_diamond'] == {'p50': 60.0, 'p75': 90.0, 'p90': 108.0}
    table = req['required_top_step']
    # T >= distance / horizon
    assert table['offset 25'] == {2: 13, 4: 7}
    assert table['offset 200'] == {2: 100, 4: 50}
    assert table['diamond p75'] == {2: 45, 4: 23}
    assert table['diamond p90'] == {2: 54, 4: 27}
    text = ds.render(ds.analyse({'cov': ds.Data(path)}))
    assert 'Top-level step T needed' in text and 'diamond p90' in text and 'stride^' not in text
    assert 'Clips per top-level step T' in text and 'Clips per depth' not in text
    # a file without a diamond episode still gives the offsets
    none = ds.required_top_step(ds.Data(write(tmp_path / 'none.h5')), horizons=(4,), offsets=(100,))
    assert none['steps_to_diamond'] is None and none['required_top_step'] == {'offset 100': {4: 25}}


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


def test_offsets_and_horizons_are_arguments(tmp_path):
    path = write(tmp_path / 'o.h5')
    result = ds.analyse({'a': ds.Data(path)}, horizons=(3,), offsets=(90,))
    table = result['files']['a']['required_top_step']['required_top_step']
    assert table == {'offset 90': {3: 30}}


# --------------------------------------------------------------------------
# the structure of the data: frequency gap, events, ladder, tasks, survival


def write_dynamics(
    path, lengths=(700,) * 4, state=True, sources=None, dead=None, updates=None
):
    """Episodes with known dynamics: iid facing and mobs, a random walk, the
    environment's clocks (food every 26 steps, drink 21, energy 31), a day of
    300 steps, and staircases (wood at 100 + e, table at 250 + e)."""
    rng = np.random.default_rng(0)
    cols = {k: [] for k in ds.Data.COLUMNS + ['dead', 'policy_updates']}
    states = []
    for e, n in enumerate(lengths):
        t = np.arange(n)
        action = rng.integers(0, 17, n).astype(np.float32)
        action[-1] = np.nan
        flags = np.zeros((n, 22), np.uint8)
        flags[100 + e :, 0] = 1
        flags[250 + e :, 1] = 1
        inventory = np.zeros((n, 12), np.int16)
        for start in (100, 250, 400, 550):
            inventory[start + e :, 0] += 1
        vitals = np.stack(
            [
                np.full(n, 9),
                9 - (t // 26) % 9,
                9 - (t // 21) % 9,
                9 - (t // 31) % 9,
            ],
            1,
        ).astype(np.int8)
        progress = (t / 300) % 1 + 0.3
        light = (1 - np.abs(np.cos(np.pi * progress)) ** 3).astype(np.float32)
        position = np.cumsum(rng.integers(-1, 2, (n, 2)), 0).astype(np.int16)
        cols['action'].append(action[:, None])
        cols['achievements'].append(flags)
        cols['inventory'].append(inventory)
        cols['vitals'].append(vitals)
        cols['facing'].append(rng.integers(1, 5, (n, 1)).astype(np.int8))
        cols['daylight'].append(light[:, None])
        cols['position'].append(position)
        cols['mobs_in_view'].append(rng.integers(0, 3, (n, 4)).astype(np.int8))
        cols['source'].append(np.full(n, 0 if sources is None else sources[e]))
        cols['seed'].append(np.full(n, e, np.int64))
        died = np.zeros(n, np.uint8)
        died[-1] = bool(dead is not None and dead[e])
        cols['dead'].append(died)
        cols['policy_updates'].append(
            np.full(n, -1 if updates is None else updates[e], np.int32)
        )
        if state:
            row = np.zeros((n, 1345), np.float32)
            blocks = np.zeros((n, 63, 21), np.float32)
            tiles = rng.integers(0, 17, (n, 63, 1))
            np.put_along_axis(blocks[:, :, :17], tiles, 1.0, 2)
            row[:, :1323] = blocks.reshape(n, -1)
            states.append(row)
    with h5py.File(path, 'w') as f:
        for name, parts in cols.items():
            f.create_dataset(name, data=np.concatenate(parts))
        if state:
            f.create_dataset('state', data=np.concatenate(states))
        f.create_dataset('ep_len', data=np.array(lengths, np.int32))
        f.create_dataset(
            'ep_offset', data=np.concatenate([[0], np.cumsum(lengths)[:-1]])
        )
    return path


@pytest.fixture(scope='module')
def dynamics(tmp_path_factory):
    path = write_dynamics(tmp_path_factory.mktemp('dyn') / 'dyn.h5')
    return ds.Data(path)


@pytest.fixture(scope='module')
def spectral(dynamics):
    return ds.spectral_analysis(dynamics, windows=(64, 256), n_boot=40)


def test_the_gap_of_known_dynamics(spectral):
    gap = spectral['gap']['all']
    assert gap['ratio'] > 10
    assert gap['fastest'] in ('facing', 'mobs_in_view')  # iid: period ~4
    assert gap['slowest'] == 'daylight'
    assert gap['lower_bound']  # the slowest entity is limited by the window
    assert gap['ci']['p5'] <= gap['ratio'] * 1.05
    assert gap['ci']['p5'] <= gap['ci']['p50'] <= gap['ci']['p95']
    assert spectral['gap']['without_daylight']['slowest'] != 'daylight'
    entities = spectral['entities']
    assert entities['facing']['period'] == pytest.approx(4, rel=0.1)
    assert not entities['facing']['limited']
    for slow in ('inventory', 'achievements', 'daylight'):
        assert entities[slow]['limited']  # staircases and a day of 300 steps
    assert spectral['subset'] == 'expert' and spectral['episodes_used'] == 4
    assert spectral['probe_groups_ordered']
    assert spectral['daylight_check'] == {'window': 256, 'limited': True}


def test_a_shorter_window_gives_a_smaller_gap_when_it_limits(spectral):
    sensitivity = spectral['sensitivity']
    gaps = {w: v['gap']['all']['ratio'] for w, v in sensitivity.items()}
    assert set(gaps) == {64, 256} and gaps[64] < gaps[256]


def test_the_view_comes_from_the_state_column(spectral, tmp_path):
    for name in ('terrain', 'view'):
        assert spectral['extra'][name]['period'] == pytest.approx(4, rel=0.15)
    plain = ds.Data(write_dynamics(tmp_path / 'n.h5', state=False))
    assert ds.spectral_analysis(plain, windows=(64,), n_boot=5)['extra'] == {}
    no_view = ds.spectral_analysis(
        plain, windows=(64,), n_boot=5, use_view=False
    )
    assert no_view['extra'] == {}


def test_episodes_shorter_than_the_window_have_no_segment(data):
    result = ds.spectral_analysis(data, windows=(64,), n_boot=5)
    assert result['primary_window'] == 256 and result['episodes_used'] == 0
    assert result['gap']['all'] is None and result['entities'] == {}
    assert ds.timescale_ladder(data, result, {}) is None


def test_the_clocks_of_the_environment_are_recovered(dynamics):
    events = ds.event_table(ds.event_masks(dynamics, np.arange(4)))
    checks = {c['event']: c for c in ds.clock_check(events)}
    assert set(checks) == {'food_drop', 'drink_drop', 'energy_drop'}
    assert all(c['ok'] and c['mode_share'] > 0.8 for c in checks.values())
    assert events['entities']['inventory']['p50'] == 150
    assert events['entities']['position']['rate'] > 0.8
    assert 'daylight' not in events['entities']  # a clock, no events


def test_a_wrong_clock_is_flagged(dynamics):
    events = ds.event_table(ds.event_masks(dynamics, np.arange(4)))
    events['types']['food_drop']['mode'] = 25
    bad = [c for c in ds.clock_check(events) if not c['ok']]
    assert [c['event'] for c in bad] == ['food_drop']


def test_the_ladder_orders_entities_by_their_events(dynamics, spectral):
    masks = ds.event_masks(dynamics, np.arange(4))
    ladder = ds.timescale_ladder(dynamics, spectral, masks)
    clusters = [c['entities'] for c in ladder['clusters']]
    assert clusters == [
        ['position', 'facing', 'mobs_in_view'],
        ['vitals'],
        ['inventory', 'achievements'],
    ]
    assert ladder['levels_upper_bound'] == 3
    fast, vitals, slow = ladder['clusters']
    assert fast['events']['p50'] == 1 and slow['events']['p50'] == 150
    assert 2 <= vitals['events']['p50'] <= 16
    assert slow['regularity'] == 'regular'
    assert slow['clips_at_p90']['T'] == 150
    assert [(b['from'], b['to']) for b in ladder['boundaries']][1] == (
        vitals['events']['p50'],
        150,
    )
    assert set(ladder['without_events']) == {'daylight'}
    verdict = ladder['verdict']
    assert verdict['gap_at_least_threshold'] and verdict['ci_above_threshold']
    assert verdict['slow_end_has_progress']


def test_gap_per_source_and_snapshot_and_for_a_mix(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, 'MIN_GROUP_EPISODES', 1)
    path = write_dynamics(
        tmp_path / 'm.h5',
        lengths=(700,) * 4,
        state=False,
        sources=[0, 0, 3, 3],
        updates=[-1, -1, 100, 200],
    )
    result = ds.spectral_analysis(
        ds.Data(path),
        windows=(256,),
        n_boot=5,
        mix={'expert': 0.7, 'early': 0.3},
        use_view=False,
    )
    groups = {'expert', 'early', 'early@100', 'early@200'}
    assert set(result['groups']) == groups
    assert result['groups']['expert']['episodes'] == 2
    assert result['subset'] == 'expert' and result['episodes'] == 2
    assert result['mix']['shares'] == {'expert': 0.7, 'early': 0.3}
    assert result['mix']['gap']['all']['ratio'] > 10


def test_mutual_information_of_chunk_and_effect():
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 5, 20000)
    same = ds.mutual_information(chunk, chunk.copy())
    assert same['mi_bits'] == pytest.approx(np.log2(5), abs=0.01)
    assert same['coverage'] == 1 and same['chunk_classes'] == 5
    noise = ds.mutual_information(chunk, rng.integers(0, 5, 20000))
    assert abs(noise['mi_bits']) < 0.01  # the correction removes the bias
    assert ds.mutual_information(chunk, chunk, min_count=10**6) is None
    rare = np.concatenate([chunk, np.arange(100, 130)])  # 30 unique chunks
    info = ds.mutual_information(rare, rare.copy(), min_count=20)
    assert info['coverage'] == pytest.approx(20000 / 20030, abs=1e-4)


def test_chunk_information_is_reported_per_length(still):
    info = ds.chunk_diversity(still)[1]['information']
    # do gives wood, noop does nothing: the chunk tells most of the effect
    assert info['interaction']['mi_bits'] > 0.5
    mi = info['interaction']
    assert mi['mi_bits'] <= mi['h_effect_bits']


def test_task_feasibility_per_offset(dynamics):
    result = ds.task_feasibility(dynamics, np.arange(4), (25, 200), 10)
    assert result['possible_tasks'] == 4 and result['short']  # 10 wanted
    o25 = result['offsets'][25]
    assert o25['episodes_with_progress'] == 4
    assert o25['share_of_starts'] == pytest.approx(50 / 675)  # 25 + 25 starts
    assert o25['required_median'] == 1
    assert dict(o25['most_required']) == {
        'collect_wood': pytest.approx(0.5),
        'place_table': pytest.approx(0.5),
    }
    assert not ds.task_feasibility(dynamics, np.arange(4), (25,), 4)['short']
    assert result['offsets'][200]['share_of_starts'] > o25['share_of_starts']


def test_achievement_ladder_per_source(tmp_path):
    path = write_dynamics(
        tmp_path / 'l.h5', state=False, sources=[0, 0, 3, 3]
    )
    ladder = ds.achievement_ladder(ds.Data(path))
    assert ladder['expert']['episodes'] == 2
    wood = ladder['expert']['achievements']['collect_wood']
    assert wood == {'share': 1.0, 'median_frame': 100.5}
    table = ladder['early']['achievements']['place_table']
    assert table['median_frame'] == 252.5
    assert ladder['early']['achievements']['collect_diamond']['share'] == 0


def test_survival_critical_runs_and_the_run_into_death(tmp_path):
    path = write_dynamics(
        tmp_path / 's.h5',
        lengths=(300, 300, 300),
        state=False,
        dead=[False, True, True],
    )
    with h5py.File(path, 'r+') as f:
        vitals = np.full((900, 4), 9, np.int8)
        vitals[100:110, 1] = 0  # episode 0: food 0 for 10 frames, recovers
        vitals[300 + 280 : 300 + 300, 1] = 1  # episode 1: 20 critical frames
        vitals[300 + 299, 0] = 0  # ... and dies of it
        f['vitals'][:] = vitals  # episode 2 dies at full health (lava)
    s = ds.survival(ds.Data(path))['expert']
    assert s['episodes'] == 3 and s['deaths'] == 2
    assert s['critical_share'] == pytest.approx(30 / 900)
    assert s['critical_runs'] == 2 and s['recovered_share'] == 0.5
    assert s['deaths_after_critical'] == 0.5
    assert s['terminal_run']['p50'] == 20


def test_survival_needs_the_dead_flag(data):
    assert ds.survival(data) is None  # the plain file has no `dead` column


def test_the_report_has_every_section_and_serialises(tmp_path):
    path = write_dynamics(
        tmp_path / 'r.h5', lengths=(600,) * 3, dead=[False, True, False]
    )
    result = ds.analyse(
        {'train': ds.Data(path)},
        windows=(64, 256),
        n_boot=10,
        num_tasks=2,
        mix={'expert': 1.0},
    )
    text = ds.render(result)
    for heading in (
        'Is a hierarchy useful',
        'Gap of the mix',
        'Distance horizon',
        'Inter-event times',
        'Time scale ladder',
        'Task feasibility',
        'Achievement ladder',
        'Survival',
        'information (interaction)',
    ):
        assert heading in text
    assert 'WARNING' not in text  # the clocks and the daylight check pass
    json.dumps(ds.to_jsonable(result))


def test_parse_mix():
    assert ds.parse_mix(None) is None
    assert ds.parse_mix(['expert=0.7', 'early=0.3']) == {
        'expert': 0.7,
        'early': 0.3,
    }
    assert ds.parse_mix(['noised=0.5', 'random=0']) == {'noised_expert': 0.5}
    with pytest.raises(SystemExit, match='unknown source'):
        ds.parse_mix(['nope=1'])
