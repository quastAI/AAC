"""Episodes that ended in death are padded with copies of their dead frame."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

import stable_worldmodel as swm  # noqa: E402


LENGTHS = [20, 12]  # frames; the first episode ends in death


@pytest.fixture()
def path(tmp_path):
    n = sum(LENGTHS)
    pixels = np.concatenate(
        [np.full((length, 4, 4, 3), 10 * e + np.arange(length)[:, None, None, None], np.uint8)
         for e, length in enumerate(LENGTHS)]
    )  # fmt: skip
    action = np.zeros((n, 1), np.float32)
    for end in np.cumsum(LENGTHS):
        action[end - 1] = np.nan
    dead = np.zeros(n, np.uint8)
    dead[LENGTHS[0] - 1] = 1
    with h5py.File(tmp_path / 'e.h5', 'w') as f:
        f.create_dataset('pixels', data=pixels)
        f.create_dataset('action', data=action)
        f.create_dataset('dead', data=dead)
        f.create_dataset('step_idx', data=np.arange(n, dtype=np.int32))
        f.create_dataset('ep_len', data=np.array(LENGTHS, np.int32))
        f.create_dataset(
            'ep_offset', data=np.array([0, LENGTHS[0]], dtype=np.int64)
        )
    return tmp_path


def make(path, pad, span=8):
    return swm.data.HDF5Dataset(
        'e',
        cache_dir=path,
        keys_to_cache=['action'],
        level1={'frameskip': 1, 'num_steps': span},
        terminal_pad=pad,
    )


def test_without_padding_clips_stay_inside_the_episodes(path):
    ds = make(path, 0)
    assert len(ds) == (20 - 8 + 1) + (12 - 8 + 1)
    assert all(start + 8 <= LENGTHS[ep] for ep, start in ds.clip_indices)


def test_a_dead_episode_gets_clips_that_end_in_copies_of_its_last_frame(path):
    ds = make(path, 5)
    # 5 more starts for the dead episode, none for the one that did not die
    assert len(ds) == (20 + 5 - 8 + 1) + (12 - 8 + 1)
    ep, start = max(
        (c for c in ds.clip_indices if c[0] == 0), key=lambda c: c[1]
    )
    assert start == 20 + 5 - 8  # the last clip has 3 real frames, 5 copies
    clip = ds[ds.clip_indices.index((ep, start))]
    pixels = clip['pixels_level1']
    assert pixels.shape[0] == 8
    values = pixels[:, 0, 0, 0].tolist()
    assert values == [17, 18, 19, 19, 19, 19, 19, 19]  # the last frame repeats
    action = clip['action_level1'][:, 0]
    # the dead frame has no action, and the copies have none either
    assert action[:2].tolist() == [0.0, 0.0] and bool(
        np.isnan(action[2:].numpy()).all()
    )
    assert clip['step_idx_level1'][:3].tolist() == [17, 18, 19]
    assert clip['step_idx_level1'][-1].item() == 19  # copies, not new steps


def test_clips_always_keep_a_real_frame(path):
    ds = make(path, 100, span=8)  # more than the span: capped at span - 1
    assert int(ds.pad_after[0]) == 7
    assert all(start <= LENGTHS[ep] - 1 for ep, start in ds.clip_indices)


def test_a_file_without_the_dead_column_is_not_padded(path):
    with h5py.File(path / 'e.h5', 'r+') as f:
        del f['dead']
    ds = make(path, 5)
    assert not ds.pad_after.any() and len(ds) == 13 + 5


def test_the_chunk_loader_for_evaluation_is_not_padded(path):
    ds = make(path, 5)
    (ep,) = ds.load_chunk(np.array([0]), np.array([16]), np.array([20]))
    assert ep['pixels'].shape[0] == 4  # real frames only


def test_auto_pad_is_one_top_level_step_minus_one_frame():
    from omegaconf import OmegaConf

    from data import dataset_kwargs

    def pad(levels, strides):
        cfg = {
            'num_levels': levels,
            'data': {'dataset': {'terminal_pad': 'auto', 'name': 'x'}},
        }
        for level in range(2, 5):
            cfg['data']['dataset'][f'level{level}'] = {
                'frameskip': strides[level - 2]
            }
        return dataset_kwargs(OmegaConf.create(cfg))['terminal_pad']

    assert pad(1, [4, 4, 4]) == 0  # level 1 alone: nothing to align
    assert pad(2, [4, 4, 4]) == 3
    assert pad(3, [4, 5, 4]) == 19
    assert pad(4, [4, 4, 4]) == 63
