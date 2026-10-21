"""Fixtures shared by the Craftax tests."""

from pathlib import Path

import h5py
import numpy as np
import pytest


FIXTURE = Path(__file__).parent / 'data' / 'craftax_classic_episode.npz'


@pytest.fixture(scope='module')
def episode():
    return dict(np.load(FIXTURE))


@pytest.fixture()
def dataset(episode, tmp_path):
    n = len(episode['pixels'])
    action = np.full((n, 1), np.nan, dtype=np.float32)
    action[:-1, 0] = episode['actions']
    columns = dict(
        pixels=episode['pixels'],
        state=episode['state'],
        full_state=episode['full_state'],
        achievements=episode['achievements'],
        reward=episode['reward'],
        action=action,
        action_policy=action.copy(),  # the collector's layout since the relabeling
        step_idx=np.arange(n, dtype=np.int32),
        ep_idx=np.zeros(n, dtype=np.int32),
        seed=np.full(n, episode['seed'], dtype=np.int64),
        source=np.full(n, 2, dtype=np.int8),
    )
    with h5py.File(tmp_path / 'craftax.h5', 'w') as f:
        for key, value in columns.items():
            f.create_dataset(key, data=value)
        f.create_dataset('ep_offset', data=np.array([0], dtype=np.int64))
        f.create_dataset('ep_len', data=np.array([n], dtype=np.int32))
    import stable_worldmodel as swm

    return swm.data.HDF5Dataset(
        'craftax',
        cache_dir=tmp_path,
        level1=dict(frameskip=1, num_steps=1),
    )
