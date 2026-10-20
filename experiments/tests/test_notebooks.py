"""Notebooks: valid JSON, code cells that parse, no committed outputs."""

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[2] / 'notebooks').glob('*.ipynb'))


def test_there_is_a_notebook():
    assert NOTEBOOKS


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse_and_carry_no_outputs(path):
    nb = json.loads(path.read_text())
    assert nb['nbformat'] == 4
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))  # SyntaxError names the cell
            assert cell['outputs'] == [] and cell['execution_count'] is None, i


def test_the_dataset_notebook_runs_the_whole_chain():
    nb = json.loads((NOTEBOOKS[0].parent / 'runpod_dataset.ipynb').read_text())
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    for needle in (
        'environment/jax-gpu-requirements.txt',
        'time_expert.py',
        'pqn_rnn_craftax.py',
        'alg.SAVE_PATH',  # an override without the prefix is ignored
        'start_new_session=True',  # detached from the kernel
        'EARLY_UPDATES',
        '"run", "D1"',
        '"run", "D0"',
        'check_replay.py',
        'JAX_PLATFORMS',
        'HDF5_USE_FILE_LOCKING',
        'LD_LIBRARY_PATH',  # the image's CUDA crashes JAX's pip CUDA
        'upload_folder',
        'path_in_repo="checkpoints"',
        'path_in_repo="data"',
        'path_in_repo="reports"',
    ):
        assert needle in text, needle
    # the resumable state is as large as the stored transitions: not uploaded
    assert '*_state.safetensors' in text


def test_the_flat_notebook_runs_the_level_one_cycle():
    nb = json.loads((NOTEBOOKS[0].parent / 'runpod_flat.ipynb').read_text())
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    for needle in (
        'environment/hjepa-requirements.txt',
        'main_hjepa.py',  # the smoke run on the real files
        '+trainer.limit_train_batches',
        'probe_report.md',
        'time_run.py',
        '--loader-batches',
        '"run", "F1"',
        '"F3", "F4", "F5"',
        '"run", "V1"',
        'trainer.max_epochs',
        'start_new_session=True',  # detached from the kernel
        'HDF5_USE_FILE_LOCKING',
        'LD_LIBRARY_PATH',
        'PYTHONPATH',
        'data/craftax_expert_',  # the dataset from the Hub
        'path_in_repo="runs"',
    ):
        assert needle in text, needle
    # the resumable training state is not uploaded
    assert '**/lightning_resume/**' in text


def test_the_flat_notebook_runs_real_jobs():
    import yaml

    nb = json.loads((NOTEBOOKS[0].parent / 'runpod_flat.ipynb').read_text())
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    spec = yaml.safe_load(
        (Path(__file__).parents[1] / 'experiments.yaml').read_text()
    )
    for exp in ('F1', 'F3', 'F4', 'F5', 'V1'):
        assert exp in text and not spec['experiments'][exp].get('blocked')


def status_script():
    """The monitor cell's inline script, tested as a subprocess."""
    nb = json.loads((NOTEBOOKS[0].parent / 'runpod_dataset.ipynb').read_text())
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'STATUS = """' in source:
            return source.split('STATUS = """')[1].split('"""')[0]
    raise AssertionError('no STATUS script in the notebook')


def run_status(home):
    import subprocess
    import sys

    frames = json.dumps({'train': 100, 'val': 50, 'test': 50})
    return subprocess.run(
        [sys.executable, '-c', status_script(), str(home), frames],
        capture_output=True,
        text=True,
    )


def test_the_monitor_survives_a_file_without_episodes_yet(tmp_path):
    import h5py

    # the collector creates the file first and writes the episode tables
    # with the first episode: a monitor run in between must not fail
    h5py.File(tmp_path / 'craftax_expert_val.h5', 'w').close()
    result = run_status(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'val' in result.stdout and 'no episodes yet' in result.stdout
    assert 'train no file yet' in result.stdout.replace('  ', ' ')


def test_the_monitor_counts_frames_per_source(tmp_path):
    import h5py
    import numpy as np

    with h5py.File(tmp_path / 'craftax_expert_train.h5', 'w') as f:
        f['ep_len'] = np.array([10, 20, 5])
        f['ep_offset'] = np.array([0, 10, 30])
        f['source'] = np.array([0] * 10 + [0] * 20 + [3] * 5)
    result = run_status(tmp_path)
    assert result.returncode == 0, result.stderr
    assert '3 episodes' in result.stdout and 'expert 30' in result.stdout
    assert 'early 5' in result.stdout
