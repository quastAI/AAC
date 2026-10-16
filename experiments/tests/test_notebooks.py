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


def test_the_pod_notebook_runs_the_whole_chain():
    nb = json.loads((NOTEBOOKS[0].parent / 'runpod.ipynb').read_text())
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
