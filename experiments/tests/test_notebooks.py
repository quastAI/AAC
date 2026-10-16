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


def test_the_d1_notebook_uses_the_runner_and_the_gpu_requirements():
    nb = json.loads((NOTEBOOKS[0].parent / 'd1_sampling.ipynb').read_text())
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    for needle in (
        'environment/jax-gpu-requirements.txt',
        'check_replay.py',
        'time_expert.py',
        '"run", "D1"',
        '"run", "D0"',
        'HDF5_USE_FILE_LOCKING',
        'JAX_PLATFORMS',
        'repo_type="dataset"',
    ):
        assert needle in text, needle
