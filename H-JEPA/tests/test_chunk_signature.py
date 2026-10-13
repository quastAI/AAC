"""Chunked evaluation resumes only from chunks of the same tasks and settings."""

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


pytest.importorskip('hydra')
pytest.importorskip('sklearn')
pytest.importorskip('transformers')
pytest.importorskip('stable_pretraining')

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

from planning_eval import (  # noqa: E402
    _check_chunk_signature,
    _chunk_signature,
    resolve_results_dir,
)


@pytest.fixture()
def tasks(tmp_path):
    path = tmp_path / 'tasks.pt'
    path.write_bytes(b'tasks of the validation worlds')
    return path


def signature(tasks, cfg=None, num_eval=10, chunk_size=5):
    cfg = OmegaConf.create(cfg or {'solver': {'num_samples': 300}})
    return _chunk_signature(cfg, tasks, num_eval, chunk_size)


def test_same_inputs_resume(tasks, tmp_path):
    chunks = tmp_path / 'chunks'
    chunks.mkdir()
    _check_chunk_signature(chunks, signature(tasks))
    (chunks / 'tasks_000-005.json').write_text('{}')
    _check_chunk_signature(chunks, signature(tasks))  # no error


@pytest.mark.parametrize(
    'change, key',
    [
        ({'tasks': b'tasks of the test worlds'}, 'tasks_sha256'),
        ({'cfg': {'solver': {'num_samples': 600}}}, 'config_sha256'),
        ({'num_eval': 20}, 'num_eval'),
        ({'chunk_size': 10}, 'chunk_size'),
    ],
)
def test_changed_inputs_are_refused(tasks, tmp_path, change, key):
    chunks = tmp_path / 'chunks'
    chunks.mkdir()
    _check_chunk_signature(chunks, signature(tasks))
    (chunks / 'tasks_000-005.json').write_text('{}')
    if 'tasks' in change:
        tasks.write_bytes(change['tasks'])
        new = signature(tasks)
    else:
        new = signature(tasks, **change)
    with pytest.raises(RuntimeError, match=key):
        _check_chunk_signature(chunks, new)


def test_chunks_without_signature_are_refused(tasks, tmp_path):
    chunks = tmp_path / 'chunks'
    chunks.mkdir()
    (chunks / 'tasks_000-005.json').write_text('{}')
    with pytest.raises(RuntimeError, match='without a signature'):
        _check_chunk_signature(chunks, signature(tasks))


def test_random_baseline_needs_an_output_directory(tmp_path):
    # there is no policy directory to put its results next to
    cfg = OmegaConf.create({'policy': 'random', 'output': {'dir': None}})
    with pytest.raises(ValueError, match='output.dir'):
        resolve_results_dir(cfg)
    cfg.output.dir = str(tmp_path / 'random')
    assert resolve_results_dir(cfg) == tmp_path / 'random'
