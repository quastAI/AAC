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
        'trainer.devices=1',  # several GPUs: DDP stops at the unused mask_token
        'start_new_session=True',  # detached from the kernel
        'HDF5_USE_FILE_LOCKING',
        'LD_LIBRARY_PATH',
        'PYTHONPATH',
        'data/craftax_expert_',  # the dataset from the Hub
        'relabel_dataset.ipynb',  # refuses files that hold the policy's actions
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


def test_the_relabel_notebook_checks_before_it_replaces_the_hub_files():
    nb = json.loads((NOTEBOOKS[0].parent / 'relabel_dataset.ipynb').read_text())
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    for needle in (
        'relabel_asleep.py',
        'check_replay.py',
        '"JAX_PLATFORMS": "cpu"',
        'revision=SOURCE_REVISION',  # the files before, named in the commit
        '"dataset_section", "compute"',
        'asleep_steps',
        'path_in_repo="data"',
        'path_in_repo="reports/dataset"',
    ):
        assert needle in text, needle
    upload = next(
        ''.join(c['source'])
        for c in nb['cells']
        if 'path_in_repo="data"' in ''.join(c['source'])
    )
    assert upload.index('all(CHECKS.values())') < upload.index('upload_folder')


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


def probe_cell_fetch_part():
    """The part of the flat notebook's probe-suite cell that gets the run."""
    nb = json.loads((NOTEBOOKS[0].parent / 'runpod_flat.ipynb').read_text())
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if source.startswith("# --- 2. The probe suite on F1's final model"):
            return source.split('health_dir = ')[0]
    raise AssertionError('no probe suite cell in the flat notebook')


def run_fetch(tmp_path, hub_files, existing=()):
    """Runs the fetch part against a fake Hub (namespace, downloads, run)."""
    import shutil

    run = tmp_path / 'ckpts' / 'craftax' / 'flat' / 'seed42'
    run.mkdir(parents=True)
    for name in existing:
        (run / name).write_text('local')
    downloaded = []

    class FakeApi:
        token = 'x'

        def list_repo_files(self, repo_id, repo_type):
            assert repo_type == 'dataset'
            return hub_files

    def fake_download(repo_id, filename, repo_type, token):
        downloaded.append(filename)
        source = tmp_path / ('hub_' + filename.replace('/', '_'))
        source.write_text('hub')
        return str(source)

    ns = {
        'run_dir': lambda model: run,
        'SEEDS': '42,43',
        'HF_REPO_ID': 'org/repo',
        'api': FakeApi(),
        'hf_hub_download': fake_download,
        'shutil': shutil,
    }
    exec(probe_cell_fetch_part(), ns)
    return ns, downloaded, run


def test_the_probe_cell_fetches_the_run_from_the_hub(tmp_path):
    hub = [
        'data/craftax_expert_train.h5',
        'runs/flat/seed42/config.yaml',
        'runs/flat/seed42/flat_object.ckpt',
        'runs/flat/seed43/config.yaml',  # another seed
        'runs/flat_h1/seed42/config.yaml',  # another model
    ]
    _, downloaded, run = run_fetch(tmp_path, hub)
    assert downloaded == [
        'runs/flat/seed42/config.yaml',
        'runs/flat/seed42/flat_object.ckpt',
    ]
    assert (run / 'config.yaml').read_text() == 'hub'
    assert (run / 'flat_object.ckpt').read_text() == 'hub'


def test_the_probe_cell_keeps_what_the_volume_has_and_finds_other_layouts(
    tmp_path,
):
    hub = ['runs/craftax/flat/seed42/flat_object.ckpt']  # one folder deeper
    _, downloaded, run = run_fetch(tmp_path, hub, existing=['config.yaml'])
    assert downloaded == ['runs/craftax/flat/seed42/flat_object.ckpt']
    assert (run / 'config.yaml').read_text() == 'local'  # not replaced
    # nothing missing: the Hub is not asked
    _, downloaded, _ = run_fetch(
        tmp_path / 'again', [], existing=['config.yaml', 'flat_object.ckpt']
    )
    assert downloaded == []


def test_the_probe_cell_stops_when_the_hub_has_no_single_match(tmp_path):
    with pytest.raises(AssertionError, match='config.yaml: 0 matches'):
        run_fetch(tmp_path, ['runs/flat/seed42/flat_object.ckpt'])
    twice = [
        'runs/a/flat/seed42/config.yaml',
        'runs/b/flat/seed42/config.yaml',
        'runs/flat/seed42/flat_object.ckpt',
    ]
    with pytest.raises(AssertionError, match='config.yaml: 2 matches'):
        run_fetch(tmp_path / 'two', twice)


def flat_cells():
    nb = json.loads((NOTEBOOKS[0].parent / 'runpod_flat.ipynb').read_text())
    return [''.join(c['source']) for c in nb['cells']]


def test_the_probe_suite_runs_detached_and_has_a_monitor_cell():
    cells = flat_cells()
    start = next(
        i
        for i, c in enumerate(cells)
        if c.startswith("# --- 2. The probe suite on F1's final model")
    )
    # detached with a live log: `sh` would show nothing until it exits
    assert 'launch("probe_suite"' in cells[start]
    assert 'sh([PYTHON, "probe_suite.py"' not in cells[start]
    assert 'quiet=True' not in cells[start]
    monitor = cells[start + 1]
    assert monitor.startswith('# --- 2. Monitor the probe suite')
    for needle in (
        'running("probe_suite")',
        'progress("probe_suite", "[probe suite]"',
        'nvidia-smi',
        'probe_report.md',
        'probe_suite.log',  # the end of the log when no report came
    ):
        assert needle in monitor, needle


def test_the_suite_prints_the_progress_lines_the_monitor_looks_for():
    suite = (
        Path(__file__).parents[2] / 'H-JEPA' / 'h_jepa' / 'probe_suite.py'
    ).read_text()
    assert 'print(f"[probe suite] {msg}", flush=True)' in suite
