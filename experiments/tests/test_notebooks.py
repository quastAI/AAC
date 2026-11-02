"""Notebooks: valid JSON, code cells that parse, no committed outputs."""

import ast
import json
import re
import sys
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).parents[2] / 'notebooks'
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob('*.ipynb'))
PODS = sorted(NOTEBOOK_DIR.glob('runpod_[VAB][0-9]*.ipynb'))  # one per pod run, named by its id
VARIANTS = [p for p in PODS if 'REFERENCE = ' in p.read_text()]  # compared with B1, planned on set A
TRACES = [p for p in PODS if 'COMPARE = ' in p.read_text()]  # the race: dense checkpoints, read after the training
POD = (NOTEBOOK_DIR / 'pod.py').read_text()
sys.path.insert(0, str(NOTEBOOK_DIR))
sys.path.insert(0, str(Path(__file__).parents[1]))

import pod  # noqa: E402
import run  # noqa: E402


def cells_of(path):
    nb = json.loads(Path(path).read_text())
    return [''.join(c['source']) for c in nb['cells']]


def text_of(name):
    return '\n'.join(cells_of(NOTEBOOK_DIR / name))


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


def test_the_baseline_notebook_runs_the_level_one_cycle():
    text = text_of('runpod_B1.ipynb') + POD  # the helpers are in pod.py
    for needle in (
        'environment/hjepa-requirements.txt',
        'main_hjepa.py',  # the smoke run on the real files
        '+trainer.limit_train_batches',
        'probe_report.md',
        'time_run.py',
        '--loader-batches',
        '"run", "B1"',  # training, probes and the cheap P1: the variants' reference
        '"P2", "P3"',
        'pod.folder_of("B1")',  # B1 keeps the folder it was trained in
        'trainer.max_epochs',
        'trainer.devices=1',  # several GPUs: DDP stops at the mask_token
        'start_new_session=True',  # detached from the kernel
        'HDF5_USE_FILE_LOCKING',
        'LD_LIBRARY_PATH',
        'PYTHONPATH',
        'data/craftax_expert_',  # the dataset from the Hub
        'relabel_dataset.ipynb',  # refuses files with the policy's actions
        'path_in_repo="runs"',
        'pod.upload',
    ):
        assert needle in text, needle
    # the resumable training state is not uploaded
    assert '**/lightning_resume/**' in POD
    # the variants have their own notebooks
    assert '"run", "V1"' not in text_of('runpod_B1.ipynb')


def test_the_baseline_notebook_runs_real_jobs():
    import yaml

    text = text_of('runpod_B1.ipynb')
    spec = yaml.safe_load(
        (Path(__file__).parents[1] / 'experiments.yaml').read_text()
    )
    for exp in ('B1', 'P2', 'P3'):
        assert exp in text and not spec['experiments'][exp].get('blocked')


def variant_config(path):
    config = cells_of(path)[1]
    exp = re.search(r'EXPERIMENT = "(\w+)"', config).group(1)
    reference = re.search(r'REFERENCE = "(\w+)"', config).group(1)
    return exp, reference


def pod_experiments(spec, planned):
    """Experiments with a trained run that run on a pod: those with planning (`protocols`: variants and ablations compared
    with B1) or without (the race runs, `trace` on their run)."""
    return {
        exp_id
        for exp_id, e in spec['experiments'].items()
        if exp_id[0] in 'VAB' and exp_id != 'B1' and not e.get('blocked')
        and bool(e.get('protocols')) == planned
        and any(r.get('train', True) for r in e.get('runs', []))
    }


def test_every_variant_has_its_own_notebook():
    spec = run.load_spec()
    found = set()
    for path in VARIANTS:
        exp, reference = variant_config(path)
        assert path.name == f'runpod_{exp}.ipynb'
        runs = spec['experiments'][exp]['runs']
        assert [r['name'] for r in runs if r.get('train', True)] == [exp]
        assert any(
            r['name'] == reference and not r.get('train', True) for r in runs
        )
        found.add(exp)
    assert found == pod_experiments(spec, planned=True)


def test_the_variant_notebooks_differ_only_in_title_and_configuration():
    first = cells_of(VARIANTS[0])
    exp0, _ = variant_config(VARIANTS[0])
    for path in VARIANTS[1:]:
        cells = cells_of(path)
        exp, _ = variant_config(path)
        assert len(cells) == len(first), path.name
        assert cells[1].replace(f'"{exp}"', '') == first[1].replace(
            f'"{exp0}"', ''
        )
        for i, (a, b) in enumerate(zip(first[2:], cells[2:]), start=2):
            assert a == b, (path.name, i)  # copies must not drift


def test_a_variant_notebook_runs_its_run_against_the_reference():
    text = '\n'.join(cells_of(VARIANTS[0]))
    for needle in (
        '"run", EXPERIMENT',
        'pod.folder_of(EXPERIMENT)',  # B1 and A1 keep their older folders
        'f"{REFERENCE_FOLDER}/seed',  # B1 from the Hub
        '"random/eval_A_val"',
        'pod.current_probe_report(reference)',
        'SPEC["config"], *SPEC.get("overrides", [])',  # the smoke run
        '+trainer.limit_train_batches',
        'launch(EXPERIMENT',
        'f"{EXPERIMENT}_*.md"',
    ):
        assert needle in text, needle
    assert 'RUN' not in re.findall(r'\b[A-Z_]+\b', text)  # the id names the run
    # the reports the notebook reads and uploads are the ones run.py writes
    exp, _ = variant_config(VARIANTS[0])
    spec = run.load_spec()
    jobs = run.build_jobs(spec, exp, [42], home='/data')
    reports = [j.output for j in jobs if j.kind == 'compare']
    assert reports and all(Path(r).name.startswith(f'{exp}_') for r in reports)
    # the Hub folder of B1's set-A results is the one the notebook checks
    ref = next(j for j in jobs if j.name == 'eval_B1_s42_A')
    assert ref.output.endswith(
        'flat/seed42/eval_A_val_craftax_plan/planning_report.md'
    )
    assert 'eval_A_val_craftax_plan' in text


def trace_config(path):
    config = cells_of(path)[1]
    exp = re.search(r'EXPERIMENT = "(\w+)"', config).group(1)
    compare = re.search(r'COMPARE = \[([^\]]*)\]', config).group(1)
    return exp, re.findall(r'"(\w+)"', compare)


def test_every_race_run_has_its_own_notebook():
    spec = run.load_spec()
    found = set()
    for path in TRACES:
        exp, compare = trace_config(path)
        assert path.name == f'runpod_{exp}.ipynb'
        runs = spec['experiments'][exp]['runs']
        trained = [r for r in runs if r.get('train', True)]
        assert [r['name'] for r in trained] == [exp] and trained[0]['trace']
        assert trained[0]['overrides'].count('dense_checkpoints.enabled=true')
        found.add(exp)
    assert found == pod_experiments(spec, planned=False) == {'B2', 'A3', 'A4'}
    # the 2x2 of the race: every run read at many steps
    traced = {
        r['name']
        for e in spec['experiments'].values()
        for r in e.get('runs', [])
        if r.get('trace') and r.get('train', True)
    }
    assert set(trace_config(TRACES[0])[1]) == traced == {'B2', 'V3', 'A3', 'A4'}


def test_the_race_notebooks_differ_only_in_title_and_configuration():
    first = cells_of(TRACES[0])
    exp0, _ = trace_config(TRACES[0])
    for path in TRACES[1:]:
        cells = cells_of(path)
        exp, _ = trace_config(path)
        assert len(cells) == len(first), path.name
        assert cells[1].replace(
            f'EXPERIMENT = "{exp}"', ''
        ) == first[1].replace(f'EXPERIMENT = "{exp0}"', '')
        for i, (a, b) in enumerate(zip(first[2:], cells[2:]), start=2):
            assert a == b, (path.name, i)  # copies must not drift


def test_a_race_notebook_smokes_the_dense_checkpoints_and_the_readout():
    text = '\n'.join(cells_of(TRACES[0]))
    for needle in (
        'launch(EXPERIMENT',
        '"dense_checkpoints.steps=[5,10]"',  # the smoke run saves dense steps
        '"dense_checkpoints.stop_after_epochs=null"',
        '"-m", "race.evaluate"',  # and reads them
        '"-m", "race.report"',  # the four runs side by side
        'only="race/"',  # the others' readouts, not their models
        'f"{EXPERIMENT}_*.png"',
    ):
        assert needle in text, needle
    exp, _ = trace_config(TRACES[0])
    jobs = run.build_jobs(run.load_spec(), exp, [42], home='/data')
    assert [j.name for j in jobs if j.kind == 'compare'][0].endswith('_trace')
    assert any(j.name == f'trace_{exp}_s42' for j in jobs)


def test_a_variant_that_is_traced_fetches_the_race_of_its_reference():
    spec = run.load_spec()
    assert pod.traced_references.__name__ == 'traced_references'
    refs = [
        r['name']
        for r in spec['experiments']['V3']['runs']
        if r.get('trace') and not r.get('train', True)
    ]
    assert refs == ['B2']  # V3's race report puts B2 next to it
    assert 'pod.traced_references(EXPERIMENT)' in '\n'.join(
        cells_of(NOTEBOOK_DIR / 'runpod_V3.ipynb')
    )


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


def test_the_bc_notebook_runs_the_entity_analysis():
    text = text_of('runpod_bc_entities.ipynb')
    for needle in (
        'pod.install',  # the PyTorch environment and the dataset
        'CHUNKS = (4, 8, 16)',
        'launch("bc"',  # detached like the training jobs
        'launch("bc_distance"',  # the temporal-distance task
        'DISTANCE = True',
        'launch("bc_inverse"',  # multi-step inverse dynamics
        'INVERSE = (4, 8)',
        'bc_entities_chunk',
        'allow_patterns=["bc_entities_*", "bc_entities.png"]',
    ):
        assert needle in text, needle
    # every option the notebook gives bc.py is one that it has
    source = (
        Path(__file__).parents[2] / 'H-JEPA' / 'h_jepa' / 'bc.py'
    ).read_text()
    options = set(re.findall(r'"(--[a-z-]+)"', text))
    assert {'--chunk', '--distance', '--inverse', '--retrain-top'} <= options
    for option in options:
        assert f'"{option}"' in source, option
    # the runs of the other notebooks are not uploaded with it
    assert 'pod.upload' not in text


def test_the_bc_tasks_run_on_their_own():
    cells = cells_of(NOTEBOOK_DIR / 'runpod_bc_entities.ipynb')

    def cell(prefix):
        return next(c for c in cells if c.startswith(prefix))

    # each task is started and waited for only when it is switched on, so
    # one can be run without the others (CHUNKS = (), DISTANCE = False,
    # INVERSE = ())
    for prefix, switch in (
        ('# --- 2. Start the models', 'if CHUNKS:'),
        ('# --- 2. Wait for it', 'if CHUNKS:'),
        ('# --- 2b. Temporal distance, detached', 'if DISTANCE:'),
        ('# --- 2b. Wait for it', 'if DISTANCE:'),
        ('# --- 2c. Multi-step inverse dynamics, detached', 'if INVERSE:'),
        ('# --- 2c. Wait for it', 'if INVERSE:'),
    ):
        assert switch in cell(prefix), prefix
    # a partial run does not replace the figure of the whole set
    figures = cell('# --- 3. Figures')
    assert 'FIGURE = "bc_entities.png" if CHUNKS and DISTANCE and INVERSE' in figures
    assert 'reports / figure' in figures


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


@pytest.fixture()
def pod_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # configure() changes to the volume
    monkeypatch.setenv('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    for folder in ('volume', 'repo'):  # /workspace and the clone exist
        (tmp_path / folder).mkdir()
    pod.configure(
        volume=tmp_path / 'volume',
        repo=tmp_path / 'repo',
        venv=tmp_path / 'venv',
        data=tmp_path / 'data',
        spt_dir=tmp_path / 'spt',
        logs=tmp_path / 'logs',
        seeds='42,43',
    )
    return tmp_path


def test_folder_of_reads_the_spec_with_the_venvs_python(pod_dirs, monkeypatch):
    monkeypatch.setattr(pod, 'REPO', Path(__file__).parents[2])
    monkeypatch.setattr(pod, 'PYTHON', sys.executable)
    assert pod.folder_of('B1') == 'flat'  # trained before the ids
    assert pod.folder_of('A1') == 'flat_balanced'
    assert pod.folder_of('V1') == 'V1'


def test_configure_links_the_volume_and_sets_the_environment(pod_dirs):
    for name in ('ckpts', 'reports', 'eval_tasks'):
        link = pod_dirs / 'data' / name
        assert link.is_symlink()
        assert link.resolve() == (pod_dirs / 'volume' / 'data' / name).resolve()
    assert 'LD_LIBRARY_PATH' not in pod.ENV  # crashes JAX's pip CUDA
    assert pod.ENV['HJEPA_HOME'] == str(pod_dirs / 'data')
    assert pod.run_dir('flat') == pod_dirs / 'data/ckpts/craftax/flat/seed42'
    assert f'spt_dir={pod_dirs / "spt"}' in pod.train_overrides(False)
    assert 'wandb.enabled=true' in pod.train_overrides(True)


def test_a_detached_job_is_followed_and_its_failure_is_seen(pod_dirs):
    pod.launch('job', [sys.executable, '-c', 'print("step 1"); print("step 2")'])
    pod.wait('job', 'step', every=0.05)
    assert not pod.running('job')
    assert pod.progress('job', 'step', 2) == ['step 1', 'step 2']
    assert pod.progress('job', 'nothing') == ['(no progress line yet)']
    script = 'print("x failed with exit code 1")'
    pod.launch('bad', [sys.executable, '-c', script])
    with pytest.raises(AssertionError, match='a job of bad failed'):
        pod.wait('bad', 'x', every=0.05)
    # a later successful start counts, not the failed one before it
    pod.launch('bad', [sys.executable, '-c', 'print("ok")'])
    pod.wait('bad', 'ok', every=0.05)


class FakeApi:
    token = 'x'

    def __init__(self, files=()):
        self.files, self.uploads = list(files), []

    def list_repo_files(self, repo_id, repo_type):
        assert repo_type == 'dataset'
        return self.files

    def upload_folder(self, **kw):
        self.uploads.append(kw)


def fake_download(tmp_path, downloaded):
    def download(repo_id, filename, repo_type, token):
        downloaded.append(filename)
        source = tmp_path / ('hub_' + filename.replace('/', '_'))
        source.write_text('hub')
        return str(source)

    return download


def test_fetch_takes_a_run_folder_from_the_hub(pod_dirs):
    hub = [
        'data/craftax_expert_train.h5',
        'runs/flat/seed42/config.yaml',
        'runs/flat/seed42/flat_object.ckpt',
        'runs/flat/seed42/eval_A_val_craftax_plan/planning_report.md',
        'runs/flat/seed43/config.yaml',  # another seed
        'runs/flat_h1/seed42/config.yaml',  # another model
    ]
    downloaded = []
    pod.fetch(
        FakeApi(hub), 'org/repo', 'flat/seed42',
        download=fake_download(pod_dirs, downloaded),
    )
    assert downloaded == hub[1:4]
    run = pod.run_dir('flat')
    assert (run / 'eval_A_val_craftax_plan' / 'planning_report.md').exists()
    assert (run / 'flat_object.ckpt').read_text() == 'hub'


def test_fetch_can_take_only_the_readouts_of_a_run(pod_dirs):
    hub = [
        'runs/B2/seed42/config.yaml',
        'runs/B2/seed42/B2_object.ckpt',
        'runs/B2/seed42/race/trace.json',
        'runs/B2/seed42/race/meta.json',
    ]
    downloaded = []
    pod.fetch(
        FakeApi(hub), 'org/repo', 'B2/seed42',
        download=fake_download(pod_dirs, downloaded), only='race/',
    )
    assert downloaded == hub[2:]
    assert (pod.run_dir('B2') / 'race' / 'trace.json').exists()
    assert not (pod.run_dir('B2') / 'B2_object.ckpt').exists()


def test_fetch_keeps_what_the_volume_has_and_finds_other_layouts(pod_dirs):
    run = pod.run_dir('flat')
    run.mkdir(parents=True)
    (run / 'config.yaml').write_text('local')
    hub = [
        'runs/craftax/flat/seed42/config.yaml',  # one folder deeper
        'runs/craftax/flat/seed42/flat_object.ckpt',
    ]
    downloaded = []
    pod.fetch(
        FakeApi(hub), 'org/repo', 'flat/seed42',
        download=fake_download(pod_dirs, downloaded),
    )
    assert downloaded == ['runs/craftax/flat/seed42/flat_object.ckpt']
    assert (run / 'config.yaml').read_text() == 'local'  # not replaced


def test_fetch_stops_when_the_hub_has_no_single_folder(pod_dirs):
    download = fake_download(pod_dirs, [])
    with pytest.raises(AssertionError, match='neither on this pod'):
        pod.fetch(FakeApi([]), 'org/repo', 'flat/seed42', download=download)
    # the random baseline may be missing: run.py runs it
    assert pod.fetch(
        FakeApi([]), 'o/r', 'random/eval_A_val', required=False,
        download=download,
    ) == []
    twice = ['runs/a/flat/seed42/config.yaml', 'runs/b/flat/seed42/x.ckpt']
    with pytest.raises(AssertionError, match='2 folders'):
        pod.fetch(FakeApi(twice), 'org/repo', 'flat/seed42', download=download)


def test_an_older_probe_report_is_moved_aside(tmp_path):
    health = tmp_path / 'health'
    health.mkdir()
    assert not pod.current_probe_report(tmp_path)  # none yet
    for text, keep in (
        ('# Probe (epoch probe)', 'probe_first_version'),
        ('# Probe suite\n### Encoder', 'probe_suite_v1'),
    ):
        (health / 'probe_report.md').write_text(text)
        (health / 'probe_summary.json').write_text('{}')
        assert not pod.current_probe_report(tmp_path)
        assert (health / keep / 'probe_report.md').read_text() == text
        assert not (health / 'probe_summary.json').exists()
    (health / 'probe_report.md').write_text(f'# Probe suite\n{pod.SUITE_MARK}')
    assert pod.current_probe_report(tmp_path)


def test_the_suite_report_still_has_the_heading_pod_looks_for():
    report = (
        Path(__file__).parents[2] / 'H-JEPA' / 'h_jepa' / 'probing' / 'report.py'
    ).read_text()
    assert f'"{pod.SUITE_MARK}"' in report
    assert '"# Probe suite' in report or "'# Probe suite" in report


def test_upload_leaves_out_the_training_state_and_smoke_runs(pod_dirs):
    api = FakeApi()
    pod.upload(api, 'org/repo', folders=['flat_x', 'flat'], reports=['V1_x_*.md'])
    runs, reports = api.uploads
    assert runs['path_in_repo'] == 'runs'
    assert runs['allow_patterns'] == ['flat_x/**', 'flat/**']
    for pattern in ('**/lightning_resume/**', '*_epoch_*_object.ckpt', '*_step_*_object.ckpt', '*_smoke/**'):
        assert pattern in runs['ignore_patterns']
    assert reports['allow_patterns'] == ['V1_x_*.md']
    pod.upload(api, 'org/repo')
    assert api.uploads[2]['allow_patterns'] is None  # every run


def flat_cells():
    nb = json.loads((NOTEBOOKS[0].parent / 'runpod_B1.ipynb').read_text())
    return [''.join(c['source']) for c in nb['cells']]


def test_the_probe_suite_runs_detached_and_has_a_monitor_cell():
    cells = flat_cells()
    start = next(
        i
        for i, c in enumerate(cells)
        if c.startswith("# --- 2. The probe suite on B1's final model")
    )
    # detached with a live log: `sh` would show nothing until it exits
    assert 'launch("probe_suite"' in cells[start]
    assert 'pod.fetch(' in cells[start]  # the run from the Hub when missing
    assert 'pod.current_probe_report(run)' in cells[start]
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
