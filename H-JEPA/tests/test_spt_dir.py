"""stable-pretraining's folder on a local disk."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

from utils import local_spt_dir  # noqa: E402


def test_none_leaves_the_run_dir_alone(tmp_path):
    run = tmp_path / 'run'
    local_spt_dir(run, None, 'craftax/flat/seed42')
    assert not run.exists()


def test_spt_becomes_a_link_to_the_local_folder(tmp_path):
    run, local = tmp_path / 'volume/run', tmp_path / 'local'
    local_spt_dir(run, str(local), 'craftax/flat/seed42')
    target = local / 'craftax/flat/seed42'
    assert (run / 'spt').is_symlink() and (run / 'spt').resolve() == target
    # what stable-pretraining writes through the link lands on the local disk
    (run / 'spt/runs').mkdir(parents=True)
    (run / 'spt/runs/metrics.csv').write_text('step\n')
    assert (target / 'runs/metrics.csv').exists()
    local_spt_dir(run, str(local), 'craftax/flat/seed42')  # a second start
    assert (run / 'spt').resolve() == target


def test_a_real_spt_folder_of_an_earlier_run_is_kept(tmp_path):
    run, local = tmp_path / 'run', tmp_path / 'local'
    (run / 'spt').mkdir(parents=True)
    (run / 'spt/old.txt').write_text('keep')
    local_spt_dir(run, str(local), 'a')
    assert (run / 'spt').is_symlink()
    assert (run / 'spt_volume/old.txt').read_text() == 'keep'
    # and again, with another real folder: nothing is overwritten
    (run / 'spt').unlink()
    (run / 'spt').mkdir()
    (run / 'spt/newer.txt').write_text('keep too')
    local_spt_dir(run, str(local), 'a')
    assert (run / 'spt_volume/old.txt').exists()
    assert (run / 'spt_volume2/newer.txt').exists()
