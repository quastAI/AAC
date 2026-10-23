"""stable-pretraining's folder on a local disk."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

from utils import local_spt_dir  # noqa: E402


def test_none_keeps_the_folder_in_the_run_dir_and_creates_nothing(tmp_path):
    run = tmp_path / 'run'
    assert local_spt_dir(run, None, 'craftax/flat/seed42') == run / 'spt'
    assert not run.exists()


def test_the_local_folder_is_returned_and_its_path_is_written(tmp_path):
    run, local = tmp_path / 'volume/run', tmp_path / 'local'
    got = local_spt_dir(run, str(local), 'craftax/flat/seed42')
    assert got == local / 'craftax/flat/seed42' and got.is_dir()
    # no link on the volume (it cannot be read back there), a text file
    assert not (run / 'spt').exists()
    assert (run / 'spt_location.txt').read_text().strip() == str(got)
    # a second start gives the same folder
    assert local_spt_dir(run, str(local), 'craftax/flat/seed42') == got


def test_a_real_spt_folder_of_an_earlier_run_is_left_alone(tmp_path):
    run, local = tmp_path / 'run', tmp_path / 'local'
    (run / 'spt').mkdir(parents=True)
    (run / 'spt/old.txt').write_text('keep')
    local_spt_dir(run, str(local), 'a')
    assert (run / 'spt/old.txt').read_text() == 'keep'
