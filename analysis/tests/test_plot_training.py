"""plot_training on a synthetic metrics CSV."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import plot_training as pt  # noqa: E402

HEADER = (
    'epoch,step,fit/loss,fit/pred_loss_level1,validate/loss_epoch,'
    'health/level1/rep/erank_frac,hparams/lr_model_opt\n'
)


def write_csv(path, rows=60):
    path.parent.mkdir(parents=True)
    lines = [HEADER]
    for step in range(rows):
        # training keys every step, the health metric every 10 steps and
        # the validation loss once: sparse rows
        health = f'{0.1 + step / 100}' if step % 10 == 0 else ''
        validation = '0.5' if step == rows - 1 else ''
        lines.append(
            f'0,{step},{1 / (step + 1)},{0.5 / (step + 1)},'
            f'{validation},{health},5e-05\n'
        )
    path.write_text(''.join(lines))
    return path


def test_read_series_keeps_only_the_rows_with_a_value(tmp_path):
    series = pt.read_series(write_csv(tmp_path / 'x/m.csv'))
    assert len(series['fit/loss'][0]) == 60
    assert series['health/level1/rep/erank_frac'][0] == [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]
    assert series['validate/loss_epoch'] == ([59.0], [0.5])
    assert 'step' not in series and 'epoch' not in series


def test_find_metrics_takes_the_newest_spt_csv(tmp_path):
    old = write_csv(tmp_path / 'spt/runs/20260101/0000/aaa/metrics.csv')
    new = write_csv(tmp_path / 'spt/runs/20260102/0000/bbb/metrics.csv')
    os.utime(old, (1, 1))
    assert pt.find_metrics(tmp_path) == new
    with pytest.raises(SystemExit):
        pt.find_metrics(tmp_path / 'empty')


def test_find_metrics_follows_spt_location_to_a_local_folder(tmp_path):
    run, local = tmp_path / 'volume/run', tmp_path / 'local/run'
    csv = write_csv(local / 'runs/20260101/0000/aaa/metrics.csv')
    run.mkdir(parents=True)
    (run / 'spt_location.txt').write_text(f'{local}\n')
    assert pt.find_metrics(run) == csv


def test_main_writes_the_figure_and_a_copy_of_the_csv(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    write_csv(tmp_path / 'spt/runs/20260101/0000/aaa/metrics.csv')
    monkeypatch.setattr(
        sys, 'argv', ['plot_training.py', '--run', str(tmp_path)]
    )
    pt.main()
    assert (tmp_path / 'health' / 'loss_curves.png').stat().st_size > 1000
    assert (tmp_path / 'health' / 'metrics.csv').exists()


def test_plot_names_the_panels_it_drew_and_refuses_an_unknown_csv(tmp_path):
    pytest.importorskip('matplotlib')
    series = pt.read_series(write_csv(tmp_path / 'x/m.csv'))
    titles = pt.plot(series, tmp_path / 'fig.png')
    assert titles == [
        'loss',
        'prediction MSE',
        'latent: rank, scale',
        'learning rate',
    ]
    with pytest.raises(SystemExit):
        pt.plot({'something/else': ([0.0], [1.0])}, tmp_path / 'none.png')
