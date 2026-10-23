"""ResumeCheckpoint: a failing write must not stop the training."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

import utils  # noqa: E402


def fake_trainer(failures):
    """A trainer whose save_checkpoint raises FileNotFoundError `failures`
    times (an fsync on a network volume), then writes."""
    calls = []

    def save_checkpoint(path):
        calls.append(path)
        if len(calls) <= failures:
            raise FileNotFoundError(2, 'No such file or directory')
        Path(path).write_text('state')

    processed = SimpleNamespace(processed=2000)
    loop = SimpleNamespace(
        epoch_loop=SimpleNamespace(
            batch_progress=SimpleNamespace(total=processed)
        )
    )
    trainer = SimpleNamespace(
        save_checkpoint=save_checkpoint, is_global_zero=True, fit_loop=loop
    )
    return trainer, calls


def callback(tmp_path):
    return utils.ResumeCheckpoint(
        tmp_path / 'resume/last.ckpt', 2000, attempts=3, wait=0
    )


def test_a_failed_write_is_tried_again(tmp_path):
    trainer, calls = fake_trainer(failures=1)
    callback(tmp_path)._save(trainer)
    assert len(calls) == 2
    assert (tmp_path / 'resume/last.ckpt').read_text() == 'state'


def test_a_write_that_keeps_failing_is_skipped_not_raised(tmp_path):
    trainer, calls = fake_trainer(failures=99)
    old = tmp_path / 'resume/last.ckpt'
    old.parent.mkdir()
    old.write_text('older state')
    callback(tmp_path)._save(trainer)  # does not raise
    assert len(calls) == 3
    assert old.read_text() == 'older state'  # the previous one is untouched


def test_other_errors_still_raise(tmp_path):
    trainer, _ = fake_trainer(failures=0)

    def broken(path):
        raise ValueError('a bug, not the volume')

    trainer.save_checkpoint = broken
    try:
        callback(tmp_path)._save(trainer)
    except ValueError:
        return
    raise AssertionError('a ValueError must not be swallowed')
