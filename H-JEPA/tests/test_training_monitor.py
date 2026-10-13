"""The training health callback, on the fixture dataset, stub trainer."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


pytest.importorskip('transformers')
pytest.importorskip('stable_pretraining')
pytest.importorskip('lightning')
hydra = pytest.importorskip('hydra')
pytest.importorskip('sklearn')

H_JEPA = Path(__file__).parents[1] / 'h_jepa'
sys.path.insert(0, str(H_JEPA))

import data as hjepa_data  # noqa: E402
from hjepa_utils import create_world_model, hjepa_forward  # noqa: E402
from training_monitor import (  # noqa: E402
    TrainingHealthCallback,
    build_health_loader,
)


def float_pixels(batch):
    for key, value in batch.items():
        if key.startswith('pixels'):
            batch[key] = value.float() / 255


class FakeModule(torch.nn.Module):
    def __init__(self, model, losses):
        super().__init__()
        self.model = model
        for name, loss in losses.items():
            setattr(self, name, loss)

    def log_dict(self, *args, **kwargs):
        pass


class Logger:
    def __init__(self):
        self.logged = {}

    def log_metrics(self, metrics, step=None):
        self.logged.update(metrics)


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    import numpy as np

    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    with hydra.initialize_config_dir(
        config_dir=str(H_JEPA / 'config' / 'train'), version_base=None
    ):
        cfg = hydra.compose(
            config_name='craftax_hjepa_l2',
            overrides=[
                'data.dataset.val_name=craftax',
                'health.drift_horizon=3',
                'health.live_every=1',
                'health.warmup_steps=0',
            ],
        )
    model, losses = create_world_model(cfg)
    return cfg, FakeModule(model, losses), np


@pytest.fixture()
def callback(setup, dataset, tmp_path):
    cfg, module, _ = setup
    transform = hjepa_data.Compose(
        hjepa_data.OneHotColumn(17, 'action', 'action')
    )
    loader, frames = build_health_loader(
        cfg,
        dataset.h5_path.parent,
        transform,
        frames=33,
        batch_size=4,
        max_clips=8,
    )
    assert loader is not None and frames == 33
    return TrainingHealthCallback(
        cfg, tmp_path, health_loader=loader, normalize_fn=float_pixels
    )


def test_evaluation_covers_every_metric_family(setup, callback):
    _, module, _ = setup
    levels = callback.evaluate(module, max_batches=2)

    assert set(levels) == {1, 2}
    for entry in levels.values():
        assert entry['rep']['erank'] > 0 and entry['rep']['dim'] == 192
        assert entry['dyn']['pred_over_copy'] is not None
        assert entry['dyn']['action_lift'] is not None
        assert set(entry['drift']) == {1, 3}  # planning and training history
        assert entry['geometry']['gaps'][0] == 1
        assert 0 <= entry['idm']['accuracy'] <= 1
        assert entry['grad']['explosions'] == 0
    assert 'vq' not in levels[1] and levels[2]['vq']['num_codes'] == 64
    assert levels[2]['vq']['groups'] == 2

    # evaluating must leave the model as it was
    assert module.model.get_level(1).target_length == 4
    assert module.training


def test_report_and_findings_from_an_untrained_model(
    setup, callback, tmp_path
):
    from health import diagnose_training, render_training_report

    _, module, _ = setup
    levels = callback.evaluate(module, max_batches=1)
    summary = callback._summary(levels)
    findings = diagnose_training(summary)
    report = render_training_report(summary, findings)
    assert '# Training report' in report
    assert 'distance ratio' in report and 'rollout mse' in report


def test_live_hook_logs_and_flags(setup, callback):
    cfg, module, _ = setup
    steps = 33
    batch = {
        'pixels_level1': torch.rand(4, steps, 3, 63, 63),
        'action_level1': torch.nn.functional.one_hot(
            torch.randint(0, 17, (4, steps)), 17
        ).float(),
    }
    module.train()
    outputs = hjepa_forward(
        module, batch, 'fit', cfg, normalize_batch=lambda b: None
    )
    logger = Logger()
    trainer = SimpleNamespace(
        global_step=0, is_global_zero=True, logger=logger
    )
    outputs['loss'].backward()
    callback.on_after_backward(trainer, module)
    callback.on_train_batch_end(trainer, module, outputs, batch, 0)

    assert 'health/level1/rep/erank' in logger.logged
    assert 'health/level2/vq/perplexity' in logger.logged
    assert 'health/level1/grad/last' in logger.logged
    assert callback.grad_norms[1] and callback.grad_norms[2]
    assert callback._seen  # an untrained model raises findings


def test_live_hook_accepts_autocast_outputs(setup, callback):
    # under precision bf16-mixed the training forward returns bf16 latents
    # while the monitor's predictor is fp32 (this crashed at the first live step)
    cfg, module, _ = setup
    steps = 33
    batch = {
        'pixels_level1': torch.rand(4, steps, 3, 63, 63),
        'action_level1': torch.nn.functional.one_hot(
            torch.randint(0, 17, (4, steps)), 17
        ).float(),
    }
    module.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outputs = hjepa_forward(
            module, batch, 'fit', cfg, normalize_batch=lambda b: None
        )
    assert outputs['embed_1'].dtype == torch.bfloat16
    logger = Logger()
    trainer = SimpleNamespace(
        global_step=0, is_global_zero=True, logger=logger
    )
    callback.on_train_batch_end(trainer, module, outputs, batch, 0)
    assert 'health/level1/rep/erank' in logger.logged


def test_loader_falls_back_to_shorter_clips(setup, dataset):
    cfg, _, _ = setup
    transform = hjepa_data.Compose(
        hjepa_data.OneHotColumn(17, 'action', 'action')
    )
    loader, frames = build_health_loader(
        cfg,
        dataset.h5_path.parent,
        transform,
        frames=200,  # longer than the 58-frame fixture episode
        batch_size=4,
        max_clips=8,
    )
    assert loader is not None and 4 <= frames < 58
    none, zero = build_health_loader(
        cfg,
        dataset.h5_path.parent,
        transform,
        frames=3,
        batch_size=4,
        max_clips=8,
    )
    assert none is None and zero == 0
