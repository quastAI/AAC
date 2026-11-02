"""The race of the first epochs: the dense checkpoints, the state encoder,
the readout of a checkpoint and the report."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

from models.encoders.build_encoder import build_encoder  # noqa: E402
from models.hjepa import HJEPA  # noqa: E402
from probing import targets as tg  # noqa: E402
from race import evaluate, readout, report  # noqa: E402
from utils import DenseCheckpoints  # noqa: E402


class FakeModule:
    """The `pl_module` of a callback: `.model` is what is saved."""

    def __init__(self):
        self.model = torch.nn.Linear(2, 2)


def fake_trainer(step=0, epoch=0, per_epoch=10):
    return SimpleNamespace(
        is_global_zero=True,
        global_step=step,
        current_epoch=epoch,
        num_training_batches=per_epoch,
        should_stop=False,
    )


def saved(folder):
    return sorted(
        int(p.name.split('_step_')[1].split('_')[0])
        for p in folder.glob('*_step_*_object.ckpt')
    )


def test_dense_checkpoints_save_the_start_the_listed_steps_and_every_n(
    tmp_path,
):
    cb = DenseCheckpoints(tmp_path, 'm', steps=[3], every=4, epochs=1)
    trainer, module = fake_trainer(), FakeModule()
    cb.on_train_start(trainer, module)
    assert saved(tmp_path) == [0]  # the untrained model
    info = json.loads((tmp_path / 'dense_checkpoints.json').read_text())
    assert info['steps_per_epoch'] == 10 and info['steps'] == [3]
    for step in range(1, 13):
        trainer.global_step, trainer.current_epoch = step, int(step > 9)
        cb.on_train_batch_end(trainer, module, None, None, 0)
    # 3 is listed, 4 and 8 are multiples in epoch 0, 12 is in epoch 1: out
    assert saved(tmp_path) == [0, 3, 4, 8]
    loaded = torch.load(tmp_path / 'm_step_3_object.ckpt', weights_only=False)
    assert isinstance(loaded, torch.nn.Linear)


def test_dense_checkpoints_do_not_overwrite_the_start_after_a_resume(tmp_path):
    cb = DenseCheckpoints(tmp_path, 'm')
    cb.on_train_start(fake_trainer(step=500), FakeModule())
    assert saved(tmp_path) == []


def test_dense_checkpoints_stop_after_the_epoch_they_are_told_to(tmp_path):
    cb = DenseCheckpoints(tmp_path, 'm', stop_after_epochs=2)
    trainer = fake_trainer()
    cb.on_train_epoch_end(trainer, FakeModule())
    assert not trainer.should_stop
    trainer.current_epoch = 1  # the second epoch has ended
    cb.on_train_epoch_end(trainer, FakeModule())
    assert trainer.should_stop
    never = fake_trainer(epoch=5)
    DenseCheckpoints(tmp_path, 'm').on_train_epoch_end(never, FakeModule())
    assert not never.should_stop


def test_the_state_encoder_reads_the_state_and_ignores_pixel_keys():
    encoder, dim = build_encoder(
        {
            'type': 'state_mlp',
            'input_dim': tg.STATE_DIM,
            'hidden_dim': 32,
            'output_dim': 16,
            'scale': 'tiny',  # a key of the pixel encoder
        }
    )
    out = encoder(torch.randn(5, tg.STATE_DIM))
    assert out.shape == (5, 16) and dim == 16


def test_a_model_reads_the_observation_it_is_told_to():
    assert HJEPA.obs_key == 'pixels'  # old checkpoints carry no obs_key
    model = HJEPA(jepas=[torch.nn.Identity()], obs_key='state')
    assert model.obs_key == 'state'
    assert HJEPA(jepas=[torch.nn.Identity()]).obs_key == 'pixels'


def test_the_checkpoints_of_a_run_by_step(tmp_path):
    for name in (
        'm_step_0_object.ckpt',
        'm_step_50_object.ckpt',
        'm_epoch_1_object.ckpt'.replace('epoch_1', 'epoch_1'),
        'm_epoch_2_object.ckpt',
        'm_step_20_object.ckpt',
        'm_object.ckpt',
        'm_step_100_object.ckpt',
    ):
        (tmp_path / name).write_text('x')
    found = evaluate.checkpoints(tmp_path, steps_per_epoch=100)
    # the epochs end at 100 and 200; a dense file of step 100 wins;
    # the final model is not a step
    assert list(found) == [0, 20, 50, 100, 200]
    assert found[100].name == 'm_step_100_object.ckpt'
    assert found[200].name == 'm_epoch_2_object.ckpt'


# --------------------------------------------------------------------------
# the readout


E_, L, DIM = 12, 8, 6
VIEW = tg.VIEW_TILES * tg.VIEW_CHANNELS


def write_val(path, episodes=E_, length=300, seed=0):
    """A validation file with random views, vitals and actions."""
    rng = np.random.default_rng(seed)
    n = episodes * length
    state = np.zeros((n, tg.STATE_DIM), np.float32)
    tiles = state[:, :VIEW].reshape(n, tg.VIEW_TILES, tg.VIEW_CHANNELS)
    blocks = rng.integers(0, tg.NUM_BLOCKS, (n, tg.VIEW_TILES, 1))
    np.put_along_axis(tiles[:, :, : tg.NUM_BLOCKS], blocks, 1, axis=2)
    tiles[:, 0, tg.NUM_BLOCKS] = rng.random(n) < 0.5  # a zombie on tile 0
    action = rng.integers(0, 17, (n, 1)).astype(np.float32)
    action[np.arange(1, episodes + 1) * length - 1] = np.nan
    with h5py.File(path, 'w') as f:
        f['ep_len'] = np.full(episodes, length, np.int32)
        f['ep_offset'] = np.arange(episodes) * length
        f['source'] = np.zeros(n, np.int8)
        f['state'] = state
        f['action'] = action
        f['inventory'] = rng.integers(0, 5, (n, 12)).astype(np.int16)
        f['vitals'] = rng.integers(0, 9, (n, 4)).astype(np.int8)
        f['daylight'] = rng.random((n, 1)).astype(np.float32)
        f['facing'] = rng.integers(1, 5, (n, 1)).astype(np.int8)
        f['achievements'] = rng.integers(0, 2, (n, 22)).astype(np.uint8)
        f['position'] = rng.integers(0, 64, (n, 2)).astype(np.int16)
        f['step_idx'] = np.tile(np.arange(length), episodes).astype(np.int32)


class CopyLevel(torch.nn.Module):
    """A level-1 stand-in: a linear encoder of the state and a predictor
    that copies the last context latent."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(tg.STATE_DIM, DIM)
        with torch.no_grad():  # axis 0 is the zombie flag of tile 0
            self.encoder.weight.zero_()
            self.encoder.weight[0, tg.NUM_BLOCKS] = 1.0
            self.encoder.weight[1:, 100:140].normal_()
        self.action_encoder = torch.nn.Identity()
        self.action_embed = torch.nn.Identity()
        self.predictor = torch.nn.Identity()

    def predict(self, emb, act_emb):
        return emb[:, -1:].expand_as(emb)


class FakeModel(torch.nn.Module):
    num_levels = 1

    def __init__(self):
        super().__init__()
        self.level = CopyLevel()

    def get_level(self, level):
        return self.level


@pytest.fixture(scope='module')
def val_file(tmp_path_factory):
    path = tmp_path_factory.mktemp('race') / 'val.h5'
    write_val(path)
    return path


def test_a_sample_holds_the_state_the_factors_and_the_neighbours(val_file):
    s = readout.Sample(val_file, obs_key='state', windows=2, length=16)
    assert s.shape == (E_, 2, 16)
    assert s.obs.shape == s.shape + (tg.STATE_DIM,)
    assert s.obs.dtype == np.float32
    assert s.near.shape == s.shape + (4 * tg.NUM_BLOCKS,)
    assert 'mob:zombie' in s.targets and 'vital:food' in s.targets
    tr, te = s.split
    assert not set(tr) & set(te)


def test_the_readout_reads_what_the_latent_holds(val_file):
    torch.manual_seed(0)
    s = readout.Sample(val_file, obs_key='state', windows=2, length=16)
    entry = readout.check(FakeModel(), s, history=3, num_actions=17)
    assert {
        'r2',
        'near',
        'rank',
        'dead_axes',
        'clip_share',
        'pred_over_copy',
    } <= set(entry)
    assert 0 < entry['rank'] <= DIM
    assert entry['pred_over_copy'] == pytest.approx(
        1.0
    )  # copying is the predictor
    assert entry['r2']['mob:zombie'] > 0.9  # axis 0 is the zombie flag
    assert (
        entry['r2']['vital:food'] < 0.2
    )  # the latent holds nothing of the vitals
    assert json.loads(json.dumps(entry))  # a trace is written as JSON


def test_the_trace_is_written_resumed_and_reported(
    val_file, tmp_path, monkeypatch
):
    from omegaconf import OmegaConf

    run = tmp_path / 'B9' / 'seed42'
    run.mkdir(parents=True)
    cfg = OmegaConf.create(
        {
            'obs_key': 'state',
            'level1': {
                'wm': {
                    'history_size': 3,
                    'num_actions': 17,
                    'detach_pred_target': True,
                },
                'loss': {'embed': {'sigreg': {'weight': 0.09}}},
            },
        }
    )
    OmegaConf.save(cfg, run / 'config.yaml')
    (run / 'dense_checkpoints.json').write_text(
        json.dumps({'steps_per_epoch': 100})
    )
    for step in (0, 10):
        (run / f'B9_step_{step}_object.ckpt').write_text('x')
    monkeypatch.setattr(evaluate, 'load_model', lambda path: FakeModel())
    monkeypatch.setattr('cpu_threads.limit_threads', lambda *a, **k: None)
    real_sample = readout.Sample
    monkeypatch.setattr(
        readout,
        'Sample',
        lambda *a, **k: real_sample(
            *a,
            windows=2,
            length=16,
            **{x: y for x, y in k.items() if x != 'windows'},
        ),
    )
    trace = evaluate.evaluate(run, val_file, device='cpu')
    assert [e['step'] for e in trace] == [0, 10]
    meta = json.loads((run / 'race' / 'meta.json').read_text())
    assert meta['obs_key'] == 'state' and meta['stop_grad'] is True
    assert (run / 'race' / 'trace_done').exists()
    (run / 'B9_step_20_object.ckpt').write_text(
        'x'
    )  # a rerun reads only what is new
    calls = []
    monkeypatch.setattr(
        readout,
        'check',
        lambda model, s, *a, **k: (
            calls.append(1)
            or {
                k: v for k, v in trace[0].items() if k not in ('step', 'epoch')
            }
        ),
    )
    evaluate.evaluate(run, val_file, device='cpu')
    assert len(calls) == 1
    out = tmp_path / 'r' / 'race.md'
    report.main(['--out', str(out), f'B9={run}'])
    text = out.read_text()
    assert '| B9 | 0 |' in text and '| B9 | 20 |' in text
    assert 'stop-grad True' in text and out.with_suffix('.png').exists()


# --------------------------------------------------------------------------
# the configs: A3 and A4 are B1 and V3 with the encoder of the state


def compose(name, overrides=()):
    import hydra
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    config_dir = Path(__file__).parents[1] / 'h_jepa' / 'config' / 'train'
    with hydra.initialize_config_dir(
        config_dir=str(config_dir), version_base=None
    ):
        return hydra.compose(config_name=name, overrides=list(overrides))


def test_the_state_config_differs_from_the_baseline_in_the_encoder_only():
    from omegaconf import OmegaConf

    base = OmegaConf.to_container(
        compose('craftax_hjepa', ['num_levels=1']), resolve=True
    )
    state = OmegaConf.to_container(
        compose('craftax_hjepa_state', ['num_levels=1']), resolve=True
    )
    assert base['obs_key'] == 'pixels' and state['obs_key'] == 'state'
    assert state['level1']['encoder']['type'] == 'state_mlp'
    assert state['level1']['encoder']['input_dim'] == tg.STATE_DIM
    for key in ('wm', 'loss', 'predictor'):  # the losses and the predictor
        if key in base['level1']:
            assert state['level1'][key] == base['level1'][key], key
    for key in ('optim', 'trainer', 'loader', 'seed'):
        if key in base:
            assert state[key] == base[key], key
    assert state['data']['dataset']['keys_to_load'] == [
        'state',
        'action',
        'source',
    ]


def test_the_state_model_builds_and_encodes_a_state():
    from hjepa_utils import create_world_model

    cfg = compose(
        'craftax_hjepa_state',
        ['num_levels=1', 'level1.wm.detach_pred_target=true'],
    )
    model = create_world_model(cfg)[0]
    assert model.obs_key == 'state'
    z = model.get_level(1).encoder(torch.randn(4, tg.STATE_DIM))
    assert z.shape == (4, 192)
