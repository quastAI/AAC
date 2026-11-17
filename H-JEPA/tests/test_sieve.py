"""The Sieve (README id V4, aac_recipe.md): its losses, its training hooks
and its metrics."""

import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

from sieve import decoder as dec  # noqa: E402
from sieve import losses as ls  # noqa: E402
from sieve import probe as pr  # noqa: E402
from sieve import report as rp  # noqa: E402

FIXTURE = Path(__file__).parent / 'data' / 'craftax_classic_episode.npz'


def compose(name, overrides=()):
    import hydra
    from omegaconf import OmegaConf

    import main_hjepa  # noqa: F401  (registers the `eval` resolver)

    assert OmegaConf.has_resolver('eval')
    config_dir = Path(__file__).parents[1] / 'h_jepa' / 'config' / 'train'
    with hydra.initialize_config_dir(
        config_dir=str(config_dir), version_base=None
    ):
        return hydra.compose(config_name=name, overrides=list(overrides))


# --------------------------------------------------------------------------
# the patches, the weights and the two stages' terms


def test_patches_follow_the_vit_order_and_invert():
    frame = torch.zeros(3, 63, 63)
    frame[:, 2 * 7 : 3 * 7, 5 * 7 : 6 * 7] = 1.0  # row 2, column 5
    patches = dec.patchify(frame[None])[0]
    assert patches.shape == (81, 147)
    assert patches.sum(-1).nonzero().flatten().tolist() == [2 * 9 + 5]
    x = torch.rand(2, 4, 3, 63, 63)
    assert torch.equal(dec.unpatchify(dec.patchify(x)), x)


def test_unit_pixels_undo_the_encoders_normalisation():
    from main_hjepa import _normalize_uint8_images_on_device

    raw = torch.randint(0, 256, (2, 3, 63, 63), dtype=torch.uint8)
    batch = {'pixels': raw.clone()}
    _normalize_uint8_images_on_device(batch)
    unit = dec.to_unit(batch['pixels'])
    assert torch.allclose(unit, raw.float() / 255, atol=1e-5)


def test_error_weights_favour_bad_patches_with_mean_one_and_a_cap():
    errors = torch.tensor([1.0, 1.0, 2.0, 1000.0])
    w = ls.error_weights(errors, power=1.0, cap=2.5)
    assert w[2] > w[0] and w.max() <= 2.5
    w = ls.error_weights(torch.rand(64) + 0.5, cap=20.0)
    assert w.mean() == pytest.approx(1.0, abs=1e-5)


def terms(stage, tau=0.05):
    torch.manual_seed(0)
    decoder = dec.PatchDecoder(8, hidden_dim=16, patch_dim=8)
    decoder.set_pixel_variance(tau / decoder.tau_scale)
    emb = torch.randn(3, 2, 8, requires_grad=True)
    pred = torch.randn(3, 8, requires_grad=True)
    pixels = torch.randn(3, 2, 3, 63, 63)
    cfg = {'stage': stage, 'recon_weight': 0.3}
    out = ls.sieve_terms(decoder, emb, pred, pixels, cfg, history=1)
    return out, emb, pred


def grads(loss, *tensors):
    return [
        g if g is not None else torch.zeros_like(t)
        for g, t in zip(
            torch.autograd.grad(loss, tensors, retain_graph=True,
                                allow_unused=True),
            tensors,
        )
    ]


def test_stage_one_reconstructs_every_frame_and_leaves_the_predictor():
    out, emb, pred = terms(1)
    g_emb, g_pred = grads(out['sieve_loss'], emb, pred)
    assert (g_emb[:, 0].abs().sum() > 0) and (g_emb[:, 1].abs().sum() > 0)
    assert g_pred.abs().sum() == 0  # the decoded prediction is only logged
    assert float(out['sieve_loss']) == pytest.approx(
        0.3 * float(out['recon_loss'])
    )
    for key in ('recon_plain_loss', 'decoded_pred_loss', 'sharp_share_loss'):
        assert key in out


def test_stage_two_decodes_the_prediction_and_gates_the_reconstruction():
    out, emb, pred = terms(2, tau=10.0)  # a wide gate: everything passes
    g_emb, g_pred = grads(out['gated_recon_loss'], emb, pred)
    assert g_pred.abs().sum() == 0  # the gate is a stop-gradient
    assert g_emb[:, 0].abs().sum() == 0  # only the next frame's latent
    assert g_emb[:, 1].abs().sum() > 0
    g_emb, g_pred = grads(out['decoded_pred_loss'], emb, pred)
    assert g_pred.abs().sum() > 0 and g_emb.abs().sum() == 0
    assert float(out['gate_open_loss']) > 0.5
    closed, _, _ = terms(2, tau=1e-6)  # nothing foreseen: nothing kept
    assert float(closed['gated_recon_loss']) < 1e-3


def test_stage_two_needs_tau():
    decoder = dec.PatchDecoder(8, hidden_dim=16, patch_dim=8)
    with pytest.raises(ValueError, match='tau'):
        ls.sieve_terms(
            decoder, torch.randn(2, 2, 8), torch.randn(2, 8),
            torch.randn(2, 2, 3, 63, 63), {'stage': 2}, history=1,
        )


def test_the_diff_gate_leaves_the_decoders_precision_out():
    torch.manual_seed(0)
    decoder = dec.PatchDecoder(8, hidden_dim=16, patch_dim=8)
    decoder.set_pixel_variance(1e-4 / decoder.tau_scale)  # a tight tau
    emb = torch.randn(3, 2, 8, requires_grad=True)
    pixels = torch.randn(3, 2, 3, 63, 63)  # nothing the decoder can draw
    same = emb[:, 1].detach().clone().requires_grad_(True)  # foreseen

    def run(pred, gate):
        cfg = {'stage': 2, 'gate': gate}
        return ls.sieve_terms(decoder, emb, pred, pixels, cfg, history=1)

    out = {g: run(same, g) for g in ('abs', 'diff')}
    assert float(out['abs']['gate_open_loss']) < 1e-3  # pixels: shut
    assert float(out['diff']['gate_open_loss']) == pytest.approx(1.0)
    g_emb, g_pred = grads(out['diff']['gated_recon_loss'], emb, same)
    assert g_pred.abs().sum() == 0  # still a stop-gradient
    assert g_emb[:, 1].abs().sum() > 0
    missed = run(torch.randn(3, 8) * 10, 'diff')  # a wrong prediction
    assert float(missed['gate_open_loss']) < 0.5
    with pytest.raises(ValueError, match='gate'):
        run(same, 'other')


def test_the_joint_gate_is_relative_uniform_when_alike_and_floored():
    import math

    errors = torch.tensor([0.1, 1.0, 1.0, 5.0, 0.0])
    g = ls.relative_gate(errors, floor=0.1, beta=1.0)
    assert torch.allclose(g, ls.relative_gate(errors * 1000, 0.1, 1.0))
    assert float(g.min()) >= 0.1 and float(g[-1]) == pytest.approx(1.0)
    assert float(g[0]) > float(g[1]) > float(g[3])  # small error: open
    alike = ls.relative_gate(torch.full((50,), 3.0), floor=0.1, beta=1.0)
    assert torch.allclose(alike, torch.full((50,), 0.1 + 0.9 * math.exp(-1)))
    sharper = ls.relative_gate(errors, floor=0.1, beta=0.25)
    assert float(sharper[3]) < float(g[3])
    for floor, beta in ((1.0, 1.0), (-0.1, 1.0), (0.1, 0.0)):
        with pytest.raises(ValueError):
            ls.relative_gate(errors, floor, beta)


def test_one_stage_decodes_the_prediction_and_gates_the_next_frame():
    torch.manual_seed(0)
    decoder = dec.PatchDecoder(8, hidden_dim=16, patch_dim=8)  # no tau
    emb = torch.randn(3, 2, 8, requires_grad=True)
    pred = torch.randn(3, 8, requires_grad=True)
    pixels = torch.randn(3, 2, 3, 63, 63)
    cfg = {'stage': 'joint', 'gate': 'diff', 'gate_floor': 0.1}
    out = ls.sieve_terms(decoder, emb, pred, pixels, cfg, history=1)
    for key in ('decoded_pred_loss', 'gated_recon_loss', 'gate_open_loss',
                'gate_shut_share_loss', 'gate_std_loss', 'recon_plain_loss'):
        assert key in out, key
    assert 'sharp_share_loss' not in out  # tau is not needed here
    assert 0.1 <= float(out['gate_open_loss']) <= 1.0
    g_emb, g_pred = grads(out['gated_recon_loss'], emb, pred)
    assert g_pred.abs().sum() == 0  # the gate is a stop-gradient
    assert g_emb[:, 0].abs().sum() == 0  # only the next frame
    assert g_emb[:, 1].abs().sum() > 0
    g_emb, g_pred = grads(out['decoded_pred_loss'], emb, pred)
    assert g_pred.abs().sum() > 0 and g_emb.abs().sum() == 0
    total = float(out['decoded_pred_loss']) + float(out['gated_recon_loss'])
    assert float(out['sieve_loss']) == pytest.approx(total)
    for bad in ({'stage': 3}, {'stage': 'one'}):
        with pytest.raises(ValueError, match='stage'):
            ls.sieve_terms(decoder, emb, pred, pixels, bad, history=1)


def test_pixel_variance_reads_runs_of_frames(tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.choice([0, 255], (200, 63, 63, 3)).astype(np.uint8)
    with h5py.File(tmp_path / 'f.h5', 'w') as f:
        f['pixels'] = pixels
        f['ep_len'] = np.array([100, 100])
        f['ep_offset'] = np.array([0, 100])
    v = dec.pixel_variance(tmp_path / 'f.h5', frames=200, run=50)
    assert v == pytest.approx(0.25, abs=0.01)  # coin flips of 0 and 1


# --------------------------------------------------------------------------
# the training: config, model, forward, schedule, fork, starting checkpoint


def test_the_config_is_b1_with_the_recipes_changes():
    from omegaconf import OmegaConf

    b1 = OmegaConf.to_container(
        compose('craftax_hjepa', ['num_levels=1']), resolve=True
    )
    sv = OmegaConf.to_container(compose('craftax_sieve'), resolve=True)
    assert sv['num_levels'] == 1
    l1 = sv['level1']
    assert l1['loss']['embed']['sigreg']['enabled'] is False
    assert l1['idm_coeff'] == 0.0
    assert l1['wm']['history_size'] == 1 and l1['wm']['rollout_n'] == 1
    assert l1['wm']['detach_pred_target'] is True
    assert l1['sieve']['stage'] == 1 and l1['sieve']['recon_weight'] == 0.3
    assert sv['optimizer']['schedule'] == 'constant'
    assert sv['loader']['batch_size'] == 256
    assert sv['data']['dataset']['level1']['num_steps'] == 2
    for key in ('encoder', 'predictor', 'action_embed'):
        assert l1[key] == b1['level1'][key], key
    assert sv['save_initial_model'] is True and sv['init_from'] is None


def test_the_one_stage_config_is_the_sieve_with_the_relative_gate():
    from omegaconf import OmegaConf

    sv = OmegaConf.to_container(compose('craftax_sieve'), resolve=True)
    jo = OmegaConf.to_container(compose('craftax_sieve_joint'), resolve=True)
    s = jo['level1']['sieve']
    assert s['stage'] == 'joint' and s['gate'] == 'diff'
    assert s['gate_floor'] == 0.1 and s['gate_beta'] == 1.0
    assert s['error_power'] == 0.0 and jo['trainer']['max_epochs'] == 10
    for key in ('encoder', 'predictor', 'action_embed', 'wm', 'idm_coeff',
                'loss', 'lr'):
        assert jo['level1'][key] == sv['level1'][key], key
    assert jo['optimizer'] == sv['optimizer']
    assert jo['loader']['batch_size'] == sv['loader']['batch_size']


def forward(model, cfg, stage):
    from hjepa_utils import hjepa_forward
    from main_hjepa import _normalize_uint8_images_on_device

    cfg.level1.sieve.stage = stage
    batch = {
        'pixels_level1': torch.randint(
            0, 256, (4, 2, 3, 63, 63), dtype=torch.uint8
        ),
        'action_level1': torch.nn.functional.one_hot(
            torch.randint(0, 17, (4, 2)), 17
        ).float(),
    }
    module = types.SimpleNamespace(model=model, log_dict=lambda d, **k: None)
    return hjepa_forward(
        module, batch, 'fit', cfg,
        normalize_batch=_normalize_uint8_images_on_device,
    )


def test_both_stages_train_the_model_through_the_usual_forward():
    from hjepa_utils import create_world_model

    cfg = compose('craftax_sieve')
    model, losses = create_world_model(cfg)
    assert losses == {}  # no SIGReg
    jepa = model.get_level(1)
    assert isinstance(jepa.decoder, dec.PatchDecoder)
    assert not hasattr(jepa, 'idm_loss_fn')
    jepa.decoder.set_pixel_variance(0.05)
    for stage, keys in (
        (1, ('recon_loss_level1', 'pred_loss_level1')),
        (2, ('decoded_pred_loss_level1', 'gated_recon_loss_level1')),
        ('joint', ('decoded_pred_loss_level1', 'gated_recon_loss_level1',
                   'gate_open_loss_level1', 'gate_std_loss_level1')),
    ):
        out = forward(model, cfg, stage)
        for key in (*keys, 'sieve_loss_level1', 'loss'):
            assert key in out, key
        out['loss'].backward()
        enc = [p.grad for p in jepa.encoder.parameters() if p.grad is not None]
        assert enc and sum(float(g.abs().sum()) for g in enc) > 0
        model.zero_grad()


def test_the_constant_schedule_warms_up_and_then_holds():
    from main_hjepa import constant_schedule

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-3)
    sched = constant_schedule(4)(opt)
    rates = []
    for _ in range(10):
        rates.append(opt.param_groups[0]['lr'])
        opt.step()
        sched.step()
    assert rates[0] == pytest.approx(2.5e-4) and rates[3:] == [1e-3] * 7
    state = sched.state_dict()  # resumable
    assert state['last_epoch'] == 10


def test_a_fork_starts_from_the_saved_weights(tmp_path):
    from hjepa_utils import create_world_model
    from main_hjepa import load_initial_weights

    cfg = compose('craftax_sieve')
    torch.manual_seed(0)
    parent = create_world_model(cfg)[0]
    parent.get_level(1).decoder.set_pixel_variance(0.07)
    torch.save(parent, tmp_path / 'p_epoch_3_object.ckpt')
    torch.manual_seed(1)
    child = create_world_model(cfg)[0]
    load_initial_weights(child, tmp_path / 'p_epoch_3_object.ckpt')
    for (k, a), (_, b) in zip(
        parent.state_dict().items(), child.state_dict().items()
    ):
        assert torch.equal(a, b), k


def test_the_starting_model_is_saved_once(tmp_path):
    from utils import ModelObjectCallBack

    cb = ModelObjectCallBack(tmp_path, 'run', save_initial=True)
    module = types.SimpleNamespace(model=torch.nn.Linear(2, 2))
    trainer = types.SimpleNamespace(is_global_zero=True, global_step=0)
    cb.on_train_start(trainer, module)
    assert (tmp_path / 'run_epoch_0_object.ckpt').exists()
    (tmp_path / 'run_epoch_0_object.ckpt').unlink()
    trainer.global_step = 500  # a resumed run keeps its start
    cb.on_train_start(trainer, module)
    assert not (tmp_path / 'run_epoch_0_object.ckpt').exists()


# --------------------------------------------------------------------------
# the counterfactual frames


def test_the_bank_renders_the_stored_frames_and_each_edit_changes_them():
    from sieve import counterfactuals as cf
    from stable_worldmodel.envs.craftax.env import unpack_state

    ep = dict(np.load(FIXTURE))
    arrays, meta = cf.build(ep['full_state'], ep['pixels'], states=6, seed=0)
    assert meta['render_match'] == 1.0  # the env's renderer is the data's
    assert meta['day_states'] == 6
    for kind in cf.EDITS:
        valid = arrays[f'valid_{kind}']
        assert valid.any(), kind
        changed = (arrays[kind] != arrays['orig']).any((1, 2, 3))
        assert changed[valid].all(), kind
    state = unpack_state(ep['full_state'][3])
    dark = state.replace(
        light_level=np.asarray(0.1, np.asarray(state.light_level).dtype)
    )
    rng = np.random.default_rng(0)
    again = cf._render(cf.edit(dark, 'noise', rng))
    assert (cf._render(dark) != again).mean() > 0.1  # another static
    day = cf._render(cf.edit(state, 'noise', rng))
    assert np.array_equal(cf._render(state), day)  # no static by day


# --------------------------------------------------------------------------
# the probes and the readiness rules


def planted_part(n=600, seed=0):
    """Frames whose latent will be the facing one-hot plus noise."""
    from probing import targets as tg

    rng = np.random.default_rng(seed)
    state = np.zeros((n, tg.STATE_DIM), np.float32)
    tiles = state[:, : tg.VIEW_TILES * tg.VIEW_CHANNELS].reshape(
        n, tg.VIEW_TILES, tg.VIEW_CHANNELS
    )
    tiles[:, :, 2] = 1  # grass everywhere
    return types.SimpleNamespace(
        state=state,
        facing=rng.integers(1, 5, n),
        inventory=rng.integers(0, 3, (n, 12)).astype(np.float32),
        vitals=rng.integers(0, 9, (n, 4)).astype(np.float32),
        daylight=rng.random(n).astype(np.float32),
    )


def test_the_probe_reads_what_the_latent_holds():
    tr, va = planted_part(seed=0), planted_part(seed=1)

    def latent(part):
        rng = np.random.default_rng(0)
        onehot = np.eye(4)[part.facing - 1]
        return np.concatenate(
            [onehot, rng.normal(size=(len(onehot), 4))], 1
        ).astype(np.float32)

    y_tr, y_va = pr.targets(tr), pr.targets(va)
    probe = pr.Probe('linear', latent(tr), y_tr, steps=300)
    s = pr.scores(probe.predict(latent(va)), y_va)
    assert s['facing'] > 0.95  # held
    assert s['inventory'] < 0.1  # not held
    assert pr.chance(y_va)['facing'] < 0.35
    assert s['map_grass'] == 1.0


def test_foresees_counts_only_what_changed():
    y_t = {'reg': np.zeros((4, 17)), 'facing': np.array([0, 0, 1, 1]),
           'faced': np.zeros(4, int), 'mobs': np.zeros((4, 4)),
           'map': np.zeros((4, 62), int)}
    y_1 = {k: v.copy() for k, v in y_t.items()}
    y_1['facing'] = np.array([0, 2, 1, 3])  # frames 1 and 3 turn
    pred = {k: v.copy() for k, v in y_1.items()}
    copy = {k: v.copy() for k, v in y_t.items()}
    out = pr.foresees(pred, copy, y_t, y_1)
    assert out['changed_facing'] == 2
    assert out['foresees_facing'] == 1.0 and out['copy_facing'] == 0.0
    assert out['changed_map'] == 0


def entry(epoch, cow, recon, rank=50.0, noise=0.5):
    e = {'epoch': epoch, 'recon_error_mean': recon,
         'recon_error_worst1pct': recon * 5, 'effective_rank': rank,
         'recon_gap': 1.0, 'latent_error_gap': 1.0, 'moves_for_noise': noise,
         'stage': 1}
    for name in pr.HEADLINE:
        e[f'holds_{name}'] = 0.5
    e['holds_mob_cow'] = cow
    e['holds_mob_cow_ci'] = [cow - 0.005, cow + 0.005]
    return e


def test_stage_one_is_ready_after_two_flat_epochs():
    entries = [entry(0, 0.5, 0.1), entry(1, 0.6, 0.05),
               entry(2, 0.7, 0.04), entry(3, 0.705, 0.0399),
               entry(4, 0.706, 0.0399)]
    checks = rp.epoch_checks(entries, stage=1)
    assert not checks[2]['passes']  # the cow still rises
    assert 'mob_cow' in checks[2]['reasons'][0]
    assert checks[3]['passes'] and checks[4]['passes']
    assert rp.ready_epoch(entries, stage=1) == 4
    collapsed = entries[:4] + [entry(4, 0.706, 0.0399, rank=5.0)]
    assert rp.ready_epoch(collapsed, stage=1) is None


def test_stage_two_stops_when_the_noise_is_released_and_probes_settle():
    entries = [entry(e, 0.8, 0.05, noise=n)
               for e, n in enumerate([0.9, 0.3, 0.05, 0.05, 0.05])]
    for e in entries:
        e['stage'] = 2
    assert rp.ready_epoch(entries, stage=2) == 4
    assert not rp.epoch_checks(entries, stage=2)[2]['passes']  # 0.3 -> 0.05


def test_one_stage_stops_when_every_probe_settles_and_the_noise_is_low():
    entries = [entry(e, c, 0.05, noise=n) for e, (c, n) in enumerate(
        [(0.5, 0.9), (0.6, 0.3), (0.7, 0.05), (0.705, 0.05), (0.706, 0.05)]
    )]
    for e in entries:
        e['stage'] = 'joint'
    checks = rp.epoch_checks(entries, stage='joint')
    assert not checks[2]['passes']  # the cow (a headline probe) still rises
    assert checks[3]['passes'] and checks[4]['passes']
    assert rp.ready_epoch(entries, stage='joint') == 4
    grown = [dict(e) for e in entries]
    grown[4]['latent_error_gap'] = 1.2  # the held-out gap grew
    assert rp.ready_epoch(grown, stage='joint') is None


# --------------------------------------------------------------------------
# the worker, end to end on a tiny file


def write_files(folder, episodes=8):
    ep = dict(np.load(FIXTURE))
    n1 = len(ep['pixels'])
    n = episodes * n1
    rng = np.random.default_rng(0)
    action = np.full((n1, 1), np.nan, np.float32)
    action[:-1, 0] = ep['actions']
    for name in ('train', 'val'):
        with h5py.File(folder / f'{name}.h5', 'w') as f:
            for key in ('pixels', 'state', 'full_state'):
                f[key] = np.concatenate([ep[key]] * episodes)
            f['action'] = np.concatenate([action] * episodes)
            f['inventory'] = rng.integers(0, 3, (n, 12)).astype(np.int16)
            f['vitals'] = rng.integers(0, 9, (n, 4)).astype(np.int8)
            f['facing'] = rng.integers(1, 5, (n, 1)).astype(np.int8)
            f['daylight'] = rng.random((n, 1)).astype(np.float32)
            f['ep_len'] = np.full(episodes, n1, np.int32)
            f['ep_offset'] = np.arange(episodes) * n1


@pytest.mark.parametrize('stage', [1, 2, 'joint'])
def test_the_worker_reads_every_epoch_and_writes_the_report(tmp_path, stage):
    from omegaconf import OmegaConf

    from hjepa_utils import create_world_model
    from sieve import counterfactuals as cf
    from sieve import watch

    write_files(tmp_path)
    cf.build_from_file(tmp_path / 'val.h5', tmp_path / 'bank.npz', states=4,
                       episodes=4, windows=2, length=8)
    cfg = compose('craftax_sieve', [f'level1.sieve.stage={stage}'])
    run = tmp_path / 'run'
    run.mkdir()
    OmegaConf.save(cfg, run / 'config.yaml')
    model = create_world_model(cfg)[0]
    model.get_level(1).decoder.set_pixel_variance(0.05)
    for epoch in (0, 1):
        torch.save(model, run / f'V4_epoch_{epoch}_object.ckpt')
    done = watch.watch(run, tmp_path / 'train.h5', tmp_path / 'val.h5',
                       bank=tmp_path / 'bank.npz', once=True, size='quick',
                       device='cpu')
    assert done == [0, 1]
    m = json.loads((run / 'sieve' / 'epoch_1.json').read_text())
    for key in ('holds_facing', 'holds_mob_cow', 'holds_map_iron',
                'holds_facing_linear', 'linear_vs_mlp', 'foresees_facing',
                'copy_facing', 'effective_rank', 'latent_error_vs_copy',
                'recon_error_mean', 'recon_error_worst1pct',
                'decoded_pred_error', 'sharp_patch_share', 'action_lift',
                'action_follow', 'action_follow_facing', 'moves_for_facing',
                'moves_for_mob', 'moves_for_tile', 'moves_for_item',
                'recon_vs_pred_gradient'):
        assert key in m, key
    assert ('fresh_decoder_error' in m) == (stage != 1)
    text = (run / 'sieve' / 'report.md').read_text()
    assert f'({rp.stage_label(stage)})' in text and 'holds_mob_cow' in text
    again = watch.watch(run, tmp_path / 'train.h5', tmp_path / 'val.h5',
                        once=True, size='quick', device='cpu')
    assert again == []  # a rerun skips the epochs done
