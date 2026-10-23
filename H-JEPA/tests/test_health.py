"""Health metrics and findings, on synthetic data with known answers."""

import math
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

import health  # noqa: E402
from models.vq import VQActionEncoder  # noqa: E402


torch.manual_seed(0)


def test_effective_rank_separates_isotropic_from_collapsed():
    isotropic = torch.randn(4, 64, 32)
    rank_one = torch.randn(4, 64, 1) * torch.randn(1, 1, 32)
    assert health.effective_rank(isotropic)['erank_frac'] > 0.8
    assert health.effective_rank(rank_one)['erank_frac'] < 0.1
    assert math.isnan(health.effective_rank(torch.randn(1, 1, 8))['erank'])


def test_dim_stats_finds_dead_dimensions():
    e = torch.randn(8, 16, 10)
    e[..., :4] = 0.5  # four constant dimensions
    stats = health.dim_stats(e)
    assert stats['dead_dim_frac'] == pytest.approx(0.4)
    assert stats['std_min'] < 1e-6
    e[0, 0, 0] = float('nan')
    assert health.dim_stats(e)['nonfinite_frac'] > 0


def test_within_clip_share_and_step_scale():
    constant_in_time = torch.randn(6, 1, 8).expand(6, 5, 8)
    iid = torch.randn(6, 5, 8)
    assert health.within_clip_share(constant_in_time) == pytest.approx(
        0, abs=1e-6
    )
    assert health.within_clip_share(iid) > 0.7
    assert health.step_to_scale(constant_in_time) == pytest.approx(0, abs=1e-6)
    assert health.step_to_scale(iid) == pytest.approx(1.0, abs=0.25)
    smooth = torch.randn(6, 1, 8) + 0.01 * torch.randn(6, 5, 8).cumsum(1)
    assert health.step_to_scale(smooth) < 0.05
    assert health.within_clip_share(torch.randn(6, 1, 8)) is None


class ToyJepa:
    """next = current + action embedding, or ignoring the action."""

    def __init__(self, use_action=True):
        self.predictor = torch.nn.Identity()
        self.use_action = use_action

    def predict(self, emb, act):
        return emb + act if self.use_action else emb


def toy_sequence(batch=8, steps=12, dim=6):
    act = torch.randn(batch, steps, dim)
    emb = torch.cat([torch.zeros(batch, 1, dim), act.cumsum(1)], dim=1)[
        :, :steps
    ]
    return emb, act  # emb[t + 1] = emb[t] + act[t]


def test_teacher_forced_metrics_see_action_use():
    emb, act = toy_sequence()
    good = health.teacher_forced_metrics(ToyJepa(True), emb, act, history=3)
    assert good['pred_mse'] == pytest.approx(0, abs=1e-8)
    assert good['pred_over_copy'] < 1e-6
    assert good['action_lift'] > 0.9  # shuffled actions are clearly worse

    blind = health.teacher_forced_metrics(ToyJepa(False), emb, act, history=3)
    assert blind['pred_over_copy'] == pytest.approx(1.0)  # it only copies
    assert abs(blind['action_lift']) < 1e-6
    assert (
        health.teacher_forced_metrics(ToyJepa(), emb[:, :2], act[:, :2], 3)
        == {}
    )


def test_drift_curve_perfect_and_blind_rollouts():
    emb, act = toy_sequence(steps=14)
    perfect = health.drift_curve(ToyJepa(True), emb, act, history=1, horizon=8)
    assert max(perfect['mse']) < 1e-8 and perfect['useful_horizon'] == 8
    blind = health.drift_curve(ToyJepa(False), emb, act, history=1, horizon=8)
    assert blind['useful_horizon'] == 0  # never better than copying
    assert (
        len(health.drift_curve(ToyJepa(), emb, act, 3, horizon=99)['mse'])
        == 11
    )


def test_cost_geometry_grows_for_a_walk_and_saturates_for_noise():
    walk = torch.randn(32, 40, 8).mul(0.2).cumsum(1)
    geo = health.cost_geometry(walk, [1, 2, 4, 8, 16, 32])
    assert geo['monotone_frac'] == 1.0 and geo['ratio'][0] < geo['ratio'][-1]
    noise = torch.randn(32, 40, 8)
    flat = health.cost_geometry(noise, [1, 2, 4, 8])
    assert flat['saturation_gap'] == 1  # unrelated from the first step
    assert health.cost_geometry(walk[:1], [1, 2]) == {}


def test_vq_metrics_report_collapse():
    class Encoder(torch.nn.Module):
        output_dim = 4

        def forward(self, x, mask=None):
            return x.mean(2)

    vq = VQActionEncoder(Encoder(), num_codes=8, groups=2)
    x = torch.randn(4, 16, 3, 4)
    vq(x)
    spread = health.vq_metrics(vq)
    assert spread['num_codes'] == 8 and spread['groups'] == 2
    vq.eval()
    vq(torch.zeros(4, 16, 3, 4))  # one input: one code per group
    collapsed = health.vq_metrics(vq)
    assert collapsed['perplexity'] == pytest.approx(1.0, abs=1e-3)
    assert collapsed['perplexity_norm'] < spread['perplexity_norm']


def healthy_level():
    return {
        'rep': {
            'erank': 100,
            'erank_frac': 0.5,
            'dim': 192,
            'std_mean': 1.0,
            'dead_dim_frac': 0.0,
            'nonfinite_frac': 0.0,
            'step_to_scale': 0.1,
        },
        'dyn': {'pred_over_copy': 0.4, 'action_lift': 0.5},
        'geometry': {
            'saturation_gap': 16,
            'monotone_frac': 1.0,
            'gaps': [1],
            'ratio': [0.1],
        },
    }


def codes(findings):
    return {f.code for f in findings}


def test_diagnose_training_is_quiet_for_a_healthy_model():
    summary = {
        'levels': {1: healthy_level()},
        'training': {
            'val_first': 1.0,
            'val_last': 0.5,
            'train_last': 0.45,
            'epochs': 5,
        },
    }
    assert health.diagnose_training(summary) == []


def test_diagnose_training_flags_each_failure():
    level = healthy_level()
    level['rep'].update(
        erank_frac=0.001, erank=0.2, dead_dim_frac=0.6, step_to_scale=0.95
    )
    level['dyn'] = {'pred_over_copy': 1.3, 'action_lift': 0.0}
    level['geometry'] = {
        'saturation_gap': 1,
        'monotone_frac': 0.2,
        'gaps': [1],
        'ratio': [1.0],
    }
    level['vq'] = {
        'perplexity': 1.0,
        'perplexity_norm': 0.01,
        'num_codes': 64,
        'reseed_rate': 0.4,
    }
    level['idm'] = {'accuracy': 0.06, 'chance': 0.0588, 'lift': 0.002}
    level['grad'] = {'explosions': 3, 'median': 1e-9}
    level['drift'] = {
        1: {'mse': [2.0, 3.0], 'copy': [1.0, 1.0], 'useful_horizon': 0},
        3: {'mse': [0.1, 0.1], 'copy': [1.0, 1.0], 'useful_horizon': 2},
    }
    summary = {
        'levels': {1: level},
        'training': {
            'val_first': 1.0,
            'val_last': 0.99,
            'train_last': 0.5,
            'epochs': 4,
            'spikes': 9,
            'nonfinite': True,
        },
    }
    found = health.diagnose_training(summary)
    expected = {
        'REP_COLLAPSE',
        'REP_DEAD_DIMS',
        'REP_INCOHERENT',
        'DYN_WORSE_THAN_COPY',
        'DYN_IGNORES_ACTIONS',
        'GEOM_SATURATES',
        'GEOM_NONMONOTONE',
        'VQ_COLLAPSE',
        'VQ_CODE_CHURN',
        'IDM_CHANCE',
        'GRAD_EXPLOSION',
        'GRAD_VANISHING',
        'DRIFT_WORSE_THAN_COPY',
        'PLAN_HISTORY_MISMATCH',
        'LOSS_NONFINITE',
        'LOSS_STALLED',
        'OVERFIT',
        'LOSS_SPIKES',
    }
    assert expected <= codes(found)
    assert health.has_critical(found)
    assert found[0].severity == 'critical'  # most severe first
    assert all(f.hint for f in found)


def test_diagnose_flags_the_latent_scale_only_where_sigreg_acts():
    shrunk = healthy_level()
    shrunk['rep']['std_mean'] = 0.05  # not constant, full rank: only scale
    summary = {'levels': {1: shrunk, 2: healthy_level()}}
    assert 'REP_SCALE' not in codes(health.diagnose_training(summary))
    summary['sigreg_levels'] = [1, 2]
    found = health.diagnose_training(summary)
    assert [(f.code, f.level) for f in found if f.code == 'REP_SCALE'] == [
        ('REP_SCALE', 1)
    ]
    shrunk['rep']['std_mean'] = 5.0
    assert 'REP_SCALE' in codes(health.diagnose_training(summary))


def test_diagnose_flags_a_level_without_abstraction():
    upper = healthy_level()
    upper['rep']['step_to_scale'] = 0.05
    summary = {'levels': {1: healthy_level(), 2: upper}}
    assert 'NO_ABSTRACTION' in codes(health.diagnose_training(summary))


def test_report_renders_and_serialises():
    summary = {'levels': {1: healthy_level()}, 'training': {'val_last': 0.5}}
    report = health.render_training_report(summary, [])
    assert '# Training report' in report and 'No findings' in report
    assert health.to_jsonable({'a': [float('nan'), torch.tensor(1.5)]}) == {
        'a': [None, 1.5]
    }


def test_majority_baseline_is_the_best_constant_guess():
    # expert actions are skewed: always guessing the commonest one is the bar
    target = torch.tensor([[0, 0, 0, 0, 1, 2, -1, -1]]).reshape(1, 8, 1)
    assert health.majority_baseline(target, 17) == pytest.approx(4 / 6)
    two_slots = torch.tensor([[[0, 5], [0, 6], [1, 7], [-1, -1]]])
    assert health.majority_baseline(two_slots, 17) == pytest.approx(
        (2 / 3 + 1 / 3) / 2
    )
    assert health.majority_baseline(torch.full((1, 4, 1), -1), 8) == 1 / 8
    # never below uniform chance
    uniform = torch.arange(8).reshape(1, 8, 1)
    assert health.majority_baseline(uniform, 8) == pytest.approx(1 / 8)


def test_idm_lift_is_undefined_when_one_class_fills_the_data():
    assert health.idm_lift(0.5, 1.0) is None
    assert health.idm_lift(None, 0.2) is None
    assert health.idm_lift(0.6, 0.2) == pytest.approx(0.5)
