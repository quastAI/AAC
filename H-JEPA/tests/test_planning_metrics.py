"""Planning metrics: summary, findings, and the recorder end to end."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


pytest.importorskip('craftax')
pytest.importorskip('transformers')
pytest.importorskip('stable_pretraining')
hydra = pytest.importorskip('hydra')
pytest.importorskip('sklearn')

H_JEPA = Path(__file__).parents[1] / 'h_jepa'
sys.path.insert(0, str(H_JEPA))

import eval_tasks  # noqa: E402
import health  # noqa: E402
import planning_eval  # noqa: E402
import planning_metrics as pm  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from hjepa_utils import create_world_model  # noqa: E402
from stable_worldmodel import PlanConfig  # noqa: E402
from stable_worldmodel.solver import CategoricalCEMSolver  # noqa: E402


def test_spearman():
    assert pm.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert pm.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert pm.spearman([1, 1, 1], [1, 2, 3]) is None  # constant
    assert pm.spearman([1, 2], [1, 2]) is None  # too short


def plan(**over):
    entry = {
        'horizon': 3,
        'pred_err': 0.1,
        'pred_err_curve': [0.05, 0.1, 0.15],
        'pred_err_random': 0.1,
        'goal0': 1.0,
        'pred_goal': 0.5,
        'real_goal': 0.5,
        'pred_goal_random': 0.9,
        'real_goal_random': 0.9,
        'rank_corr': 0.8,
    }
    return {**entry, **over}


def record(progress, dist, plan_entry=None, actions=(1, 2, 3, 4), **over):
    solves = [
        {
            'step': 0,
            'flops': 2e9,
            'levels': {
                '1': {
                    'horizon': 3,
                    'wall_clock': 0.1,
                    'best_cost': 0.5,
                    'first_cost': 1.0,
                    'entropy': 0.3,
                }
            },
        }
    ]
    if plan_entry:
        solves[0]['plan'] = plan_entry
    done = progress[-1] >= 1
    rec = {
        'step': list(range(len(progress))),
        'progress': list(progress),
        'required': 1,
        'achievements': [0] * len(progress),
        'terminated': [False] * (len(progress) - 1) + [done],
        'dist': {'1': list(dist)},
        'actions': list(actions),
        'solves': solves,
        'subgoals': [],
    }
    return {**rec, **over}


def codes(findings):
    return {f.code for f in findings}


def test_summary_of_a_healthy_run_raises_no_critical_finding():
    records = [
        record([0, 0, 1], [1.0, 0.5, 0.1], plan()),
        record([0, 1], [1.0, 0.2], plan(), actions=(1, 2, 3, 5, 6)),
    ]
    summary = pm.summarize_planning(records)
    assert summary['success_rate'] == 1.0 and summary['progress_mean'] == 1.0
    assert summary['dist'][1]['decreased_frac'] == 1.0
    assert summary['planner'][1]['cem_improvement'] == pytest.approx(0.5)
    assert summary['compute']['tflops_per_task'] == pytest.approx(0.002)
    assert summary['plans']['err_ratio_vs_random'] == pytest.approx(1.0)
    assert summary['plans']['rank_corr'] == pytest.approx(0.8)
    assert not health.has_critical(health.diagnose_planning(summary))


def test_findings_name_the_cause_of_a_failed_planner():
    bad = plan(
        pred_err=0.5,
        pred_goal=0.1,
        real_goal=1.2,
        rank_corr=0.0,
        offsupport_plan=0.8,
        offsupport_random=0.3,
        offsupport_real=0.05,
    )
    records = [
        record([0, 0, 0], [1.0, 1.1, 1.3], bad, actions=(0, 0, 0, 0, 0)),
        record([0, 0, 0], [1.0, 1.2, 1.2], bad, actions=(3,) * 19 + (1,)),
    ]
    for r in records:
        r['solves'][0]['levels']['1'].update(
            first_cost=1.0, best_cost=0.999, entropy=0.99
        )
    found = health.diagnose_planning(pm.summarize_planning(records))
    expected = {
        'PLAN_NO_PROGRESS',
        'DIST_NOT_DECREASING',
        'CEM_NOT_OPTIMIZING',
        'CEM_NOT_CONVERGED',
        'PLAN_EXPLOITATION',
        'PLAN_OPTIMISM',
        'COST_RANKING',
        'OFF_SUPPORT',
        'PLAN_DEGENERATE',
        'PREDICTED_GAIN_NOT_REAL',
    }
    assert expected <= codes(found)
    assert (
        found[0].code == 'PLAN_NO_PROGRESS' and found[0].severity == 'critical'
    )
    assert all(f.hint for f in found)


def test_partial_progress_is_reported_apart_from_failure():
    summary = pm.summarize_planning([record([0, 0.5], [1.0, 0.6], plan())])
    found = health.diagnose_planning(summary)
    assert 'PLAN_PARTIAL_ONLY' in codes(
        found
    ) and 'PLAN_NO_PROGRESS' not in codes(found)


def test_diamond_rate_and_unlocks_per_required_achievement():
    def flags(*unlocked):
        out = [0] * 22
        for a in unlocked:
            out[a] = 1
        return out

    diamond = eval_tasks.DIAMOND
    records = [
        record([0, 0.5], [1, 0.5], plan(), flags=flags(0, diamond)),
        record([0, 0.5], [1, 0.5], plan(), flags=flags(0)),
        record([0, 1.0], [1, 0.1], plan(), flags=flags(1)),
    ]
    info = [
        {'required': [0, diamond]},
        {'required': [0, diamond]},
        {'required': [1]},
    ]
    summary = pm.summarize_planning(records, info)
    assert summary['diamond'] == {'n': 2, 'rate': 0.5}
    by = summary['by_achievement']
    assert by['collect_wood'] == {'n': 2, 'unlocked': 1.0}
    assert by['collect_diamond'] == {'n': 2, 'unlocked': 0.5}
    assert by['place_table'] == {'n': 1, 'unlocked': 1.0}
    report = health.render_planning_report(summary, [])
    assert 'Diamond: unlocked in 0.5 of the 2 tasks' in report
    assert 'collect_diamond' in report
    assert 'diamond' not in pm.summarize_planning(records[2:], info[2:])


def test_groups_by_required_count_and_task_info():
    records = [
        record([0, 1], [1, 0.1], plan(), required=1),
        record([0, 0], [1, 1], plan(), required=3),
    ]
    info = [{'offset': 25}, {'offset': 50}]
    groups = pm.summarize_planning(records, info)['by_group']
    assert groups['required_1']['success'] == 1.0
    assert groups['required_3']['success'] == 0.0
    assert (
        groups['offset_25']['n'] == 1 and groups['offset_50']['progress'] == 0
    )
    found = health.diagnose_planning(pm.summarize_planning(records))
    assert 'FAR_GOALS_FAIL' in codes(found)


def test_report_renders():
    summary = pm.summarize_planning([record([0, 1], [1.0, 0.1], plan())])
    report = health.render_planning_report(
        summary, health.diagnose_planning(summary)
    )
    assert '# Planning report' in report and 'Plan reliability' in report


def test_recording_solver_delegates_and_counts_flops():
    class Inner:
        solve_records = [1]
        level_solvers = {1: object()}

        def configure(self, **kw):
            self.cfg = kw

        def solve(self, info, **kw):
            return {'actions': torch.ones(2, 3, 1) @ torch.ones(1, 4)}

    inner = Inner()
    proxy = pm.RecordingSolver(inner)
    out = proxy({'x': 1}, steps_taken=0)
    assert proxy.last['outputs'] is out and proxy.last['flops'] > 0
    assert proxy.level_solvers is inner.level_solvers
    proxy.solve_records = []
    assert inner.solve_records == []


# --- the recorder against the real env and a tiny untrained model ----


def compose_train(name):
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    with hydra.initialize_config_dir(
        config_dir=str(H_JEPA / 'config' / 'train'), version_base=None
    ):
        return hydra.compose(config_name=name)


CALLABLES = [
    {'method': '_set_state', 'args': {'full_state': {'value': 'full_state'}}},
    {
        'method': '_set_goal_state',
        'args': {'full_state': {'value': 'goal_full_state'}},
    },
]


def run_eval(policy, dataset, budget=8):
    # the fixture episode is a random one (source 2): a plumbing test, so no source filter
    tasks = eval_tasks.make_tasks(
        dataset,
        mode='achievements',
        num_tasks=1,
        depths=(1,),
        seed=0,
        sources=None,
    )
    payload = eval_tasks.build_payload(dataset, tasks, eval_budget=budget)
    world = swm.World(
        'swm/CraftaxClassic-v1',
        num_envs=1,
        image_shape=(63, 63),
        max_episode_steps=2 * budget,
        verbose=0,
    )
    world.set_policy(policy)
    try:
        world.evaluate_from_dataset(
            None,
            None,
            None,
            goal_offset_steps=payload['goal_offset_steps'],
            eval_budget=budget,
            callables=CALLABLES,
            load_eval_trajs_path=payload,
        )
    finally:
        world.close()
    return tasks


def image_transforms():
    cfg = OmegaConf.create({'eval': {'img_size': 63}})
    return {
        'pixels': planning_eval.img_transform(cfg),
        'goal': planning_eval.img_transform(cfg),
    }


def check_records(policy, tasks, levels):
    (record,) = policy.records
    assert len(record['progress']) >= 2 and record['required'] == 1
    # the state after the last action is recorded too (the budget is 8)
    assert record['terminated'][-1] or record['step'][-1] == 8
    assert len(record['flags']) == 22
    assert set(record['dist']) == {str(level) for level in levels}
    solve = record['solves'][0]
    assert solve['flops'] > 0 and set(solve['levels']) == {
        str(level) for level in levels
    }
    assert (
        solve['levels']['1']['first_cost'] >= solve['levels']['1']['best_cost']
    )
    plan_entry = solve['plan']
    assert len(plan_entry['pred_err_curve']) == plan_entry['horizon']
    assert plan_entry['pred_err_random'] > 0 and 'rank_corr' in plan_entry
    summary = pm.summarize_planning(policy.records, tasks)
    assert (
        summary['plans']['n'] >= 1
        and summary['compute']['tflops_per_task'] > 0
    )
    report = health.render_planning_report(
        summary, health.diagnose_planning(summary)
    )
    assert '# Planning report' in report
    return summary


def test_recorder_end_to_end_flat(dataset):
    jepa = create_world_model(compose_train('craftax_lewm'))[0].get_level(1)
    jepa.eval()
    solver = pm.RecordingSolver(
        CategoricalCEMSolver(jepa, num_samples=8, n_iters=2, topk=2, seed=0)
    )
    bank = (torch.randn(64, 192), 1.0)
    policy = pm.RecordingPolicy(
        solver,
        PlanConfig(horizon=3, receding_horizon=2),
        metrics_cfg={'diag_every': 1, 'rank_candidates': 4},
        bank=bank,
        process={},
        transform=image_transforms(),
        eval_budget=8,
    )
    tasks = run_eval(policy, dataset)
    summary = check_records(policy, tasks, levels=[1])
    assert 'offsupport_plan' in summary['plans']  # the support metric ran
    assert summary['actions']['distinct_mean'] >= 1


def test_recorder_end_to_end_hierarchical_through_build_policy(dataset):
    model = create_world_model(compose_train('craftax_hjepa_l2'))[0]
    model.eval()
    cfg = planning_eval.load_eval_config(config_name='craftax_l2')
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for level in (1, 2):
        cfg.solver.solvers[f'level{level}'].update(
            device='cpu', num_samples=6, n_iters=2, topk=2, batch_size=2
        )
    cfg.planning_metrics.support_bank_dataset = None
    cfg.planning_metrics.rank_candidates = 4
    cfg.eval.eval_budget = 8
    policy = planning_eval._build_policy(
        cfg, {}, image_transforms(), model=model
    )
    assert isinstance(policy, pm.RecordingPolicy)
    tasks = run_eval(policy, dataset)
    summary = check_records(policy, tasks, levels=[1, 2])
    assert (
        summary['subgoals']['n'] >= 1
    )  # the upper-level subgoal was followed up
    assert set(summary['planner']) == {1, 2}


def test_write_planning_health_creates_the_report(dataset, tmp_path):
    summary = planning_eval.write_planning_health(
        tmp_path,
        [record([0, 1], [1.0, 0.1], plan())],
        None,
        metrics := {},
    )
    assert (
        (tmp_path / 'planning_report.md').read_text().startswith('# Planning')
    )
    assert (tmp_path / 'planning_health.json').exists()
    assert metrics['progress_mean'] == summary['progress_mean'] == 1.0
    assert np.isfinite(metrics['tflops_per_task'])


def test_random_baseline_records_like_the_planner(dataset):
    policy = pm.RandomRecordingPolicy(seed=0)
    tasks = run_eval(policy, dataset)
    (record,) = policy.records
    assert len(record['progress']) >= 2 and record['required'] == 1
    assert len(record['actions']) >= 2 and not record['solves']
    summary = pm.summarize_planning(policy.records, tasks)
    assert summary['num_tasks'] == 1 and 'plans' not in summary
    assert summary['compute']['tflops_per_task'] == 0
    report = health.render_planning_report(
        summary, health.diagnose_planning(summary)
    )
    assert '# Planning report' in report
