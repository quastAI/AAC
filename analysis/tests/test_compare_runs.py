"""compare_runs on synthetic result directories."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import compare_runs as cr  # noqa: E402


def planning_run(
    path, success, progress, tflops, groups=None, findings=(), tasks=None
):
    path.mkdir(parents=True)
    summary = {
        'num_tasks': 10,
        'success_rate': success,
        'progress_mean': progress,
        'compute': {'tflops_per_task': tflops},
        'by_group': groups
        or {'required_1': {'n': 10, 'success': success, 'progress': progress}},
        'plans': {
            'err_ratio_vs_random': 1.2,
            'optimism_chosen': 1.5,
            'rank_corr': 0.4,
        },
    }
    (path / 'planning_health.json').write_text(
        json.dumps(
            {
                'summary': summary,
                'findings': list(findings),
                'tasks_sha256': tasks,
            }
        )
    )
    return path


def args(**kw):
    base = dict(
        planning=None, scaling=None, training=None, probing=None, baseline=None
    )
    return Namespace(**{**base, **kw})


def test_planning_table_compares_with_the_baseline(tmp_path):
    rand = planning_run(tmp_path / 'random', 0.3, 0.3, 0.0)
    model = planning_run(
        tmp_path / 'model',
        0.5,
        0.6,
        2.0,
        findings=[{'severity': 'warn'}, {'severity': 'critical'}],
    )
    text = cr.build(
        args(planning=[f'random={rand}', f'model={model}'], baseline='random')
    )
    row = next(
        line for line in text.splitlines() if line.startswith('| model')
    )
    assert '0.3' in row  # progress 0.6 - 0.3 over the baseline
    assert row.rstrip().endswith('| 1 | 1 |')
    assert '(0.3)' in text  # the baseline success is shown next to the group


def test_scaling_flags_progress_that_falls_with_compute(tmp_path):
    runs = [
        f'{label}={planning_run(tmp_path / label, 0.5, progress, tflops)}'
        for label, progress, tflops in (
            ('a', 0.6, 1.0),
            ('b', 0.4, 4.0),
            ('c', 0.2, 16.0),
        )
    ]
    text = cr.build(args(scaling=runs))
    assert 'Progress falls as the planner gets more compute' in text
    ordered = [
        line.split('|')[1].strip()
        for line in text.splitlines()
        if line.startswith('| ') and line.split('|')[1].strip() in 'abc'
    ]
    assert ordered == ['a', 'b', 'c']

    rising = [
        f'{label}={planning_run(tmp_path / f"r{label}", 0.5, progress, tf)}'
        for label, progress, tf in (
            ('a', 0.2, 1.0),
            ('b', 0.4, 4.0),
            ('c', 0.6, 16.0),
        )
    ]
    assert 'Progress falls' not in cr.build(args(scaling=rising))


def test_training_and_probing_tables(tmp_path):
    run = tmp_path / 'run'
    (run / 'health').mkdir(parents=True)
    level = {
        'rep': {'erank': 80.0, 'dim': 192},
        'dyn': {'pred_over_copy': 0.4, 'action_lift': 0.5},
        'idm': {'accuracy': 0.9},
        'drift': {'1': {'useful_horizon': 5}},
    }
    (run / 'health' / 'summary.json').write_text(
        json.dumps(
            {
                'summary': {'levels': {'1': level}, 'plan_history': 1},
                'findings': [{'severity': 'warn'}],
            }
        )
    )
    probe = {
        'levels': {
            '1': {
                'groups': {'fast': 0.8, 'medium': 0.5, 'slow': 0.4},
                'control_groups': {'fast': 0.1, 'medium': 0.1, 'slow': 0.1},
            }
        }
    }
    (run / 'health' / 'probe_summary.json').write_text(
        json.dumps({'summary': probe, 'findings': []})
    )
    text = cr.build(args(training=[f'm={run}'], probing=[f'm={run}']))
    assert '| m | 1 | 80 / 192 | 0.4 | 0.5 | 5 |' in text
    assert '0.8 (0.1)' in text


def test_missing_results_are_skipped_and_bad_labels_rejected(tmp_path):
    assert (
        cr.build(args(planning=[f'x={tmp_path / "nothing"}'])).strip()
        == '# Run comparison'
    )
    with pytest.raises(SystemExit):
        cr.build(args(planning=['no-label']))


def test_planning_table_refuses_to_compare_other_tasks(tmp_path):
    rand = planning_run(tmp_path / 'random', 0.3, 0.3, 0.0, tasks='a' * 64)
    same = planning_run(tmp_path / 'same', 0.5, 0.6, 2.0, tasks='a' * 64)
    other = planning_run(tmp_path / 'other', 0.5, 0.9, 2.0, tasks='b' * 64)

    def row(model, label):
        text = cr.build(
            args(planning=[f'random={rand}', f'{label}={model}'], baseline='random')
        )
        line = next(l for l in text.splitlines() if l.startswith(f'| {label}'))
        return text, line

    text, line = row(same, 'same')
    assert 'WARNING' not in text and '| 0.3 |' in line  # gain 0.6 - 0.3
    assert 'aaaaaaaa' in line

    text, line = row(other, 'other')
    assert 'WARNING' in text and 'not evaluated on the same tasks' in text
    assert '| 0.6 |' not in line  # no gain across different task files
    assert 'bbbbbbbb' in line
