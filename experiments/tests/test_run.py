"""The experiment runner: commands, dependencies, seeds, scripts."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

import run  # noqa: E402


@pytest.fixture(scope='module')
def spec():
    return run.load_spec()


def jobs_of(spec, exp, seeds=(42,), overrides=(), home='/data'):
    return run.build_jobs(spec, exp, list(seeds), overrides, home)


def by_name(jobs):
    return {j.name: j for j in jobs}


def test_every_experiment_builds_or_is_blocked(spec):
    for exp_id, exp in spec['experiments'].items():
        if exp.get('blocked'):
            with pytest.raises(SystemExit, match='blocked'):
                jobs_of(spec, exp_id)
        else:
            jobs = jobs_of(spec, exp_id)
            assert jobs, exp_id
            names = [j.name for j in jobs]
            assert len(names) == len(set(names)), exp_id  # unique job names
            for j in jobs:
                assert set(j.after) <= set(names[: names.index(j.name)])


def test_e1_commands_and_dependencies(spec):
    jobs = by_name(jobs_of(spec, 'E1'))
    train = jobs['train_e1_hjepa_l2_s42']
    assert train.command[:4] == ['python', 'main_hjepa.py', '--config-name', 'craftax_hjepa_l2']
    assert 'output_model_name=e1_hjepa_l2' in train.command and 'seed=42' in train.command
    assert 'data.dataset.name=craftax_expert_train' in train.command
    ev = jobs['eval_e1_hjepa_l2_s42_A']
    assert ev.after == ['tasks_A', 'train_e1_hjepa_l2_s42']
    assert 'policy=ckpts/craftax/e1_hjepa_l2/seed42' in ev.command
    assert 'output.dir=seed42/eval_A_val_craftax_l2' in ev.command
    assert 'load_eval_trajs_path=/data/eval_tasks/A_val.pt' in ev.command
    # tasks A (offsets with progress) and B (full episodes) and a random baseline for each
    assert '--offsets' in jobs['tasks_A'].command and '--mode' in jobs['tasks_B'].command
    base = jobs['eval_random_B']
    assert 'policy=random' in base.command and 'output.dir=/data/ckpts/craftax/random/eval_B_val' in base.command
    # the comparison of a protocol waits for its evaluations and names the baseline
    compare = jobs['compare_E1_A']
    assert set(compare.after) == {n for n in jobs if n.startswith('eval_') and n.endswith('_A')}
    assert '--baseline' in compare.command and any(c.startswith('random=') for c in compare.command)
    assert 'compare_E1_models' in jobs and '--probing' in jobs['compare_E1_models'].command


def test_seeds_multiply_the_runs_and_overrides_reach_training_only(spec):
    one = jobs_of(spec, 'E1', seeds=[42])
    three = jobs_of(spec, 'E1', seeds=[42, 43, 44], overrides=['level2.stride=6'])
    count = lambda jobs, kind: sum(j.kind == kind for j in jobs)  # noqa: E731
    assert count(three, 'train') == 3 * count(one, 'train')
    trains = [j for j in three if j.kind == 'train' and 'e1_hjepa_l2' in j.name]
    assert all('level2.stride=6' in j.command for j in trains)
    assert not any('level2.stride=6' in j.command for j in three if j.kind != 'train')


def test_reused_models_are_not_retrained(spec):
    jobs = by_name(jobs_of(spec, 'E6'))
    assert not any(j.kind == 'train' for j in jobs.values())
    ev = jobs['eval_e6_hjepa_l2_project_s42_A']
    assert ev.after == ['tasks_A']
    assert 'policy=ckpts/craftax/e1_hjepa_l2/seed42' in ev.command
    assert '--config-name' in ev.command and 'craftax_l2_project' in ev.command


def test_ablations_set_their_overrides(spec):
    e5 = by_name(jobs_of(spec, 'E5'))
    assert 'level1.idm_coeff=0' in e5['train_e5_lewm_noidm_s42'].command
    e3 = by_name(jobs_of(spec, 'E3'))
    assert 'level2.wm.rollout_n=4' in e3['train_e3_hjepa_l2_rollout4_s42'].command


def test_probe_experiment_has_no_training(spec):
    jobs = jobs_of(spec, 'E4')
    probes = [j for j in jobs if j.kind == 'probe']
    assert len(probes) == 4 and not any(j.kind == 'train' for j in jobs)
    assert '--run' in probes[0].command and '/data/ckpts/craftax/e1_lewm/seed42' in probes[0].command


def test_planner_scale_changes_samples_and_iterations_together(spec):
    assert run.scaled_overrides('craftax_l2', 1) == run.scaled_overrides('craftax_l2', 1.0)
    base = {o.split('=')[0]: int(o.split('=')[1]) for o in run.scaled_overrides('craftax_l2', 1)}
    big = {o.split('=')[0]: int(o.split('=')[1]) for o in run.scaled_overrides('craftax_l2', 4)}
    assert big['solver.solvers.level1.num_samples'] == 2 * base['solver.solvers.level1.num_samples']
    assert big['solver.solvers.level2.n_iters'] == 2 * base['solver.solvers.level2.n_iters']
    flat = [o.split('=')[0] for o in run.scaled_overrides('craftax_flat', 0.25)]
    assert flat == ['solver.num_samples', 'solver.n_iters', 'solver.topk']
    jobs = by_name(jobs_of(spec, 'C1'))
    ev = jobs['eval_c1_x4_s42_A']
    assert 'output.dir=seed42/eval_A_val_craftax_l2_x4' in ev.command
    assert 'solver.solvers.level1.num_samples=%d' % big['solver.solvers.level1.num_samples'] in ev.command
    assert '--scaling' in jobs['compare_C1_A'].command


def test_local_run_skips_finished_jobs_and_stops_on_failure(spec, tmp_path):
    jobs = run.build_jobs(spec, 'D0', [42], home='$HJEPA_HOME')
    calls = []

    class Result:
        returncode = 0

    ran = run.run_local(jobs, str(tmp_path), runner=lambda cmd, **kw: calls.append((cmd, kw)) or Result())
    assert ran == ['analysis'] and calls[0][1]['env']['HJEPA_HOME'] == str(tmp_path)
    assert str(tmp_path / 'craftax_expert_train.h5') in calls[0][0]  # $HJEPA_HOME expanded

    out = Path(run.resolve(jobs[0], str(tmp_path)).output)
    out.parent.mkdir(parents=True)
    out.write_text('done')
    assert run.run_local(jobs, str(tmp_path), runner=lambda *a, **k: 1 / 0) == []  # skipped
    Result.returncode = 3
    with pytest.raises(SystemExit, match='failed'):
        run.run_local(jobs, str(tmp_path), force=True, runner=lambda *a, **k: Result())


def test_sbatch_scripts_chain_dependencies(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'HERE', tmp_path)
    slurm = yaml.safe_load((Path(run.__file__).parent / 'slurm.yaml').read_text())
    slurm.update(partition='gpu', account='lab', venv='/venvs/h')
    jobs = jobs_of(spec, 'E6')
    counter = iter(range(100, 200))

    class Result:
        def __init__(self, i):
            self.returncode, self.stdout, self.stderr = 0, f'{i};cluster', ''

    submitted = []

    def fake_sbatch(cmd, **kw):
        submitted.append(Path(cmd[-1]).read_text())
        return Result(next(counter))

    ids = run.submit_slurm(jobs, slurm, str(tmp_path / 'home'), 'E6', submit=True, runner=fake_sbatch)
    assert ids['tasks_A'] == '100'
    evaluation = next(t for t in submitted if '--job-name=eval_e6_hjepa_l2_project_s42_A' in t)
    assert '#SBATCH --dependency=afterok:100' in evaluation
    assert '#SBATCH --gres=gpu:1' in evaluation and '#SBATCH --partition=gpu' in evaluation
    assert 'source /venvs/h/bin/activate' in evaluation and 'eval.py' in evaluation
    compare = next(t for t in submitted if '--job-name=compare_E6_A' in t)
    assert 'gres' not in compare  # no GPU for the comparison
    assert 'afterok:' in compare and compare.count(':') >= 3  # waits for both evaluations

    dry = run.submit_slurm(jobs, slurm, str(tmp_path / 'home'), 'E6', submit=False)
    assert all(v.startswith('<') for v in dry.values())  # written, not submitted
    assert (tmp_path / 'generated' / 'E6').is_dir()


def test_cli_list_and_plan(spec, capsys):
    run.main(['list'])
    out = capsys.readouterr().out
    assert 'E1:' in out and '[blocked' in out
    run.main(['plan', 'E2', '--final'])
    plan = capsys.readouterr().out
    assert 'seed=43' in plan and 'seed=44' in plan
    assert 'e2_hwm_l3' not in plan  # optional runs only with --include-optional
    run.main(['plan', 'E2', '--include-optional'])
    assert 'e2_hwm_l3' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run.main(['plan', 'E7'])


def test_final_runs_take_their_tasks_from_the_test_split(spec):
    jobs = by_name(run.build_jobs(spec, 'E1', [42], home='/data', final=True))
    assert 'craftax_expert_test' in jobs['tasks_A'].command
    assert jobs['tasks_A'].output == '/data/eval_tasks/A_test.pt'
    ev = jobs['eval_e1_hjepa_l2_s42_A']
    assert 'eval.dataset_name=craftax_expert_test' in ev.command
    # training still validates on the validation worlds
    assert 'data.dataset.val_name=craftax_expert_val' in jobs['train_e1_hjepa_l2_s42'].command


def test_final_and_exploration_results_never_share_a_directory(spec):
    # a headline run must not skip jobs, or reuse chunks, of the exploration run
    explore = run.build_jobs(spec, 'E1', [42], home='/data')
    final = run.build_jobs(spec, 'E1', [42], home='/data', final=True)
    outputs = lambda jobs: {  # noqa: E731
        j.name: j.output for j in jobs if j.kind in ('eval', 'compare') and j.output
    }
    out_e, out_f = outputs(explore), outputs(final)
    assert out_e.keys() == out_f.keys()
    for name in out_e:
        if name.endswith('_models'):
            continue  # probes and training reports do not depend on the task split
        assert out_e[name] != out_f[name], name
    ev = {j.name: j for j in final}['eval_e1_hjepa_l2_s42_A']
    assert 'output.dir=seed42/eval_A_test_craftax_l2' in ev.command


def test_collection_mixes_experts_with_a_quarter_of_other_episodes(spec):
    jobs = run.build_jobs(spec, 'D1', [42], home='/data', checkpoint='/ckpt/expert.safetensors')
    c = spec['collection']
    for split, dataset in (('train', 'craftax_expert_train'), ('val', 'craftax_expert_val'), ('test', 'craftax_expert_test')):
        expert, noised, rand = (by_name(jobs)[f'collect_{split}_{k}'] for k in ('expert', 'noised', 'random'))
        total = c['episodes'][split]
        count = lambda job: int(job.command[job.command.index('--num-episodes') + 1])  # noqa: E731
        assert count(expert) + count(noised) + count(rand) == total
        assert count(noised) + count(rand) == round(total * c['nonexpert_share'])
        assert all('/data/' + dataset + '.h5' in j.command for j in (expert, noised, rand))
        # one file, so the runs are chained; seeds of the three runs do not overlap
        assert noised.after == [expert.name] and rand.after == [noised.name]
        starts = [int(j.command[j.command.index('--seed-start') + 1]) for j in (expert, noised, rand)]
        assert starts == [c['seed_start'][split], c['seed_start'][split] + 500_000, c['seed_start'][split] + 750_000]
        assert '--no-require-diamond' not in expert.command and '--no-require-diamond' in noised.command
        assert '--checkpoint' not in rand.command and rand.command[rand.command.index('--eps') + 1] == '1.0'
    ranges = [c['seed_start'][s] for s in ('train', 'val', 'test')]
    assert ranges == sorted(ranges) and ranges[1] - ranges[0] >= 1_000_000  # disjoint splits


def test_collection_run_needs_the_checkpoint(spec):
    with pytest.raises(SystemExit, match='expert-checkpoint'):
        run.main(['run', 'D1', '--mode', 'slurm'])


def test_dataset_analysis_checks_the_seed_overlap_of_all_three_files(spec):
    jobs = run.build_jobs(spec, 'D0', [42], home='/data')
    command = jobs[0].command
    assert command.count('--data') == 3
    labels = [command[i + 1] for i, c in enumerate(command) if c == '--label']
    assert labels == ['train', 'val', 'test']
