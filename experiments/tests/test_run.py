"""The experiment runner: commands, dependencies, seeds, scripts."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

import run  # noqa: E402


@pytest.fixture()
def spec():  # a fresh copy per test: some tests change the collection plan
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


def test_the_baseline_is_trained_probed_and_planned(spec):
    jobs = by_name(jobs_of(spec, 'B1'))
    train = jobs['train_B1_s42']
    assert train.command[:4] == ['python', 'main_hjepa.py', '--config-name', 'craftax_hjepa']
    assert 'num_levels=1' in train.command  # the flat model is level 1 alone
    # B1 was trained before the ids: it keeps its folder and checkpoint name
    assert 'output_model_name=flat' in train.command and 'seed=42' in train.command
    assert train.output == '/data/ckpts/craftax/flat/seed42/flat_object.ckpt'
    assert 'data.dataset.name=craftax_expert_train' in train.command
    probe = jobs['probe_B1_s42']
    assert probe.after == ['train_B1_s42']  # trained in the same experiment: probed after it
    assert '--run' in probe.command and '/data/ckpts/craftax/flat/seed42' in probe.command
    assert '--probing' in jobs['compare_B1_models'].command
    assert set(jobs['compare_B1_models'].after) == {'train_B1_s42', 'probe_B1_s42'}
    # the cheap planning P1 (set A) is part of the baseline, after its training
    ev = jobs['eval_B1_s42_A']
    assert ev.after == ['tasks_A', 'train_B1_s42']
    assert 'policy=ckpts/craftax/flat/seed42' in ev.command
    assert 'output.dir=seed42/eval_A_val_craftax_plan' in ev.command
    assert 'load_eval_trajs_path=/data/eval_tasks/A_val.pt' in ev.command
    assert '--offsets' in jobs['tasks_A'].command
    base = jobs['eval_random_A']
    assert 'policy=random' in base.command and 'output.dir=/data/ckpts/craftax/random/eval_A_val' in base.command
    # the comparison of a protocol waits for its evaluations and names the baseline
    compare = jobs['compare_B1_A']
    assert set(compare.after) == {'eval_B1_s42_A', 'eval_random_A'}
    assert '--baseline' in compare.command and any(c.startswith('random=') for c in compare.command)
    assert compare.output == '/data/reports/B1_A_val.md'


def test_planning_to_the_diamond_reuses_the_baseline(spec):
    full = by_name(jobs_of(spec, 'P3'))
    assert not any(j.kind == 'train' for j in full.values())
    assert full['tasks_B'].command[full['tasks_B'].command.index('--mode') + 1] == 'episode'
    assert 'eval_random_B' in full
    assert 'policy=ckpts/craftax/flat/seed42' in full['eval_B1_s42_B'].command
    chain = by_name(jobs_of(spec, 'P2'))['tasks_C'].command
    assert chain[chain.index('--mode') + 1] == 'chain'
    assert chain[chain.index('--offsets') + 1] == '50' and '--subgoal-budget' in chain


def test_every_trained_run_is_named_by_its_experiment_id(spec):
    for exp_id, exp in spec['experiments'].items():
        trained = [r['name'] for r in exp.get('runs', []) if r.get('train', True)]
        assert trained in ([], [exp_id]), exp_id
        assert exp_id[0] in 'BVAPD', exp_id
    # the folders of the runs trained before the ids, and of a new run
    assert run.folder_of(spec, 'B1') == 'flat'
    assert run.folder_of(spec, 'A1') == 'flat_balanced'
    assert run.folder_of(spec, 'V1') == 'V1'


def test_seeds_multiply_the_runs_and_overrides_reach_training_only(spec):
    one = jobs_of(spec, 'V1', seeds=[42])
    three = jobs_of(spec, 'V1', seeds=[42, 43, 44], overrides=['level1.lr=1e-4'])
    count = lambda jobs, kind: sum(j.kind == kind for j in jobs)  # noqa: E731
    assert count(three, 'train') == 3 * count(one, 'train')
    assert all('level1.lr=1e-4' in j.command for j in three if j.kind == 'train')
    assert not any('level1.lr=1e-4' in j.command for j in three if j.kind != 'train')


def test_variants_set_their_overrides(spec):
    v1 = by_name(jobs_of(spec, 'V1'))
    rollout = v1['train_V1_s42'].command
    assert 'level1.wm.history_size=1' in rollout
    assert 'level1.wm.rollout_n=4' in rollout
    assert 'output_model_name=V1' in rollout  # a new run trains into its id
    v2 = by_name(jobs_of(spec, 'V2'))
    sigreg = v2['train_V2_s42'].command
    assert 'level1.loss.embed.sigreg.weight=0.03' in sigreg
    assert 'level1.wm.rollout_n=4' not in sigreg  # one change against B1
    assert v1['probe_V1_s42'].after == ['train_V1_s42']
    # the cheap planning set A only: no episode or chain tasks for a variant
    assert not any(n.startswith('tasks_') and n != 'tasks_A' for n in v1)
    assert v2['eval_V2_s42_A'].after == ['tasks_A', 'train_V2_s42']


def test_the_map_decoder_ablation_is_one_change_against_the_baseline(spec):
    a2 = by_name(jobs_of(spec, 'A2'))
    command = a2['train_A2_s42'].command
    assert 'level1.map_coeff=0.1' in command and 'num_levels=1' in command
    # the state column is loaded by the data config itself, not by an override
    assert not any('keys_to_load' in c for c in command)
    # B1's settings otherwise: SIGReg stays 0.09, no rollout or history change
    assert not any('sigreg' in c or 'rollout' in c or 'history' in c for c in command)
    assert 'train_B1_s42' not in a2  # B1 is the reference
    assert a2['probe_A2_s42'].after == ['train_A2_s42']
    assert a2['eval_A2_s42_A'].after == ['tasks_A', 'train_A2_s42']
    assert a2['eval_B1_s42_A'].output == (
        by_name(jobs_of(spec, 'B1'))['eval_B1_s42_A'].output
    )
    assert a2['compare_A2_A'].output == '/data/reports/A2_A_val.md'


def test_variants_are_compared_against_the_baseline(spec):
    v1 = by_name(jobs_of(spec, 'V1'))
    # B1 is planned on set A in the same directory as in its own experiment: done once
    b1 = by_name(jobs_of(spec, 'B1'))
    assert v1['eval_B1_s42_A'].output == b1['eval_B1_s42_A'].output
    assert 'train_B1_s42' not in v1
    assert v1['probe_B1_s42'].after == []  # trained before: its report
    plan = v1['compare_V1_A'].command
    assert any(c.startswith('B1_s42=') for c in plan)
    models = v1['compare_V1_models'].command
    training = models[models.index('--training') + 1 : models.index('--probing')]
    assert 'B1_s42=/data/ckpts/craftax/flat/seed42' in training
    assert 'V1_s42=/data/ckpts/craftax/V1/seed42' in training
    assert len(training) == 2


def test_one_run_per_pod(spec):
    # every experiment trains one run: --run only names it, the reports keep the experiment's name
    tag = run.select_runs(spec, 'V2', ['V2'])
    jobs = by_name(run.build_jobs(spec, 'V2', [42], home='/data', tag=tag))
    for name in ('tasks_A', 'eval_B1_s42_A', 'eval_random_A', 'probe_B1_s42',
                 'train_V2_s42', 'probe_V2_s42', 'eval_V2_s42_A'):
        assert name in jobs, name
    assert jobs['compare_V2_A'].output == '/data/reports/V2_A_val.md'
    assert jobs['compare_V2_models'].output == '/data/reports/V2_models.md'
    with pytest.raises(SystemExit, match='no trained run'):
        run.select_runs(run.load_spec(), 'V2', ['B1'])  # the reference


def test_cli_runs_one_variant(capsys):
    run.main(['plan', 'V1'])
    plan = capsys.readouterr().out
    assert 'train_V1_s42' in plan and 'sigreg' not in plan
    assert 'V1_A_val.md' in plan


def test_the_balanced_ablation_is_one_change_against_the_baseline(spec):
    a1 = by_name(jobs_of(spec, 'A1'))
    train = a1['train_A1_s42'].command
    assert 'data.balance.enabled=true' in train
    assert 'num_levels=1' in train
    # trained before the ids: its folder stays
    assert 'output_model_name=flat_balanced' in train
    # B1's settings otherwise: SIGReg stays 0.09, no rollout or history change
    assert not any('sigreg' in c or 'rollout' in c or 'history' in c for c in train)
    # B1 is the reference, planned and probed as for the variants
    assert 'train_B1_s42' not in a1
    assert a1['probe_A1_s42'].after == ['train_A1_s42']
    assert '/data/ckpts/craftax/flat_balanced/seed42' in a1['probe_A1_s42'].command
    assert a1['eval_B1_s42_A'].output == (
        by_name(jobs_of(spec, 'B1'))['eval_B1_s42_A'].output
    )
    assert a1['compare_A1_A'].output.endswith('A1_A_val.md')


def test_the_state_mlp_runs_are_trained_by_their_script_and_compared(spec):
    a3 = by_name(jobs_of(spec, 'A3'))
    train = a3['train_A3_s42']
    assert train.command[:3] == ['python', '-m', 'state_jepa.train']
    assert '--stop-grad' not in train.command
    assert '/data/craftax_expert_train.h5' in train.command
    assert '/data/craftax_expert_val.h5' in train.command  # read on held-out worlds
    out = train.command[train.command.index('--out') + 1]
    assert out == '/data/ckpts/craftax/A3/seed42'
    # the trace grows during the run: only the model at its end means done
    assert train.output == '/data/ckpts/craftax/A3/seed42/model.pt'
    # no planning, no probe suite, no Hydra overrides for these
    assert not any(j.kind in ('eval', 'probe', 'tasks') for j in a3.values())
    assert 'compare_A3_models' not in a3
    a4 = by_name(jobs_of(spec, 'A4', overrides=['level1.lr=1e-4']))
    assert '--stop-grad' in a4['train_A4_s42'].command  # the one change
    assert 'level1.lr=1e-4' not in a4['train_A4_s42'].command
    assert 'train_A3_s42' not in a4  # A3 is the reference, trained in A3
    report = a4['compare_A4_trace']
    assert report.command[:3] == ['python', '-m', 'state_jepa.report']
    assert 'A3_s42=/data/ckpts/craftax/A3/seed42' in report.command
    assert 'A4_s42=/data/ckpts/craftax/A4/seed42' in report.command
    assert report.after == ['train_A4_s42'] and report.output == '/data/reports/A4_trace.md'
    assert spec['experiments']['A3'].get('local') and spec['experiments']['A4'].get('local')


def test_a_probe_of_an_existing_run_has_no_training(spec):
    spec['experiments']['P9'] = {'title': 'probes only', 'probe': ['B1']}
    jobs = jobs_of(spec, 'P9')
    probes = [j for j in jobs if j.kind == 'probe']
    assert len(probes) == 1 and probes[0].after == []
    assert '/data/ckpts/craftax/flat/seed42' in probes[0].command
    assert not any(j.kind == 'train' for j in jobs)


def test_planner_scale_changes_samples_and_iterations_together(spec, tmp_path, monkeypatch):
    assert run.scaled_overrides('craftax_plan', 1) == run.scaled_overrides('craftax_plan', 1.0)
    base = {o.split('=')[0]: int(o.split('=')[1]) for o in run.scaled_overrides('craftax_plan', 1)}
    big = {o.split('=')[0]: int(o.split('=')[1]) for o in run.scaled_overrides('craftax_plan', 4)}
    assert big['solver.solvers.level1.num_samples'] == 2 * base['solver.solvers.level1.num_samples']
    assert big['solver.solvers.level2.n_iters'] == 2 * base['solver.solvers.level2.n_iters']
    # an eval config with a single solver (the other benchmarks' flat configs)
    flat_dir = tmp_path / 'config' / 'eval'
    flat_dir.mkdir(parents=True)
    (flat_dir / 'flat.yaml').write_text(
        'solver: {num_samples: 100, n_iters: 4, topk: 10}\n'
    )
    monkeypatch.setattr(run, 'H_JEPA', tmp_path)
    flat = [o.split('=')[0] for o in run.scaled_overrides('flat', 0.25)]
    assert flat == ['solver.num_samples', 'solver.n_iters', 'solver.topk']
    monkeypatch.undo()
    spec['experiments']['S'] = {
        'title': 'compute scaling',
        'protocols': ['A'],
        'compare': 'scaling',
        'runs': [
            {'name': f'B1_x{s}', 'train': False, 'model': 'B1', 'eval': 'craftax_plan', 'planner_scale': s}
            for s in (1, 4)
        ],
    }
    jobs = by_name(jobs_of(spec, 'S'))
    ev = jobs['eval_B1_x4_s42_A']
    assert 'policy=ckpts/craftax/flat/seed42' in ev.command  # the model of B1, in its folder
    assert 'output.dir=seed42/eval_A_val_craftax_plan_x4' in ev.command
    assert 'solver.solvers.level1.num_samples=%d' % big['solver.solvers.level1.num_samples'] in ev.command
    assert '--scaling' in jobs['compare_S_A'].command
    assert '--planning' in by_name(jobs_of(spec, 'B1'))['compare_B1_A'].command


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
    jobs = jobs_of(spec, 'P2')
    counter = iter(range(100, 200))

    class Result:
        def __init__(self, i):
            self.returncode, self.stdout, self.stderr = 0, f'{i};cluster', ''

    submitted = []

    def fake_sbatch(cmd, **kw):
        submitted.append(Path(cmd[-1]).read_text())
        return Result(next(counter))

    ids = run.submit_slurm(jobs, slurm, str(tmp_path / 'home'), 'P2', submit=True, runner=fake_sbatch)
    assert ids['tasks_C'] == '100'
    evaluation = next(t for t in submitted if '--job-name=eval_B1_s42_C' in t)
    assert '#SBATCH --dependency=afterok:100' in evaluation
    assert '#SBATCH --gres=gpu:1' in evaluation and '#SBATCH --partition=gpu' in evaluation
    assert 'source /venvs/h/bin/activate' in evaluation and 'eval.py' in evaluation
    compare = next(t for t in submitted if '--job-name=compare_P2_C' in t)
    assert 'gres' not in compare  # no GPU for the comparison
    assert 'afterok:' in compare and compare.count(':') >= 3  # waits for both evaluations

    dry = run.submit_slurm(jobs, slurm, str(tmp_path / 'home'), 'P2', submit=False)
    assert all(v.startswith('<') for v in dry.values())  # written, not submitted
    assert (tmp_path / 'generated' / 'P2').is_dir()


def test_cli_list_and_plan(spec, capsys, tmp_path):
    run.main(['list'])
    out = capsys.readouterr().out
    assert 'B1:' in out and '[blocked' in out
    run.main(['plan', 'B1', '--final'])
    plan = capsys.readouterr().out
    assert 'seed=43' in plan and 'seed=44' in plan
    variant = next(r for r in spec['experiments']['V2']['runs'] if r['name'] == 'V2')
    variant['optional'] = True
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(spec))
    run.main(['plan', 'V2', '--spec', str(path)])
    assert 'train_V2' not in capsys.readouterr().out  # optional runs only with --include-optional
    run.main(['plan', 'V2', '--spec', str(path), '--include-optional'])
    assert 'train_V2' in capsys.readouterr().out
    with pytest.raises(SystemExit, match='blocked'):
        run.main(['plan', 'P0'])


def test_final_runs_take_their_tasks_from_the_test_split(spec):
    jobs = by_name(run.build_jobs(spec, 'V1', [42], home='/data', final=True))
    assert 'craftax_expert_test' in jobs['tasks_A'].command
    assert jobs['tasks_A'].output == '/data/eval_tasks/A_test.pt'
    ev = jobs['eval_V1_s42_A']
    assert 'eval.dataset_name=craftax_expert_test' in ev.command
    # training still validates on the validation worlds
    train = jobs['train_V1_s42'].command
    assert 'data.dataset.val_name=craftax_expert_val' in train


def test_final_and_exploration_results_never_share_a_directory(spec):
    # a headline run must not skip jobs, or reuse chunks, of the exploration run
    explore = run.build_jobs(spec, 'V1', [42], home='/data')
    final = run.build_jobs(spec, 'V1', [42], home='/data', final=True)
    outputs = lambda jobs: {  # noqa: E731
        j.name: j.output for j in jobs if j.kind in ('eval', 'compare') and j.output
    }
    out_e, out_f = outputs(explore), outputs(final)
    assert out_e.keys() == out_f.keys()
    for name in out_e:
        if name.endswith('_models'):
            continue  # probes and training reports do not depend on the task split
        assert out_e[name] != out_f[name], name
    ev = {j.name: j for j in final}['eval_V1_s42_A']
    assert 'output.dir=seed42/eval_A_test_craftax_plan' in ev.command


def collect_args(job, flag):
    return int(job.command[job.command.index(flag) + 1])


SNAPSHOTS = ['/ckpt/pqn_u0000381.safetensors', '/ckpt/pqn_u0001143.safetensors']


def collection(spec, **kwargs):
    return run.build_jobs(
        spec, 'D1', [42], home='/data', checkpoint='/ckpt/expert.safetensors', **kwargs
    )


def test_collection_mixes_expert_and_early_checkpoints_by_frames(spec):
    jobs = by_name(collection(spec, early_checkpoints=SNAPSHOTS))
    c = spec['collection']
    for split in ('train', 'val'):
        share = c['share'][split]
        total = c['frames'][split]
        expert, e1, e2 = (jobs[f'collect_{split}_{k}'] for k in ('expert', 'early1', 'early2'))
        # the mix is a share of frames, not of episodes (episodes differ a lot in length)
        assert collect_args(expert, '--target-frames') == round(total * share['expert'])
        early = collect_args(e1, '--target-frames') + collect_args(e2, '--target-frames')
        assert early == round(total * share['early'])  # split equally among the snapshots
        assert abs(collect_args(e1, '--target-frames') - collect_args(e2, '--target-frames')) <= 1
        assert [j.command[j.command.index('--checkpoint') + 1] for j in (e1, e2)] == SNAPSHOTS
        assert all('--num-episodes' not in j.command for j in (expert, e1, e2))
        assert all('--source' in j.command and j.command[j.command.index('--source') + 1] == 'early' for j in (e1, e2))
        assert '--no-require-diamond' not in expert.command and '--no-require-diamond' in e1.command
        assert '--source' not in expert.command  # derived: the final expert is source 0
        # one file: the runs are chained, and their seeds do not overlap
        assert e1.after == [expert.name] and e2.after == [e1.name]
        start = c['seed_start'][split]
        starts = [collect_args(j, '--seed-start') for j in (expert, e1, e2)]
        ends = [collect_args(j, '--seed-end') for j in (expert, e1, e2)]
        assert starts == [start, start + 500_000, start + 750_000]
        assert ends == starts[1:] + [start + 1_000_000]  # the last stops where the next split begins
    ranges = [c['seed_start'][s] for s in ('train', 'val', 'test')]
    assert ranges == sorted(ranges) and ranges[1] - ranges[0] >= 1_000_000  # disjoint splits


def test_the_test_split_holds_only_the_final_expert(spec):
    jobs = collection(spec, early_checkpoints=SNAPSHOTS)
    test = [j.name for j in jobs if j.name.startswith('collect_test_')]
    assert test == ['collect_test_expert']
    assert collect_args(by_name(jobs)['collect_test_expert'], '--target-frames') == spec['collection']['frames']['test']


def test_without_snapshots_the_plan_shows_a_placeholder(spec):
    jobs = by_name(collection(spec))
    assert '<early-checkpoint>' in jobs['collect_train_early1'].command
    assert 'collect_train_early2' not in jobs


def test_noised_and_random_remain_optional_fallbacks(spec):
    spec['collection']['share']['train'] = {'expert': 0.5, 'early': 0.2, 'noised': 0.15, 'random': 0.15}
    jobs = by_name(collection(spec, early_checkpoints=SNAPSHOTS[:1]))
    names = ['expert', 'early1', 'noised', 'random']
    chain = [jobs[f'collect_train_{n}'] for n in names]
    assert [j.after for j in chain] == [[]] + [[chain[i].name] for i in range(3)]
    starts = [collect_args(j, '--seed-start') for j in chain]
    ends = [collect_args(j, '--seed-end') for j in chain]
    assert ends[:-1] == starts[1:] and ends[-1] == 1_000_000 and starts[1] == 500_000
    noised, random = chain[2], chain[3]
    assert noised.command[noised.command.index('--eps') + 1] == '0.3' and '--checkpoint' in noised.command
    assert random.command[random.command.index('--eps') + 1] == '1.0' and '--checkpoint' not in random.command


def test_collection_run_needs_the_checkpoints(spec):
    with pytest.raises(SystemExit, match='expert-checkpoint'):
        run.main(['run', 'D1', '--mode', 'slurm'])
    with pytest.raises(SystemExit, match='early-checkpoint'):
        run.main(['run', 'D1', '--mode', 'slurm', '--expert-checkpoint', '/ckpt/e.safetensors'])


def test_dataset_analysis_checks_the_seed_overlap_of_all_three_files(spec):
    jobs = run.build_jobs(spec, 'D0', [42], home='/data')
    command = jobs[0].command
    assert command.count('--data') == 3
    labels = [command[i + 1] for i, c in enumerate(command) if c == '--label']
    assert labels == ['train', 'val', 'test']


def test_collection_shares_must_sum_to_one(spec):
    spec['collection']['share']['val'] = {'expert': 0.5, 'early': 0.1}
    with pytest.raises(SystemExit, match='sum to 1'):
        run.build_jobs(spec, 'D1', [42], home='/data', checkpoint='/c')


def test_unknown_collection_sources_are_rejected(spec):
    spec['collection']['share']['train'] = {'expert': 0.5, 'early': 0.4, 'noisy': 0.1}
    with pytest.raises(SystemExit, match='unknown collection sources'):
        run.build_jobs(spec, 'D1', [42], home='/data', checkpoint='/c')


def test_dataset_analysis_runs_the_dataset_section_on_all_three_files(spec):
    job = run.build_jobs(spec, 'D0', [42], home='/data')[0]
    assert job.command[1:4] == ['-m', 'dataset_section', 'compute']
    assert job.cwd == run.ROOT / 'analysis'
    assert job.output == '/data/reports/dataset/dataset_numbers.json'
    assert '/data/reports/dataset' == job.command[job.command.index('--out') + 1]


def placeholder_collection(spec):
    """The collection jobs with the $HJEPA_HOME placeholder, as `main` builds them."""
    return run.build_jobs(
        spec,
        'D1',
        [42],
        home=run.HOME_VAR,
        checkpoint='/ckpt/expert.safetensors',
        early_checkpoints=SNAPSHOTS,
    )


def test_a_restarted_collection_skips_the_finished_runs(spec, tmp_path):
    # the collector appends and --target-frames counts one run: repeating a
    # finished run would add another full share on new seeds
    jobs = placeholder_collection(spec)
    home = str(tmp_path)
    calls = []

    class Result:
        returncode = 0

    def runner(command, **kwargs):
        calls.append(command)
        return Result()

    first = run.run_local(jobs, home, runner=runner)
    assert first == [j.name for j in jobs] and len(calls) == len(jobs)
    marker = Path(run.resolve(jobs[0], home).output)
    assert marker.parent == tmp_path / 'collect_done' and marker.exists()
    assert run.run_local(jobs, home, runner=runner) == []  # nothing repeated
    assert len(calls) == len(jobs)
    assert run.run_local(jobs, home, force=True, runner=runner)  # --force


def test_a_failed_collection_run_has_no_marker(spec, tmp_path):
    jobs = placeholder_collection(spec)

    class Result:
        def __init__(self, code):
            self.returncode = code

    def runner(command, **kwargs):
        return Result(1 if '--source' in command else 0)  # the first early run

    with pytest.raises(SystemExit, match='failed'):
        run.run_local(jobs, str(tmp_path), runner=runner)
    done = sorted(p.name for p in (tmp_path / 'collect_done').iterdir())
    assert done == ['collect_train_expert']


def test_the_slurm_script_of_a_collection_run_writes_the_marker(spec, tmp_path):
    job = collection(spec, early_checkpoints=SNAPSHOTS)[0]
    slurm = yaml.safe_load(
        (Path(run.__file__).parent / 'slurm.yaml').read_text()
    )
    script = run.render_sbatch(job, slurm, '/data', [])
    assert 'echo done > /data/collect_done/collect_train_expert' in script


def test_the_cheap_planning_set_is_small_and_checked_against_the_budget(spec):
    a = spec['tasks']['A']
    assert a['mode'] == 'offset' and a['offsets'] == [8]
    assert a['eval_budget'] >= 2 * max(a['offsets'])  # room to reach the goal
    # one task per episode: the 128 validation episodes that reach the
    # diamond are all there are (the 95% interval of a success rate: +-9)
    assert a['num_tasks'] == 128
