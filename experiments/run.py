"""Plan, run, or submit the Craftax experiments listed in experiments.yaml.

    python run.py list
    python run.py plan F1                     # print every command, with dependencies
    python run.py plan F1 --final             # three seeds instead of one
    python run.py run F1 --mode local         # run them here, in order
    python run.py run F1 --mode slurm         # write sbatch scripts (and submit with --submit)

`HJEPA_HOME` must point at the directory with the datasets and checkpoints. Jobs whose
output exists are skipped (`--force` runs them again). Experiments whose code is not
built yet are marked `blocked` in the spec and cannot be run.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ROOT = HERE.parent
H_JEPA = ROOT / "H-JEPA" / "h_jepa"
HOME_VAR = "$HJEPA_HOME"


@dataclass
class Job:
    name: str
    kind: str  # tasks | train | eval | probe | analysis | compare
    command: list[str]
    after: list[str] = field(default_factory=list)
    output: str | None = None  # a path whose existence means the job is done
    cwd: Path = H_JEPA


def _run_dir(home, model, seed) -> str:
    return f"{home}/ckpts/craftax/{model}/seed{seed}"


def _eval_dir(home, model, seed, protocol, split, eval_name, suffix="") -> str:
    # the split is part of the path: exploration (val worlds) and headline (test worlds) results never share a directory
    return f"{_run_dir(home, model, seed)}/eval_{protocol}_{split}_{eval_name}{suffix}"


def scaled_overrides(eval_name: str, scale: float) -> list[str]:
    """Overrides that multiply the planner's samples x iterations by `scale` (each by its square root)."""
    cfg = yaml.safe_load((H_JEPA / "config" / "eval" / f"{eval_name}.yaml").read_text())
    factor = scale**0.5
    solvers = cfg["solver"].get("solvers") or {"": cfg["solver"]}
    out = []
    for level, s in solvers.items():
        prefix = f"solver.solvers.{level}" if level else "solver"
        samples = max(2, round(s["num_samples"] * factor))
        iters = max(1, round(s["n_iters"] * factor))
        topk = max(1, min(samples, round(s["topk"] * factor)))
        out += [f"{prefix}.num_samples={samples}", f"{prefix}.n_iters={iters}", f"{prefix}.topk={topk}"]
    return out


COLLECTION_KINDS = ("expert", "early", "noised", "random")


def collection_shares(c: dict, split: str) -> dict:
    """Share of the split's frames per source. `share` is one mapping for all splits, or one per split."""
    share = c["share"][split] if split in c["share"] else c["share"]
    unknown = set(share) - set(COLLECTION_KINDS)
    if unknown:
        raise SystemExit(f"unknown collection sources {sorted(unknown)}; use {COLLECTION_KINDS}")
    share = {k: float(share.get(k, 0.0)) for k in COLLECTION_KINDS}
    if abs(sum(share.values()) - 1.0) > 1e-6:
        raise SystemExit(f"collection shares must sum to 1, got {share} for {split}")
    return share


def collection_jobs(spec: dict, home: str, checkpoint: str | None, python: str = "python",
                    early_checkpoints: list[str] | None = None) -> list[Job]:  # fmt: skip
    """Collector runs per split: the final expert's diamond episodes first, then the others.

    `early` runs act with earlier snapshots of the expert's training (the early share is split equally among
    the snapshots, each with its own seeds); `noised` and `random` are optional fallbacks. The seeds of the
    split's 1M: the first 500k for the expert, the rest equally among the other runs.
    """
    c, st = spec["collection"], spec["settings"]
    jax_dir = ROOT / "purejaxql"
    script = "scripts/collect_craftax_episodes.py"
    snapshots = list(early_checkpoints or []) or ["<early-checkpoint>"]
    jobs = []
    for split, dataset in (("train", st["train_dataset"]), ("val", st["val_dataset"]), ("test", st["test_dataset"])):
        frames, share = int(c["frames"][split]), collection_shares(c, split)
        target = {name: round(frames * share[name]) for name in COLLECTION_KINDS}
        out = f"{home}/{dataset}.h5"
        start = int(c["seed_start"][split])
        base = [python, script, "--out", out]
        expert = checkpoint or "<expert-checkpoint>"
        steps = [("expert", [*base, "--checkpoint", expert, "--target-frames", str(target["expert"])], target["expert"], (start, start + 500_000))]  # fmt: skip
        others = []  # (name, command without seeds, frames)
        if target["early"] > 0:
            for k, snapshot in enumerate(snapshots):
                part = target["early"] // len(snapshots) + (1 if k < target["early"] % len(snapshots) else 0)
                others.append((f"early{k + 1}", [*base, "--checkpoint", snapshot, "--source", "early", "--no-require-diamond",
                                                  "--target-frames", str(part)], part))  # fmt: skip
        if target["noised"] > 0:
            others.append(("noised", [*base, "--checkpoint", expert, "--eps", str(c["noise_eps"]), "--no-require-diamond",
                                      "--target-frames", str(target["noised"])], target["noised"]))  # fmt: skip
        if target["random"] > 0:
            others.append(("random", [*base, "--eps", "1.0", "--no-require-diamond", "--target-frames", str(target["random"])], target["random"]))  # fmt: skip
        width = 500_000 // max(1, len(others))
        for k, (name, command, count) in enumerate(others):
            lo = start + 500_000 + k * width
            hi = start + 1_000_000 if k == len(others) - 1 else lo + width
            steps.append((name, command, count, (lo, hi)))
        previous = None
        for name, command, count, (lo, hi) in steps:
            if count <= 0:
                continue
            command = [*command, "--seed-start", str(lo), "--seed-end", str(hi)]
            # the runs append to one file, so they run one after the other
            job_name = f"collect_{split}_{name}"
            # The runs append to one file and --target-frames counts the frames of one run, so a repeated run adds
            # another full share on new seeds. The marker (written after the run, see `mark_done`) keeps a restarted
            # D1 from repeating the finished steps. A run that was killed has no marker and starts again: its partial
            # episodes stay in the file (the seeds are skipped), the share is then exceeded by at most that run's part.
            jobs.append(Job(job_name, "collect", command, after=[previous] if previous else [],
                            output=f"{home}/collect_done/{job_name}", cwd=jax_dir))  # fmt: skip
            previous = job_name
    return jobs


def build_jobs(spec: dict, exp_id: str, seeds: list[int], overrides=(), home: str = HOME_VAR, python: str = "python",
               final: bool = False, checkpoint: str | None = None, early_checkpoints: list[str] | None = None) -> list[Job]:  # fmt: skip
    """All jobs of one experiment, in an order that respects their dependencies."""
    exp = spec["experiments"][exp_id]
    if exp.get("blocked"):
        raise SystemExit(f"{exp_id} is blocked: {exp['blocked']}")
    st = spec["settings"]
    if exp.get("collection"):
        return collection_jobs(spec, home, checkpoint, python, early_checkpoints)
    train_ds, val_ds = st["train_dataset"], st["val_dataset"]
    task_ds = st["test_dataset"] if final else val_ds  # headline results use held-out test worlds
    split = "test" if final else "val"
    jobs: list[Job] = []

    if exp.get("analysis"):
        out = f"{home}/reports/dataset"
        jobs.append(Job("analysis", "analysis", [python, "-m", "dataset_section", "compute",
                        "--data", f"{home}/{train_ds}.h5", "--label", "train", "--data", f"{home}/{val_ds}.h5",
                        "--label", "val", "--data", f"{home}/{st['test_dataset']}.h5", "--label", "test", "--out", out],
                        output=f"{out}/dataset_numbers.json", cwd=ROOT / "analysis"))  # fmt: skip
        return jobs

    protocols = exp.get("protocols", [])
    task_jobs = {}
    for p in protocols:
        t = spec["tasks"][p]
        path = f"{home}/{st['tasks_dir']}/{p}_{split}.pt"
        cmd = [python, "eval_tasks.py", "--dataset", task_ds, "--mode", t["mode"], "--num-tasks", str(t["num_tasks"]),
               "--eval-budget", str(t["eval_budget"]), "--out", path]  # fmt: skip
        if t.get("offsets"):
            cmd += ["--offsets", *map(str, t["offsets"])]
        task_jobs[p] = Job(f"tasks_{p}", "tasks", cmd, output=path)
        jobs.append(task_jobs[p])

    eval_dirs: dict[str, list[str]] = {}
    for run in exp.get("runs", []):
        model = run.get("model", run["name"])
        for seed in seeds:
            train_name = None
            if run.get("train", True):
                cmd = [python, "main_hjepa.py", "--config-name", run["config"], f"output_model_name={run['name']}",
                       f"seed={seed}", f"data.dataset.name={train_ds}", f"data.dataset.val_name={val_ds}",
                       *run.get("overrides", []), *overrides]  # fmt: skip
                train_name = f"train_{run['name']}_s{seed}"
                jobs.append(Job(train_name, "train", cmd, output=f"{_run_dir(home, run['name'], seed)}/{run['name']}_object.ckpt"))
            for p in protocols:
                t = spec["tasks"][p]
                suffix = f"_x{run['planner_scale']}" if "planner_scale" in run else ""
                directory = _eval_dir(home, model, seed, p, split, run["eval"], suffix)
                cmd = [python, "eval.py", "--config-name", run["eval"], f"policy=ckpts/craftax/{model}/seed{seed}",
                       f"load_eval_trajs_path={task_jobs[p].output}", f"eval.num_eval={t['num_tasks']}",
                       f"eval.goal_offset_steps={t['eval_budget']}", f"eval.eval_budget={t['eval_budget']}",
                       f"eval.chunk_size={st['chunk_size']}", f"eval.dataset_name={task_ds}",
                       f"planning_metrics.support_bank_dataset={train_ds}",
                       f"output.dir=seed{seed}/eval_{p}_{split}_{run['eval']}{suffix}"]  # fmt: skip
                if "planner_scale" in run:
                    cmd += scaled_overrides(run["eval"], float(run["planner_scale"]))
                after = [task_jobs[p].name] + ([train_name] if train_name else [])
                name = f"eval_{run['name']}_s{seed}_{p}"
                jobs.append(Job(name, "eval", cmd, after=after, output=f"{directory}/planning_report.md"))
                eval_dirs.setdefault(p, []).append(f"{run['name']}_s{seed}={directory}")

    if exp.get("baseline"):
        for p in protocols:
            t = spec["tasks"][p]
            directory = f"{home}/ckpts/craftax/random/eval_{p}_{split}"
            cmd = [python, "eval.py", "--config-name", "craftax_plan", "policy=random", f"load_eval_trajs_path={task_jobs[p].output}",
                   f"eval.num_eval={t['num_tasks']}", f"eval.goal_offset_steps={t['eval_budget']}",
                   f"eval.eval_budget={t['eval_budget']}", f"eval.chunk_size={st['chunk_size']}",
                   f"eval.dataset_name={task_ds}", f"output.dir={directory}"]  # fmt: skip
            jobs.append(Job(f"eval_random_{p}", "eval", cmd, after=[task_jobs[p].name], output=f"{directory}/planning_report.md"))
            eval_dirs.setdefault(p, []).append(f"random={directory}")

    probes = []
    for model in exp.get("probe", []):
        for seed in seeds:
            cmd = [python, "probe_eval.py", "--run", _run_dir(home, model, seed), "--train-dataset", train_ds,
                   "--val-dataset", val_ds, "--max-clips", "512"]  # fmt: skip
            train = f"train_{model}_s{seed}"  # a model trained in the same experiment is probed after its training
            after = [train] if any(j.name == train for j in jobs) else []
            jobs.append(Job(f"probe_{model}_s{seed}", "probe", cmd, after=after,
                            output=f"{_run_dir(home, model, seed)}/health/probe_report.md"))  # fmt: skip
            probes.append(f"{model}_s{seed}={_run_dir(home, model, seed)}")

    flag = "--scaling" if exp.get("compare") == "scaling" else "--planning"
    for p, dirs in eval_dirs.items():  # one comparison per protocol: the baseline differs between task sets
        out = f"{home}/reports/{exp_id}_{p}_{split}.md"
        cmd = [python, str(ROOT / "analysis" / "compare_runs.py"), "--out", out, flag, *dirs]
        if any(d.startswith("random=") for d in dirs):
            cmd += ["--baseline", "random"]
        after = [j.name for j in jobs if j.kind == "eval" and j.name.endswith(f"_{p}")]
        jobs.append(Job(f"compare_{exp_id}_{p}", "compare", cmd, after=after, output=out, cwd=ROOT))
    trained = [r for r in exp.get("runs", []) if r.get("train", True)]
    if trained or probes:
        out = f"{home}/reports/{exp_id}_models.md"
        cmd = [python, str(ROOT / "analysis" / "compare_runs.py"), "--out", out]
        if trained:
            cmd += ["--training", *[f"{r['name']}_s{s}={_run_dir(home, r['name'], s)}" for r in trained for s in seeds]]
        probed = probes or [f"{r['name']}_s{s}={_run_dir(home, r['name'], s)}" for r in trained for s in seeds]
        cmd += ["--probing", *probed]
        after = [j.name for j in jobs if j.kind in ("train", "probe")]
        jobs.append(Job(f"compare_{exp_id}_models", "compare", cmd, after=after, output=out, cwd=ROOT))
    return jobs


# --------------------------------------------------------------------------
# running


def shell(job: Job) -> str:
    return " ".join(shlex.quote(c) if "$" not in c else c for c in job.command)


def resolve(job: Job, home: str) -> Job:
    expand = lambda s: s.replace(HOME_VAR, home)  # noqa: E731
    return Job(job.name, job.kind, [expand(c) for c in job.command], job.after, expand(job.output) if job.output else None, job.cwd)


def mark_done(job: Job):
    """The marker of a finished collection run (the collector appends, so the file itself says nothing)."""
    Path(job.output).parent.mkdir(parents=True, exist_ok=True)
    Path(job.output).write_text(shell(job) + "\n")


def run_local(jobs: list[Job], home: str, force=False, runner=subprocess.run) -> list[str]:
    ran = []
    env = {**os.environ, "HJEPA_HOME": home, "PYTHONPATH": f"{H_JEPA.parent}:{H_JEPA}"}
    for job in map(lambda j: resolve(j, home), jobs):
        if job.output and Path(job.output).exists() and not force:
            print(f"skip {job.name} (done: {job.output})")
            continue
        print(f"run  {job.name}: {shell(job)}")
        result = runner(job.command, cwd=job.cwd, env=env)
        if result.returncode:
            raise SystemExit(f"{job.name} failed with exit code {result.returncode}")
        if job.kind == "collect":
            mark_done(job)
        ran.append(job.name)
    return ran


def render_sbatch(job: Job, slurm: dict, home: str, dependency_ids: list[str]) -> str:
    lines = ["#!/bin/bash", f"#SBATCH --job-name={job.name}", f"#SBATCH --time={slurm['time'][job.kind]}",
             f"#SBATCH --cpus-per-task={slurm['cpus_per_task']}", f"#SBATCH --mem={slurm['mem']}",
             f"#SBATCH --output={HERE / 'generated'}/%x-%j.log"]  # fmt: skip
    if job.kind in ("train", "eval", "probe") and slurm.get("gpus"):
        lines.append(f"#SBATCH --gres=gpu:{slurm['gpus']}")
    for key, flag in (("partition", "partition"), ("account", "account"), ("qos", "qos")):
        if slurm.get(key):
            lines.append(f"#SBATCH --{flag}={slurm[key]}")
    if dependency_ids:
        lines.append("#SBATCH --dependency=afterok:" + ":".join(dependency_ids))
    lines += ["", "set -euo pipefail", f"export HJEPA_HOME={shlex.quote(home)}",
              f"export PYTHONPATH={H_JEPA.parent}:{H_JEPA}"]  # fmt: skip
    if slurm.get("venv"):
        lines.append(f"source {slurm['venv']}/bin/activate")
    lines += list(slurm.get("extra") or [])
    lines += [f"cd {job.cwd}", shell(resolve(job, home))]
    if job.kind == "collect":
        marker = shlex.quote(resolve(job, home).output)
        lines += [f"mkdir -p $(dirname {marker})", f"echo done > {marker}"]
    return "\n".join(lines) + "\n"


def submit_slurm(jobs, slurm, home, exp_id, submit=False, force=False, runner=subprocess.run) -> dict[str, str]:
    """Write one script per job; with `submit`, sbatch them with dependencies. Returns name -> job id."""
    out_dir = HERE / "generated" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ids: dict[str, str] = {}
    for i, job in enumerate(jobs):
        if job.output and Path(resolve(job, home).output).exists() and not force:
            print(f"skip {job.name} (done)")
            continue
        deps = [ids[a] for a in job.after if a in ids]
        script = out_dir / f"{i:02d}_{job.name}.sbatch"
        script.write_text(render_sbatch(job, slurm, home, deps))
        if not submit:
            print(f"wrote {script}")
            ids[job.name] = f"<{job.name}>"
            continue
        result = runner(["sbatch", "--parsable", str(script)], capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f"sbatch failed for {job.name}: {result.stderr}")
        ids[job.name] = result.stdout.strip().split(";")[0]
        print(f"submitted {job.name} as {ids[job.name]}")
    return ids


def load_spec(path=HERE / "experiments.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "plan", "run"])
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("--spec", default=str(HERE / "experiments.yaml"))
    parser.add_argument("--final", action="store_true", help="the seeds of the headline results instead of one")
    parser.add_argument("--seeds", help="comma-separated seeds, overrides --final")
    parser.add_argument("--override", nargs="*", default=[], help="Hydra overrides for every training run")
    parser.add_argument("--mode", choices=["local", "slurm"], default="local")
    parser.add_argument("--submit", action="store_true", help="slurm: submit instead of only writing the scripts")
    parser.add_argument("--slurm-config", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--expert-checkpoint", help="the expert's .safetensors, for D1 (data collection)")
    parser.add_argument("--early-checkpoint", action="append", default=[], metavar="SNAPSHOT",
                        help="a snapshot of the expert's training (..._vmap0_u<updates>.safetensors) whose rollouts "
                             "make the early share of D1; repeat it for several snapshots")  # fmt: skip
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    if args.command == "list":
        for exp_id, e in spec["experiments"].items():
            print(f"{exp_id}: {e['title']}" + (f"  [blocked: {e['blocked']}]" if e.get("blocked") else ""))
        return
    if not args.experiment or args.experiment not in spec["experiments"]:
        parser.error(f"give an experiment: {', '.join(spec['experiments'])}")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else spec["settings"]["seeds_final" if args.final else "seeds_explore"]
    if not args.include_optional:
        for e in spec["experiments"].values():
            dropped = {r["name"] for r in e.get("runs", []) if r.get("optional")}
            e["runs"] = [r for r in e.get("runs", []) if not r.get("optional")]
            if "probe" in e:
                e["probe"] = [m for m in e["probe"] if m not in dropped]

    home = os.environ.get("HJEPA_HOME")
    jobs = build_jobs(spec, args.experiment, seeds, args.override, home or HOME_VAR, final=args.final,
                      checkpoint=args.expert_checkpoint, early_checkpoints=args.early_checkpoint)  # fmt: skip
    if args.command == "run" and args.experiment == "D1":
        if not args.expert_checkpoint:
            raise SystemExit("D1 needs --expert-checkpoint")
        if not args.early_checkpoint and any(collection_shares(spec["collection"], s)["early"] > 0 for s in ("train", "val", "test")):
            raise SystemExit("D1 needs at least one --early-checkpoint (a snapshot of the expert's training)")
    if args.command == "plan":
        for job in jobs:
            print(f"[{job.kind}] {job.name}" + (f"  after {', '.join(job.after)}" if job.after else ""))
            print(f"    cd {job.cwd.relative_to(ROOT)} && {shell(job)}")
        return
    if not home:
        raise SystemExit("set HJEPA_HOME")
    if args.mode == "local":
        run_local(jobs, home, args.force)
    else:
        config = args.slurm_config or (HERE / "slurm.local.yaml" if (HERE / "slurm.local.yaml").exists() else HERE / "slurm.yaml")
        submit_slurm(jobs, yaml.safe_load(Path(config).read_text()), home, args.experiment, args.submit, args.force)


if __name__ == "__main__":
    sys.exit(main())
