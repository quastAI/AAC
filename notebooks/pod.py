"""Helpers of the RunPod notebooks of the flat phase (`runpod_flat.ipynb` and one notebook per variant).

A notebook calls `configure()` once, then imports what it uses (`from pod import sh, launch, ...`). The paths: the network
volume at `/workspace` refuses `chmod` (`git clone`, `venv` and `pip` fail on it) and stalls on the random reads of the
training file, so only what must survive a pod stop lives there (checkpoints, reports, task files); the clone, the venv,
the episode files, the logs and stable-pretraining's folder are on the local disk.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

VOLUME = Path("/workspace")  # network volume: survives a pod stop, but refuses chmod and stalls on random reads
REPO = Path("/root/AAC")  # the git clone, on the local disk: git fails on the volume
VENV = Path("/root/venvs/hjepa")  # PyTorch environment, on the local disk (pip needs chmod); installed again on a new pod
DATA = Path("/root/data")  # HJEPA_HOME on the local disk: the episode files (the volume stalled the dataset build)
SPT_DIR = Path("/root/spt")  # stable-pretraining's logs: written at every logging step, which stalled the training 10 s each on the volume
LOGS = Path("/root/logs")  # the notebooks' logs: written at every step, and lost with the pod (the training resumes from the volume)
PERSIST = VOLUME / "data"  # what must survive a pod stop: DATA/ckpts, reports and eval_tasks link to it
SEEDS = "42"
PYTHON = VENV / "bin" / "python"
H_JEPA = REPO / "H-JEPA" / "h_jepa"
RUNNER = [PYTHON, REPO / "experiments" / "run.py"]
ENV: dict[str, str] = {}
PROCS: dict[str, subprocess.Popen] = {}
LAUNCH_MARK = "##### launched"  # written to a log at every launch: wait() only looks at the output of the last one
SUITE_MARK = "### View concepts"  # a heading of the current probe suite's report (added on 2026-10-27)


def cpu_threads(limit=8, cpu_max="/sys/fs/cgroup/cpu.max"):
    """Threads for torch/BLAS: the container's CPU quota (the cgroup), at most `limit`. torch starts os.cpu_count() threads,
    128 on a pod that may use 13.6 cores: they throttle each other, the 512-clip probe's regressions ran 40x slower."""
    try:
        quota, period = Path(cpu_max).read_text().split()
        cpus = int(quota) // int(period) if quota != "max" else os.cpu_count()
    except (OSError, ValueError):
        cpus = os.cpu_count()
    return max(1, min(limit, cpus or 1))


def configure(volume=VOLUME, repo=REPO, venv=VENV, data=DATA, spt_dir=SPT_DIR, logs=LOGS, seeds="42"):
    """Set the paths and the environment of every command, link the volume's folders into DATA."""
    global VOLUME, REPO, VENV, DATA, SPT_DIR, LOGS, PERSIST, SEEDS, PYTHON, H_JEPA, RUNNER, ENV
    VOLUME, REPO, VENV, DATA, SPT_DIR, LOGS = map(Path, (volume, repo, venv, data, spt_dir, logs))
    PERSIST, SEEDS = VOLUME / "data", seeds
    PYTHON, H_JEPA = VENV / "bin" / "python", REPO / "H-JEPA" / "h_jepa"
    RUNNER = [PYTHON, REPO / "experiments" / "run.py"]
    os.chdir(VOLUME)  # a deleted and re-cloned repo leaves the kernel in a directory that no longer exists
    threads = str(cpu_threads())
    ENV = {
        # LD_LIBRARY_PATH is dropped: the RunPod image puts its own CUDA 12.8 there, next to the pip-installed CUDA libraries
        **{k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"},
        "PATH": f"{VENV / 'bin'}:{os.environ['PATH']}",  # `python` in the runner's commands is the venv's
        "HJEPA_HOME": str(DATA),
        "PYTHONPATH": f"{H_JEPA.parent}:{H_JEPA}",  # h_jepa modules import each other as top-level modules
        "HDF5_USE_FILE_LOCKING": "FALSE",  # HDF5 can fail to lock files on a network volume
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "OMP_NUM_THREADS": threads,  # see cpu_threads
        "MKL_NUM_THREADS": threads,
        "MPLBACKEND": "agg",  # the kernel's inline backend does not exist in the venv, matplotlib would fail at import
    }
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("ckpts", "reports", "eval_tasks"):  # the real folders are on the volume: a pod stop keeps them
        (PERSIST / name).mkdir(parents=True, exist_ok=True)
        if not (DATA / name).is_symlink() and not (DATA / name).exists():
            (DATA / name).symlink_to(PERSIST / name)
    LOGS.mkdir(parents=True, exist_ok=True)


def train_overrides(use_wandb: bool, project: str = "hjepa-flat") -> list[str]:
    """Hydra overrides for every training run; one GPU: on several, DDP stops at the unused ViT mask_token."""
    out = ["trainer.devices=1", f"spt_dir={SPT_DIR}"]
    return out + (["wandb.enabled=true", f"wandb.config.project={project}"] if use_wandb else [])


def sh(cmd, cwd=None, env=None, check=True, quiet=False, log=None):
    """Run a command and stream its output into the cell; `quiet` shows the last lines only if it fails, `log` also writes all of
    the output to LOGS/<log>.log (the tail of a failing run can be one harmless traceback that hides the real error)."""
    cmd = [str(c) for c in cmd]
    print("$", " ".join((c.splitlines() or [""])[0][:100] + (" ..." if len(c.splitlines()) > 1 or len(c) > 100 else "") for c in cmd))
    proc = subprocess.Popen(cmd, env=env or ENV, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if not quiet:
            print(line, end="")
    code = proc.wait()
    if log:
        (LOGS / f"{log}.log").write_text("".join(lines))
    if code and quiet:
        print("".join(lines[-80:]), end="")
    if check and code:
        raise RuntimeError(f"{cmd[0]} exited with {code}" + (f", all of its output is in {LOGS / (log + '.log')}" if log else ""))
    return code


def launch(name, cmd, cwd=None):
    """Start a command detached from the kernel (new session), log to LOGS/<name>.log, pid in LOGS/<name>.pid."""
    if running(name):
        return print(f"{name} is already running (pid {(LOGS / (name + '.pid')).read_text()})")
    with open(LOGS / f"{name}.log", "a") as log:
        log.write(f"{LAUNCH_MARK} {name} {time.strftime('%H:%M:%S')}\n")
        log.flush()
        PROCS[name] = subprocess.Popen([str(c) for c in cmd], env=ENV, cwd=cwd or REPO, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)  # fmt: skip
    (LOGS / f"{name}.pid").write_text(str(PROCS[name].pid))
    print(f"started {name}, pid {PROCS[name].pid}, log {LOGS / (name + '.log')}")


def running(name):
    if name in PROCS:  # poll() also reaps a finished child
        return PROCS[name].poll() is None
    try:  # after a kernel restart the process is no longer our child
        os.kill(int((LOGS / f"{name}.pid").read_text()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False


def progress(name, key, n=1):
    """The last `n` log lines that contain `key`; progress bars overwrite their line with \\r, so both split lines."""
    path = LOGS / f"{name}.log"
    lines = [line for line in re.split(r"[\r\n]", path.read_text(errors="replace")) if key in line] if path.exists() else []
    return [line.strip()[:200] for line in lines[-n:]] or ["(no progress line yet)"]


def wait(name, key, every=120):
    while running(name):
        time.sleep(every)
        print(time.strftime("%H:%M"), progress(name, key)[0])
    print(f"{name} finished")
    last_launch = (LOGS / f"{name}.log").read_text(errors="replace").rsplit(LAUNCH_MARK, 1)[-1]  # an earlier failed start stays in the file
    assert "failed with exit code" not in last_launch, f"a job of {name} failed: read {LOGS / (name + '.log')}, then run its start cell again"


def run_dir(model, seed=None):
    return DATA / "ckpts" / "craftax" / model / f"seed{seed or SEEDS.split(',')[0]}"


def show(path, lines=60):
    path = Path(path)
    print("\n".join(path.read_text().splitlines()[:lines]) if path.exists() else f"({path} does not exist yet)")


def login(repo_id, private=True, use_wandb=False):
    """The Hugging Face API (token from HF_TOKEN or asked), the dataset repo, the W&B key for the training subprocesses."""
    assert "your-user" not in repo_id, "set HF_REPO_ID in the configuration cell to your Hub dataset repo (the one runpod_dataset.ipynb filled)"
    sh([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub>=2"], env=os.environ)
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face token (write access): "))
    print("logged in as", api.whoami()["name"])
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    if use_wandb:  # the training subprocess reads it from ENV; without it the training stops at its start
        ENV["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY") or getpass.getpass("Weights & Biases API key: ")
    return api


def install(api, repo_id, use_wandb=False):
    """The pod, the PyTorch environment (installed again on a new pod) and the dataset from the Hub (skipped when there)."""
    from huggingface_hub import hf_hub_download

    assert REPO.exists(), f"clone the repository to {REPO} (the local disk, not /workspace: git fails on the volume)"
    for command in (["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], ["nproc"], ["df", "-h", str(VOLUME), str(DATA)]):
        sh(command, env=os.environ, check=False)
    sh(["git", "-C", REPO, "log", "-1", "--oneline"], env=os.environ)

    files = [REPO / "environment" / "hjepa-requirements.txt"]
    stamp = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]
    marker = VENV / ".installed"
    if not (marker.exists() and marker.read_text() == stamp):
        if not PYTHON.exists():
            sh([sys.executable, "-m", "venv", VENV], env=os.environ)
        sh([PYTHON, "-m", "pip", "install", "-q", "-U", "pip"])
        sh([PYTHON, "-m", "pip", "install", "-q", "-r", "environment/hjepa-requirements.txt"], cwd=REPO)
        marker.write_text(stamp)
    cuda = subprocess.run([PYTHON, "-c", "import torch; print(torch.cuda.is_available(), torch.__version__)"], env=ENV, capture_output=True, text=True).stdout.strip()
    assert cuda.startswith("True"), f"torch does not see the GPU ({cuda}): check nvidia-smi and the torch install in the venv"
    print("torch sees the GPU:", cuda)
    if use_wandb:  # a wrong key would stop the training at its start
        sh([PYTHON, "-c", "import os, wandb; assert wandb.login(key=os.environ['WANDB_API_KEY'], verify=True)"], quiet=True)
        print("Weights & Biases login ok")

    # the dataset of runpod_dataset.ipynb, from the Hub to the local disk (about 9 GB, skipped when the files are there)
    assert DATA.name == "data", "the files are downloaded to <DATA's parent>/data/, as in the Hub repo"
    for split in ("train", "val", "test"):
        if not (DATA / f"craftax_expert_{split}.h5").exists():
            hf_hub_download(repo_id, f"data/craftax_expert_{split}.h5", repo_type="dataset", local_dir=DATA.parent, token=api.token)
        print(split, f"{(DATA / f'craftax_expert_{split}.h5').stat().st_size / 2**30:.2f} GiB")

    # the files must hold the executed actions (noop while asleep): relabel_dataset.ipynb wrote them to the Hub
    relabeled = "import h5py, sys; f = h5py.File(sys.argv[1], 'r'); print('relabel' in f.attrs and 'action_policy' in f)"
    for split in ("train", "val", "test"):
        out = subprocess.run([PYTHON, "-c", relabeled, DATA / f"craftax_expert_{split}.h5"], env=ENV, capture_output=True, text=True)
        assert out.stdout.strip() == "True", f"{split} is not relabeled: run notebooks/relabel_dataset.ipynb (or delete the local copy to download it again)"
    print("the files hold the executed actions (action_policy: the policy's choice)")


def run_spec(experiment, run):
    """A run of experiments/experiments.yaml (config, overrides, ...), read with the venv's yaml."""
    script = ("import json, sys, yaml; spec = yaml.safe_load(open(sys.argv[1])); "
              "print(json.dumps(next(r for r in spec['experiments'][sys.argv[2]]['runs'] if r['name'] == sys.argv[3])))")  # fmt: skip
    out = subprocess.run([PYTHON, "-c", script, REPO / "experiments" / "experiments.yaml", experiment, run],
                         env=ENV, capture_output=True, text=True)  # fmt: skip
    assert out.returncode == 0, f"{experiment} has no run {run} in experiments.yaml: {out.stderr.strip().splitlines()[-1:]}"
    return json.loads(out.stdout)


def fetch(api, repo_id, folder, required=True, download=None):
    """Copy a folder of ckpts/craftax (a run `flat/seed42`, the baseline `random/eval_A_val`) from `runs/` of the Hub repo,
    where the notebooks upload them, to this pod; files the volume has already are kept. Returns the copied Hub paths."""
    if download is None:
        from huggingface_hub import hf_hub_download as download
    key = f"/{folder.strip('/')}/"
    hub = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.startswith("runs/") and key in "/" + f]
    prefixes = sorted({("/" + f).split(key)[0] for f in hub})
    if not hub:
        assert not required, f"{folder} is neither on this pod nor under runs/ in {repo_id}"
        return []
    assert len(prefixes) == 1, f"{folder}: {len(prefixes)} folders under runs/ in {repo_id}: {prefixes}; copy it by hand"
    target, copied = DATA / "ckpts" / "craftax" / folder, []
    for f in hub:
        local = target / ("/" + f).split(key, 1)[1]
        if local.exists():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        # copyfile: the network volume refuses chmod and file times, which shutil.copy and copy2 set
        shutil.copyfile(download(repo_id, f, repo_type="dataset", token=api.token), local)
        copied.append(f)
    print(f"{folder}: {len(copied)} files from the Hub, {len(hub) - len(copied)} were here")
    return copied


def current_probe_report(run) -> bool:
    """Whether the run's probe report is the current suite's; an older one is moved to health/probe_<version>/, so that the
    probe job of run.py writes a new one (`probe_first_version`: before the suite, `probe_suite_v1`: before 2026-10-27)."""
    health = Path(run) / "health"
    report = health / "probe_report.md"
    if not report.exists():
        return False
    text = report.read_text()
    if SUITE_MARK in text:
        return True
    keep = health / ("probe_suite_v1" if text.startswith("# Probe suite") else "probe_first_version")
    keep.mkdir(exist_ok=True)
    for name in ("probe_report.md", "probe_summary.json"):
        if (health / name).exists():
            shutil.move(str(health / name), str(keep / name))
    print(f"the probe report of {run} is from an older suite: moved to {keep}")
    return False


UPLOAD_IGNORE = ["**/lightning_resume/**", "*_epoch_*_object.ckpt", "*_smoke/**", "**/hydra/**", "**/spt/**", "**/spt_volume*/**"]


def upload(api, repo_id, folders=None, reports=("F*.md", "V*.md", "A*.md")):
    """Runs (`folders` of ckpts/craftax, all by default) to runs/ and the comparison reports to reports/ of the Hub repo.
    Left out: the resumable training state, the per-epoch models and the smoke runs."""
    allow = [f"{f.strip('/')}/**" for f in folders] if folders else None
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(DATA / "ckpts" / "craftax"), path_in_repo="runs",
                      allow_patterns=allow, ignore_patterns=UPLOAD_IGNORE)  # fmt: skip
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(DATA / "reports"), path_in_repo="reports",
                      allow_patterns=list(reports))  # fmt: skip
    print(f"done: https://huggingface.co/datasets/{repo_id}/tree/main/runs")
