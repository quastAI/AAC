"""Lightning callback: live, per-epoch and end-of-training health of a world model.

Live (every `live_every` steps, from the training batch): representation, the
copy and shuffled-action baselines of the predictor, codebook use, IDM
accuracy, raw gradient norms per level, loss spikes.

Per epoch and at the end: an evaluation pass over long validation clips, which
adds open-loop rollout drift, the cost geometry (latent distance against time
gap) and IDM accuracy on held-out worlds. At the end it also writes
`health/report.md` and `health/summary.json` in the run directory, with the
findings of `health.diagnose_training`.

Everything is logged under `health/...`; findings are printed as they appear.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger as logging
from torch.utils.data import DataLoader, Subset

import health
from data import build_hdf5_dataset


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _average(dicts: list[dict]) -> dict:
    """Mean of the numeric entries over a list of dicts."""
    keys = {k for d in dicts for k in d}
    return {k: _mean([d.get(k) for d in dicts]) for k in keys}


def build_health_loader(
    cfg, cache_dir, transform, *, frames, batch_size, max_clips, seed=0, name=None, columns=("pixels", "action")
):
    """Long clips (level-1 frames only; the model builds the upper levels).

    `name` is the dataset (default: the validation set), `columns` what to load.

    Falls back to shorter clips when the validation episodes are too short, and
    returns None when there are no clips at all.
    """
    base = {
        k: v
        for k, v in cfg.data.dataset.items()
        if k not in ("val_name", "total_transitions", "val_total_transitions", "keys_to_merge")
    }
    base = {k: (dict(v) if hasattr(v, "items") else v) for k, v in base.items()}
    base = {k: v for k, v in base.items() if not (k.startswith("level") and k != "level1")}
    base.pop("terminal_pad", None)  # the long validation clips are real frames only
    base["name"] = name or cfg.data.dataset.val_name
    base["keys_to_load"] = list(columns)
    base["keys_to_cache"] = ["action"]
    frameskip = int(base["level1"]["frameskip"])
    while frames >= 4:
        base["level1"] = {"frameskip": frameskip, "num_steps": frames}
        dataset = build_hdf5_dataset(dict(base), cache_dir=cache_dir)
        if len(dataset) >= 2:
            dataset.transform = transform
            order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
            subset = Subset(dataset, order[:max_clips].tolist())
            return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0), frames
        frames //= 2
    return None, 0


class TrainingHealthCallback(Callback):
    def __init__(
        self, cfg, run_dir, *, health_loader=None, normalize_fn=None, thresholds=None, cache_dir=None, transform=None
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.transform = transform
        self.cfg = cfg
        self.hcfg = cfg.health
        self.run_dir = Path(run_dir) / "health"
        self.loader = health_loader
        self.normalize = normalize_fn or (lambda batch: None)
        self.thr = thresholds or health.Thresholds(**dict(self.hcfg.get("thresholds", {}) or {}))
        self.num_levels = int(cfg.num_levels)
        self.plan_history = int(self.hcfg.get("plan_history", 1))
        self.generator = torch.Generator().manual_seed(0)

        self.live: dict[int, dict] = {n: {} for n in range(1, self.num_levels + 1)}
        self.grad_norms = {n: deque(maxlen=200) for n in self.live}
        self.explosions = {n: 0 for n in self.live}
        self.losses = deque(maxlen=200)
        self.spikes = 0
        self.nonfinite = False
        self.epoch_history: list[dict] = []
        self.val_losses: list[float] = []
        self.train_loss = None
        self._seen: set = set()
        self._critical_streak = 0

    # ---------------------------------------------------------------- helpers

    def _jepa(self, pl_module, n):
        return pl_module.model.get_level(n)

    def _history(self, n) -> int:
        return int(self.cfg[f"level{n}"].wm.history_size)

    def _log(self, trainer, metrics: dict, step=None):
        metrics = {k: v for k, v in metrics.items() if v is not None}
        if trainer.is_global_zero and trainer.logger is not None and metrics:
            trainer.logger.log_metrics(metrics, step=trainer.global_step if step is None else step)

    @staticmethod
    def _flat(prefix: str, d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(TrainingHealthCallback._flat(f"{prefix}/{k}", v))
            elif isinstance(v, (int, float)) and v is not True and v is not False:
                out[f"{prefix}/{k}"] = float(v)
        return out

    def _summary(self, levels: dict) -> dict:
        return {"levels": levels, "plan_history": self.plan_history, "training": self._training_summary()}

    def _training_summary(self) -> dict:
        return {
            "epochs": len(self.val_losses),
            "val_first": self.val_losses[0] if self.val_losses else None,
            "val_last": self.val_losses[-1] if self.val_losses else None,
            "train_last": self.train_loss,
            "spikes": self.spikes,
            "nonfinite": self.nonfinite,
        }

    def _report_findings(self, findings, tag):
        """Print each finding the first time it appears, and a one-line summary."""
        new = [f for f in findings if (f.code, f.level) not in self._seen]
        for f in new:
            self._seen.add((f.code, f.level))
            (logging.error if f.severity == "critical" else logging.warning)(f"[health:{tag}] {f.line()}")
        return new

    # ------------------------------------------------------------------- live

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        every = int(self.hcfg.live_every)
        loss = outputs.get("loss") if isinstance(outputs, dict) else None
        if loss is not None:
            value = float(loss.detach())
            if not torch.isfinite(torch.tensor(value)):
                self.nonfinite = True
            if len(self.losses) >= 50:
                median = sorted(self.losses)[len(self.losses) // 2]
                if value > self.thr.spike_factor * max(median, 1e-12):
                    self.spikes += 1
            self.losses.append(value)
        if trainer.global_step % every or not isinstance(outputs, dict):
            return

        metrics = {}
        for n in range(1, self.num_levels + 1):
            emb, act = outputs.get(f"embed_{n}"), outputs.get(f"action_{n}")
            if emb is None:
                continue
            # the training forward runs under autocast: these are bf16, the metrics run in fp32
            emb, act = emb.detach().float(), act.detach().float()
            jepa = self._jepa(pl_module, n)
            entry = {"rep": health.representation_summary(emb)}
            entry["dyn"] = health.teacher_forced_metrics(jepa, emb, act, self._history(n), self.generator)
            vq = getattr(jepa.action_encoder, "num_codes", None) and health.vq_metrics(jepa.action_encoder)
            if vq:
                entry["vq"] = vq
            accuracy = outputs.get(f"idm_accuracy_loss_level{n}")
            if accuracy is not None and getattr(jepa, "idm_loss_fn", None) is not None:
                chance = self._idm_chance(outputs, n, jepa)
                accuracy = float(accuracy)
                entry["idm"] = {"accuracy": accuracy, "chance": chance, "lift": health.idm_lift(accuracy, chance)}
            entry["grad"] = self._grad_entry(n)
            self.live[n] = entry
            metrics.update(self._flat(f"health/level{n}", {k: v for k, v in entry.items() if k != "grad"}))
            metrics.update(self._flat(f"health/level{n}/grad", entry["grad"]))
        self._log(trainer, metrics)

        if trainer.global_step >= int(self.hcfg.warmup_steps):
            findings = health.diagnose_training(self._summary(self.live), self.thr)
            self._report_findings(findings, f"step {trainer.global_step}")

    def _idm_chance(self, outputs, n, jepa) -> float:
        """Accuracy of the best constant guess on this batch (see health.majority_baseline)."""
        target = health_idm_target(outputs, n, self.cfg)
        classes = jepa.idm_loss_fn.num_classes
        return 1.0 / classes if target is None else health.majority_baseline(target.cpu(), classes)

    def _grad_entry(self, n) -> dict:
        norms = self.grad_norms[n]
        return {
            "last": norms[-1] if norms else None,
            "median": sorted(norms)[len(norms) // 2] if norms else None,
            "explosions": self.explosions[n],
        }

    def on_after_backward(self, trainer, pl_module):
        every = int(self.hcfg.live_every)
        if trainer.global_step % every:
            return
        for n in range(1, self.num_levels + 1):
            grads = [p.grad.detach().float().norm() for p in self._jepa(pl_module, n).parameters() if p.grad is not None]
            if not grads:
                continue
            norm = torch.stack(grads).norm().item()
            norms = self.grad_norms[n]
            if len(norms) >= 20:
                median = sorted(norms)[len(norms) // 2]
                if norm > self.thr.grad_explosion_factor * max(median, 1e-12):
                    self.explosions[n] += 1
            norms.append(norm)

    # ------------------------------------------------------- evaluation passes

    @torch.no_grad()
    def evaluate(self, pl_module, max_batches: int) -> dict:
        """Metrics per level over long validation clips; empty if there are none."""
        if self.loader is None:
            return {}
        model = pl_module.model
        was_training = model.training
        model.eval()
        jepas = [model.get_level(n) for n in range(1, self.num_levels + 1)]
        saved = [getattr(j, "target_length", None) for j in jepas]
        for j in jepas:
            j.target_length = None  # keep the whole sequence of every level
        embs = {n: [] for n in range(1, self.num_levels + 1)}
        acts = {n: [] for n in embs}
        codes = {n: [] for n in embs}
        idm = {n: [] for n in embs}
        idm_targets = {n: [] for n in embs}
        try:
            device = next(model.parameters()).device
            for i, batch in enumerate(self.loader):
                if i >= max_batches:
                    break
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                self.normalize(batch)
                out = model.encode_hierarchical(batch)
                for n in embs:
                    embs[n].append(out[f"embed_{n}"].float().cpu())
                    acts[n].append(out[f"action_{n}"].float().cpu())
                    if f"action_index_{n}" in out:
                        codes[n].append(out[f"action_index_{n}"].cpu())
                    jepa = jepas[n - 1]
                    loss_fn = getattr(jepa, "idm_loss_fn", None)
                    if loss_fn is not None:
                        target = health_idm_target(out, n, self.cfg)
                        if target is not None:
                            loss_fn(out[f"embed_{n}"], target, out.get(f"duration_index_{n}"))
                            idm[n].append(float(loss_fn.last_accuracy))
                            idm_targets[n].append(target.cpu())
        finally:
            for j, t in zip(jepas, saved):
                j.target_length = t
            model.train(was_training)

        levels = {}
        for n in embs:
            if not embs[n]:
                continue
            emb, act = torch.cat(embs[n]), torch.cat(acts[n])
            jepa = jepas[n - 1]
            device = next(jepa.parameters()).device
            emb_d, act_d = emb.to(device), act.to(device)
            history = self._history(n)
            entry = {"rep": health.representation_summary(emb)}
            windows = [
                health.teacher_forced_metrics(jepa, emb_d[:, s : s + history + 1], act_d[:, s : s + history + 1], history, self.generator)
                for s in range(0, max(emb.shape[1] - history, 1), history + 1)
            ][:4]
            entry["dyn"] = _average([w for w in windows if w])
            horizon = int(self.hcfg.drift_horizon)
            entry["drift"] = {
                h: health.drift_curve(jepa, emb_d, act_d, h, horizon) for h in sorted({self.plan_history, history})
            }
            gaps = [g for g in (1, 2, 4, 8, 16, 32, 64) if g < emb.shape[1]]
            entry["geometry"] = health.cost_geometry(emb, gaps)
            vq_module = jepa.action_encoder
            if codes[n] and getattr(vq_module, "num_codes", None):
                entry["vq"] = health.vq_metrics(vq_module, torch.cat(codes[n]))
            if idm[n]:
                chance = health.majority_baseline(torch.cat(idm_targets[n]), jepa.idm_loss_fn.num_classes)
                accuracy = _mean(idm[n])
                entry["idm"] = {"accuracy": accuracy, "chance": chance, "lift": health.idm_lift(accuracy, chance)}
            entry["grad"] = self._grad_entry(n)
            levels[n] = entry
        return levels

    # ----------------------------------------------------------------- epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        for key in ("validate/loss_epoch", "validate/loss"):
            if key in metrics:
                self.val_losses.append(float(metrics[key]))
                break
        for key in ("fit/loss_epoch", "train/loss_epoch", "fit/loss"):
            if key in metrics:
                self.train_loss = float(metrics[key])
                break

        levels = self.evaluate(pl_module, int(self.hcfg.epoch_batches))
        if not levels:
            return
        summary = self._summary(levels)
        findings = health.diagnose_training(summary, self.thr)
        epoch = trainer.current_epoch
        flat = {}
        for n, entry in levels.items():
            flat.update(self._flat(f"health_val/level{n}", {k: v for k, v in entry.items() if k not in ("drift", "grad")}))
            for hist, curve in entry.get("drift", {}).items():
                if curve:
                    flat[f"health_val/level{n}/drift_h{hist}/useful_horizon"] = float(curve["useful_horizon"])
                    flat[f"health_val/level{n}/drift_h{hist}/ratio_last"] = curve["mse"][-1] / (curve["copy"][-1] + health.EPS)
        flat["health_val/critical"] = float(sum(f.severity == "critical" for f in findings))
        flat["health_val/warnings"] = float(sum(f.severity == "warn" for f in findings))
        self._log(trainer, flat)
        self._print_epoch(epoch, levels, findings)
        self._report_findings(findings, f"epoch {epoch}")

        if trainer.is_global_zero:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / f"epoch_{epoch:04d}.json").write_text(
                json.dumps({"summary": health.to_jsonable(summary), "findings": health.findings_to_json(findings)}, indent=1)
            )
        self.epoch_history.append(health.to_jsonable(summary))

        patience = int(self.hcfg.get("stop_after_critical_epochs", 0))
        self._critical_streak = self._critical_streak + 1 if health.has_critical(findings) else 0
        if patience and self._critical_streak >= patience:
            logging.error(f"[health] critical findings for {patience} epochs in a row: stopping training")
            trainer.should_stop = True

    def _print_epoch(self, epoch, levels, findings):
        for n, m in sorted(levels.items()):
            rep, dyn, vq, idm = (m.get(k) or {} for k in ("rep", "dyn", "vq", "idm"))
            drift = (m.get("drift") or {}).get(self.plan_history) or {}
            geo = m.get("geometry") or {}
            logging.info(
                f"[health epoch {epoch}] level {n}: erank {health._fmt(rep.get('erank'))}/{rep.get('dim')} "
                f"pred/copy {health._fmt(dyn.get('pred_over_copy'))} action-lift {health._fmt(dyn.get('action_lift'))} "
                f"useful-horizon {health._fmt(drift.get('useful_horizon'))} saturation-gap {health._fmt(geo.get('saturation_gap'))} "
                f"vq-perplexity {health._fmt(vq.get('perplexity'))} idm-acc {health._fmt(idm.get('accuracy'))}"
            )
        n_crit = sum(f.severity == "critical" for f in findings)
        n_warn = sum(f.severity == "warn" for f in findings)
        logging.info(f"[health epoch {epoch}] {n_crit} critical, {n_warn} warnings")

    # -------------------------------------------------------------------- end

    def on_train_end(self, trainer, pl_module):
        levels = self.evaluate(pl_module, int(self.hcfg.final_batches)) or {
            n: dict(entry) for n, entry in self.live.items() if entry
        }
        summary = self._summary(levels)
        findings = health.diagnose_training(summary, self.thr)
        if trainer.is_global_zero:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "report.md").write_text(health.render_training_report(summary, findings))
            (self.run_dir / "summary.json").write_text(
                json.dumps({"summary": health.to_jsonable(summary), "findings": health.findings_to_json(findings),
                            "epochs": self.epoch_history}, indent=1)
            )
            logging.info(f"[health] report written to {self.run_dir / 'report.md'}")
        logging.info("[health] final findings:\n" + (health.findings_block(findings)))
        if self.hcfg.get("probe_at_end", False) and trainer.is_global_zero:
            self._probe(pl_module)

    def _probe(self, pl_module):
        """Probe every level on held-out worlds; skipped when the data lacks the target columns."""
        import h5py

        from probe_eval import COLUMNS, run_probing

        path = Path(self.cache_dir or ".") / f"{self.cfg.data.dataset.val_name}.h5"
        try:
            with h5py.File(path, "r") as f:
                missing = [c for c in COLUMNS if c not in f]
        except OSError:
            missing = ["(dataset not readable)"]
        if missing:
            logging.warning(f"[health] probing skipped, the validation data lacks columns {missing}; collect it with the current collector")
            return
        try:
            device = next(pl_module.model.parameters()).device
            summary, findings = run_probing(
                pl_module.model, self.cfg, self.cache_dir, transform=self.transform, normalize=self.normalize,
                device=device, frames=int(self.hcfg.long_clip_frames), max_clips=int(self.hcfg.get("probe_clips", 256)),
            )  # fmt: skip
        except Exception as error:  # a failed probe must not lose a finished training run
            logging.warning(f"[health] probing failed: {error!r}")
            return
        (self.run_dir / "probe_report.md").write_text(health.render_probing_report(summary, findings))
        (self.run_dir / "probe_summary.json").write_text(
            json.dumps({"summary": health.to_jsonable(summary), "findings": health.findings_to_json(findings)}, indent=1)
        )
        logging.info("[health] probing findings:\n" + health.findings_block(findings))


def health_idm_target(output: dict, level: int, cfg):
    """The class indices the IDM of this level must predict (see hjepa_utils)."""
    from hjepa_utils import _discrete_idm_target

    return _discrete_idm_target(output, level, cfg[f"level{level}"])
