"""Cross-entropy method over discrete actions or action codes for planning."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .solver import (
    Costable,
    expand_info_dict_for_samples,
    get_planning_horizon,
)


class CategoricalCEMSolver(torch.nn.Module):
    """CEM over a sequence of categorical actions (forward passes only).

    Every planned step holds a few independent categorical distributions
    ("slots"), which start uniform. Each iteration samples `num_samples` action
    sequences, scores them with `model.get_cost`, and moves the distributions
    toward the `topk` cheapest sequences.

    Level 1 plans the environment's discrete actions: `action_block` slots of
    `num_actions` classes, passed to the model as one-hot vectors of size
    `num_actions * action_block`, the encoding the world model is trained on.

    Levels above 1 plan the action tokens of the VQ action encoder
    (`model.get_action_spec()`): one slot per code group, plus one slot for the
    step duration k when the model has a band of durations. The model receives
    the code vectors of the groups, followed by the one-hot of k. These levels
    need `action_block == 1`.

    `solve` returns the best sequence as `action_indices` (one index per slot);
    `actions` holds the indices at level 1 and the token vectors above it, and
    `durations` the chosen k of every step when the model has durations.

    `allowed_durations` limits what the planner may choose for k. A free k lets
    the planner pick long jumps through poorly modelled territory, so the safe
    default pins every step to the model's default duration (the fixed stride).
    'all' makes k a free choice; a list of ints allows that subset.

    Args:
        model: World model implementing the Costable protocol.
        num_samples: Action sequences sampled per iteration and environment.
        n_iters: CEM iterations per solve.
        topk: Number of elite sequences used to refit the distributions.
        batch_size: Number of environments processed in parallel.
        alpha: Weight of the elite frequencies in the distribution update.
        min_prob: Probability mass of each slot that is spread uniformly over
            its classes, so no class is ruled out for good. A mass, not a
            per-class floor: a per-class floor grows with the number of classes
            (0.01 per class caps a 64-code slot at 0.55), so large slots never
            converged.
        proposal_fraction: Share of the samples of every iteration that are
            whole code sequences from the proposal bank (`set_proposal`), the
            sequences of the training data. They keep the joint structure of
            codes (which codes occur together and in what order) that the
            independent per-slot distributions lose, and keep candidates on the
            data support; without a bank this has no effect. Upper levels only.
        allowed_durations: 'default' (pin k to the default duration), 'all',
            or a list of allowed durations. Only used by models with durations.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Costable,
        num_samples: int = 300,
        n_iters: int = 10,
        topk: int = 30,
        batch_size: int | None = None,
        alpha: float = 0.5,
        min_prob: float = 0.05,
        proposal_fraction: float = 0.0,
        allowed_durations: str | list[int] = 'default',
        device: str | torch.device = 'cpu',
        seed: int = 1234,
    ) -> None:
        super().__init__()
        if not 0 < topk <= num_samples:
            raise ValueError(
                f'topk must be in (0, num_samples], got {topk} / {num_samples}'
            )
        self.model = model
        self.num_samples = int(num_samples)
        self.n_iters = int(n_iters)
        self.topk = int(topk)
        self.batch_size = batch_size
        self.alpha = float(alpha)
        self.min_prob = float(min_prob)
        if not 0.0 <= float(proposal_fraction) < 1.0:
            raise ValueError(
                f'proposal_fraction must be in [0, 1), got {proposal_fraction}'
            )
        self.proposal_fraction = float(proposal_fraction)
        self._bank = None  # (M, H_bank, groups) code sequences, or None
        # a list from a Hydra config is a ListConfig: keep plain values (it ends up in JSON records)
        self.allowed_durations = (
            allowed_durations
            if isinstance(allowed_durations, str)
            else [int(k) for k in allowed_durations]
        )
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self._n_envs = None
        self._slot_sizes = None
        self._duration_classes = None  # allowed classes of the duration slot
        self._config = None
        # One entry per solve(), same layout as GradientSolver.solve_records.
        self.solve_records = []

    def configure(
        self,
        *,
        action_space: gym.Space,
        n_envs: int,
        config: Any,
        level: int = 1,
    ) -> None:
        """Configure the solver with environment specifications."""
        self.level = int(level)
        block = int(getattr(config, 'action_block', 1))
        if self.level == 1:
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                num_actions = int(np.max(action_space.nvec))
            elif isinstance(action_space, gym.spaces.Discrete):
                num_actions = int(action_space.n)
            else:
                raise TypeError(
                    f'Discrete action space required, got {action_space}.'
                )
            slot_sizes = (num_actions,) * block
        else:
            spec = self.model.get_action_spec()
            if spec is None:
                raise ValueError(
                    f'Level {self.level} has no VQ action encoder; CEM needs '
                    'discrete actions (use the gradient solver otherwise).'
                )
            if block != 1:
                raise ValueError(
                    f'VQ levels need action_block 1, got {block}.'
                )
            groups, num_codes, _ = spec.codebook.shape
            slot_sizes = (num_codes,) * groups
            if spec.durations is not None:
                slot_sizes += (len(spec.durations),)
                self._duration_classes = self._resolve_durations(spec)
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._slot_sizes = slot_sizes

    def set_proposal(self, bank: torch.Tensor | None) -> None:
        """Code sequences of the data, (M, H_bank, groups) integer indices, or None to remove them."""
        if bank is None:
            self._bank = None
            return
        bank = torch.as_tensor(bank, dtype=torch.long, device=self.device)
        if bank.ndim != 3 or bank.shape[0] == 0:
            raise ValueError(
                f'proposal bank must be (M, H, groups) with M > 0, got {tuple(bank.shape)}'
            )
        self._bank = bank

    def _resolve_durations(self, spec) -> list[int]:
        """Classes of the duration slot the planner may choose from."""
        mode = self.allowed_durations
        if mode == 'all':
            return list(range(len(spec.durations)))
        if mode == 'default':
            if spec.default_duration is None:
                raise ValueError(
                    'The model has no default duration to pin k to.'
                )
            mode = [spec.default_duration]
        unknown = [k for k in mode if k not in spec.durations]
        if unknown or not mode:
            raise ValueError(
                f'allowed_durations {mode} not within {spec.durations}'
            )
        return [spec.durations.index(k) for k in mode]

    def _mix_in_proposals(self, samples, bs: int, horizon: int) -> None:
        """Replace the code slots of some samples by sequences of the proposal bank (in place).

        The incumbent (sample 0) stays; a duration slot keeps what the distribution drew.
        Positions beyond the bank's horizon keep the distribution's draw.
        """
        if self._bank is None or self.proposal_fraction <= 0:
            return
        k = min(self.num_samples - 1, round(self.proposal_fraction * self.num_samples))
        if k <= 0:
            return
        groups = self._bank.shape[2]
        if self.level == 1 or groups != self._n_code_slots():
            raise ValueError(
                f'the proposal bank has {groups} code slots, the model has {self._n_code_slots()}'
            )
        pick = torch.randint(
            0, self._bank.shape[0], (bs, k), generator=self.torch_gen, device=self.device
        )
        length = min(horizon, self._bank.shape[1])
        samples[:, 1 : 1 + k, :length, :groups] = self._bank[pick][:, :, :length]

    def _n_code_slots(self) -> int:
        """Slots that are codes: all but a trailing duration slot."""
        return len(self._slot_sizes) - (1 if self._duration_classes is not None else 0)

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Size of the action vector fed to the model, incl. action_block."""
        if self.level == 1:
            return sum(self._slot_sizes)
        spec = self.model.get_action_spec()
        groups, _, group_dim = spec.codebook.shape
        n_durations = 0 if spec.durations is None else len(spec.durations)
        return groups * group_dim + n_durations

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def _decode(self, indices: torch.Tensor) -> torch.Tensor:
        """(..., slots) indices -> (..., action vector) the model consumes."""
        if self.level == 1:
            num_actions = self._slot_sizes[0]
            return F.one_hot(indices, num_actions).float().flatten(-2)
        spec = self.model.get_action_spec()
        codebook = spec.codebook.detach().to(indices.device).float()
        groups = codebook.shape[0]
        which = torch.arange(groups, device=indices.device)
        vector = codebook[which, indices[..., :groups]].flatten(-2)
        if spec.durations is not None:
            onehot = F.one_hot(indices[..., groups], len(spec.durations))
            vector = torch.cat([vector, onehot.float()], dim=-1)
        return vector

    def _update_probs(
        self,
        probs: torch.Tensor,
        samples: torch.Tensor,
        costs: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Refit the categorical distributions to the elite samples."""
        elite = costs.topk(self.topk, dim=1, largest=False).indices
        index = elite[:, :, None, None].expand(-1, -1, *samples.shape[2:])
        elite_samples = samples.gather(1, index)  # (b, k, H, slots)
        freq = F.one_hot(elite_samples, probs.shape[-1]).float().mean(1)
        probs = self.alpha * freq + (1 - self.alpha) * probs
        # mix in uniform mass over a slot's classes; the others stay at zero
        uniform = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
        probs = (1 - self.min_prob) * probs + self.min_prob * uniform
        return probs / probs.sum(-1, keepdim=True)

    @staticmethod
    def _entropy(probs: torch.Tensor, valid: torch.Tensor) -> float:
        """Mean normalised entropy: 1 is uniform, 0 is converged.

        Slots with a single allowed class (a pinned duration) are left out.
        """
        classes = valid.sum(-1)  # (slots,)
        keep = classes > 1
        if not keep.any():
            return 0.0
        entropy = -(probs * (probs + 1e-12).log()).sum(-1)  # (b, H, slots)
        return (entropy[..., keep] / classes[keep].float().log()).mean().item()

    @torch.inference_mode()
    def _rollout_final_actions(
        self, info_dict: dict, actions: torch.Tensor
    ) -> dict:
        """Roll out the model once with the chosen actions."""
        candidates = self._decode(actions.to(self.device).unsqueeze(1))
        info_dict = expand_info_dict_for_samples(
            info_dict, 0, self.n_envs, 1, self.device
        )
        rollout_info = self.model.rollout(info_dict, candidates)

        predictions: dict[str, torch.Tensor] = {}
        for k, v in rollout_info.items():
            if not (k.startswith('predicted') and torch.is_tensor(v)):
                continue
            value = v.detach().cpu()
            if value.ndim > 1 and value.shape[1] == 1:
                value = value[:, 0]
            predictions[k] = value
        return predictions

    @torch.inference_mode()
    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        planning_horizon: int | None = None,
        steps_taken: int | None = None,
        eval_budget: int | None = None,
    ) -> dict:
        """Plan with CEM. `init_action` is unused: there is no warm start."""
        start_time = time.time()
        cost_last_n = max(1, int(self._config.cost_last_n))
        intermediate_cost_weight = float(self._config.intermediate_cost_weight)
        block = int(self._config.action_block)
        if planning_horizon is None:
            planning_horizon = get_planning_horizon(
                configured_horizon=self.horizon,
                replanning_interval=int(self._config.receding_horizon) * block,
                steps_taken=steps_taken,
                eval_budget=eval_budget,
                schedule=getattr(
                    self._config, 'horizon_schedule', 'decreasing'
                ),
            )

        n_slots, max_classes = len(self._slot_sizes), max(self._slot_sizes)
        sizes = torch.tensor(self._slot_sizes, device=self.device)
        valid = torch.arange(max_classes, device=self.device) < sizes[:, None]
        if self._duration_classes is not None:  # the duration is the last slot
            allowed = torch.zeros(
                max_classes, dtype=torch.bool, device=self.device
            )
            allowed[self._duration_classes] = True
            valid[-1] &= allowed
        uniform = valid.float() / valid.sum(-1, keepdim=True)

        batch_size = self.batch_size or self.n_envs
        best_actions_list = []
        curves, entropies = [], []
        per_env_time = []
        for start in range(0, self.n_envs, batch_size):
            end = min(start + batch_size, self.n_envs)
            bs = end - start
            batch_t0 = time.time()
            expanded = expand_info_dict_for_samples(
                info_dict, start, end, self.num_samples, self.device
            )
            probs = uniform.expand(bs, planning_horizon, -1, -1).clone()
            best = torch.zeros(
                (bs, planning_horizon, n_slots),
                dtype=torch.long,
                device=self.device,
            )
            best_cost = torch.full((bs,), float('inf'), device=self.device)
            curve = []  # best cost so far, after every iteration

            for it in range(self.n_iters):
                samples = torch.multinomial(
                    probs.reshape(-1, max_classes),
                    self.num_samples,
                    replacement=True,
                    generator=self.torch_gen,
                )  # (bs * H * slots, S)
                samples = samples.reshape(
                    bs, planning_horizon, n_slots, self.num_samples
                ).permute(0, 3, 1, 2)  # (bs, S, H, slots)
                if (
                    it > 0
                ):  # keep the incumbent: the best cost never gets worse
                    samples[:, 0] = best
                self._mix_in_proposals(samples, bs, planning_horizon)

                current = expanded.copy()
                current['cost_last_n'] = cost_last_n
                current['intermediate_cost_weight'] = intermediate_cost_weight
                costs = self.model.get_cost(current, self._decode(samples))
                for k in ('embed_0', 'goal_embed_0'):  # encode once per solve
                    expanded.setdefault(k, current[k])
                assert costs.shape == (bs, self.num_samples), (
                    f'Cost should be ({bs}, {self.num_samples}), '
                    f'got {tuple(costs.shape)}'
                )

                min_cost, argmin = costs.min(dim=1)
                better = min_cost < best_cost
                best_cost = torch.where(better, min_cost, best_cost)
                best = torch.where(
                    better[:, None, None],
                    samples[torch.arange(bs), argmin],
                    best,
                )
                probs = self._update_probs(probs, samples, costs, valid)
                curve.append(best_cost.clone())

            best_actions_list.append(best.cpu())
            curves.append(torch.stack(curve, dim=1).cpu())  # (bs, n_iters)
            entropies.append(self._entropy(probs, valid))
            per_env_time.extend([time.time() - batch_t0] * bs)

        indices = torch.cat(best_actions_list, dim=0)  # (B, H, slots)
        curve = torch.cat(curves, dim=0)  # (B, n_iters), best cost so far
        outputs = {'action_indices': indices, 'cost': curve[:, -1]}
        outputs['actions'] = (
            indices if self.level == 1 else self._decode(indices).cpu()
        )
        if self.level > 1:
            spec = self.model.get_action_spec()
            if spec.durations is not None:
                table = torch.tensor(spec.durations)
                outputs['durations'] = table[indices[..., -1]]
        outputs['predictions'] = self._rollout_final_actions(
            info_dict, indices
        )
        end_time = time.time()
        self.solve_records.append(
            {
                'level': self.level,
                'solver': 'categorical_cem',
                'backward': False,  # forward passes only, unlike GD
                'horizon': int(planning_horizon),
                'num_samples': int(self.num_samples),
                'allowed_durations': self.allowed_durations,
                'proposal_fraction': self.proposal_fraction if self._bank is not None else 0.0,
                'n_iters': [self.n_iters] * self.n_envs,
                'first_cost': curve[:, 0].tolist(),
                'best_cost': curve[:, -1].tolist(),
                'final_entropy': float(sum(entropies) / len(entropies)),
                'wall_clock': end_time - start_time,
                'wall_clock_per_env': per_env_time,
            }
        )
        return outputs
