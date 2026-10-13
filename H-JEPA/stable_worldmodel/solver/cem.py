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

    Every planned step holds one independent categorical distribution, which
    starts uniform. Each iteration samples `num_samples` action sequences,
    scores them with `model.get_cost`, and moves the distributions toward the
    `topk` cheapest sequences.

    Level 1 plans the environment's discrete actions and passes them to the
    model as one-hot vectors of size `num_actions * action_block`, the encoding
    the world model is trained on. Levels above 1 plan the codes of the VQ
    action encoder (`model.get_action_codebook()`) and pass the code vectors.
    `solve` returns the best sequence as `action_indices`; `actions` holds the
    indices at level 1 and the code vectors above it.

    Args:
        model: World model implementing the Costable protocol.
        num_samples: Action sequences sampled per iteration and environment.
        n_iters: CEM iterations per solve.
        topk: Number of elite sequences used to refit the distributions.
        batch_size: Number of environments processed in parallel.
        alpha: Weight of the elite frequencies in the distribution update.
        min_prob: Probability floor, so no action is ruled out for good.
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
        min_prob: float = 0.01,
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
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self._n_envs = None
        self._num_actions = None
        self._code_dim = None
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
        if self.level == 1:
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                num_actions = int(np.max(action_space.nvec))
            elif isinstance(action_space, gym.spaces.Discrete):
                num_actions = int(action_space.n)
            else:
                raise TypeError(
                    f'Discrete action space required, got {action_space}.'
                )
        else:
            codebook = self.model.get_action_codebook()
            if codebook is None:
                raise ValueError(
                    f'Level {self.level} has no VQ action codebook; CEM needs '
                    'discrete actions (use the gradient solver otherwise).'
                )
            num_actions, self._code_dim = codebook.shape
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._num_actions = int(num_actions)

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Action size fed to the model, including action_block."""
        size = self._num_actions if self.level == 1 else self._code_dim
        return size * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def _decode(self, indices: torch.Tensor) -> torch.Tensor:
        """(b, S, H, block) indices -> (b, S, H, block * action size)."""
        if self.level == 1:
            return F.one_hot(indices, self._num_actions).float().flatten(-2)
        codebook = self.model.get_action_codebook().detach()
        return codebook.to(indices.device).float()[indices].flatten(-2)

    def _update_probs(
        self,
        probs: torch.Tensor,
        samples: torch.Tensor,
        costs: torch.Tensor,
    ) -> torch.Tensor:
        """Refit the categorical distributions to the elite samples."""
        elite = costs.topk(self.topk, dim=1, largest=False).indices
        index = elite[:, :, None, None].expand(-1, -1, *samples.shape[2:])
        elite_samples = samples.gather(1, index)  # (b, k, H, block)
        freq = F.one_hot(elite_samples, self._num_actions).float().mean(1)
        probs = self.alpha * freq + (1 - self.alpha) * probs
        probs = probs.clamp_min(self.min_prob)
        return probs / probs.sum(-1, keepdim=True)

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
            )

        batch_size = self.batch_size or self.n_envs
        best_actions_list = []
        per_env_time = []
        for start in range(0, self.n_envs, batch_size):
            end = min(start + batch_size, self.n_envs)
            bs = end - start
            batch_t0 = time.time()
            expanded = expand_info_dict_for_samples(
                info_dict, start, end, self.num_samples, self.device
            )
            shape = (bs, planning_horizon, block, self._num_actions)
            probs = torch.full(
                shape, 1.0 / self._num_actions, device=self.device
            )
            best = torch.zeros(
                shape[:-1], dtype=torch.long, device=self.device
            )
            best_cost = torch.full((bs,), float('inf'), device=self.device)

            for it in range(self.n_iters):
                flat = probs.reshape(-1, self._num_actions)
                samples = torch.multinomial(
                    flat,
                    self.num_samples,
                    replacement=True,
                    generator=self.torch_gen,
                )  # (bs * H * block, S)
                samples = samples.reshape(
                    bs, planning_horizon, block, self.num_samples
                ).permute(0, 3, 1, 2)  # (bs, S, H, block)
                if (
                    it > 0
                ):  # keep the incumbent: the best cost never gets worse
                    samples[:, 0] = best

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
                probs = self._update_probs(probs, samples, costs)

            best_actions_list.append(best.cpu())
            per_env_time.extend([time.time() - batch_t0] * bs)

        indices = torch.cat(best_actions_list, dim=0)
        outputs = {'action_indices': indices}
        outputs['actions'] = (
            indices if self.level == 1 else self._decode(indices).cpu()
        )
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
                'n_iters': [self.n_iters] * self.n_envs,
                'wall_clock': end_time - start_time,
                'wall_clock_per_env': per_env_time,
            }
        )
        return outputs
