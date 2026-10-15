"""METRA skill-discovery reward, with the ``phi`` training loop inside it.

Port of ``mjlab_hand.mdp.metra.reward.MetraReward`` for the contacts-only
setting: ``phi`` sees, per fingertip, whether it touches the object and the
touched region's centre and normal in the object frame (5 x 7 = 35-D). Each
env step the env hands in those features; this pays
``(phi(s') - phi(s))^T z``, stores the transition, and every ``update_every``
steps trains ``phi`` on the stored transitions.

With ``update_every`` equal to the runner's ``num_steps_per_env``, ``phi``
changes only on the last step of each rollout, after that step's reward, so
every reward in one PPO batch comes from the same ``phi``. A transition never
straddles an episode boundary: :meth:`reset` invalidates the cached previous
features, so an env's first step of a new episode pays nothing and stores
nothing.
"""

from __future__ import annotations

import torch

from newton_hand.config import MetraCfg
from newton_hand.metra.phi import MetraPhi

CONTACT_FEATURES_PER_PRIMARY = 7
"""[touching, region centre (3), region normal (3)] per fingertip."""


class MetraReward:
    def __init__(self, cfg: MetraCfg, num_envs: int, num_primaries: int, device: torch.device | str):
        self.cfg = cfg
        self._device = torch.device(device)
        self._num_envs = num_envs
        self.skill_dim = cfg.skill_dim
        self.feature_dim = CONTACT_FEATURES_PER_PRIMARY * num_primaries

        # Positions are a few centimetres; scale them to O(1) alongside the flags and normals.
        per_primary = torch.tensor([1.0] + [cfg.position_scale] * 3 + [1.0] * 3)
        self.phi = MetraPhi(
            self.feature_dim,
            self.skill_dim,
            hidden_dims=tuple(cfg.hidden_dims),
            input_scale=per_primary.repeat(num_primaries),
            learning_rate=cfg.learning_rate,
            dual_lambda_init=cfg.dual_lambda_init,
            dual_learning_rate=cfg.dual_learning_rate,
            constraint_eps=cfg.constraint_eps,
            device=self._device,
        )

        capacity = num_envs * max(1, cfg.replay_steps)
        self._capacity = capacity
        self._states = torch.zeros((capacity, self.feature_dim), device=self._device)
        self._next_states = torch.zeros_like(self._states)
        self._skills = torch.zeros((capacity, self.skill_dim), device=self._device)
        self._head = 0
        self._size = 0

        self._prev = torch.zeros((num_envs, self.feature_dim), device=self._device)
        self._prev_valid = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        self._step_count = 0
        self._last_reward = 0.0
        self._last_contact_rate = 0.0
        self.metrics: dict[str, float] = {}
        print(
            f"[METRA] phi input {self.feature_dim}-D ({num_primaries} contacts x "
            f"{CONTACT_FEATURES_PER_PRIMARY}) -> {self.skill_dim}-D skills; replay {capacity} transitions."
        )

    def __call__(self, features: torch.Tensor, skills: torch.Tensor) -> torch.Tensor:
        reward = torch.zeros(self._num_envs, device=self._device)
        finite = torch.isfinite(features).all(dim=-1)
        valid = self._prev_valid & finite
        if valid.any():
            idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            states, next_states, z = self._prev[idx], features[idx], skills[idx]
            reward[idx] = self.phi.reward(states, next_states, z) * self.cfg.reward_scale
            self._push(states, next_states, z)
        self._last_reward = float(reward.mean())
        self._last_contact_rate = float(features[:, ::CONTACT_FEATURES_PER_PRIMARY].mean())

        self._prev.copy_(features)
        self._prev_valid.copy_(finite)

        self._step_count += 1
        if self._should_update():
            self._update_phi()
        return reward

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev_valid[env_ids] = False

    @torch.no_grad()
    def _push(self, states: torch.Tensor, next_states: torch.Tensor, skills: torch.Tensor) -> None:
        n = int(states.shape[0])
        if n >= self._capacity:
            states, next_states, skills = (
                states[-self._capacity :],
                next_states[-self._capacity :],
                skills[-self._capacity :],
            )
            n = self._capacity
        slots = (torch.arange(n, device=self._device) + self._head) % self._capacity
        self._states[slots] = states
        self._next_states[slots] = next_states
        self._skills[slots] = skills
        self._head = (self._head + n) % self._capacity
        self._size = min(self._capacity, self._size + n)

    def _should_update(self) -> bool:
        cfg = self.cfg
        return (
            cfg.train_phi
            and self._step_count > cfg.warmup_steps
            and self._step_count % max(1, cfg.update_every) == 0
            and self._size >= cfg.batch_size
        )

    def _update_phi(self) -> None:
        with torch.inference_mode(False), torch.enable_grad():
            stats: dict[str, float] = {}
            for _ in range(max(1, self.cfg.updates_per_round)):
                idx = torch.randint(0, self._size, (self.cfg.batch_size,), device=self._device)
                stats = self.phi.update(
                    self._states.index_select(0, idx).clone(),
                    self._next_states.index_select(0, idx).clone(),
                    self._skills.index_select(0, idx).clone(),
                )
        self.metrics = {f"metra/{k}": v for k, v in stats.items()}
        self.metrics["metra/reward"] = self._last_reward
        self.metrics["metra/contact_rate"] = self._last_contact_rate
        self.metrics["metra/replay_size"] = float(self._size)

    def state_dict(self) -> dict:
        return {"phi": self.phi.training_state(), "step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self.phi.load_training_state(state["phi"])
        self._step_count = int(state.get("step_count", 0))


def metra_checkpoint_path(policy_checkpoint: str) -> str:
    """``.../model_100.pt`` -> ``.../metra_phi_100.pt``, kept clear of the ``model_*.pt`` pattern."""
    directory, sep, name = policy_checkpoint.rpartition("/")
    return f"{directory}{sep}metra_phi_{name.removeprefix('model_')}"
