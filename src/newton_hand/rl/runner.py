"""rsl-rl glue: the ``VecEnv`` wrapper and a runner that checkpoints METRA's ``phi``."""

from __future__ import annotations

import os
from typing import Any

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from newton_hand.envs.grasp_env import NUM_ACTIONS, GraspEnv
from newton_hand.metra.reward import metra_checkpoint_path


class RslRlVecEnv(VecEnv):
    """Same contract as mjlab's ``RslRlVecEnvWrapper``: clipped actions, ``time_outs`` in extras."""

    def __init__(self, env: GraspEnv, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.num_envs
        self.num_actions = getattr(env, "num_actions", NUM_ACTIONS)
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.cfg = env.cfg

    @property
    def unwrapped(self) -> GraspEnv:
        return self.env

    @property
    def episode_length_buf(self) -> torch.Tensor:  # type: ignore[override]
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        return TensorDict(self.env.obs, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs, rew, terminated, time_outs, extras = self.env.step(actions)
        dones = (terminated | time_outs).to(dtype=torch.long)
        return TensorDict(obs, batch_size=[self.num_envs]), rew, dones, extras


class LoggingOnPolicyRunner(OnPolicyRunner):
    """Drops the empty ``extras["log"]`` of steps without resets: rsl-rl takes
    the metric keys from the first entry of an iteration, so an empty first
    entry silently drops every episode metric (mjlab_hand's
    ``AmpOnPolicyRunner`` fix)."""

    def __init__(self, env: RslRlVecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        original = self.logger.process_env_step

        def process_env_step(rewards, dones, extras: dict, intrinsic_rewards=None) -> Any:
            if "episode" not in extras and not extras.get("log"):
                extras = {k: v for k, v in extras.items() if k != "log"}
            return original(rewards, dones, extras, intrinsic_rewards)

        self.logger.process_env_step = process_env_step  # type: ignore[method-assign]


class MetraOnPolicyRunner(LoggingOnPolicyRunner):
    """Saves ``phi`` to ``metra_phi_<N>.pt`` next to every ``model_<N>.pt``, and loads it back."""

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        torch.save(self.env.unwrapped.metra_state_dict(), metra_checkpoint_path(path))

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
        infos = super().load(path, load_cfg, strict, map_location)
        phi_path = metra_checkpoint_path(path)
        if os.path.isfile(phi_path):
            self.env.unwrapped.load_metra_state_dict(
                torch.load(phi_path, map_location=map_location, weights_only=False)
            )
        else:
            print(f"[METRA] No phi checkpoint at {phi_path}; phi starts untrained.")
        return infos
