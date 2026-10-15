"""Train the random-object METRA grasp task on Newton.

    CUDA_VISIBLE_DEVICES=0 uv run train-newton \\
        --env.num-envs 4096 --agent.max-iterations 20000 \\
        --agent.logger wandb --agent.run-name metra-random-object-newton --video True

Logs and checkpoints go to ``logs/rsl_rl/<experiment>/<date>_<run-name>/``,
with ``metra_phi_<N>.pt`` next to every ``model_<N>.pt``. Resume with
``--agent.resume True --agent.load-run <run folder> --agent.load-checkpoint model_<N>.pt``;
``--agent.max-iterations`` then counts the additional iterations.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import tyro

from newton_hand.config import EnvCfg, PpoCfg
from newton_hand.envs.grasp_env import GraspEnv
from newton_hand.rl.runner import MetraOnPolicyRunner, RslRlVecEnv
from newton_hand.rl.video import TrainVideoVecEnv

LOG_ROOT = Path("logs") / "rsl_rl"


@dataclass
class TrainArgs:
    env: EnvCfg = field(default_factory=EnvCfg)
    agent: PpoCfg = field(default_factory=PpoCfg)
    device: str = "cuda:0"
    video: bool = False
    """Record a close-up of world 0 every ``video-interval`` env steps into ``<run>/videos/train``
    (uploaded to W&B with ``--agent.logger wandb``)."""
    video_length: int = 500
    """Steps per clip; each clip starts at world 0's next reset, so 500 is one whole episode."""
    video_interval: int = 2000


def resolve_checkpoint(experiment_dir: Path, load_run: str, load_checkpoint: str) -> Path:
    """Latest run matching ``load_run`` and, in it, the highest-sorting ``load_checkpoint`` match (mjlab's rule)."""
    runs = sorted(p for p in experiment_dir.iterdir() if p.is_dir() and re.fullmatch(load_run, p.name))
    if not runs:
        raise FileNotFoundError(f"No run matching {load_run!r} in {experiment_dir}")
    run = runs[-1]
    ckpts = sorted(
        (p for p in run.iterdir() if re.fullmatch(load_checkpoint, p.name)),
        key=lambda p: int(m.group(1)) if (m := re.search(r"(\d+)", p.name)) else -1,
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint matching {load_checkpoint!r} in {run}")
    return ckpts[-1]


def main() -> None:
    # Booleans take a value (`--agent.resume True`), as in mjlab's CLI.
    args = tyro.cli(TrainArgs, config=(tyro.conf.FlagConversionOff,))
    if args.env.metra.update_every != args.agent.num_steps_per_env:
        raise ValueError("env.metra.update-every must equal agent.num-steps-per-env so each PPO batch sees one phi.")

    experiment_dir = LOG_ROOT / args.agent.experiment_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = experiment_dir / (f"{stamp}_{args.agent.run_name}" if args.agent.run_name else stamp)
    resume_path = None
    if args.agent.resume:
        resume_path = resolve_checkpoint(experiment_dir, args.agent.load_run, args.agent.load_checkpoint)

    env = GraspEnv(args.env, device=args.device)
    if args.video:
        vec_env = TrainVideoVecEnv(env, log_dir / "videos" / "train", args.agent.clip_actions,
                                   interval=args.video_interval, length=args.video_length)
    else:
        vec_env = RslRlVecEnv(env, clip_actions=args.agent.clip_actions)
    print(f"[newton_hand] actor obs {env.obs['actor'].shape[1]}-D, critic {env.obs['critic'].shape[1]}-D, "
          f"{vec_env.num_actions}-D actions, {env.num_envs} envs.")

    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=1, default=str)

    runner = MetraOnPolicyRunner(vec_env, args.agent.to_rsl_rl(), str(log_dir), device=args.device)
    if resume_path is not None:
        print(f"[newton_hand] Resuming from {resume_path}")
        runner.load(str(resume_path))
    torch.backends.cuda.matmul.allow_tf32 = True
    runner.learn(num_learning_iterations=args.agent.max_iterations, init_at_random_ep_len=True)
    if args.video:
        vec_env.close()


if __name__ == "__main__":
    main()
