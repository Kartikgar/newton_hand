"""Train a mimic task (mjlab_hand's ``Mimic-Panda-*``) on Newton.

    CUDA_VISIBLE_DEVICES=0 uv run train-mimic --env.task Mimic-Panda-Allegro-Apple \\
        --env.num-envs 4096 --agent.max-iterations 20000 \\
        --agent.logger wandb --agent.run-name mimic-panda-allegro-apple-newton --video True

The task must have been exported from mjlab_hand first
(``scripts/export_mimic_from_mjlab.py``). Logs and checkpoints go to
``logs/rsl_rl/<experiment>/<date>_<run-name>/``; the experiment defaults to the
task id in lower case with underscores, as in mjlab. Resume with
``--agent.resume True --agent.load-run <run folder> --agent.load-checkpoint model_<N>.pt``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import torch
import tyro

from newton_hand.cli.train import LOG_ROOT, resolve_checkpoint
from newton_hand.mimic.config import MimicEnvCfg, MimicPpoCfg, check_against_mjlab
from newton_hand.mimic.env import MimicEnv
from newton_hand.mimic.scene import load_meta
from newton_hand.rl.runner import LoggingOnPolicyRunner, RslRlVecEnv
from newton_hand.rl.video import TrainVideoVecEnv


@dataclass
class TrainMimicArgs:
    env: MimicEnvCfg = field(default_factory=MimicEnvCfg)
    agent: MimicPpoCfg = field(default_factory=MimicPpoCfg)
    device: str = "cuda:0"
    video: bool = False
    """Record world 0 every ``video-interval`` env steps into ``<run>/videos/train`` (uploaded to W&B)."""
    video_length: int = 500
    video_interval: int = 8000
    video_distance: float = 1.6
    """Camera distance; the arm needs more room than the floating hand's close-up."""


def main() -> None:
    args = tyro.cli(TrainMimicArgs, config=(tyro.conf.FlagConversionOff,))
    agent = args.agent
    if not agent.experiment_name:
        agent.experiment_name = args.env.task.lower().replace("-", "_")
    diffs = check_against_mjlab(args.env, agent, load_meta(args.env.task), strict=False)
    for diff in diffs:
        print(f"[newton_hand] differs from mjlab (command line?): {diff}")

    experiment_dir = LOG_ROOT / agent.experiment_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = experiment_dir / (f"{stamp}_{agent.run_name}" if agent.run_name else stamp)
    resume_path = resolve_checkpoint(experiment_dir, agent.load_run, agent.load_checkpoint) if agent.resume else None

    env = MimicEnv(args.env, device=args.device)
    if args.video:
        vec_env = TrainVideoVecEnv(env, log_dir / "videos" / "train", agent.clip_actions, interval=args.video_interval,
                                   length=args.video_length, follow_distance=args.video_distance)
    else:
        vec_env = RslRlVecEnv(env, clip_actions=agent.clip_actions)
    print(f"[newton_hand] {args.env.task}: actor obs {env.obs['actor'].shape[1]}-D, critic "
          f"{env.obs['critic'].shape[1]}-D, {vec_env.num_actions}-D actions, {env.num_envs} envs.")

    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=1, default=str)

    runner = LoggingOnPolicyRunner(vec_env, agent.to_rsl_rl(), str(log_dir), device=args.device)
    if resume_path is not None:
        print(f"[newton_hand] Resuming from {resume_path}")
        runner.load(str(resume_path))
    torch.backends.cuda.matmul.allow_tf32 = True
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
    if args.video:
        vec_env.close()


if __name__ == "__main__":
    main()
