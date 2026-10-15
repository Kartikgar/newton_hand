"""Evaluate a mimic policy on whole clips (mjlab_hand's ``mjlab_hand.eval.mimic_eval``, on Newton).

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/eval_mimic.py --task Mimic-FR3-Wuji-Omni \\
        --checkpoint logs/rsl_rl/mimic_fr3_wuji_omni/<run>/model_19999.pt \\
        --out ~/mjlab_hand/renders/mimic_eval/newton_omni

Every episode starts at the clip's first frame (no reference state init at
random frames) and has no 10s cap, so it runs until the clip ends (success) or
a failure termination fires. Everything else is training's: reset joint noise,
observation noise and fingertip friction randomization. Actions are the
policy's mean.

Writes ``rollout.npz`` in mjlab_hand's format and prints the summary. Render it
from mjlab_hand (same model, checked by ``check_mimic_model.py``):

    cd ~/mjlab_hand && MUJOCO_GL=egl uv run python -m mjlab_hand.eval.mimic_eval --task <task> \\
        --render-only <out>/rollout.npz --out <out> --label "newton | ..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from newton_hand.math_utils import quat_error_magnitude
from newton_hand.mimic.config import MimicEnvCfg, MimicPpoCfg
from newton_hand.mimic.env import MimicEnv
from newton_hand.mimic.scene import asset_dir
from newton_hand.rl.runner import LoggingOnPolicyRunner, RslRlVecEnv

NAMES = ("motion_end", "root_pos", "object_pos", "nan")
"""Termination reasons; failures win over ``motion_end`` when both fire."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = MimicEnvCfg(task=args.task, num_envs=args.num_episodes, seed=args.seed)
    cfg.command.start_prob = 1.0  # every episode from frame 0
    env = MimicEnv(cfg)
    env.max_episode_length = 10**9  # no 10s cap
    agent = MimicPpoCfg()
    vec = RslRlVecEnv(env, clip_actions=agent.clip_actions)
    runner = LoggingOnPolicyRunner(vec, agent.to_rsl_rl(), log_dir=None, device=str(env.device))
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=str(env.device))
    policy = runner.get_inference_policy(device=str(env.device))

    env.reset()
    # The state is frame 0's; reset's command update has already moved the reference to frame 1, as in training.
    assert (env.time_steps == 1).all(), "episodes must start at frame 0"
    obs = vec.get_observations()
    n, dev = env.num_envs, env.device
    motion_length = int(env.motion_length.max())
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    reason = torch.full((n,), -1, dtype=torch.long, device=dev)
    rec = {k: [] for k in ("robot_q", "obj_pos", "obj_quat", "ref_palm", "ref_obj", "palm_err", "obj_err", "obj_rot_err")}

    def record() -> None:
        palm_pos, _ = env._palm()
        obj_pos, obj_quat = env._pose(env.ix.object_body)
        values = {
            "robot_q": env.robot_q,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "ref_palm": env.ref("root_pos"),
            "ref_obj": env.ref("obj_pos"),
            "palm_err": (env.ref("root_pos") - palm_pos).norm(dim=-1),
            "obj_err": (env.ref("obj_pos") - obj_pos).norm(dim=-1),
            "obj_rot_err": quat_error_magnitude(env.ref("obj_quat"), obj_quat),
        }
        for k, v in values.items():
            rec[k].append(v.detach().clone().cpu())

    record()
    with torch.inference_mode():
        for t in range(motion_length + 5):
            obs, _, dones, _ = vec.step(policy(obs))
            terms = env.last_terminations
            new = dones.bool() & ~finished
            for i, name in reversed(list(enumerate(NAMES))):
                reason[new & terms[name] & (reason < 0)] = i
            length[new] = t + 1
            finished |= new
            # Done envs are already reset; their last recorded state stays the episode's last.
            record()
            if finished.all():
                break

    motion = np.load(asset_dir(args.task) / "motions" / env.meta["motion_files"][0])
    out = {k: torch.stack(v).numpy() for k, v in rec.items()}
    out.update(
        length=length.cpu().numpy(),
        reason=np.array([NAMES[i] if i >= 0 else "unfinished" for i in reason.cpu().numpy()]),
        joint_names=np.array(env.meta["joint_names"]),
        motion_length=np.array(motion_length),
        step_dt=np.array(env.step_dt),
        env_origins=np.zeros((n, 3)),  # every world at the origin
        base_pos=np.tile(motion["base_pos"], (n, 1)),
        base_quat=np.tile(motion["base_quat"], (n, 1)),
    )
    np.savez_compressed(args.out / "rollout.npz", **out)
    reasons, counts = np.unique(out["reason"], return_counts=True)
    print(json.dumps({"success_rate": float((out["reason"] == "motion_end").mean()),
                      "terminations": dict(zip(reasons.tolist(), counts.tolist()))}, indent=1))
    print(f"Wrote {args.out / 'rollout.npz'}")


if __name__ == "__main__":
    main()
