"""Replay ``mjlab_mimic_trace.py``'s rollout in Newton and compare.

    uv run python scripts/parity_mimic.py --task Mimic-Panda-Allegro-Apple

Same start frames, no reset or observation noise, fingertip friction 1, no
early terminations, and the same scripted "oracle" wrist policy, computed from
each sim's own state. Reports:

- the observation after the first step, term by term (the states then differ by ~0.01mm);
- palm and object tracking error over the rollout, in both sims;
- each reward term's mean per step, in both sims.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from newton_hand.math_utils import axis_angle_from_quat, quat_conjugate, quat_mul
from newton_hand.mimic.config import MimicEnvCfg
from newton_hand.mimic.env import MimicEnv
from newton_hand.mimic.scene import asset_dir


def oracle(env: MimicEnv) -> torch.Tensor:
    palm_pos, palm_quat = env._palm()
    anchor = env._anchor[:, None]
    tp = torch.where(anchor, palm_pos, env._pos_target)
    tq = torch.where(anchor, palm_quat, env._quat_target)
    ac = env.cfg.action
    a = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    a[:, :3] = ((env.ref("root_pos") - tp) / ac.pos_step).clamp(-1, 1)
    a[:, 3:6] = (axis_angle_from_quat(quat_mul(env.ref("root_quat"), quat_conjugate(tq))) / ac.rot_step).clamp(-1, 1)
    return a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Mimic-Panda-Allegro-Apple")
    args = parser.parse_args()
    trace = np.load(asset_dir(args.task) / "mjlab_trace.npz")
    n, steps = len(trace["start"]), len(trace["palm"])

    cfg = MimicEnvCfg(task=args.task, num_envs=n, obs_noise=False, keypoint_friction=(1.0, 1.0))
    cfg.command.joint_pos_noise = cfg.command.arm_joint_noise = 0.0
    cfg.termination.root_pos_threshold = cfg.termination.object_pos_threshold = 1e9
    env = MimicEnv(cfg)
    ids = torch.arange(n, device=env.device)
    env.motion_ids[:] = 0
    env.time_steps[:] = torch.as_tensor(trace["start"], device=env.device)
    env._write_reference_state(ids)
    env.actions[:] = 0.0
    env.prev_actions[:] = 0.0
    env._anchor[:] = True
    env.episode_length_buf[:] = 0
    env._update_command()
    env.obs = env._compute_obs()

    names = list(trace["reward_names"])
    palm, objp, rewards, actor = [], [], [], []
    for _ in range(steps):
        obs, *_ = env.step(oracle(env))
        actor.append(obs["actor"].cpu().numpy())
        palm.append(env._palm()[0].cpu().numpy())
        objp.append(env._pose(env.ix.object_body)[0].cpu().numpy())
        r = env._episode_sums  # per-step values: difference of the running sums
        rewards.append(np.stack([r[k].cpu().numpy() for k in names], -1))
    palm, objp = np.stack(palm), np.stack(objp)
    rewards = np.diff(np.concatenate([np.zeros_like(rewards[0])[None], np.stack(rewards)]), axis=0)

    print("Observation after the first step, max |newton - mjlab| per actor term:")
    col = 0
    for term in env.meta["observations"]["actor"]:
        d = np.abs(actor[0][:, col : col + term["dim"]] - trace["actor"][0][:, col : col + term["dim"]]).max()
        print(f"  {term['name']:14s} {d:.2e}")
        col += term["dim"]

    frames = trace["start"][None, :] + np.arange(1, steps + 1)[:, None]
    m = env.motions
    ref_root = m.root_pos.cpu().numpy()[frames]
    ref_obj = m.obj_pos.cpu().numpy()[frames]

    def err(x: np.ndarray, ref: np.ndarray) -> str:
        e = np.linalg.norm(x - ref, axis=-1)
        return f"mean {e.mean() * 100:5.2f}cm  p95 {np.percentile(e, 95) * 100:5.2f}cm"

    print(f"\nOracle wrist rollout, {n} envs x {steps} steps, error vs reference:")
    print(f"  palm   mjlab {err(trace['palm'], ref_root)} | newton {err(palm, ref_root)}")
    print(f"  object mjlab {err(trace['obj'], ref_obj)} | newton {err(objp, ref_obj)}")
    d_palm = np.linalg.norm(palm - trace["palm"], axis=-1)
    d_obj = np.linalg.norm(objp - trace["obj"], axis=-1)
    print(f"  newton vs mjlab: palm mean {d_palm.mean() * 100:.2f}cm (first 10 steps {d_palm[:10].mean() * 100:.3f}cm), "
          f"object mean {d_obj.mean() * 100:.2f}cm (first 10 steps {d_obj[:10].mean() * 100:.3f}cm)")
    print("\nReward per step (x1000), mean over envs and steps:")
    for i, name in enumerate(names):
        print(f"  {name:20s} mjlab {trace['rewards'][..., i].mean() * 1e3:8.3f} | newton {rewards[..., i].mean() * 1e3:8.3f}")


if __name__ == "__main__":
    main()
