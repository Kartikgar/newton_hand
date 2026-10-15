"""Record an mjlab mimic rollout for ``parity_mimic.py`` to replay in Newton.

Run it in **mjlab_hand's** environment:

    cd ~/mjlab_hand && uv run python ~/newton_hand/scripts/mjlab_mimic_trace.py --task Mimic-Panda-Allegro-Apple

Deterministic setup, mirrored by ``parity_mimic.py``: every env starts at its
own fixed frame (spread over the clip) with no reset noise, no observation
noise, fingertip friction fixed at 1, and no early terminations. The policy is
a scripted "oracle" that steps the wrist target toward the reference palm with
zero finger offsets. Saves the start frames and, per
step, palm/object positions, per-term rewards and the observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

NEWTON_ROOT = Path(__file__).resolve().parents[1]


def oracle(cmd, act, pos_step: float, rot_step: float, num_actions: int, device) -> torch.Tensor:
    from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_inv, quat_mul

    n = cmd.num_envs
    anchor = act._anchor[:, None]
    tp = torch.where(anchor, cmd.palm_pos_w, act._pos_target)
    tq = torch.where(anchor, cmd.palm_quat_w, act._quat_target)
    a = torch.zeros(n, num_actions, device=device)
    a[:, :3] = ((cmd.ref_w("root_pos") - tp) / pos_step).clamp(-1, 1)
    a[:, 3:6] = (axis_angle_from_quat(quat_mul(cmd.ref("root_quat"), quat_inv(tq))) / rot_step).clamp(-1, 1)
    return a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Mimic-Panda-Allegro-Apple")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjlab_hand.registration import register_tasks

    register_tasks()
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cmd_cfg = cfg.commands["motion"]
    cmd_cfg.joint_pos_noise = cmd_cfg.arm_joint_noise = 0.0
    cfg.observations["actor"].enable_corruption = False
    cfg.events["keypoint_friction"].params["ranges"] = (1.0, 1.0)
    for name in ("root_pos", "object_pos"):
        cfg.terminations.pop(name)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    cmd = env.command_manager.get_term("motion")
    act = env.action_manager.get_term("hand")
    length = int(cmd.motions.length[0])
    start = torch.linspace(0, length - args.steps - 3, args.num_envs, device=env.device).long()

    # Place every env at its start frame, then step as usual.
    env.reset()
    ids = torch.arange(args.num_envs, device=env.device)
    cmd.motion_ids[:] = 0
    cmd.time_steps[:] = start
    cmd._write_reference_state(ids)
    act.reset(ids)
    env.action_manager._action[:] = 0.0
    env.action_manager._prev_action[:] = 0.0
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.command_manager.compute(dt=0.0)

    out = {"start": start.cpu().numpy()}
    names = list(env.reward_manager.active_terms)
    palm, objp, rewards, actor = [], [], [], []
    with torch.inference_mode():
        for _ in range(args.steps):
            a = oracle(cmd, act, act.cfg.pos_step, act.cfg.rot_step, env.action_manager.total_action_dim, env.device)
            obs, _, _, _, _ = env.step(a)
            palm.append(cmd.palm_pos_w.cpu().numpy() - env.scene.env_origins.cpu().numpy())
            objp.append(cmd.obj.data.root_link_pos_w.cpu().numpy() - env.scene.env_origins.cpu().numpy())
            rewards.append(env.reward_manager._step_reward.cpu().numpy() * env.step_dt)
            actor.append(obs["actor"].cpu().numpy())
    out.update(palm=np.stack(palm), obj=np.stack(objp), rewards=np.stack(rewards), actor=np.stack(actor),
               reward_names=np.array(names))
    path = NEWTON_ROOT / "src" / "newton_hand" / "assets" / "mimic" / args.task / "mjlab_trace.npz"
    np.savez(path, **out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
