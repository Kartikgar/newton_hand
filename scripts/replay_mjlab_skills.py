"""Replay chosen mjlab envs in Newton: same skill, same start state, same scripted lift.

A METRA skill only reproduces together with its env's reset (object pose, hand
pose, lift direction and speed). This takes an mjlab rollout log written by
mjlab_hand's ``docs/weekly_2026-10-08/pen_contacts_eval.py`` and, for each
chosen env k, starts Newton world i from env k's logged state at step 0: hand
joints and velocities, object pose and velocity, last action, skill ``z_k``. The
log has no record of the lift draw, so the lift velocity is fitted from the base
slides once the lift runs (base actions no longer count then; see
:func:`fit_lift_velocity`). Then the mjlab policy drives both, and the outcome is
compared step for step with what mjlab logged.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/replay_mjlab_skills.py \\
        --log ~/mjlab_hand/docs/weekly_2026-10-08/pen_contacts_drop10_47000.npz \\
        --checkpoint <mjlab run>/model_47000.pt --envs 88 160 196 304 323 \\
        --video renders/pen_skills_47000/skill.mp4
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from newton_hand.cli.play import VideoRecorder
from newton_hand.config import EnvCfg, PpoCfg
from newton_hand.envs.grasp_env import GraspEnv
from newton_hand.math_utils import wxyz_to_xyzw
from newton_hand.objects.random_primitive import RandomObjectCfg
from newton_hand.rl.runner import MetraOnPolicyRunner, RslRlVecEnv

LIFT_START_STEP = 250  # WUJI_LIFT_START_S / step_dt


@dataclass
class Args:
    log: str
    checkpoint: str
    envs: tuple[int, ...] = (88, 160, 196, 304, 323)
    steps: int = 500
    video: str | None = None
    out: str | None = None
    """JSON summary; defaults to next to the video, or ``replay_<log stem>.json``."""
    width: int = 800
    height: int = 600
    collision: str = "newton"
    device: str = "cuda:0"


def fit_lift_velocity(slides: np.ndarray, alive_until: int, dt: float, axes: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """World-frame lift velocity from one env's base slide trajectory ``(T, 3)``.

    After the lift starts each slide target moves at a constant rate, but a slide
    only follows once its target is free: one pinned at a limit before the lift
    (the policy pressing down into the table) stays put until the target crosses
    the actual position, and the whole lift stops once any target reaches a
    limit. So fit 50-step windows from the end backwards and keep the latest one
    that is clear of the limits and gives a velocity the lift could have drawn
    (within 45 degrees of vertical, 0.05-0.2 m/s).
    """
    last = min(alive_until, slides.shape[0] - 1)
    for end in range(last, LIFT_START_STEP + 50, -5):
        window = slides[end - 50 : end]
        if np.abs(window).max() > 0.49:
            continue
        v = np.polyfit(np.arange(50) * dt, window, 1)[0] @ axes
        speed = float(np.linalg.norm(v))
        if 0.05 <= speed <= 0.2 and v[2] / speed >= math.cos(math.radians(45.0)):
            return v, (end - 50, end)
    raise ValueError("No window of the lift fits a velocity the task could have drawn.")


def start_from_log(env: GraspEnv, log, envs: list[int]) -> dict:
    """Put Newton world i in mjlab env ``envs[i]``'s logged state after its first step.

    Sets hand joints and velocities, object pose and velocity, action targets, last
    action, skill, episode step 1 and the fitted lift velocity. Returns the lift fits.
    """
    dt = float(log["step_dt"])
    dev, ix = env.device, env.ix

    def t(a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)

    # Start state: mjlab logs after its first step, so these worlds start at episode step 1.
    hand_q, hand_qd = t(log["joint_pos"][0, envs]), t(log["joint_vel"][0, envs])
    pen_pos, pen_quat = t(log["pen_pos"][0, envs]), t(log["pen_quat"][0, envs])
    env._joint_q[ix.hand_q] = hand_q
    env._joint_qd[ix.hand_qd] = hand_qd
    env._joint_q[ix.object_q] = torch.cat([pen_pos, wxyz_to_xyzw(pen_quat)], dim=-1)
    env._joint_qd[ix.object_qd] = torch.cat([t(log["pen_linvel"][0, envs]), t(log["pen_angvel"][0, envs])], dim=-1)
    env._forward()
    env._base_target[:] = env._base_ema[:] = hand_q[:, :6]
    env._finger_target[:] = env._finger_ema[:] = hand_q[:, 6:]
    env.actions[:] = t(log["action"][0, envs])
    env._apply_targets()
    env.skill[:] = t(log["skills"][envs])
    env.episode_length_buf[:] = 1

    axes = env._slide_axes.cpu().numpy()  # row i: world direction of base slide i
    lifts = {}
    for i, k in enumerate(envs):
        v, window = fit_lift_velocity(log["joint_pos"][:, k, :3], int(log["last_step"][k]), dt, axes)
        env._lift_velocity[i] = t(v)
        lifts[k] = {"velocity": np.round(v, 4).tolist(), "speed": round(float(np.linalg.norm(v)), 3),
                    "cone_deg": round(math.degrees(math.acos(v[2] / np.linalg.norm(v))), 1), "fit_steps": window}
    env.obs = env._compute_obs()
    return lifts


def main() -> None:
    args = tyro.cli(Args, config=(tyro.conf.FlagConversionOff,))
    log = np.load(Path(args.log).expanduser(), mmap_mode="r", allow_pickle=True)
    envs = list(args.envs)
    n = len(envs)

    cfg = EnvCfg(num_envs=n, seed=0, obs_noise=False, episode_length_s=1e9)
    cfg.metra.train_phi = False
    cfg.sim.collision = args.collision
    cfg.object = RandomObjectCfg(shape_weights=(0.0, 0.0, 1.0))
    env = GraspEnv(cfg, device=args.device)
    vec = RslRlVecEnv(env, clip_actions=1.0)
    runner = MetraOnPolicyRunner(vec, PpoCfg().to_rsl_rl(), log_dir=None, device=args.device)
    runner.load(str(Path(args.checkpoint).expanduser()), load_cfg={"actor": True})
    policy = runner.get_inference_policy(device=args.device)

    lifts = start_from_log(env, log, envs)
    dev, ix = env.device, env.ix

    recorder = None
    if args.video is not None:
        recorder = VideoRecorder(env, args.video, args.width, args.height, follow=True,
                                 names=[f"skill{k}" for k in envs])

    done = torch.zeros(n, dtype=torch.bool, device=dev)
    end_reason = ["running"] * n
    end_step = [args.steps - 1] * n
    pen_z = np.zeros((args.steps, n), dtype=np.float32)
    obs = vec.get_observations()
    with torch.inference_mode():
        # Logged step s is mjlab's state after s + 1 steps; ours after the loop's step s is too.
        for s in range(1, args.steps):
            obs, _, dones, _ = vec.step(policy(obs))
            for i in (dones.bool() & ~done).nonzero().flatten().tolist():
                terms = env.last_terminations
                end_reason[i] = next(name for name in ("nan_term", "object_fallen", "palm_object_distance", "object_spin", "time_out") if bool(terms[name][i]))
                end_step[i] = s
            done |= dones.bool()
            pen_z[s] = env._pose(ix.object_body)[0][:, 2].cpu().numpy()
            if recorder is not None:
                recorder.capture(s)
    if recorder is not None:
        recorder.close()

    term_names = list(log["term_names"])
    summary = {"log": args.log, "checkpoint": args.checkpoint, "collision": args.collision, "skills": {}}
    for i, k in enumerate(envs):
        mj_z = np.asarray(log["pen_pos"][:, k, 2])
        mj_alive = bool(log["alive"][-1, k])
        mj_rise = float(mj_z[-1] - mj_z[30]) if mj_alive else None
        nt_alive = not bool(done[i])
        nt_rise = float(pen_z[-1, i] - pen_z[30, i]) if nt_alive else None
        reason = int(log["term_reason"][k])
        summary["skills"][k] = {
            "lift_fit": lifts[k],
            "mjlab": {"held": bool(log["success"][k]), "end_step": int(log["last_step"][k]),
                      "end": term_names[reason] if 0 <= reason < len(term_names) else "running",
                      "pen_rise_cm": None if mj_rise is None else round(100 * mj_rise, 1)},
            "newton": {"held": bool(nt_alive and nt_rise > 0.05), "end_step": end_step[i], "end": end_reason[i],
                       "pen_rise_cm": None if nt_rise is None else round(100 * nt_rise, 1),
                       "max_pen_rise_cm": round(100 * float(pen_z[30:end_step[i] + 1, i].max() - pen_z[30, i]), 1)
                       if end_step[i] >= 30 else None},
        }
    out = args.out or (str(Path(args.video).with_suffix("")) + "_summary.json" if args.video else f"replay_{Path(args.log).stem}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["skills"], indent=1))


if __name__ == "__main__":
    main()
