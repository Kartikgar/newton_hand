"""Roll out a checkpoint (or zero actions) and watch it, live or as a video.

    # Live, in Newton's OpenGL viewer (needs a display):
    uv run play-newton --checkpoint logs/rsl_rl/<exp>/<run>/model_<N>.pt

    # Headless, to an mp4:
    MUJOCO_GL=egl uv run play-newton --checkpoint ... --video renders/play.mp4 --steps 500

Without ``--checkpoint`` the policy outputs zeros, which shows the reset, the
hover pose and the scripted lift on their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
import warp as wp

from newton_hand.config import EnvCfg, PpoCfg
from newton_hand.envs.grasp_env import GraspEnv
from newton_hand.rl.runner import MetraOnPolicyRunner, RslRlVecEnv


def _play_env_cfg() -> EnvCfg:
    cfg = EnvCfg(num_envs=16, obs_noise=False)
    cfg.metra.train_phi = False
    return cfg


@dataclass
class PlayArgs:
    env: EnvCfg = field(default_factory=_play_env_cfg)
    checkpoint: str | None = None
    steps: int = 500
    video: str | None = None
    """Write an mp4 here instead of opening a window."""
    width: int = 1280
    height: int = 720
    spacing: float = 0.8
    """Visual spacing between worlds; the simulation itself does not space them."""
    follow_each: bool = False
    """With ``--video``: one close-up video per env (``<video>_env<k>.mp4``), each
    showing that world alone from a camera that follows its fingertips."""
    follow_distance: float = 0.55
    """Camera distance from the fingertips in follow mode, metres."""
    device: str = "cuda:0"


class VideoRecorder:
    """Headless mp4 recording: one grid video of every world, or with ``follow`` one
    close-up per world (``<path>_env<k>.mp4``), each showing that world alone from a
    camera that tracks an EMA of its fingertip centroid."""

    def __init__(
        self,
        env: GraspEnv,
        path: str,
        width: int = 1280,
        height: int = 720,
        follow: bool = False,
        spacing: float = 0.8,
        follow_distance: float = 0.55,
        names: list[str] | None = None,
    ):
        import imageio.v2 as imageio
        import pyglet

        # Render through EGL with no X display; must be set before the viewer's window import.
        pyglet.options["headless"] = True
        import newton.viewer

        self.env, self.follow, self.follow_distance = env, follow, follow_distance
        self.viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
        self.viewer.set_model(env.model)
        if not follow:
            self.viewer.set_world_offsets((spacing, spacing, 0.0))
            extent = spacing * math.ceil(math.sqrt(env.num_envs))
            self.viewer.set_camera(wp.vec3(-0.5 - 0.8 * extent, 0.0, 0.75 + 0.35 * extent), -18.0, 0.0)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fps = round(1.0 / env.step_dt)
        if follow:
            stem = Path(path).with_suffix("")
            names = names or [f"env{k}" for k in range(env.num_envs)]
            self.paths = [f"{stem}_{name}.mp4" for name in names]
        else:
            self.paths = [path]
        self.writers = [imageio.get_writer(p, fps=fps, macro_block_size=8) for p in self.paths]
        # Behind-left and above the fingertips.
        view_dir = torch.tensor([-0.75, -0.45, 0.48], device=env.device)
        self.view_dir = view_dir / view_dir.norm()
        self.focus = None
        self.frames = 0

    def capture(self, step: int) -> None:
        env, viewer = self.env, self.viewer
        self.frames += 1
        if not self.follow:
            viewer.begin_frame(step * env.step_dt)
            viewer.log_state(env.state)
            viewer.end_frame()
            self.writers[0].append_data(viewer.get_frame().numpy())
            return
        tips = env._pose(env.ix.keypoint_bodies)[0].mean(dim=1)
        self.focus = tips if self.focus is None else 0.9 * self.focus + 0.1 * tips
        for k, writer in enumerate(self.writers):
            viewer.set_visible_worlds([k])
            eye = self.focus[k] + self.follow_distance * self.view_dir
            d = (self.focus[k] - eye).tolist()
            yaw = math.degrees(math.atan2(d[1], d[0]))
            pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
            viewer.set_camera(wp.vec3(*eye.tolist()), pitch, yaw)
            viewer.begin_frame(step * env.step_dt)
            viewer.log_state(env.state)
            viewer.end_frame()
            writer.append_data(viewer.get_frame().numpy())

    def close(self) -> None:
        for writer in self.writers:
            writer.close()
        self.viewer.close()
        print(f"Wrote {self.frames} frames to {', '.join(self.paths)}")


def main() -> None:
    args = tyro.cli(PlayArgs, config=(tyro.conf.FlagConversionOff,))
    env = GraspEnv(args.env, device=args.device)
    vec_env = RslRlVecEnv(env, clip_actions=1.0)

    policy = None
    if args.checkpoint:
        runner = MetraOnPolicyRunner(vec_env, PpoCfg(logger="tensorboard").to_rsl_rl(), log_dir=None, device=args.device)
        runner.load(args.checkpoint, load_cfg={"actor": True})
        policy = runner.get_inference_policy(device=args.device)

    if args.video is None:
        import newton.viewer

        viewer = newton.viewer.ViewerGL(width=args.width, height=args.height)
        viewer.set_model(env.model)
        viewer.set_world_offsets((args.spacing, args.spacing, 0.0))
        extent = args.spacing * math.ceil(math.sqrt(env.num_envs))
        viewer.set_camera(wp.vec3(-0.5 - 0.8 * extent, 0.0, 0.75 + 0.35 * extent), -18.0, 0.0)
    else:
        recorder = VideoRecorder(env, args.video, args.width, args.height, args.follow_each, args.spacing, args.follow_distance)

    obs = vec_env.get_observations()
    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy(obs) if policy is not None else torch.zeros(env.num_envs, vec_env.num_actions, device=env.device)
            obs, _, _, _ = vec_env.step(actions)
            if args.video is not None:
                recorder.capture(step)
                continue
            viewer.begin_frame(step * env.step_dt)
            viewer.log_state(env.state)
            viewer.end_frame()
            if not viewer.is_running():
                break

    if args.video is not None:
        recorder.close()
        return
    viewer.close()
    wp.synchronize()


if __name__ == "__main__":
    main()
