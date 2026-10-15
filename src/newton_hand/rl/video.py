"""Videos during training, as mjlab's ``--video``: every ``interval`` env steps, a close-up of world 0.

Each clip starts at world 0's next reset and runs ``length`` steps, so with the
default 500 it shows one whole episode, scripted lift included. Clips land in
``<run>/videos/train/rl-video-step-<N>.mp4``; rsl-rl uploads every mp4 in the
run folder to W&B at the next log. A clip is encoded in a temporary folder and
moved in once complete, so rsl-rl never uploads a half-written file.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import torch
import warp as wp

from newton_hand.rl.runner import RslRlVecEnv


class TrainVideoVecEnv(RslRlVecEnv):
    def __init__(
        self,
        env,
        video_dir: str | Path,
        clip_actions: float | None = None,
        interval: int = 2000,
        length: int = 500,
        width: int = 640,
        height: int = 480,
        follow_distance: float = 0.55,
    ):
        super().__init__(env, clip_actions)
        import pyglet

        # Render through EGL with no X display; must be set before the viewer's window import.
        pyglet.options["headless"] = True
        import newton.viewer

        # One viewer for the whole run: a second ViewerGL in the same process renders black.
        self.viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
        self.viewer.set_model(env.model)
        self.viewer.set_visible_worlds([0])
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="newton_hand_video_"))
        self.interval, self.length, self.follow_distance = interval, length, follow_distance
        # Behind-left and above the fingertips, as play-newton's follow camera.
        view_dir = torch.tensor([-0.75, -0.45, 0.48], device=env.device)
        self.view_dir = view_dir / view_dir.norm()
        self.step_count = 0
        self.pending: int | None = None  # step that triggered the clip waiting for world 0's reset
        self.writer = None
        self.frames = 0
        self.focus: torch.Tensor | None = None

    def step(self, actions: torch.Tensor):
        if self.step_count % self.interval == 0 and self.pending is None and self.writer is None:
            self.pending = self.step_count
        out = super().step(actions)
        dones = out[2]
        self.step_count += 1
        if self.writer is not None:
            if bool(dones[0]):
                self._refresh_shapes()
            self._capture()
            if self.frames >= self.length:
                self._finish()
        # World 0's reset already happened inside this step, so the next frame is its episode start.
        if self.pending is not None and bool(dones[0]):
            self._start()
        return out

    def _start(self) -> None:
        import imageio.v2 as imageio

        self.name = f"rl-video-step-{self.pending}.mp4"
        self.writer = imageio.get_writer(str(self.tmp_dir / self.name), fps=round(1.0 / self.env.step_dt), macro_block_size=8)
        self.pending, self.frames, self.focus = None, 0, None
        self._refresh_shapes()

    def _refresh_shapes(self) -> None:
        """Rebuild the drawn shapes from the model. The viewer reads shape sizes and
        local transforms only in ``set_model``, but every reset redraws the object.

        ``set_model`` walks every world's shapes (6.5s at 4096 mimic worlds), so
        envs whose shapes never change (``shapes_change_on_reset = False``) skip it:
        on the mimic task it made recorded iterations take 7-14s instead of 0.65s.
        """
        if not getattr(self.env, "shapes_change_on_reset", True):
            return
        self.viewer.set_model(self.env.model)
        self.viewer.set_visible_worlds([0])

    def _capture(self) -> None:
        env, viewer = self.env, self.viewer
        tips = env._pose(env.ix.keypoint_bodies[:1])[0].mean(dim=1)[0]
        self.focus = tips if self.focus is None else 0.9 * self.focus + 0.1 * tips
        eye = self.focus + self.follow_distance * self.view_dir
        d = (self.focus - eye).tolist()
        yaw = math.degrees(math.atan2(d[1], d[0]))
        pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
        viewer.set_camera(wp.vec3(*eye.tolist()), pitch, yaw)
        viewer.begin_frame(self.frames * env.step_dt)
        viewer.log_state(env.state)
        viewer.end_frame()
        self.writer.append_data(viewer.get_frame().numpy())
        self.frames += 1

    def _finish(self) -> None:
        self.writer.close()
        self.writer = None
        shutil.move(str(self.tmp_dir / self.name), str(self.video_dir / self.name))

    def close(self) -> None:
        if self.writer is not None:
            self._finish()
        self.viewer.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
