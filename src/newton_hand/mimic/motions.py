"""Motion files concatenated along time: mjlab_hand's ``MotionLibrary``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

FIELDS = (
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "keypoint_pos",
    "keypoint_obj_dist",
    "obj_pos",
    "obj_quat",
    "obj_lin_vel",
    "obj_ang_vel",
    "arm_joint_pos",
    "arm_joint_vel",
)


class MotionLibrary:
    def __init__(self, files: list[Path], device: torch.device, fps: float):
        data = [np.load(f, allow_pickle=True) for f in files]
        for f, d in zip(files, data):
            if abs(float(d["fps"]) - fps) > 1e-6:
                raise ValueError(f"{f} is at {float(d['fps'])} fps; the env steps at {fps}.")
        self.names = tuple(Path(f).stem for f in files)
        self.joint_names = tuple(str(j) for j in data[0]["joint_names"])
        self.arm_joint_names = tuple(str(j) for j in data[0]["arm_joint_names"])
        self.keypoint_names = tuple(str(k) for k in data[0]["keypoint_names"])
        lengths = [len(d["root_pos"]) for d in data]
        self.length = torch.tensor(lengths, dtype=torch.long, device=device)
        self.start = torch.cumsum(self.length, 0) - self.length
        for name in FIELDS:
            stacked = np.concatenate([d[name] for d in data], axis=0)
            setattr(self, name, torch.tensor(stacked, dtype=torch.float32, device=device))
        # Fingertip velocities, which the motion files do not store.
        keypoint_vel = np.concatenate([np.gradient(d["keypoint_pos"], 1.0 / fps, axis=0) for d in data], axis=0)
        self.keypoint_lin_vel = torch.tensor(keypoint_vel, dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return len(self.names)
