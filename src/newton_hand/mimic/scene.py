"""Build the Newton model for a mimic task: per world the Panda carrying the hand, the table and the object.

Everything is loaded from ``assets/mimic/<task>/``, exported from mjlab_hand by
``scripts/export_mimic_from_mjlab.py`` *after* mjlab applied its collision and
actuator config, so masses, friction, solref/solimp, armature, damping,
gravcomp and the finger actuators' gains are mjlab's
(``scripts/check_mimic_model.py`` verifies it). The arm base stands where the
motion file puts it, as mjlab's mocap base does; every motion of an exported
task shares it.

Worlds are separate MuJoCo worlds at the same origin, so every world's frame is
the env frame of mjlab's motions and no env-origin correction is needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp

import newton

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "mimic"


def asset_dir(task: str) -> Path:
    path = ASSET_ROOT / task
    if not (path / "mjlab.json").exists():
        raise FileNotFoundError(
            f"No exported task at {path}. Export it from mjlab_hand:\n"
            f"  cd ~/mjlab_hand && uv run python ~/newton_hand/scripts/export_mimic_from_mjlab.py --task {task}"
        )
    return path


def load_meta(task: str) -> dict:
    return json.loads((asset_dir(task) / "mjlab.json").read_text())


@dataclass
class SceneIndex:
    """Per-world indices into Newton's flat arrays, ``(num_envs, ...)`` long tensors."""

    palm_body: torch.Tensor  # (N,)
    keypoint_bodies: torch.Tensor  # (N, K)
    object_body: torch.Tensor  # (N,)
    robot_q: torch.Tensor  # (N, J) joint_q indices, in mjlab's joint order
    robot_qd: torch.Tensor  # (N, J) joint_qd indices
    object_q: torch.Tensor  # (N, 7) free joint: pos, quat xyzw
    object_qd: torch.Tensor  # (N, 6) free joint: COM linear, angular (world)
    pad_shapes: torch.Tensor  # (N, K) fingertip collision geoms, mjlab's keypoint_contacts primaries
    object_shapes: torch.Tensor  # (N, S) the object's collision shapes
    arm_joints: torch.Tensor  # (N, 7) Newton joint indices of the arm hinges
    shape_table_primary: torch.Tensor  # (shape_count,) primary slot of a hand body touching the table, or -1
    shape_is_table: torch.Tensor  # (shape_count,) 1 for the table's shape
    num_table_primaries: int


def _world_builder(meta: dict, task_dir: Path) -> newton.ModelBuilder:
    motion = np.load(task_dir / "motions" / meta["motion_files"][0])
    pos, quat = motion["base_pos"], motion["base_quat"]  # quat w, x, y, z
    b = newton.ModelBuilder()
    # No detection gap, as in mjlab (Newton's default 10cm is forwarded to MuJoCo-Warp).
    b.rigid_gap = 0.0
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    base = wp.transform(wp.vec3(*pos.tolist()), wp.quat(float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])))
    b.add_mjcf(str(task_dir / "robot.xml"), xform=base, floating=False, enable_self_collisions=True)
    b.add_mjcf(str(task_dir / "table.xml"), floating=False)
    b.add_mjcf(str(task_dir / "object.xml"))
    return b


def build_model(cfg, meta: dict, device: str) -> tuple[newton.Model, SceneIndex, Path]:
    task_dir = asset_dir(cfg.task)
    world = _world_builder(meta, task_dir)
    scene = newton.ModelBuilder()
    scene.rigid_gap = 0.0
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.add_ground_plane()
    scene.replicate(world, cfg.num_envs)
    model = scene.finalize(device=device)
    return model, _index(model, meta, cfg.num_envs, device), task_dir


def _short(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _per_world(labels: list[str], world_of: np.ndarray, name: str, num_envs: int) -> np.ndarray:
    hits = np.asarray([i for i, label in enumerate(labels) if _short(label) == name])
    hits = hits[np.argsort(world_of[hits], kind="stable")] if len(hits) else hits
    if len(hits) != num_envs or not np.array_equal(world_of[hits], np.arange(num_envs)):
        raise RuntimeError(f"Expected one {name!r} per world, found {len(hits)}.")
    return hits


def _index(model: newton.Model, meta: dict, num_envs: int, device: str) -> SceneIndex:
    body_world = model.body_world.numpy()
    shape_world = model.shape_world.numpy()
    shape_body = model.shape_body.numpy()
    joint_world = model.joint_world.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    def bodies(name: str) -> np.ndarray:
        return _per_world(model.body_label, body_world, name, num_envs)

    def joints(name: str) -> np.ndarray:
        return _per_world(model.joint_label, joint_world, name, num_envs)

    robot_joints = np.stack([joints(n) for n in meta["joint_names"]], axis=-1)
    object_body = bodies("object")
    object_joint = np.flatnonzero(np.isin(model.joint_child.numpy(), object_body))
    object_joint = object_joint[np.argsort(joint_world[object_joint])]

    # Collision shapes only: the ones that reach MuJoCo.
    collides = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    pattern = re.compile(meta["keypoint_geom_pattern"])
    pads = np.asarray([i for i, label in enumerate(model.shape_label) if pattern.fullmatch(_short(label))])
    pads = pads[np.argsort(shape_world[pads], kind="stable")].reshape(num_envs, -1)
    obj_shapes = np.flatnonzero(np.isin(shape_body, object_body) & collides)
    obj_shapes = obj_shapes[np.argsort(shape_world[obj_shapes], kind="stable")].reshape(num_envs, -1)

    # mjlab's table_contacts primaries: every hand body except the fingertip ones,
    # one slot per body name. Only bodies with collision shapes can ever touch.
    excluded = set(meta["table_contact_excluded_bodies"])
    robot_bodies = [n for n in meta["body_names"] if n not in excluded]
    slot_of = {}
    shape_primary = np.full(model.shape_count, -1, dtype=np.int32)
    body_label = [_short(label) for label in model.body_label]
    for s in np.flatnonzero(collides):
        b = shape_body[s]
        if b >= 0 and body_label[b] in robot_bodies:
            shape_primary[s] = slot_of.setdefault(body_label[b], len(slot_of))
    shape_is_table = np.zeros(model.shape_count, dtype=np.int32)
    table_body = bodies("table")
    shape_is_table[np.flatnonzero(np.isin(shape_body, table_body) & collides)] = 1

    def t(a: np.ndarray, dtype=torch.long) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    return SceneIndex(
        palm_body=t(bodies(meta["palm_body"])),
        keypoint_bodies=t(np.stack([bodies(n) for n in meta["keypoint_bodies"]], axis=-1)),
        object_body=t(object_body),
        robot_q=t(q_start[robot_joints]),
        robot_qd=t(qd_start[robot_joints]),
        object_q=t(q_start[object_joint][:, None] + np.arange(7)),
        object_qd=t(qd_start[object_joint][:, None] + np.arange(6)),
        pad_shapes=t(pads),
        object_shapes=t(obj_shapes),
        arm_joints=t(robot_joints[:, :7]),
        shape_table_primary=t(shape_primary, torch.int32),
        shape_is_table=t(shape_is_table, torch.int32),
        num_table_primaries=max(len(slot_of), 1),
    )
