"""Build the Newton model: per world a floating Wuji hand, a table and the random object.

The hand comes from ``assets/wuji/wuji_right_floating.xml``, exported from
mjlab_hand *after* mjlab applied ``FULL_COLLISION`` and the position
actuators, so collision flags, friction, solref/solimp, priorities, armature
and actuator gains are those the mjlab task trained with. The palm hangs off
the world on six base joints (three slides, three hinges), which Newton merges
into one D6 joint.

Worlds are not spaced out: each is a separate MuJoCo world, so they never
collide and every world's frame is the env frame mjlab's observations use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp

import newton
from newton_hand.objects.random_primitive import HIDDEN_SIZE, RandomObjectCfg

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
WUJI_XML = ASSET_DIR / "wuji" / "wuji_right_floating.xml"

PALM_BODY = "right_palm_link"
KEYPOINT_BODIES = tuple(f"right_finger{i}_link4" for i in range(1, 6))
# The fingertip pads: WUJI_KEYPOINT_GEOMS (condim 6, priority 1, solref 0.01).
PAD_SHAPES = tuple(f"right_finger{i}_link4_collision_1" for i in range(1, 6))
NUM_BASE_DOFS = 6
NUM_FINGER_DOFS = 20

# WUJI_HOME_JOINT_POS, finger by finger (joint1..4); the base defaults to 0.
FINGER_HOME = (1.5, 0.0, 0.5, 0.5) + (0.3, 0.0, 0.4, 0.4) * 4

# The grasp table: a 60x60x4cm box whose centre is 40cm up, top at 42cm.
TABLE_POS = (0.0, 0.0, 0.4)
TABLE_HALF = (0.3, 0.3, 0.02)
TABLE_COLOR = (0.5, 0.35, 0.2)  # brown, mjlab's table rgba

# The table's contact stiffness; the object uses it too (mjlab_hand STIFF_SOLREF/SOLIMP).
STIFF_SOLREF = (0.005, 1.0)
STIFF_SOLIMP = (0.99, 0.99, 0.001, 0.5, 2.0)
# Use the authored solref as-is. Otherwise Newton derives it from the shape's
# ke/kd and the stiff solref above never reaches MuJoCo.
SOLREF_MODE_RAW = 1
STIFF_CONTACT_ATTRS = {
    "mujoco:solref": wp.vec2(*STIFF_SOLREF),
    "mujoco:solref_mode": SOLREF_MODE_RAW,
    "mujoco:geom_solimp": list(STIFF_SOLIMP),
}


@dataclass
class SceneIndex:
    """Where each world's pieces live in Newton's flat arrays, all ``(num_envs, ...)`` long tensors."""

    palm_body: torch.Tensor  # (N,)
    keypoint_bodies: torch.Tensor  # (N, 5)
    object_body: torch.Tensor  # (N,)
    hand_q: torch.Tensor  # (N, 26) joint_q indices: base 6, fingers 20
    hand_qd: torch.Tensor  # (N, 26) joint_qd indices
    object_q: torch.Tensor  # (N, 7) free joint: pos, quat xyzw
    object_qd: torch.Tensor  # (N, 6) free joint: linear, angular
    cylinder_shape: torch.Tensor  # (N,)
    sphere_shape: torch.Tensor  # (N,)
    pad_shapes: torch.Tensor  # (N, 5)


def _world_builder(object_cfg: RandomObjectCfg) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    # No detection gap, as in mjlab. Newton's default 10cm gap is forwarded to
    # MuJoCo-Warp and makes every pair within 10cm a contact candidate, which
    # overflowed its broadphase.
    b.rigid_gap = 0.0
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(str(WUJI_XML), enable_self_collisions=True)
    # MuJoCo collides a mesh geom as its convex hull. Newton keeps the triangle mesh,
    # which its CollisionPipeline handled ~20x slower and with different contacts,
    # so give it the hulls MuJoCo (and mjlab) uses. Visual meshes keep their detail.
    colliders = [i for i, label in enumerate(b.shape_label) if "_collision" in label.rsplit("/", 1)[-1]]
    b.approximate_meshes("convex_hull", shape_indices=colliders, raise_on_failure=True)

    table_cfg = newton.ModelBuilder.ShapeConfig(mu=1.0, mu_torsional=0.005, mu_rolling=0.0001)
    b.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(*TABLE_POS), wp.quat_identity()),
        hx=TABLE_HALF[0],
        hy=TABLE_HALF[1],
        hz=TABLE_HALF[2],
        cfg=table_cfg,
        label="table_surface",
        color=TABLE_COLOR,
        custom_attributes=dict(STIFF_CONTACT_ATTRS),
    )

    # Mass and inertia are rewritten at every reset; these are placeholders.
    obj = b.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        mass=0.05,
        inertia=wp.mat33(*np.diag([1e-5, 1e-5, 1e-5]).flatten().tolist()),
        lock_inertia=True,
        label="object",
    )
    # condim 6, friction 1, priority 1, table-stiff contacts: the ContactDB objects' settings.
    obj_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0, mu_torsional=0.002, mu_rolling=0.002)
    obj_attrs = {**STIFF_CONTACT_ATTRS, "mujoco:condim": 6, "mujoco:geom_priority": 1}
    # Built at the largest size each shape takes, so MuJoCo's bounding radius
    # (compiled once, never updated by Newton) covers every draw.
    r, h = object_cfg.max_cylinder_size()
    b.add_shape_cylinder(obj, radius=r, half_height=h, cfg=obj_cfg, label="object_cylinder", custom_attributes=obj_attrs)
    b.add_shape_sphere(
        obj,
        radius=max(object_cfg.sphere_radius[1], HIDDEN_SIZE),
        cfg=obj_cfg,
        label="object_sphere",
        custom_attributes=obj_attrs,
    )
    return b


def build_model(num_envs: int, object_cfg: RandomObjectCfg, device: str) -> tuple[newton.Model, SceneIndex]:
    world = _world_builder(object_cfg)
    scene = newton.ModelBuilder()
    scene.rigid_gap = 0.0
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.add_ground_plane()
    scene.replicate(world, num_envs)
    model = scene.finalize(device=device)
    return model, _index(model, num_envs, device)


def _per_world(labels: list[str], world_of: np.ndarray, suffix: str, num_envs: int) -> np.ndarray:
    """Index of the entity whose label ends in ``/suffix`` (or is ``suffix``) in each world."""
    hits = np.asarray([i for i, label in enumerate(labels) if label == suffix or label.endswith("/" + suffix)])
    hits = hits[np.argsort(world_of[hits], kind="stable")] if len(hits) else hits
    if len(hits) != num_envs or not np.array_equal(world_of[hits], np.arange(num_envs)):
        raise RuntimeError(f"Expected one {suffix!r} per world, found {len(hits)}.")
    return hits


def _index(model: newton.Model, num_envs: int, device: str) -> SceneIndex:
    body_world = model.body_world.numpy()
    shape_world = model.shape_world.numpy()
    joint_world = model.joint_world.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    def bodies(name: str) -> np.ndarray:
        return _per_world(model.body_label, body_world, name, num_envs)

    def shapes(name: str) -> np.ndarray:
        return _per_world(model.shape_label, shape_world, name, num_envs)

    # The hand's root D6 joint is followed by its 20 finger joints, one DOF each,
    # in MJCF order (finger1 joint1..4, ..., finger5 joint4), as in mjlab.
    hand_root = _per_world(
        model.joint_label, joint_world, "base_x_base_y_base_z_base_roll_base_pitch_base_yaw", num_envs
    )
    hand_q = q_start[hand_root][:, None] + np.arange(NUM_BASE_DOFS + NUM_FINGER_DOFS)
    hand_qd = qd_start[hand_root][:, None] + np.arange(NUM_BASE_DOFS + NUM_FINGER_DOFS)
    # Every finger joint is one coordinate, so the block must end where the next joint starts.
    assert (q_start[hand_root + NUM_FINGER_DOFS + 1] == hand_q[:, -1] + 1).all(), "unexpected hand joint layout"

    object_joint = np.flatnonzero(np.isin(model.joint_child.numpy(), bodies("object")))
    object_joint = object_joint[np.argsort(joint_world[object_joint])]
    object_q = q_start[object_joint][:, None] + np.arange(7)
    object_qd = qd_start[object_joint][:, None] + np.arange(6)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.long, device=device)

    return SceneIndex(
        palm_body=t(bodies(PALM_BODY)),
        keypoint_bodies=t(np.stack([bodies(n) for n in KEYPOINT_BODIES], axis=-1)),
        object_body=t(bodies("object")),
        hand_q=t(hand_q),
        hand_qd=t(hand_qd),
        object_q=t(object_q),
        object_qd=t(object_qd),
        cylinder_shape=t(shapes("object_cylinder")),
        sphere_shape=t(shapes("object_sphere")),
        pad_shapes=t(np.stack([shapes(n) for n in PAD_SHAPES], axis=-1)),
    )
