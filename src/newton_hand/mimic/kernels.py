"""Per-substep warp kernels for the mimic env, launched inside the captured decimation loop."""

from __future__ import annotations

import warp as wp


@wp.kernel
def ideal_pd_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    q_idx: wp.array2d[wp.int32],
    qd_idx: wp.array2d[wp.int32],
    pos_target: wp.array2d[wp.float32],
    vel_target: wp.array2d[wp.float32],
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    effort_limit: wp.array[wp.float32],
    # output
    joint_f: wp.array[wp.float32],
):
    """mjlab's ``IdealPdActuator``: ``clip(kp (q* - q) + kd (qd* - qd), +-limit)``, from the current state."""
    world, j = wp.tid()
    qi = q_idx[world, j]
    di = qd_idx[world, j]
    tau = stiffness[j] * (pos_target[world, j] - joint_q[qi]) + damping[j] * (vel_target[world, j] - joint_qd[di])
    joint_f[di] = wp.clamp(tau, -effort_limit[j], effort_limit[j])


@wp.kernel
def table_force_kernel(
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    force: wp.array[wp.spatial_vector],
    shape_world: wp.array[wp.int32],
    shape_primary: wp.array[wp.int32],
    shape_is_table: wp.array[wp.int32],
    num_primaries: int,
    # output
    net_force: wp.array[wp.vec3],
):
    """Sum, per world and hand body, the contact forces between that body and the table."""
    c = wp.tid()
    if c >= count[0]:
        return
    a = shape0[c]
    b = shape1[c]
    if a < 0 or b < 0:
        return
    f = wp.spatial_top(force[c])  # world-frame force on shape0
    primary = shape_primary[a]
    other = b
    if primary < 0:
        primary = shape_primary[b]
        other = a
        f = -f
    if primary < 0 or shape_is_table[other] == 0:
        return
    wp.atomic_add(net_force, shape_world[a] * num_primaries + primary, f)


@wp.kernel
def max_force_norm_kernel(net_force: wp.array[wp.vec3], peak: wp.array[wp.float32]):
    """Keep the largest net-force magnitude seen this policy step (mjlab's force history, ``amax``)."""
    i = wp.tid()
    peak[i] = wp.max(peak[i], wp.length(net_force[i]))
