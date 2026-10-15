"""Quaternion helpers in torch, ``(w, x, y, z)`` like mjlab.

Newton and warp store quaternions as ``(x, y, z, w)``. Everything in this
package that is not a warp kernel works in ``(w, x, y, z)``, so the code reads
like the mjlab version it was ported from; convert at the boundary with
:func:`xyzw_to_wxyz` and :func:`wxyz_to_xyzw`.
"""

from __future__ import annotations

import torch


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:4], q[..., :3]], dim=-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by unit quaternion ``q``; broadcasts over leading dims."""
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    m = torch.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Extrinsic XYZ (= intrinsic ZYX) Euler angles to a quaternion, as mjlab's helper."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dim=-1,
    )


def sample_uniform(lo: float, hi: float, shape, device) -> torch.Tensor:
    return torch.rand(shape, device=device) * (hi - lo) + lo


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Rotation vector of a quaternion, shortest arc; mjlab's ``axis_angle_from_quat``."""
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_over_angle = torch.where(angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48)
    return quat[..., 1:4] / sin_half_over_angle.unsqueeze(-1)


def quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """Quaternion of a rotation vector; mjlab_hand's ``_quat_from_rotvec`` (zero vector -> identity)."""
    angle = rotvec.norm(dim=-1)
    axis = rotvec / angle.clamp_min(1e-9).unsqueeze(-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    half = (angle / 2).unsqueeze(-1)
    return torch.cat([half.cos(), axis * half.sin()], dim=-1)


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angle between two orientations; mjlab's ``quat_error_magnitude``."""
    return axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2))).norm(dim=-1)
