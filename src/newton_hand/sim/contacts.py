"""Fingertip-pad vs object contacts, the Newton stand-in for mjlab's ``keypoint_contacts`` sensor.

mjlab's sensor (``reduce="netforce"``, one slot per fingertip pad) reports per
pad whether it touches the object and where. Here one kernel walks Newton's
``Contacts`` buffer -- filled by Newton's ``CollisionPipeline``, or converted
from MuJoCo-Warp's contacts by ``SolverMuJoCo.update_contacts`` -- and, for
every penetrating contact between a pad and the object, counts it and sums its
world position (the midpoint between the two surfaces, like MuJoCo's
``contact.pos``). The touch point is the mean of those positions.
"""

from __future__ import annotations

import torch
import warp as wp

import newton


@wp.kernel
def _pad_object_contacts(
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[wp.float32],
    margin1: wp.array[wp.float32],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_pad: wp.array[wp.int32],
    shape_is_object: wp.array[wp.int32],
    num_pads: int,
    # outputs
    found: wp.array[wp.float32],
    pos_sum: wp.array[wp.vec3],
):
    c = wp.tid()
    if c >= count[0]:
        return
    a = shape0[c]
    b = shape1[c]
    if a < 0 or b < 0:
        return
    pad = shape_pad[a]
    other = b
    if pad < 0:
        pad = shape_pad[b]
        other = a
    if pad < 0 or shape_is_object[other] == 0:
        return

    pa = point0[c]
    body_a = shape_body[a]
    if body_a >= 0:
        pa = wp.transform_point(body_q[body_a], pa)
    pb = point1[c]
    body_b = shape_body[b]
    if body_b >= 0:
        pb = wp.transform_point(body_q[body_b], pb)
    # Signed gap between the surfaces; only touching contacts count, as in MuJoCo.
    dist = wp.dot(normal[c], pb - pa) - (margin0[c] + margin1[c])
    if dist > 0.0:
        return

    slot = shape_world[a] * num_pads + pad
    wp.atomic_add(found, slot, 1.0)
    wp.atomic_add(pos_sum, slot, 0.5 * (pa + pb))


class PadContactSensor:
    """Per world and fingertip pad: touching (0/1) and the mean touch point in the world frame."""

    def __init__(self, model: newton.Model, pad_shapes: torch.Tensor, object_shapes: torch.Tensor, num_envs: int):
        self.model = model
        self.num_envs = num_envs
        self.num_pads = int(pad_shapes.shape[1])
        device = model.device

        shape_pad = torch.full((model.shape_count,), -1, dtype=torch.int32)
        for pad in range(self.num_pads):
            shape_pad[pad_shapes[:, pad].cpu()] = pad
        shape_is_object = torch.zeros(model.shape_count, dtype=torch.int32)
        shape_is_object[object_shapes.reshape(-1).cpu()] = 1
        self._shape_pad = wp.from_torch(shape_pad.to(str(device)))
        self._shape_is_object = wp.from_torch(shape_is_object.to(str(device)))

        self._found = wp.zeros(num_envs * self.num_pads, dtype=wp.float32, device=device)
        self._pos_sum = wp.zeros(num_envs * self.num_pads, dtype=wp.vec3, device=device)
        self.found = wp.to_torch(self._found).view(num_envs, self.num_pads)
        self._pos_sum_t = wp.to_torch(self._pos_sum).view(num_envs, self.num_pads, 3)

    def update(self, contacts: newton.Contacts, state: newton.State) -> None:
        self._found.zero_()
        self._pos_sum.zero_()
        wp.launch(
            _pad_object_contacts,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.model.shape_body,
                self.model.shape_world,
                state.body_q,
                self._shape_pad,
                self._shape_is_object,
                self.num_pads,
            ],
            outputs=[self._found, self._pos_sum],
            device=self.model.device,
        )

    @property
    def touching(self) -> torch.Tensor:
        """``(N, 5)`` bool."""
        return self.found > 0

    @property
    def pos(self) -> torch.Tensor:
        """``(N, 5, 3)`` mean touch point, world frame; zero where not touching."""
        return self._pos_sum_t / self.found.clamp_min(1.0).unsqueeze(-1)
