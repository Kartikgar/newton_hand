"""DeepMimic-style imitation of retargeted GRAB motions with a Panda-mounted hand, on Newton.

Port of mjlab_hand's ``Mimic-Panda-*`` tasks (``mimic/mimic_env_cfg.py`` and
``mimic/mdp/``). mjlab assembles the task from manager terms; here it is one
vectorized env whose pieces are named after the mjlab terms they replace, and
whose step runs them in mjlab's order:

1. **Action** (``_process_action``, ``MimicArmHandAction``): the first 6 numbers
   move a wrist target (palm position and world rotation vector, ``pos_step`` /
   ``rot_step`` per step at 1, kept within ``max_lead_*`` of the palm and
   re-anchored on it after a reset). Damped-least-squares differential IK turns
   it into arm joint targets, with a nullspace pull toward the reference arm
   posture; the commanded twist becomes the velocity target. The rest are
   finger offsets from the reference joint angles (``finger_scale``).
2. **Physics**: 4 substeps. Each one runs mjlab's ``IdealPdActuator`` on the
   arm (a warp kernel writing ``joint_f``) and the fingers' MuJoCo position
   actuators, then records the net force each hand body presses on the table.
3. **Terminations**: time-out, ``motion_end`` (both time-outs), NaN, palm or
   object more than 10cm from the reference.
4. **Rewards** (all times dt): BeyondMimic's anchor, relative-body and velocity
   tracking terms plus object, contact, action-rate, arm joint-limit and
   table-contact terms, see ``_rewards``.
5. **Resets** (``MimicCommand._resample_command``): a motion and a start frame
   (frame 0 with ``start_prob``, else BeyondMimic's adaptive sampling: 1s bins
   weighted by where episodes recently failed), then hand, arm and object are
   written at the reference state with small joint noise. Episodes last at
   most 10s (mjlab's ``episode_length_s``), or until the clip ends.
6. **Command update**: metrics, then the reference advances one frame.
7. **Observations**: the same terms, sizes and order as mjlab (``_compute_obs``).

Positions are world-frame and every world's origin is the env origin.
"""

from __future__ import annotations

import re

import numpy as np
import torch
import warp as wp

import newton
from newton_hand.math_utils import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_rotvec,
    quat_mul,
    sample_uniform,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from newton_hand.mimic.config import MimicEnvCfg, check_against_mjlab
from newton_hand.mimic.kernels import ideal_pd_kernel, max_force_norm_kernel, table_force_kernel
from newton_hand.mimic.motions import MotionLibrary
from newton_hand.mimic.scene import build_model, load_meta
from newton_hand.sim.contacts import PadContactSensor

REWARD_NAMES = (
    "root_pos",
    "root_rot",
    "keypoint_pos",
    "joint_pos",
    "body_lin_vel",
    "root_ang_vel",
    "object_pos",
    "object_rot",
    "contacts",
    "action_rate",
    "undesired_contacts",
    "joint_limit",
)
TERMINATION_NAMES = ("time_out", "motion_end", "nan", "root_pos", "object_pos")
TIME_OUT_TERMS = ("time_out", "motion_end")
METRIC_NAMES = (
    "error_root_pos",
    "error_root_rot",
    "error_joint_pos",
    "error_keypoint_pos",
    "error_obj_pos",
    "error_obj_rot",
    "progress",
    "sampling_entropy",
    "sampling_top1_prob",
    "sampling_top1_bin",
    "error_arm_joint_pos",
)


def _clip_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm.clamp_min(max_norm))


def _kernel(sq_err: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-sq_err / std**2)


def _rot6d(quat: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, mjlab_hand's ``_rot6d``."""
    return matrix_from_quat(quat)[..., :, :2].reshape(*quat.shape[:-1], 6)


class MimicEnv:
    shapes_change_on_reset = False
    """The scene's shapes are fixed, so training videos need not reload them (``rl/video.py``)."""

    def __init__(self, cfg: MimicEnvCfg, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = N = cfg.num_envs
        self.step_dt = cfg.step_dt
        torch.manual_seed(cfg.seed)

        self.meta = meta = load_meta(cfg.task)
        for diff in check_against_mjlab(cfg, None, meta, strict=False):
            print(f"[newton_hand] differs from mjlab: {diff}")
        self.max_episode_length = int(meta["env"]["max_episode_length"])
        self.max_episode_length_s = float(meta["env"]["episode_length_s"])

        wp.init()
        self.model, self.ix, task_dir = build_model(cfg, meta, device)
        model, ix, sim = self.model, self.ix, cfg.sim
        self.solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_contacts=True,  # mjlab's collision detection; also the one that reports contact forces
            solver="newton",
            integrator=sim.integrator,
            cone=sim.cone,
            impratio=sim.impratio,
            iterations=sim.iterations,
            ls_iterations=sim.ls_iterations,
            njmax=sim.njmax,
            nconmax=sim.nconmax,
        )
        self.state = model.state()
        self._state_tmp = model.state()
        self.control = model.control()
        self.contacts = newton.Contacts(
            self.solver.mjw_data.naconmax, 0, device=model.device, requested_attributes={"force"}
        )
        self.pads = PadContactSensor(model, ix.pad_shapes, ix.object_shapes, N)

        # Torch views on Newton's buffers. ``self.state`` is always the latest state.
        self._joint_q = wp.to_torch(self.state.joint_q)
        self._joint_qd = wp.to_torch(self.state.joint_qd)
        self._body_q = wp.to_torch(self.state.body_q)
        self._body_qd = wp.to_torch(self.state.body_qd)
        self._target_q = wp.to_torch(self.control.joint_target_q)
        assert self._target_q.shape[0] == model.joint_coord_count, "expected joint_q-layout targets"
        self._body_com = wp.to_torch(model.body_com)

        # Motions and the joint layout, checked against mjlab's.
        self.motions = MotionLibrary([task_dir / "motions" / f for f in meta["motion_files"]], self.device,
                                     1.0 / self.step_dt)
        names = list(meta["joint_names"])
        if list(self.motions.arm_joint_names) != names[:7] or set(self.motions.joint_names) != set(names[7:]):
            raise ValueError(f"Motion joints do not match mjlab's {names}")
        self.arm_ids = torch.tensor([names.index(n) for n in self.motions.arm_joint_names], device=self.device)
        self.finger_ids = torch.tensor([names.index(n) for n in self.motions.joint_names], device=self.device)
        self._default_q = torch.tensor(meta["default_joint_pos"], device=self.device).expand(N, -1)
        self._soft_limits = torch.tensor(meta["soft_joint_pos_limits"], device=self.device).expand(N, -1, -1)
        self.lookahead = torch.tensor(cfg.command.lookahead, dtype=torch.long, device=self.device)

        # Arm joint frames for the palm Jacobian: hinge axis and anchor in each child body's frame.
        arm_joint = ix.arm_joints[0].cpu().numpy()
        X_c = model.joint_X_c.numpy()[arm_joint]
        axis = model.joint_axis.numpy()[model.joint_qd_start.numpy()[arm_joint]]
        self._arm_child = torch.as_tensor(model.joint_child.numpy()[ix.arm_joints.cpu().numpy()], device=self.device)
        self._arm_anchor_c = torch.tensor(X_c[:, :3], device=self.device)
        x_c_quat = xyzw_to_wxyz(torch.tensor(X_c[:, 3:7], device=self.device))
        self._arm_axis_c = quat_apply(x_c_quat, torch.tensor(axis, device=self.device))
        self._eye6 = torch.eye(6, device=self.device)
        self._eye_arm = torch.eye(7, device=self.device)

        # mjlab's IdealPdActuator gains per arm joint.
        kp, kd, limit = [], [], []
        for joint in self.motions.arm_joint_names:
            act = next(a for a in meta["arm_actuators"] if any(re.fullmatch(p, joint) for p in a["joints"]))
            kp.append(act["stiffness"]), kd.append(act["damping"]), limit.append(act["effort_limit"])
        self._kp, self._kd, self._effort = (wp.array(np.asarray(v, np.float32), device=model.device) for v in (kp, kd, limit))
        self._arm_q_idx = wp.from_torch(ix.robot_q[:, self.arm_ids].to(torch.int32).contiguous())
        self._arm_qd_idx = wp.from_torch(ix.robot_qd[:, self.arm_ids].to(torch.int32).contiguous())
        self._arm_pos_target = torch.zeros(N, 7, device=self.device)
        self._arm_vel_target = torch.zeros(N, 7, device=self.device)
        self._arm_pos_target_wp = wp.from_torch(self._arm_pos_target)
        self._arm_vel_target_wp = wp.from_torch(self._arm_vel_target)

        # Table contact forces, per world and hand body.
        P = ix.num_table_primaries
        self._net_force = wp.zeros(N * P, dtype=wp.vec3, device=model.device)
        self._peak_force = wp.zeros(N * P, dtype=wp.float32, device=model.device)
        self._peak_force_t = wp.to_torch(self._peak_force).view(N, P)
        self._shape_primary = wp.from_torch(ix.shape_table_primary)
        self._shape_is_table = wp.from_torch(ix.shape_is_table)

        # Command state.
        self.motion_ids = torch.zeros(N, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(N, dtype=torch.long, device=self.device)
        self.metrics = {name: torch.zeros(N, device=self.device) for name in METRIC_NAMES}

        # Adaptive sampling: 1s bins per motion, laid out motion after motion (mjlab_hand's MimicCommand).
        cc = cfg.command
        self._bin_count = self.motions.length // int(round(1.0 / self.step_dt)) + 1
        self._bin_start = torch.cumsum(self._bin_count, 0) - self._bin_count
        total_bins = int(self._bin_count.sum())
        self.bin_failed_count = torch.zeros(total_bins, device=self.device)
        self._current_bin_failed = torch.zeros(total_bins, device=self.device)
        kernel = torch.tensor([cc.adaptive_lambda**i for i in range(cc.adaptive_kernel_size)], device=self.device)
        self._kernel = (kernel / kernel.sum()).view(1, 1, -1)
        width = int(self._bin_count.max())
        j = torch.arange(width + cc.adaptive_kernel_size - 1, device=self.device)
        self._bin_gather = self._bin_start[:, None] + torch.minimum(j[None], self._bin_count[:, None] - 1)
        self._bin_valid = torch.arange(width, device=self.device)[None] < self._bin_count[:, None]

        # Action state.
        self.num_actions = 6 + len(self.finger_ids)
        self.actions = torch.zeros(N, self.num_actions, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self._pos_target = torch.zeros(N, 3, device=self.device)
        self._quat_target = torch.zeros(N, 4, device=self.device)
        self._quat_target[:, 0] = 1.0
        self._lin_vel_cmd = torch.zeros(N, 3, device=self.device)
        self._ang_vel_cmd = torch.zeros(N, 3, device=self.device)
        self._anchor = torch.ones(N, dtype=torch.bool, device=self.device)

        # Episode bookkeeping.
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self._episode_sums = {name: torch.zeros(N, device=self.device) for name in REWARD_NAMES}
        self.extras: dict = {"log": {}}
        self._world_mask = wp.zeros(N + 1, dtype=wp.bool, device=model.device)
        self._world_mask_t = wp.to_torch(self._world_mask)

        self._randomize_keypoint_friction()
        self._graph = None
        if sim.cuda_graph and model.device.is_cuda:
            self._simulate()  # compile every kernel before capture
            with wp.ScopedCapture(device=model.device) as capture:
                self._simulate()
            self._graph = capture.graph

        self.reset()

    # Setup.

    def _randomize_keypoint_friction(self) -> None:
        """``keypoint_friction``: fingertip sliding friction, uniform, per world and geom."""
        pads = self.ix.pad_shapes.reshape(-1)
        mu = wp.to_torch(self.model.shape_material_mu)
        mu[pads] = sample_uniform(*self.cfg.keypoint_friction, pads.numel(), self.device)
        self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    # Simulation.

    def _simulate(self) -> None:
        states = (self.state, self._state_tmp)
        dt, device = self.cfg.sim.dt, self.model.device
        N, P = self.num_envs, self.ix.num_table_primaries
        self._peak_force.zero_()
        for i in range(self.cfg.sim.decimation):
            src, dst = states[i % 2], states[(i + 1) % 2]
            wp.launch(
                ideal_pd_kernel,
                dim=(N, 7),
                inputs=[src.joint_q, src.joint_qd, self._arm_q_idx, self._arm_qd_idx, self._arm_pos_target_wp,
                        self._arm_vel_target_wp, self._kp, self._kd, self._effort],
                outputs=[self.control.joint_f],
                device=device,
            )
            self.solver.step(src, dst, self.control, None, dt)
            self.solver.update_contacts(self.contacts, dst)
            self._net_force.zero_()
            wp.launch(
                table_force_kernel,
                dim=self.contacts.rigid_contact_max,
                inputs=[self.contacts.rigid_contact_count, self.contacts.rigid_contact_shape0,
                        self.contacts.rigid_contact_shape1, self.contacts.force, self.model.shape_world,
                        self._shape_primary, self._shape_is_table, P],
                outputs=[self._net_force],
                device=device,
            )
            wp.launch(max_force_norm_kernel, dim=N * P, inputs=[self._net_force], outputs=[self._peak_force],
                      device=device)
        if self.cfg.sim.decimation % 2:
            self.state.assign(self._state_tmp)
        self.pads.update(self.contacts, self.state)

    def _forward(self) -> None:
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    # Kinematics, in mjlab's conventions: quaternions (w, x, y, z), body-origin velocities.

    def _pose(self, bodies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bq = self._body_q[bodies]
        return bq[..., :3], xyzw_to_wxyz(bq[..., 3:7])

    def _link_velocity(self, bodies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Body-origin linear velocity and angular velocity, world frame (Newton's ``body_qd`` is at the COM)."""
        pos, quat = self._pose(bodies)
        qd = self._body_qd[bodies]
        lin, ang = qd[..., :3], qd[..., 3:6]
        com_w = pos + quat_apply(quat, self._body_com[bodies])
        return lin + torch.cross(ang, pos - com_w, dim=-1), ang

    @property
    def robot_q(self) -> torch.Tensor:
        return self._joint_q[self.ix.robot_q]

    @property
    def robot_qd(self) -> torch.Tensor:
        return self._joint_qd[self.ix.robot_qd]

    def _palm(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pose(self.ix.palm_body)

    # Reference lookup (MimicCommand).

    @property
    def frame(self) -> torch.Tensor:
        return self.motions.start[self.motion_ids] + self.time_steps

    @property
    def motion_length(self) -> torch.Tensor:
        return self.motions.length[self.motion_ids]

    def ref(self, name: str, offsets: torch.Tensor | None = None) -> torch.Tensor:
        values = getattr(self.motions, name)
        if offsets is None:
            return values[self.frame]
        last = (self.motion_length - 1)[:, None]
        steps = torch.minimum(self.time_steps[:, None] + offsets[None, :], last)
        return values[self.motions.start[self.motion_ids][:, None] + steps]

    @property
    def ref_contact(self) -> torch.Tensor:
        return self.ref("keypoint_obj_dist") < self.cfg.command.contact_threshold

    def _ref_keypoint_pos_relative(self, palm_pos: torch.Tensor, palm_quat: torch.Tensor) -> torch.Tensor:
        """Reference fingertips carried onto the simulated palm (BeyondMimic's ``body_pos_relative_w``)."""
        k = self.ix.keypoint_bodies.shape[1]
        delta = quat_mul(palm_quat[:, None].expand(-1, k, -1),
                         quat_conjugate(self.ref("root_quat")[:, None].expand(-1, k, -1)))
        offset = self.ref("keypoint_pos") - self.ref("root_pos")[:, None]
        return palm_pos[:, None] + quat_apply(delta, offset)

    def _command_obs(self, palm_pos: torch.Tensor, palm_quat: torch.Tensor) -> torch.Tensor:
        n, k = self.num_envs, len(self.lookahead)
        pp = palm_pos[:, None, :].expand(n, k, 3)
        pq = palm_quat[:, None, :].expand(n, k, 4)
        inv = quat_conjugate(pq)
        root_pos = quat_apply_inverse(pq, self.ref("root_pos", self.lookahead) - pp)
        root_quat = quat_mul(inv, self.ref("root_quat", self.lookahead))
        obj_pos = quat_apply_inverse(pq, self.ref("obj_pos", self.lookahead) - pp)
        obj_quat = quat_mul(inv, self.ref("obj_quat", self.lookahead))
        return torch.cat(
            [
                root_pos.reshape(n, -1),
                _rot6d(root_quat).reshape(n, -1),
                obj_pos.reshape(n, -1),
                _rot6d(obj_quat).reshape(n, -1),
                self.ref("joint_pos", self.lookahead).reshape(n, -1),
                self.ref("arm_joint_pos", self.lookahead).reshape(n, -1),
            ],
            dim=-1,
        )

    # RL interface.

    def reset(self) -> dict[str, torch.Tensor]:
        self.extras = {"log": {}}
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_command()
        self.obs = self._compute_obs()
        return self.obs

    def step(self, actions: torch.Tensor):
        self.extras = {"log": {}}
        self._process_action(actions.to(self.device))
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        self.episode_length_buf += 1
        terms = self._terminations()
        self.last_terminations = terms
        time_outs = terms["time_out"] | terms["motion_end"]
        terminated = terms["nan"] | terms["root_pos"] | terms["object_pos"]
        rewards = self._rewards()

        reset_ids = (terminated | time_outs).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._log_episodes(reset_ids, terms)
            self._reset_idx(reset_ids, failed=terminated[reset_ids])
        self._update_command()
        self.obs = self._compute_obs()
        self.extras["time_outs"] = time_outs
        return self.obs, rewards, terminated, time_outs, self.extras

    # Reset (MimicCommand._resample_command and the action term's reset).

    def _reset_idx(self, env_ids: torch.Tensor, failed: torch.Tensor | None = None) -> None:
        cc, ix, dev, m = self.cfg.command, self.ix, self.device, self.motions
        n = len(env_ids)

        adaptive = cc.sampling_mode == "adaptive"
        if adaptive and failed is not None and torch.any(failed):
            # Episodes that ended in failure count against the bin they ended in.
            bins = self._bin_of(env_ids)[failed]
            self._current_bin_failed[:] = torch.bincount(bins, minlength=self.bin_failed_count.numel())
        self.motion_ids[env_ids] = (torch.rand(n, device=dev) * len(m)).long()
        # Leave at least one step to play.
        last = (m.length[self.motion_ids[env_ids]] - 2).clamp_min(0)
        if adaptive:
            steps = self._adaptive_steps(env_ids).clamp(max=last)
        else:
            steps = (torch.rand(n, device=dev) * (last + 1).float()).long()
        from_start = torch.rand(n, device=dev) < cc.start_prob
        self.time_steps[env_ids] = torch.where(from_start, torch.zeros_like(steps), steps)
        self._write_reference_state(env_ids)

        # Action term: zeroed actions, wrist target re-anchored on the palm at the next step.
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._anchor[env_ids] = True

        self.episode_length_buf[env_ids] = 0
        for sums in self._episode_sums.values():
            sums[env_ids] = 0.0

    def _bin_of(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Global bin index of each env's current frame."""
        m = self.motion_ids[env_ids]
        count = self._bin_count[m]
        local = torch.minimum(self.time_steps[env_ids] * count // self.motions.length[m].clamp_min(1), count - 1)
        return self._bin_start[m] + local

    def _adaptive_steps(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Start frames drawn from each env's motion's failure-weighted bins (mjlab_hand's ``_adaptive_steps``)."""
        cc = self.cfg.command
        uniform = cc.adaptive_uniform_ratio / self._bin_count.float()
        weight = self.bin_failed_count[self._bin_gather] + uniform[:, None]
        weight = torch.nn.functional.conv1d(weight[:, None], self._kernel).squeeze(1)
        prob = weight * self._bin_valid
        prob = prob / prob.sum(-1, keepdim=True)

        m = self.motion_ids[env_ids]
        rows = prob[m]
        bins = torch.multinomial(rows, 1).squeeze(-1)
        count, length = self._bin_count[m], self.motions.length[m]
        steps = ((bins + torch.rand(len(env_ids), device=self.device)) / count * (length - 1)).long()

        entropy = -(rows * (rows + 1e-12).log()).sum(-1)
        top_prob, top_bin = rows.max(-1)
        self.metrics["sampling_entropy"][env_ids] = torch.where(
            count > 1, entropy / count.float().log().clamp_min(1e-12), torch.ones_like(entropy)
        )
        self.metrics["sampling_top1_prob"][env_ids] = top_prob
        self.metrics["sampling_top1_bin"][env_ids] = top_bin.float() / count
        return steps

    def _write_reference_state(self, env_ids: torch.Tensor) -> None:
        """Hand, arm and object at each env's current reference frame, with the joint reset noise."""
        cc, ix, dev, m = self.cfg.command, self.ix, self.device, self.motions
        n = len(env_ids)
        f = self.frame[env_ids]
        q = torch.zeros(n, len(self.meta["joint_names"]), device=dev)
        qd = torch.zeros_like(q)
        q[:, self.finger_ids] = m.joint_pos[f] + sample_uniform(-cc.joint_pos_noise, cc.joint_pos_noise,
                                                                m.joint_pos[f].shape, dev)
        qd[:, self.finger_ids] = m.joint_vel[f]
        q[:, self.arm_ids] = m.arm_joint_pos[f] + sample_uniform(-cc.arm_joint_noise, cc.arm_joint_noise,
                                                                 m.arm_joint_pos[f].shape, dev)
        qd[:, self.arm_ids] = m.arm_joint_vel[f]
        soft = self._soft_limits[env_ids]
        self._joint_q[ix.robot_q[env_ids]] = torch.clip(q, soft[..., 0], soft[..., 1])
        self._joint_qd[ix.robot_qd[env_ids]] = qd

        # The object at the reference: mjlab writes the body-origin velocity,
        # Newton's free joint takes the COM's.
        obj_quat = m.obj_quat[f]
        ang = m.obj_ang_vel[f]
        com_offset = quat_apply(obj_quat, self._body_com[ix.object_body[env_ids]])
        lin = m.obj_lin_vel[f] + torch.cross(ang, com_offset, dim=-1)
        self._joint_q[ix.object_q[env_ids]] = torch.cat([m.obj_pos[f], wxyz_to_xyzw(obj_quat)], dim=-1)
        self._joint_qd[ix.object_qd[env_ids]] = torch.cat([lin, ang], dim=-1)

        self._forward()
        self._world_mask_t.zero_()
        self._world_mask_t[env_ids] = True
        self.solver.reset(self.state, world_mask=self._world_mask, flags=0)

    # Action (MimicArmHandAction).

    def _palm_jacobian(self, palm_pos: torch.Tensor) -> torch.Tensor:
        """``(N, 6, 7)`` world-frame palm Jacobian of the arm hinges (mjlab reads ``xaxis``/``xanchor``)."""
        pos, quat = self._pose(self._arm_child)
        anchor = pos + quat_apply(quat, self._arm_anchor_c.expand_as(pos))
        axis = quat_apply(quat, self._arm_axis_c.expand_as(pos))
        lever = palm_pos[:, None] - anchor
        return torch.cat([torch.cross(axis, lever, dim=-1), axis], dim=-1).transpose(1, 2)

    def _process_action(self, actions: torch.Tensor) -> None:
        ac, dt, ix = self.cfg.action, self.step_dt, self.ix
        self.prev_actions[:] = self.actions
        self.actions[:] = actions
        a = actions.clamp(-1.0, 1.0)
        palm_pos, palm_quat = self._palm()

        # Wrist target.
        if self._anchor.any():
            self._pos_target[self._anchor] = palm_pos[self._anchor]
            self._quat_target[self._anchor] = palm_quat[self._anchor]
            self._anchor[:] = False
        dpos, drot = a[:, :3] * ac.pos_step, a[:, 3:6] * ac.rot_step
        self._pos_target = self._pos_target + dpos
        self._quat_target = quat_mul(quat_from_rotvec(drot), self._quat_target)
        self._lin_vel_cmd, self._ang_vel_cmd = dpos / dt, drot / dt
        self._pos_target = palm_pos + _clip_norm(self._pos_target - palm_pos, ac.max_lead_pos)
        lead_rot = axis_angle_from_quat(quat_mul(self._quat_target, quat_conjugate(palm_quat)))
        self._quat_target = quat_mul(quat_from_rotvec(_clip_norm(lead_rot, ac.max_lead_rot)), palm_quat)

        # Differential IK.
        jac = self._palm_jacobian(palm_pos)
        jac_t = jac.transpose(1, 2)
        pinv = jac_t @ torch.linalg.inv(jac @ jac_t + ac.ik_damping**2 * self._eye6)
        err = torch.cat(
            [self._pos_target - palm_pos, axis_angle_from_quat(quat_mul(self._quat_target, quat_conjugate(palm_quat)))],
            dim=-1,
        )
        q_all = self.robot_q
        q = q_all[:, self.arm_ids]
        posture = ac.nullspace_gain * (self.ref("arm_joint_pos") - q)
        null = self._eye_arm - pinv @ jac
        dq = (pinv @ err[..., None] + null @ posture[..., None]).squeeze(-1)
        dq = dq.clamp(-ac.max_joint_step, ac.max_joint_step)
        twist = torch.cat([self._lin_vel_cmd, self._ang_vel_cmd], dim=-1)
        qdot = (pinv @ twist[..., None]).squeeze(-1)

        # Fingers: residual on the reference, within the soft limits; then every target within them.
        soft = self._soft_limits
        fingers = self.ref("joint_pos") + a[:, 6:] * ac.finger_scale
        fingers = torch.clip(fingers, soft[:, self.finger_ids, 0], soft[:, self.finger_ids, 1])
        target = torch.empty_like(q_all)
        target[:, self.arm_ids] = q + dq
        target[:, self.finger_ids] = fingers
        target = torch.clip(target, soft[..., 0], soft[..., 1])

        self._arm_pos_target.copy_(target[:, self.arm_ids])
        self._arm_vel_target.copy_(qdot)
        self._target_q[ix.robot_q[:, self.finger_ids]] = target[:, self.finger_ids]

    # Terminations, rewards, command update.

    def _terminations(self) -> dict[str, torch.Tensor]:
        ix, tc = self.ix, self.cfg.termination
        palm_pos, _ = self._palm()
        obj_pos, _ = self._pose(ix.object_body)
        state = torch.cat([self.robot_q, self.robot_qd, self._joint_q[ix.object_q], self._joint_qd[ix.object_qd]], -1)
        return {
            "time_out": self.episode_length_buf >= self.max_episode_length,
            "motion_end": self.time_steps >= self.motion_length - 1,
            "nan": ~torch.isfinite(state).all(dim=-1),
            "root_pos": (self.ref("root_pos") - palm_pos).norm(dim=-1) > tc.root_pos_threshold,
            "object_pos": (self.ref("obj_pos") - obj_pos).norm(dim=-1) > tc.object_pos_threshold,
        }

    def _rewards(self) -> torch.Tensor:
        r, ix = self.cfg.rewards, self.ix
        palm_pos, palm_quat = self._palm()
        palm_lin, palm_ang = self._link_velocity(ix.palm_body)
        tips, _ = self._pose(ix.keypoint_bodies)
        tips_lin, _ = self._link_velocity(ix.keypoint_bodies)
        obj_pos, obj_quat = self._pose(ix.object_body)
        q = self.robot_q

        ref_lin = torch.cat([self.ref("root_lin_vel")[:, None], self.ref("keypoint_lin_vel")], dim=1)
        lin = torch.cat([palm_lin[:, None], tips_lin], dim=1)
        ref_c = self.ref_contact
        touching = self.pads.touching
        arm_q = q[:, self.arm_ids]
        arm_soft = self._soft_limits[:, self.arm_ids]
        out_of_limits = (-(arm_q - arm_soft[..., 0]).clip(max=0.0) + (arm_q - arm_soft[..., 1]).clip(min=0.0)).sum(-1)

        values = {
            "root_pos": _kernel((self.ref("root_pos") - palm_pos).square().sum(-1), r.root_pos.std),
            "root_rot": _kernel(quat_error_magnitude(self.ref("root_quat"), palm_quat).square(), r.root_rot.std),
            "keypoint_pos": _kernel(
                (self._ref_keypoint_pos_relative(palm_pos, palm_quat) - tips).square().sum(-1).mean(-1),
                r.keypoint_pos.std,
            ),
            "joint_pos": _kernel((self.ref("joint_pos") - q[:, self.finger_ids]).square().mean(-1), r.joint_pos.std),
            "body_lin_vel": _kernel((ref_lin - lin).square().sum(-1).mean(-1), r.body_lin_vel.std),
            "root_ang_vel": _kernel((self.ref("root_ang_vel") - palm_ang).square().sum(-1), r.root_ang_vel.std),
            "object_pos": _kernel((self.ref("obj_pos") - obj_pos).square().sum(-1), r.object_pos.std),
            "object_rot": _kernel(quat_error_magnitude(self.ref("obj_quat"), obj_quat).square(), r.object_rot.std),
            "contacts": (ref_c & touching).sum(-1).float() / ref_c.sum(-1).clamp_min(1).float(),
            "action_rate": (self.actions - self.prev_actions).square().sum(-1),
            "undesired_contacts": (self._peak_force_t > r.undesired_contact_threshold).sum(-1).float(),
            "joint_limit": out_of_limits,
        }
        total = torch.zeros(self.num_envs, device=self.device)
        for name, value in values.items():
            value = getattr(r, name).weight * value * self.step_dt
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            total += value
            self._episode_sums[name] += value
        return total

    def _update_command(self) -> None:
        """mjlab's command compute: metrics, the failure average, then the reference advances one frame."""
        if self.cfg.command.sampling_mode == "adaptive":
            alpha = self.cfg.command.adaptive_alpha
            self.bin_failed_count = alpha * self._current_bin_failed + (1 - alpha) * self.bin_failed_count
            self._current_bin_failed.zero_()
        palm_pos, palm_quat = self._palm()
        obj_pos, obj_quat = self._pose(self.ix.object_body)
        tips, _ = self._pose(self.ix.keypoint_bodies)
        q = self.robot_q
        self.metrics["error_root_pos"] = (self.ref("root_pos") - palm_pos).norm(dim=-1)
        self.metrics["error_root_rot"] = quat_error_magnitude(self.ref("root_quat"), palm_quat)
        self.metrics["error_joint_pos"] = (self.ref("joint_pos") - q[:, self.finger_ids]).norm(dim=-1)
        self.metrics["error_arm_joint_pos"] = (self.ref("arm_joint_pos") - q[:, self.arm_ids]).norm(dim=-1)
        self.metrics["error_keypoint_pos"] = (self.ref("keypoint_pos") - tips).norm(dim=-1).mean(-1)
        self.metrics["error_obj_pos"] = (self.ref("obj_pos") - obj_pos).norm(dim=-1)
        self.metrics["error_obj_rot"] = quat_error_magnitude(self.ref("obj_quat"), obj_quat)
        self.metrics["progress"] = self.time_steps.float() / (self.motion_length - 1).float()
        self.time_steps = torch.minimum(self.time_steps + 1, self.motion_length - 1)

    def _log_episodes(self, env_ids: torch.Tensor, terms: dict[str, torch.Tensor]) -> None:
        log = self.extras["log"]
        for name, sums in self._episode_sums.items():
            log[f"Episode_Reward/{name}"] = sums[env_ids].mean() / self.max_episode_length_s
        for name, value in self.metrics.items():
            log[f"Metrics/motion/{name}"] = value[env_ids].mean()
        for name, value in terms.items():
            log[f"Episode_Termination/{name}"] = value[env_ids].sum().float()

    # Observations.

    _obs_checked = False

    def _check_obs_layout(self, terms: dict[str, torch.Tensor]) -> None:
        """Term names, sizes, order and noise must be mjlab's (``mjlab.json``)."""
        ours = [(name, v.shape[1], _ACTOR_NOISE.get(name)) for name, v in terms.items()]
        theirs = [(t["name"], t["dim"], None if t["noise"] is None else t["noise"][1])
                  for t in self.meta["observations"]["actor"]]
        if ours != theirs:
            raise ValueError(f"Actor observations differ from mjlab's:\n  newton {ours}\n  mjlab  {theirs}")
        critic = [t["name"] for t in self.meta["observations"]["critic"]]
        if critic != [*terms, "object_vel", "contacts", "reference_contacts", "tracking_errors"]:
            raise ValueError(f"Critic observations differ from mjlab's: {critic}")
        self._obs_checked = True

    def _compute_obs(self) -> dict[str, torch.Tensor]:
        """mjlab's actor terms in order. The critic adds object velocity and fingertip contacts (hard to
        measure on a real robot, so the actor doesn't get them), reference contacts and tracking errors."""
        ix, n = self.ix, self.num_envs
        palm_pos, palm_quat = self._palm()
        palm_lin, palm_ang = self._link_velocity(ix.palm_body)
        tips, _ = self._pose(ix.keypoint_bodies)
        obj_pos, obj_quat = self._pose(ix.object_body)
        obj_lin, obj_ang = self._link_velocity(ix.object_body)
        k = tips.shape[1]
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(n, 3)

        terms = {
            "reference": self._command_obs(palm_pos, palm_quat),
            "phase": (self.time_steps.float() / (self.motion_length - 1).float())[:, None],
            "joint_pos": self.robot_q - self._default_q,
            "joint_vel": self.robot_qd,
            "palm_vel": torch.cat([quat_apply_inverse(palm_quat, palm_lin), quat_apply_inverse(palm_quat, palm_ang)], -1),
            "palm_gravity": quat_apply_inverse(palm_quat, down),
            "keypoint_pos": quat_apply_inverse(palm_quat[:, None].expand(-1, k, -1), tips - palm_pos[:, None]),
            "object_pose": torch.cat(
                [
                    quat_apply_inverse(palm_quat, obj_pos - palm_pos),
                    matrix_from_quat(quat_mul(quat_conjugate(palm_quat), obj_quat))[..., :2].reshape(n, 6),
                ],
                dim=-1,
            ),
            "actions": self.actions,
        }
        terms = {name: v.reshape(n, -1) for name, v in terms.items()}
        if not self._obs_checked:
            self._check_obs_layout(terms)
        critic = torch.cat(
            [
                *terms.values(),
                quat_apply_inverse(palm_quat, obj_lin),
                quat_apply_inverse(palm_quat, obj_ang),
                self.pads.touching.float(),
                self.ref_contact.float(),
                torch.cat(
                    [
                        self.ref("root_pos") - palm_pos,
                        quat_error_magnitude(self.ref("root_quat"), palm_quat)[:, None],
                        self.ref("obj_pos") - obj_pos,
                        quat_error_magnitude(self.ref("obj_quat"), obj_quat)[:, None],
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        if self.cfg.obs_noise:
            for name, scale in _ACTOR_NOISE.items():
                terms[name] = terms[name] + sample_uniform(-scale, scale, terms[name].shape, self.device)
        actor = torch.cat(list(terms.values()), dim=-1)
        if not (torch.isfinite(actor).all() and torch.isfinite(critic).all()):
            raise RuntimeError("Non-finite observation; mjlab's nan_policy='error'.")
        return {"actor": actor, "critic": critic}


# Uniform noise half-widths on the actor's observations (mjlab's Unoise terms).
_ACTOR_NOISE = {
    "joint_pos": 0.01,
    "joint_vel": 0.5,
    "palm_vel": 0.05,
    "object_pose": 0.005,
}

__all__ = ["MimicEnv"]
