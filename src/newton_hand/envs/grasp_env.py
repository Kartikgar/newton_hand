"""The random-object METRA grasp task on Newton.

Port of mjlab_hand's ``Grasp-METRA-Hover-Lift-Contacts-Push-RandomObject-Wuji``.
mjlab assembles that task from manager terms spread over a dozen files; here it
is one vectorized env, with each piece named after the mjlab term it replaces:

- **Reset** (``_reset_idx``): ``reset_joints_by_offset`` on every hand joint,
  ``randomize_primitive_object``, ``reset_object_on_table``,
  ``reset_hand_above_object`` (palm 5cm above the object's top, fingers over
  it via ``palm_offset``), then the action, skill, METRA and push state.
- **Action** (``_process_action``): ``FingerDeltaScriptedLiftAction`` with
  ``control_base=True``. 26-D in [-1, 1]: base translation and rotation
  velocities, then 20 finger joint velocities, integrated into position
  targets and EMA-smoothed. From ``lift_start_s`` the base actions stop
  counting and the base moves along a direction drawn per episode within
  ``lift_cone_deg`` of vertical.
- **Push** (``_push_step``): ``push_object_in_air``. While the object is 3cm
  above where it settled, a random 3-7N force for 0.1s every 1.5s.
- **Reward**: METRA ``(phi(s') - phi(s))^T z`` on contacts-only phi, the L1
  finger-velocity smoothness penalty, and a one-shot drop penalty of 10 when
  the episode ends because the object fell or left the 22cm palm radius.
- **Terminations**: time-out (10s), NaN state, object fallen, palm-object
  distance, and object spin (a time-out, guarding against divergence).
- **Observations**: the same 177 actor / 178 critic numbers in the same order
  as mjlab (see ``_compute_obs``).

Positions are world-frame and every world's origin is the env origin, so no
env-origin correction is needed anywhere.
"""

from __future__ import annotations

import math

import torch
import warp as wp

import newton
from newton_hand.config import EnvCfg
from newton_hand.math_utils import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from newton_hand.metra.reward import CONTACT_FEATURES_PER_PRIMARY, MetraReward
from newton_hand.objects.random_primitive import HIDDEN_SIZE, PEN, PEN_QUAT_WXYZ, SPHERE, RandomPrimitiveState
from newton_hand.sim.contacts import PadContactSensor
from newton_hand.sim.scene import (
    FINGER_HOME,
    NUM_BASE_DOFS,
    NUM_FINGER_DOFS,
    TABLE_HALF,
    TABLE_POS,
    build_model,
)

NUM_ACTIONS = NUM_BASE_DOFS + NUM_FINGER_DOFS
SOFT_LIMIT_FACTOR = 0.9  # mjlab's soft_joint_pos_limit_factor for the Wuji

TERMINATION_NAMES = ("time_out", "nan_term", "object_fallen", "palm_object_distance", "object_spin")
TIME_OUT_TERMS = ("time_out", "object_spin")
DROP_TERMS = ("object_fallen", "palm_object_distance")
REWARD_NAMES = ("smoothness", "metra", "object_dropped")


class GraspEnv:
    def __init__(self, cfg: EnvCfg, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = N = cfg.num_envs
        self.step_dt = cfg.step_dt
        self.max_episode_length = cfg.max_episode_length
        torch.manual_seed(cfg.seed)

        wp.init()
        self.model, self.ix = build_model(N, cfg.object, device)
        model = self.model
        sim = cfg.sim
        self.solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_contacts=sim.collision == "mujoco",
            enable_multiccd=sim.multiccd,
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
        if sim.collision == "newton":
            self.pipeline = newton.CollisionPipeline(model)
            self.contacts = self.pipeline.contacts()
        else:
            self.pipeline = None
            self.contacts = newton.Contacts(
                self.solver.mjw_data.naconmax, 0, device=model.device, requested_attributes=set()
            )
        self.pads = PadContactSensor(
            model, self.ix.pad_shapes, torch.stack([self.ix.cylinder_shape, self.ix.sphere_shape], -1), N
        )

        # Torch views on Newton's buffers. ``self.state`` is always the latest state.
        self._joint_q = wp.to_torch(self.state.joint_q)
        self._joint_qd = wp.to_torch(self.state.joint_qd)
        self._body_q = wp.to_torch(self.state.body_q)
        self._body_qd = wp.to_torch(self.state.body_qd)
        self._body_f = (wp.to_torch(self.state.body_f), wp.to_torch(self._state_tmp.body_f))
        self._target_q = wp.to_torch(self.control.joint_target_q)
        # Newton 1.6 lays targets out like joint_q; the hand's coordinates and DOFs coincide.
        assert self._target_q.shape[0] == model.joint_coord_count, "expected joint_q-layout targets"
        self._body_com = wp.to_torch(model.body_com)
        self._shape_scale = wp.to_torch(model.shape_scale)
        self._shape_xform = wp.to_torch(model.shape_transform)
        self._shape_radius = wp.to_torch(model.shape_collision_radius)
        self._body_mass = wp.to_torch(model.body_mass)
        self._body_inv_mass = wp.to_torch(model.body_inv_mass)
        self._body_inertia = wp.to_torch(model.body_inertia)
        self._body_inv_inertia = wp.to_torch(model.body_inv_inertia)

        # Joint defaults and limits, per world: (N, 26).
        lower = wp.to_torch(model.joint_limit_lower)[self.ix.hand_qd]
        upper = wp.to_torch(model.joint_limit_upper)[self.ix.hand_qd]
        self._limits = torch.stack([lower, upper], dim=-1)
        mid, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
        self._soft_limits = torch.stack([mid - SOFT_LIMIT_FACTOR * half, mid + SOFT_LIMIT_FACTOR * half], dim=-1)
        home = torch.tensor((0.0,) * NUM_BASE_DOFS + FINGER_HOME, device=self.device)
        self._default_q = home.expand(N, -1).clone()

        # World direction of each base slide joint, (3, 3) rows. The slides come
        # first in the joint chain, so they are fixed by the palm's mount
        # orientation and never turn with the hand.
        joint_X_p = wp.to_torch(model.joint_X_p)[self._hand_root_joint(0)]
        mount = xyzw_to_wxyz(joint_X_p[3:7])
        self._slide_axes = quat_apply(mount.expand(3, 4), torch.eye(3, device=self.device))

        self.objects = RandomPrimitiveState(N, str(self.device), cfg.object, seed=cfg.seed)

        mc = cfg.metra
        self.metra = MetraReward(mc, N, num_primaries=self.ix.pad_shapes.shape[1], device=self.device)
        self.skill = torch.zeros(N, mc.skill_dim, device=self.device)

        # Action state.
        self.actions = torch.zeros(N, NUM_ACTIONS, device=self.device)
        self._finger_target = torch.zeros(N, NUM_FINGER_DOFS, device=self.device)
        self._finger_ema = torch.zeros_like(self._finger_target)
        self._base_target = torch.zeros(N, NUM_BASE_DOFS, device=self.device)
        self._base_ema = torch.zeros_like(self._base_target)
        self._lift_velocity = torch.zeros(N, 3, device=self.device)
        ac = cfg.action
        self._base_vel_max = torch.tensor([ac.trans_vel] * 3 + [ac.rot_vel] * 3, device=self.device)

        # Push state.
        self._push_rest_z = torch.zeros(N, device=self.device)
        self._push_initialized = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._push_active = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._push_time_left = torch.zeros(N, device=self.device)
        self._push_interval_left = torch.full((N,), cfg.push.period_s, device=self.device)
        self._push_force = torch.zeros(N, 3, device=self.device)

        # Episode bookkeeping.
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self._episode_sums = {name: torch.zeros(N, device=self.device) for name in REWARD_NAMES}
        self._contact_features = torch.zeros(N, CONTACT_FEATURES_PER_PRIMARY * 5, device=self.device)
        self.extras: dict = {}
        # One entry per world plus a trailing False for the global world.
        self._world_mask = wp.zeros(N + 1, dtype=wp.bool, device=model.device)
        self._world_mask_t = wp.to_torch(self._world_mask)

        self._randomize_pad_friction()
        self._graph = None
        if sim.cuda_graph and model.device.is_cuda:
            self._simulate()  # Compile every kernel before capture.
            with wp.ScopedCapture(device=model.device) as capture:
                self._simulate()
            self._graph = capture.graph

        self.reset()

    # Setup.

    def _hand_root_joint(self, world: int) -> int:
        q_start = self.model.joint_q_start.numpy()
        return int((q_start == int(self.ix.hand_q[world, 0])).nonzero()[0][0])

    def _randomize_pad_friction(self) -> None:
        """Startup DR on the fingertip pads: slide uniform, spin and roll log-uniform, per world."""
        dr, pads = self.cfg.friction_dr, self.ix.pad_shapes
        n = pads.numel()

        def log_uniform(lo: float, hi: float) -> torch.Tensor:
            return torch.exp(sample_uniform(math.log(lo), math.log(hi), n, self.device))

        wp.to_torch(self.model.shape_material_mu)[pads.reshape(-1)] = sample_uniform(*dr.slide, n, self.device)
        wp.to_torch(self.model.shape_material_mu_torsional)[pads.reshape(-1)] = log_uniform(*dr.spin)
        wp.to_torch(self.model.shape_material_mu_rolling)[pads.reshape(-1)] = log_uniform(*dr.roll)
        self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    # Simulation.

    def _simulate(self) -> None:
        """``decimation`` physics substeps, leaving the latest state in ``self.state``."""
        states = (self.state, self._state_tmp)
        dt = self.cfg.sim.dt
        for i in range(self.cfg.sim.decimation):
            src, dst = states[i % 2], states[(i + 1) % 2]
            if self.pipeline is not None:
                self.pipeline.collide(src, self.contacts)
            self.solver.step(src, dst, self.control, self.contacts, dt)
        if self.cfg.sim.decimation % 2:
            self.state.assign(self._state_tmp)
        if self.pipeline is None:
            self.solver.update_contacts(self.contacts, self.state)
        self.pads.update(self.contacts, self.state)

    # Kinematics, in mjlab's conventions: quaternions (w, x, y, z), link-origin velocities.

    def _pose(self, bodies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bq = self._body_q[bodies]
        return bq[..., :3], xyzw_to_wxyz(bq[..., 3:7])

    def _link_velocity(self, bodies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Linear velocity of the body origin and angular velocity, world frame.

        Newton's ``body_qd`` carries the centre-of-mass velocity; mjlab's
        ``body_link_vel_w`` is the body origin's.
        """
        pos, quat = self._pose(bodies)
        qd = self._body_qd[bodies]
        lin, ang = qd[..., :3], qd[..., 3:6]
        com_w = pos + quat_apply(quat, self._body_com[bodies])
        return lin + torch.cross(ang, pos - com_w, dim=-1), ang

    @property
    def hand_q(self) -> torch.Tensor:
        return self._joint_q[self.ix.hand_q]

    @property
    def hand_qd(self) -> torch.Tensor:
        return self._joint_qd[self.ix.hand_qd]

    # RL interface.

    def reset(self) -> dict[str, torch.Tensor]:
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.obs = self._compute_obs()
        return self.obs

    def step(self, actions: torch.Tensor):
        self.extras = {"log": {}}
        self._process_action(actions.to(self.device))
        self._apply_targets()
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        self.episode_length_buf += 1
        terms = self._terminations()
        self.last_terminations = terms  # for evaluation scripts
        terminated = terms["nan_term"] | terms["object_fallen"] | terms["palm_object_distance"]
        time_outs = terms["time_out"] | terms["object_spin"]
        dones = terminated | time_outs

        self._contact_features = self._compute_contact_features()
        rewards = self._rewards(terms)

        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._log_episodes(reset_ids, terms)
            self._reset_idx(reset_ids)
            # Hand and object were just pulled apart; last substep's contacts are stale.
            self._contact_features[reset_ids] = 0.0

        self._push_step()
        self.obs = self._compute_obs()
        self.extras["time_outs"] = time_outs
        return self.obs, rewards, terminated, time_outs, self.extras

    # Reset.

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        cfg, ix, dev = self.cfg, self.ix, self.device
        n = len(env_ids)
        rc = cfg.reset

        # reset_joints_by_offset on all 26 hand joints (it overrides reset_hand_base's
        # orientation noise in mjlab too, so that event has no separate effect).
        q = self._default_q[env_ids] + sample_uniform(-rc.joint_offset, rc.joint_offset, (n, NUM_ACTIONS), dev)
        soft = self._soft_limits[env_ids]
        q = torch.clamp(q, soft[..., 0], soft[..., 1])
        self._joint_q[ix.hand_q[env_ids]] = q
        self._joint_qd[ix.hand_qd[env_ids]] = 0.0

        # randomize_primitive_object: shape, size, mass.
        self.objects.sample(env_ids)
        self._write_object_model(env_ids)

        # reset_object_on_table: random pose, resting `clearance` above the table top.
        roll = sample_uniform(-rc.object_roll_pitch, rc.object_roll_pitch, n, dev)
        pitch = sample_uniform(-rc.object_roll_pitch, rc.object_roll_pitch, n, dev)
        yaw = sample_uniform(-rc.object_yaw, rc.object_yaw, n, dev)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        pos = torch.tensor(TABLE_POS, device=dev).repeat(n, 1)
        pos[:, :2] += sample_uniform(-rc.object_xy, rc.object_xy, (n, 2), dev)
        reach = self.objects.world_z_extent(quat, env_ids)
        pos[:, 2] = TABLE_POS[2] + TABLE_HALF[2] + reach + rc.object_clearance
        self._joint_q[ix.object_q[env_ids]] = torch.cat([pos, wxyz_to_xyzw(quat)], dim=-1)
        self._joint_qd[ix.object_qd[env_ids]] = 0.0

        # reset_hand_above_object: needs the palm pose these joint values give.
        self._forward()
        palm_pos, palm_quat = self._pose(ix.palm_body[env_ids])
        target = pos.clone()
        target[:, 2] = pos[:, 2] + reach + rc.hover_height
        target += sample_uniform(-rc.hover_noise, rc.hover_noise, (n, 3), dev)
        offset = torch.tensor(rc.palm_offset, device=dev).expand(n, 3)
        target += quat_apply(palm_quat, offset)
        delta = (target - palm_pos) @ self._slide_axes.T
        self._joint_q[ix.hand_q[env_ids, :3]] += delta
        self._forward()

        # Solver warm start, applied forces and actuator state of these worlds.
        self._world_mask_t.zero_()
        self._world_mask_t[env_ids] = True
        self.solver.reset(self.state, world_mask=self._world_mask, flags=0)

        # Action term: targets start where the joints are; draw the lift.
        hand_q = self.hand_q[env_ids]
        self._base_target[env_ids] = hand_q[:, :NUM_BASE_DOFS]
        self._base_ema[env_ids] = hand_q[:, :NUM_BASE_DOFS]
        self._finger_target[env_ids] = hand_q[:, NUM_BASE_DOFS:]
        self._finger_ema[env_ids] = hand_q[:, NUM_BASE_DOFS:]
        self.actions[env_ids] = 0.0
        self._sample_lift(env_ids)
        self._apply_targets()

        # Skill command: uniform on the unit sphere, held for the episode.
        z = torch.randn(n, cfg.metra.skill_dim, device=dev)
        self.skill[env_ids] = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.metra.reset(env_ids)

        # Push term.
        self._push_initialized[env_ids] = False
        self._push_active[env_ids] = False
        self._push_time_left[env_ids] = 0.0
        self._push_interval_left[env_ids] = cfg.push.period_s
        self._set_push_force(env_ids, torch.zeros(n, 3, device=dev))

        self.episode_length_buf[env_ids] = 0
        for sums in self._episode_sums.values():
            sums[env_ids] = 0.0

    def _forward(self) -> None:
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def _write_object_model(self, env_ids: torch.Tensor) -> None:
        """Write each world's drawn shape into the model and tell the solver."""
        st, ix, dev = self.objects, self.ix, self.device
        shape, radius, half = st.shape[env_ids], st.radius[env_ids], st.half_length[env_ids]
        sphere = shape == SPHERE
        hidden = torch.full_like(radius, HIDDEN_SIZE)
        zero = torch.zeros_like(radius)

        cyl, sph = ix.cylinder_shape[env_ids], ix.sphere_shape[env_ids]
        cyl_r, cyl_h = torch.where(sphere, hidden, radius), torch.where(sphere, hidden, half)
        self._shape_scale[cyl] = torch.stack([cyl_r, cyl_h, zero], dim=-1)
        sph_r = torch.where(sphere, radius, hidden)
        self._shape_scale[sph] = torch.stack([sph_r, zero, zero], dim=-1)
        self._shape_radius[cyl] = torch.sqrt(cyl_r**2 + cyl_h**2)
        self._shape_radius[sph] = sph_r

        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        pen_turn = torch.tensor(PEN_QUAT_WXYZ, device=dev)
        quat = torch.where((shape == PEN).unsqueeze(-1), pen_turn, identity)
        self._shape_xform[cyl, 3:7] = wxyz_to_xyzw(quat)

        body = ix.object_body[env_ids]
        mass, inertia = st.mass[env_ids], st.inertia[env_ids]
        self._body_mass[body] = mass
        self._body_inv_mass[body] = 1.0 / mass
        self._body_inertia[body] = torch.diag_embed(inertia)
        self._body_inv_inertia[body] = torch.diag_embed(1.0 / inertia)
        self.solver.notify_model_changed(
            newton.ModelFlags.SHAPE_PROPERTIES | newton.ModelFlags.BODY_INERTIAL_PROPERTIES
        )

    # Action.

    def _sample_lift(self, env_ids: torch.Tensor) -> None:
        """Each world's lift direction (uniform on the cap within the cone) and speed."""
        ac, n = self.cfg.action, len(env_ids)
        cos_min = math.cos(math.radians(ac.lift_cone_deg))
        cos_t = 1.0 - torch.rand(n, device=self.device) * (1.0 - cos_min)
        sin_t = torch.sqrt((1.0 - cos_t**2).clamp_min(0.0))
        az = torch.rand(n, device=self.device) * (2.0 * math.pi)
        direction = torch.stack([sin_t * torch.cos(az), sin_t * torch.sin(az), cos_t], dim=-1)
        speed = sample_uniform(*ac.lift_speed_range, n, self.device)
        self._lift_velocity[env_ids] = direction * speed.unsqueeze(-1)

    def _process_action(self, actions: torch.Tensor) -> None:
        ac, dt = self.cfg.action, self.step_dt
        self.actions[:] = actions
        a = torch.clamp(actions, -1.0, 1.0)

        self._finger_target = self._finger_target + a[:, NUM_BASE_DOFS:] * ac.joint_vel * dt
        lifting = (self.episode_length_buf * dt >= ac.lift_start_s).unsqueeze(-1)
        # The lift is scripted, so the base actions stop counting once it starts.
        base_delta = a[:, :NUM_BASE_DOFS] * self._base_vel_max * dt * ~lifting
        self._base_target = self._base_target + base_delta

        if lifting.any():
            # Along this episode's lift velocity; the slide axes are orthonormal.
            step = (self._lift_velocity * dt) @ self._slide_axes.T * lifting
            # Stop the whole lift once any slide would pass its limit, so the base
            # only ever moves along the drawn direction.
            limits = self._limits[:, :3]
            new = self._base_target[:, :3] + step
            inside = ((new >= limits[..., 0]) & (new <= limits[..., 1])).all(dim=-1, keepdim=True)
            self._base_target[:, :3] = self._base_target[:, :3] + step * inside

        lim = self._limits
        self._finger_target = torch.clamp(
            self._finger_target, lim[:, NUM_BASE_DOFS:, 0], lim[:, NUM_BASE_DOFS:, 1]
        )
        self._base_target = torch.clamp(self._base_target, lim[:, :NUM_BASE_DOFS, 0], lim[:, :NUM_BASE_DOFS, 1])
        alpha = ac.ema_alpha
        self._finger_ema = alpha * self._finger_target + (1 - alpha) * self._finger_ema
        self._base_ema = alpha * self._base_target + (1 - alpha) * self._base_ema

    def _apply_targets(self) -> None:
        self._target_q[self.ix.hand_q] = torch.cat([self._base_ema, self._finger_ema], dim=-1)

    # Push.

    def _set_push_force(self, env_ids: torch.Tensor, force: torch.Tensor) -> None:
        self._push_force[env_ids] = force
        self._write_push_force()

    def _write_push_force(self) -> None:
        """Copy every world's push into both states' ``body_f`` (the substeps alternate between them)."""
        wrench = torch.cat([self._push_force, torch.zeros_like(self._push_force)], dim=-1)
        for body_f in self._body_f:
            body_f[self.ix.object_body] = wrench

    def _push_step(self) -> None:
        """``push_object_in_air``, branch-free over all worlds so it never waits on the GPU."""
        pc, dt = self.cfg.push, self.step_dt
        if not pc.enabled:
            return
        obj_z = self._body_q[self.ix.object_body, 2]
        # Rest height is taken on the first step after reset, once the object is on the table.
        self._push_rest_z = torch.where(self._push_initialized, self._push_rest_z, obj_z)
        self._push_initialized.fill_(True)
        in_air = obj_z - self._push_rest_z > pc.lift_threshold

        self._push_time_left = torch.where(self._push_active, self._push_time_left - dt, self._push_time_left)
        expired = self._push_active & (self._push_time_left <= 0)
        self._push_active &= ~expired

        # The clock only runs in the air; touching down restarts it.
        self._push_interval_left = torch.where(in_air, self._push_interval_left - dt, pc.period_s)
        trigger = in_air & ~self._push_active & (self._push_interval_left <= 0)

        direction = torch.randn(self.num_envs, 3, device=self.device)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        magnitude = sample_uniform(*pc.force_range, (self.num_envs, 1), self.device)
        t = trigger.unsqueeze(-1)
        self._push_force = torch.where(t, direction * magnitude, torch.where(expired.unsqueeze(-1), 0.0, self._push_force))
        self._push_time_left = torch.where(trigger, pc.duration_s, self._push_time_left)
        self._push_active |= trigger
        self._push_interval_left = torch.where(trigger, pc.period_s, self._push_interval_left)
        self._write_push_force()

    # Terminations and rewards.

    def _terminations(self) -> dict[str, torch.Tensor]:
        tc, ix = self.cfg.termination, self.ix
        obj_pos, _ = self._pose(ix.object_body)
        palm_pos, _ = self._pose(ix.palm_body)
        ang = self._body_qd[ix.object_body, 3:6].norm(dim=-1)
        state = torch.cat([self._joint_q[ix.hand_q], self._joint_q[ix.object_q], self._joint_qd[ix.hand_qd]], -1)
        return {
            "time_out": self.episode_length_buf >= self.max_episode_length,
            "nan_term": ~torch.isfinite(state).all(dim=-1),
            "object_fallen": obj_pos[:, 2] - TABLE_POS[2] < tc.fall_height,
            # The object's centre of mass is its body origin.
            "palm_object_distance": (palm_pos - obj_pos).norm(dim=-1) > tc.palm_radius,
            "object_spin": ~(ang <= tc.max_object_ang_vel),
        }

    def _rewards(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        rc = self.cfg.reward
        finger_vel = self._joint_qd[self.ix.hand_qd[:, NUM_BASE_DOFS:]]
        dropped = terms["object_fallen"] | terms["palm_object_distance"]
        values = {
            "smoothness": rc.smoothness_weight * -self.cfg.reward.smoothness_scale * finger_vel.abs().sum(-1),
            "metra": rc.metra_weight * self.metra(self._contact_features, self.skill),
            "object_dropped": -rc.drop_penalty * dropped.float(),
        }
        total = torch.zeros(self.num_envs, device=self.device)
        for name, value in values.items():
            # A world that diverged this step ends on nan_term; keep its NaN out of PPO.
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            total += value
            self._episode_sums[name] += value
        return total

    def _log_episodes(self, env_ids: torch.Tensor, terms: dict[str, torch.Tensor]) -> None:
        log = self.extras["log"]
        for name, sums in self._episode_sums.items():
            log[f"Episode_Reward/{name}"] = sums[env_ids].mean() / self.cfg.episode_length_s
        for name, value in terms.items():
            log[f"Episode_Termination/{name}"] = value[env_ids].sum().float()
        log["Episode/length_s"] = self.episode_length_buf[env_ids].float().mean() * self.step_dt
        log.update(self.metra.metrics)

    # Observations.

    def _compute_contact_features(self) -> torch.Tensor:
        """Per pad ``[touching, region centre, region normal]``, object frame, ``(N, 35)``."""
        obj_pos, obj_quat = self._pose(self.ix.object_body)
        touching = self.pads.touching
        ids = self.objects.nearest_region_ids(torch.nan_to_num(self.pads.pos), obj_pos, obj_quat)
        centers, normals = self.objects.region_centers_normals(ids)
        t = touching.unsqueeze(-1)
        return (torch.cat([t.float(), centers, normals], dim=-1) * t).reshape(self.num_envs, -1)

    def _compute_obs(self) -> dict[str, torch.Tensor]:
        """mjlab's term order: grasp base terms, skill, contact regions, fingertips, object shape."""
        ix = self.ix
        q, qd = self.hand_q, self.hand_qd
        palm_pos, palm_quat = self._pose(ix.palm_body)
        palm_lin, palm_ang = self._link_velocity(ix.palm_body)
        obj_pos, obj_quat = self._pose(ix.object_body)
        obj_lin, obj_ang = self._link_velocity(ix.object_body)
        kp_pos, _ = self._pose(ix.keypoint_bodies)
        inv_palm = quat_conjugate(palm_quat)

        terms = {
            "hand_dof_pos": (q - self._default_q)[:, NUM_BASE_DOFS:],
            "hand_dof_vel": qd[:, NUM_BASE_DOFS:],
            "hand_root_pos": palm_pos,
            "hand_root_quat": palm_quat,
            "hand_root_linvel": palm_lin * 0.2,
            "hand_root_angvel": palm_ang * 0.2,
            "object_pos": obj_pos,
            "object_quat": obj_quat,
            "object_pos_rel_hand": quat_apply(inv_palm, obj_pos - palm_pos),
            "object_quat_rel_hand": quat_mul(inv_palm, obj_quat),
            "object_linvel": obj_lin * 0.2,
            "object_angvel": obj_ang * 0.2,
            "keypoint_pos_rel": quat_apply(inv_palm.unsqueeze(1).expand(-1, 5, -1), kp_pos - palm_pos.unsqueeze(1)),
            "actions": self.actions,
            "skill": self.skill,
            "contact_regions": self._contact_features,
            "keypoint_pos_object": quat_apply_inverse(
                obj_quat.unsqueeze(1).expand(-1, 5, -1), kp_pos - obj_pos.unsqueeze(1)
            ),
            "object_shape": self.objects.shape_params(),
        }
        terms = {k: v.reshape(self.num_envs, -1) for k, v in terms.items()}
        critic = torch.cat([*terms.values(), self.objects.mass_obs()], dim=-1)
        if self.cfg.obs_noise:
            for name, scale in _ACTOR_NOISE.items():
                terms[name] = terms[name] + sample_uniform(-scale, scale, terms[name].shape, self.device)
        actor = torch.cat(list(terms.values()), dim=-1)
        if not (torch.isfinite(actor).all() and torch.isfinite(critic).all()):
            raise RuntimeError("Non-finite observation; mjlab's nan_policy='error'.")
        return {"actor": actor, "critic": critic}

    # Checkpointing of the env-side learner.

    def metra_state_dict(self) -> dict:
        return self.metra.state_dict()

    def load_metra_state_dict(self, state: dict) -> None:
        self.metra.load_state_dict(state)


# Uniform noise half-widths on the actor's observations (mjlab's Unoise terms).
_ACTOR_NOISE = {
    "hand_dof_pos": 0.01,
    "hand_dof_vel": 1.5,
    "hand_root_pos": 0.005,
    "object_pos": 0.01,
    "object_pos_rel_hand": 0.01,
}

__all__ = ["GraspEnv", "NUM_ACTIONS"]
