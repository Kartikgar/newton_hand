"""Task and training settings for the Newton random-object METRA grasp task.

Every default reproduces ``Grasp-METRA-Hover-Lift-Contacts-Push-RandomObject-Wuji``
in mjlab_hand (``grasp/config/wuji/env_cfgs.py`` and the factories it calls);
the comments name where each number came from when it is not obvious. All of
it is a tyro CLI: ``--env.push.force-range 1.0 3.0`` and so on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from newton_hand.objects.random_primitive import RandomObjectCfg


@dataclass
class SimCfg:
    # mjlab's grasp SimulationCfg: 5ms physics, 4 substeps per 20ms control step.
    dt: float = 0.005
    decimation: int = 4
    iterations: int = 50
    ls_iterations: int = 50
    impratio: float = 10.0
    cone: Literal["elliptic", "pyramidal"] = "elliptic"
    integrator: Literal["implicitfast", "euler"] = "implicitfast"
    # mjlab used 200; Newton keeps more candidate pairs and overflowed MuJoCo-Warp's broadphase at 255.
    nconmax: int = 400
    njmax: int = 1000
    collision: Literal["mujoco", "newton"] = "newton"
    """``newton``: Newton's ``CollisionPipeline`` detects contacts and hands them to the
    MuJoCo solver, which still applies each geom's condim, priority, solref, solimp and
    friction -- the setup Newton's own hand examples use.
    ``mujoco``: MuJoCo-Warp's collision detection, exactly as mjlab."""
    multiccd: bool = False
    """With ``collision=mujoco``: let MuJoCo's convex collision return up to 4 contacts per
    geom pair instead of 1 (``enable_multiccd``). MuJoCo-Warp only does this for box and
    mesh pairs, so it does not change the cylinder/pen on the table. mjlab runs with it off."""
    cuda_graph: bool = True
    """Capture the decimation loop as one CUDA graph."""


@dataclass
class ActionCfg:
    """``FingerDeltaScriptedLiftAction`` with ``control_base=True``: 26-D, [base 6, fingers 20]."""

    trans_vel: float = 0.2  # m/s at a full-scale base action
    rot_vel: float = math.pi / 4  # rad/s
    joint_vel: float = 2.5  # rad/s, apply_hover_lift's joint_vel_limit
    ema_alpha: float = 0.3
    # Randomized scripted lift: WUJI_LIFT_START_S, _SPEED_RANGE, _CONE_DEG.
    lift_start_s: float = 5.0
    lift_speed_range: tuple[float, float] = (0.05, 0.2)
    lift_cone_deg: float = 45.0


@dataclass
class ResetCfg:
    joint_offset: float = 0.1  # reset_joints_by_offset on every joint, base included
    object_xy: float = 0.05  # object placed within +-5cm of the table centre
    object_roll_pitch: float = 1.57
    object_yaw: float = 3.14
    object_clearance: float = 0.002  # gap under the object's lowest point
    hover_height: float = 0.05  # palm target above the object's highest point
    hover_noise: float = 0.01  # per-axis uniform noise on the hover target
    # Palm-frame shift putting the fingers, not the wrist, over the object (WUJI_PALM_OFFSET).
    palm_offset: tuple[float, float, float] = (0.0, 0.0, -0.14)


@dataclass
class PushCfg:
    """``push_object_in_air`` with the WUJI_PUSH_* settings."""

    enabled: bool = True
    force_range: tuple[float, float] = (3.0, 7.0)  # N; the 0.16kg object weighs 1.6N
    duration_s: float = 0.1
    period_s: float = 1.5
    lift_threshold: float = 0.03


@dataclass
class TerminationCfg:
    palm_radius: float = 0.22  # WUJI_PALM_RADIUS
    fall_height: float = -0.2  # object below table reference
    max_object_ang_vel: float = 1000.0  # rad/s; a time-out, guards against divergence


@dataclass
class RewardCfg:
    smoothness_weight: float = 1.0e-4
    smoothness_scale: float = 0.1
    drop_penalty: float = 10.0  # once, when object_fallen or palm_object_distance ends the episode
    metra_weight: float = 1.0


@dataclass
class MetraCfg:
    """``MetraReward`` with contacts-only phi: 5 fingertips x 7 = 35-D."""

    skill_dim: int = 8
    position_scale: float = 10.0
    hidden_dims: tuple[int, ...] = (512, 512)
    learning_rate: float = 1.0e-4
    dual_lambda_init: float = 30.0
    dual_learning_rate: float = 1.0e-4
    constraint_eps: float = 1.0e-3
    replay_steps: int = 50
    batch_size: int = 4096
    update_every: int = 24  # must equal the runner's num_steps_per_env
    updates_per_round: int = 32
    warmup_steps: int = 0
    reward_scale: float = 1.0
    train_phi: bool = True


@dataclass
class FrictionDRCfg:
    """Startup randomization of the fingertip pads' friction, per world."""

    slide: tuple[float, float] = (0.5, 1.0)  # uniform
    spin: tuple[float, float] = (1e-4, 2e-2)  # log-uniform
    roll: tuple[float, float] = (1e-5, 5e-3)  # log-uniform


@dataclass
class EnvCfg:
    num_envs: int = 4096
    seed: int = 42
    episode_length_s: float = 10.0
    obs_noise: bool = True
    """Uniform noise on the actor's observations, as mjlab's enable_corruption."""
    sim: SimCfg = field(default_factory=SimCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    reset: ResetCfg = field(default_factory=ResetCfg)
    push: PushCfg = field(default_factory=PushCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    metra: MetraCfg = field(default_factory=MetraCfg)
    friction_dr: FrictionDRCfg = field(default_factory=FrictionDRCfg)
    object: RandomObjectCfg = field(default_factory=RandomObjectCfg)

    @property
    def step_dt(self) -> float:
        return self.sim.dt * self.sim.decimation

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self.episode_length_s / self.step_dt)


@dataclass
class PpoCfg:
    """rsl-rl 5.2.0 settings, as ``wuji_metra_hover_object_grasp_ppo_runner_cfg``."""

    experiment_name: str = "wuji_metra_hover_lift_contacts_push_random_object_grasp_newton"
    run_name: str = ""
    max_iterations: int = 20_000
    num_steps_per_env: int = 24
    save_interval: int = 500
    logger: Literal["wandb", "tensorboard"] = "tensorboard"
    wandb_project: str = "mjlab-amp"
    seed: int = 42
    clip_actions: float = 1.0
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    init_std: float = 1.0
    learning_rate: float = 3e-4
    entropy_coef: float = 0.0
    value_loss_coef: float = 2.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    clip_param: float = 0.2
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    def to_rsl_rl(self) -> dict:
        """The ``train_cfg`` dict ``rsl_rl.runners.OnPolicyRunner`` takes."""
        model = {"hidden_dims": list(self.hidden_dims), "activation": "elu", "obs_normalization": True}
        return {
            "seed": self.seed,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
            "save_interval": self.save_interval,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "logger": self.logger,
            "wandb_project": self.wandb_project,
            "clip_actions": self.clip_actions,
            "actor": {
                **model,
                "class_name": "MLPModel",
                "distribution_cfg": {
                    "class_name": "GaussianDistribution",
                    "init_std": self.init_std,
                    "std_type": "log",
                },
            },
            "critic": {**model, "class_name": "MLPModel"},
            "algorithm": {
                "class_name": "PPO",
                "num_learning_epochs": self.num_learning_epochs,
                "num_mini_batches": self.num_mini_batches,
                "learning_rate": self.learning_rate,
                "schedule": "adaptive",
                "gamma": self.gamma,
                "lam": self.lam,
                "entropy_coef": self.entropy_coef,
                "desired_kl": self.desired_kl,
                "max_grad_norm": self.max_grad_norm,
                "value_loss_coef": self.value_loss_coef,
                "use_clipped_value_loss": True,
                "clip_param": self.clip_param,
                "normalize_advantage_per_mini_batch": False,
                "optimizer": "adam",
                "share_cnn_encoders": False,
            },
        }
