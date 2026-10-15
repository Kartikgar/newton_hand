"""Settings for the Newton port of mjlab_hand's mimic tasks (``Mimic-Panda-*``).

Every default reproduces mjlab_hand's ``mimic/mimic_env_cfg.py``,
``mimic/mdp/{actions,commands}.py`` and ``mimic/config/rl_cfg.py``.
:func:`check_against_mjlab` compares them at startup with the values the
exporter read from the live mjlab env (``assets/mimic/<task>/mjlab.json``), so
a change on either side fails loudly instead of drifting. All of it is a tyro
CLI: ``--env.rewards.object-pos.weight 3.0`` and so on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MimicSimCfg:
    # mjlab's mimic SimulationCfg: 5ms physics, 4 substeps per 20ms control step.
    dt: float = 0.005
    decimation: int = 4
    iterations: int = 50
    ls_iterations: int = 50
    impratio: float = 10.0
    cone: Literal["elliptic", "pyramidal"] = "elliptic"
    integrator: Literal["implicitfast", "euler"] = "implicitfast"
    nconmax: int = 200
    njmax: int = 1000
    cuda_graph: bool = True
    """Capture the decimation loop as one CUDA graph."""


@dataclass
class CommandCfg:
    """``MimicCommandCfg``: reference lookup and reference state initialization."""

    lookahead: tuple[int, ...] = (0, 2, 5, 10)
    sampling_mode: Literal["adaptive", "uniform"] = "adaptive"
    """BeyondMimic's adaptive start sampling (1s bins weighted by recent failures), or uniform."""
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    start_prob: float = 0.2  # chance an episode starts at frame 0 rather than a sampled frame
    arm_joint_noise: float = 0.01  # reset noise on the arm joints (rad)
    joint_pos_noise: float = 0.05  # reset noise on the finger joints (rad)
    contact_threshold: float = 0.005  # reference fingertip within this of the object = in contact


@dataclass
class ActionCfg:
    """``MimicArmHandActionCfg``: wrist pose deltas through differential IK, finger residuals."""

    pos_step: float = 0.04  # palm translation per step (m) at action 1
    rot_step: float = 0.15  # palm rotation per step (rad) at action 1
    max_lead_pos: float = 0.05
    max_lead_rot: float = 0.3
    finger_scale: float = 0.2  # finger offset from the reference (rad) at action 1
    ik_damping: float = 0.05
    nullspace_gain: float = 0.2
    max_joint_step: float = 0.3


@dataclass
class RewardTerm:
    weight: float
    std: float = 0.0


@dataclass
class RewardsCfg:
    """BeyondMimic-style terms; each tracking term is ``weight * exp(-err^2 / std^2)``, all scaled by dt."""

    root_pos: RewardTerm = field(default_factory=lambda: RewardTerm(0.5, 0.05))
    root_rot: RewardTerm = field(default_factory=lambda: RewardTerm(0.5, 0.3))
    keypoint_pos: RewardTerm = field(default_factory=lambda: RewardTerm(1.0, 0.02))
    joint_pos: RewardTerm = field(default_factory=lambda: RewardTerm(1.0, 0.2))
    body_lin_vel: RewardTerm = field(default_factory=lambda: RewardTerm(1.0, 0.5))
    root_ang_vel: RewardTerm = field(default_factory=lambda: RewardTerm(1.0, 3.14))
    object_pos: RewardTerm = field(default_factory=lambda: RewardTerm(2.0, 0.05))
    object_rot: RewardTerm = field(default_factory=lambda: RewardTerm(1.0, 0.4))
    contacts: RewardTerm = field(default_factory=lambda: RewardTerm(0.5))
    action_rate: RewardTerm = field(default_factory=lambda: RewardTerm(-0.1))
    undesired_contacts: RewardTerm = field(default_factory=lambda: RewardTerm(-0.1))
    joint_limit: RewardTerm = field(default_factory=lambda: RewardTerm(-10.0))
    undesired_contact_threshold: float = 1.0  # N, net force per body over the policy step


@dataclass
class TerminationCfg:
    root_pos_threshold: float = 0.10  # palm this far from the reference palm (m)
    object_pos_threshold: float = 0.10  # object this far from the reference object (m)


@dataclass
class MimicEnvCfg:
    task: str = "Mimic-Panda-Allegro-Apple"
    """An exported task: ``assets/mimic/<task>/`` (see ``scripts/export_mimic_from_mjlab.py``)."""
    num_envs: int = 4096
    seed: int = 42
    obs_noise: bool = True
    """Uniform noise on the actor's observations, as mjlab's enable_corruption."""
    keypoint_friction: tuple[float, float] = (0.7, 1.2)
    """Startup DR: fingertip sliding friction, uniform, per world and fingertip geom."""
    sim: MimicSimCfg = field(default_factory=MimicSimCfg)
    command: CommandCfg = field(default_factory=CommandCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)

    @property
    def step_dt(self) -> float:
        return self.sim.dt * self.sim.decimation


@dataclass
class MimicPpoCfg:
    """rsl-rl 5.2.0 settings, as ``mimic_ppo_runner_cfg`` (Sprout's noise setup)."""

    experiment_name: str = ""
    """Defaults to the task id, lower case with underscores, as in mjlab."""
    run_name: str = ""
    max_iterations: int = 20_000
    num_steps_per_env: int = 24
    save_interval: int = 500
    logger: Literal["wandb", "tensorboard"] = "tensorboard"
    wandb_project: str = "mjlab-amp"
    seed: int = 42
    clip_actions: float | None = None
    """None, as mjlab: ``action_rate`` must see the raw sampled noise to hold the std down."""
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    init_std: float = 0.5
    std_type: Literal["scalar", "log"] = "scalar"
    learning_rate: float = 1e-3
    entropy_coef: float = 0.005
    value_loss_coef: float = 1.0
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
                    "std_type": self.std_type,
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


def check_against_mjlab(env: MimicEnvCfg, ppo: MimicPpoCfg | None, meta: dict, strict: bool = True) -> list[str]:
    """Differences between these settings and the exported mjlab ones; raises if ``strict``.

    Command-line overrides are deliberate, so the trainer passes ``strict=False``
    and only prints them.
    """
    diffs: list[str] = []

    def same(name: str, ours, theirs) -> None:
        if isinstance(ours, (tuple, list)) or isinstance(theirs, (tuple, list)):
            ok = list(ours) == list(theirs)
        elif isinstance(ours, float) or isinstance(theirs, float):
            ok = theirs is not None and abs(float(ours) - float(theirs)) <= 1e-9 * max(1.0, abs(float(theirs)))
        else:
            ok = ours == theirs
        if not ok:
            diffs.append(f"{name}: newton {ours!r} vs mjlab {theirs!r}")

    sim, msim = env.sim, meta["env"]["sim"]
    same("sim.dt", sim.dt, msim["mujoco"]["timestep"])
    same("sim.decimation", sim.decimation, meta["env"]["decimation"])
    same("sim.iterations", sim.iterations, msim["mujoco"]["iterations"])
    same("sim.ls_iterations", sim.ls_iterations, msim["mujoco"]["ls_iterations"])
    same("sim.impratio", sim.impratio, msim["mujoco"]["impratio"])
    same("sim.cone", sim.cone, msim["mujoco"]["cone"])
    same("sim.integrator", sim.integrator, msim["mujoco"]["integrator"])
    same("sim.nconmax", sim.nconmax, msim["nconmax"])
    same("sim.njmax", sim.njmax, msim["njmax"])
    same("scale_rewards_by_dt", True, meta["env"]["scale_rewards_by_dt"])

    c, mc = env.command, meta["command"]
    for name in ("lookahead", "sampling_mode", "adaptive_kernel_size", "adaptive_lambda", "adaptive_uniform_ratio",
                 "adaptive_alpha", "start_prob", "arm_joint_noise", "joint_pos_noise", "contact_threshold"):
        same(f"command.{name}", getattr(c, name), mc[name])
    a, ma = env.action, meta["action"]
    for name in ("pos_step", "rot_step", "max_lead_pos", "max_lead_rot", "finger_scale", "ik_damping",
                 "nullspace_gain", "max_joint_step"):
        same(f"action.{name}", getattr(a, name), ma[name])

    mr = meta["rewards"]
    if set(mr) != {"root_pos", "root_rot", "keypoint_pos", "joint_pos", "body_lin_vel", "root_ang_vel", "object_pos",
                   "object_rot", "contacts", "action_rate", "undesired_contacts", "joint_limit"}:
        diffs.append(f"reward terms: mjlab has {sorted(mr)}")
    for name, term in mr.items():
        ours = getattr(env.rewards, name, None)
        if ours is None:
            continue
        same(f"rewards.{name}.weight", ours.weight, term["weight"])
        if "std" in term["params"]:
            same(f"rewards.{name}.std", ours.std, term["params"]["std"])
    if "undesired_contacts" in mr:
        same("rewards.undesired_contact_threshold", env.rewards.undesired_contact_threshold,
             mr["undesired_contacts"]["params"]["threshold"])

    mt = meta["terminations"]
    if set(mt) != {"time_out", "motion_end", "nan", "root_pos", "object_pos"}:
        diffs.append(f"termination terms: mjlab has {sorted(mt)}")
    same("termination.root_pos_threshold", env.termination.root_pos_threshold, mt["root_pos"]["params"]["threshold"])
    same("termination.object_pos_threshold", env.termination.object_pos_threshold,
         mt["object_pos"]["params"]["threshold"])

    fr = meta["events"]["keypoint_friction"]["params"]
    same("keypoint_friction", env.keypoint_friction, fr["ranges"])
    same("keypoint_friction axes", [0], fr["axes"])

    if ppo is not None:
        mp = meta["ppo"]
        ours = ppo.to_rsl_rl()
        same("ppo.clip_actions", ours["clip_actions"], mp["clip_actions"])
        same("ppo.num_steps_per_env", ours["num_steps_per_env"], mp["num_steps_per_env"])
        same("ppo.seed", ours["seed"], mp["seed"])
        for k in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg"):
            same(f"ppo.actor.{k}", ours["actor"][k], mp["actor"][k])
            if k != "distribution_cfg":
                same(f"ppo.critic.{k}", ours["critic"][k], mp["critic"][k])
        for k, v in ours["algorithm"].items():
            if k != "class_name":
                same(f"ppo.algorithm.{k}", v, mp["algorithm"][k])

    if diffs and strict:
        raise ValueError("Settings differ from the exported mjlab task:\n  " + "\n  ".join(diffs))
    return diffs
