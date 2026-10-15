"""Export an mjlab_hand mimic task's model, motions and settings for newton_hand.

Run it in **mjlab_hand's** environment, which has mjlab:

    cd ~/mjlab_hand && uv run python ~/newton_hand/scripts/export_mimic_from_mjlab.py \\
        --task Mimic-Panda-Allegro-Apple

It writes ``src/newton_hand/assets/mimic/<task>/``:

- ``robot.xml``: the arm (Franka Panda or FR3) carrying the hand, from the entity spec *after* mjlab
  applied the collision and actuator config. The finger actuators are rewritten
  as ``<position kp kv>`` (what mjlab's builtin position actuators compile to).
  The arm's ideal-PD ``<motor>``s are dropped: mjlab computes their torques in
  torch every substep, and newton_hand does the same (``mimic/env.py``).
- ``object.xml``, ``table.xml``: the GRAB object (free body, CoACD hulls) and the table.
- ``meshes/``: every mesh those files use.
- ``motions/<seq>.npz``: the task's motion files, as mjlab loads them.
- ``mjlab.json``: the values newton_hand must reproduce, read from the live
  mjlab env: joint order, default pose, soft limits, actuator gains, and every
  reward, termination, observation, command, action, sim and PPO setting.
  ``newton_hand.mimic.config`` is checked against it at startup.

Only tasks whose motions share one object and one arm base are supported (the
Newton worlds are built identical).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

NEWTON_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = NEWTON_ROOT / "src" / "newton_hand" / "assets" / "mimic"


def _export_xml(spec, out: Path, prefix: str) -> str:
    """The spec's XML with every mesh written to ``out/meshes`` and pointed at there."""
    keys = list(spec.assets)
    files = {}
    for mesh in spec.meshes:
        hits = [k for k in keys if k == mesh.file or k.split("/")[-1] == mesh.file.split("/")[-1]]
        assert len(hits) == 1, f"{mesh.name}: {mesh.file} -> {hits}"
        name = f"{prefix}_{Path(mesh.file).name}"
        (out / "meshes" / name).write_bytes(spec.assets[hits[0]])
        files[mesh.file] = name
    xml = spec.to_xml()
    xml = re.sub(r'file="([^"]+)"', lambda m: f'file="{files.get(m.group(1), m.group(1))}"', xml)
    xml = re.sub(r'<compiler([^>]*?) meshdir="[^"]*"', r"<compiler\1", xml)
    return xml.replace("<compiler", '<compiler meshdir="meshes"', 1)


def _name_attached_defaults(xml: str, palm_body: str) -> str:
    """Make the hand's own default class reach the hand.

    Each attached spec's top-level default is written as an unnamed ``<default>``
    one level inside an unnamed wrapper: the arm's first (the FR3's XML has its
    own top-level default; the Panda's classes sit directly in the wrapper), the
    hand's last. Nothing refers to them, and
    in XML the hand's elements nested inside ``panda_link0`` (``childclass=
    "panda_panda"``) would inherit the arm's joint armature and damping instead
    (it happened to the Wuji). Name them and give the palm the hand's.
    """
    xml, n_root = re.subn(r"(?m)^    <default>$", '    <default class="attached_main">', xml)
    # An arm XML without defaults (the Vega's) leaves an empty ``<default/>``.
    inner = list(re.finditer(r"(?m)^      <default(/?)>$", xml))
    if n_root != 1 or not inner:
        raise RuntimeError(f"Unexpected default layout in the exported XML ({n_root}, {len(inner)}).")
    # The hand is attached last, so its default comes last; earlier ones are the arm's.
    names = [f"arm_main_{k}" for k in range(len(inner) - 1)] + ["hand_main"]
    for match, name in zip(reversed(inner), reversed(names)):
        xml = xml[: match.start()] + f'      <default class="{name}"{match.group(1)}>' + xml[match.end():]
    tag = re.search(rf'<body name="{re.escape(palm_body)}"[^>]*>', xml)
    assert tag, f"palm body {palm_body} not found"
    if "childclass=" not in tag.group(0):  # a palm with its own childclass (Allegro) already sits under hand_main
        xml = xml.replace(tag.group(0), tag.group(0).replace(f'name="{palm_body}"', f'name="{palm_body}" childclass="hand_main"'), 1)
    return xml


def _roundtrip_check(path: Path, reference) -> None:
    """Compile the written XML with MuJoCo and compare it with the spec mjlab compiled."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    errors = []
    for j in range(reference.njnt):
        name = reference.joint(j).name
        k = model.joint(name).id
        a, b = reference.jnt_dofadr[j], model.jnt_dofadr[k]
        for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
            if not np.isclose(getattr(reference, field)[a], getattr(model, field)[b]):
                errors.append(f"{name} {field}: {getattr(model, field)[b]} vs {getattr(reference, field)[a]}")
        if not np.allclose(reference.jnt_range[j], model.jnt_range[k]):
            errors.append(f"{name} range")
    for field in ("body_mass", "body_inertia", "body_ipos", "body_gravcomp"):
        if not np.allclose(getattr(reference, field), getattr(model, field), atol=1e-7):
            errors.append(field)
    for field in ("geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_priority", "geom_contype",
                  "geom_conaffinity", "geom_margin", "geom_gap", "geom_size", "geom_pos", "geom_quat"):
        if not np.allclose(getattr(reference, field), getattr(model, field), atol=1e-6):
            errors.append(field)
    if errors:
        raise RuntimeError(f"{path.name} does not reproduce mjlab's model: {errors[:10]}")


def _finish_xml(xml: str) -> str:
    # MuJoCo's export drops a solref's default damping ratio; Newton would read it as 0.
    return re.sub(r'solref="([^" ]+)"', r'solref="\1 1"', xml)


def _position_actuators(spec) -> str:
    """The builtin position actuators as ``<position>`` elements; arm motors are dropped."""
    lines = []
    for act in spec.actuators:
        kp, bias = float(act.gainprm[0]), act.biasprm
        if bias[1] == 0.0:  # a motor: mjlab's IdealPd, applied in torch
            continue
        assert np.isclose(bias[1], -kp), f"{act.name} is not a position actuator"
        attrs = [f'name="{act.name}"', f'joint="{act.target}"', f'kp="{kp:.9g}"', f'kv="{-float(bias[2]):.9g}"']
        if act.forcelimited == 1 or (act.forcelimited == 2 and act.forcerange[0] < act.forcerange[1]):
            attrs.append(f'forcerange="{act.forcerange[0]:.9g} {act.forcerange[1]:.9g}"')
        if act.ctrllimited == 1 or (act.ctrllimited == 2 and act.ctrlrange[0] < act.ctrlrange[1]):
            attrs.append(f'ctrlrange="{act.ctrlrange[0]:.9g} {act.ctrlrange[1]:.9g}"')
        lines.append(f"    <position {' '.join(attrs)}/>")
    return "  <actuator>\n" + "\n".join(lines) + "\n  </actuator>\n"


def _explicit_inertials(spec) -> None:
    """Write each body's compiled mass, COM and inertia into the spec.

    Bodies without ``<inertial>`` (the Allegro's) get them from their geoms'
    density in MuJoCo, and Newton computes that differently (fingertips 2.9g vs
    MuJoCo's 7.5g). Spelling out MuJoCo's result makes both use the same.
    """
    model = spec.compile()
    for body in spec.bodies[1:]:
        b = model.body(body.name).id
        body.explicitinertial = True
        body.mass = float(model.body_mass[b])
        body.ipos = model.body_ipos[b].copy()
        body.iquat = model.body_iquat[b].copy()
        body.inertia = model.body_inertia[b].copy()
        body.fullinertia = [float("nan")] * 6


def _strip(spec) -> None:
    for key in list(spec.keys):
        spec.delete(key)
    for act in list(spec.actuators):
        spec.delete(act)


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if callable(value):
        return getattr(value, "__name__", repr(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _strip_entity(name: str) -> str:
    return name.split("/", 1)[1] if "/" in name else name


def _mj_params(mj) -> dict:
    """Physics parameters of the compiled mjlab scene, keyed by element name.

    Collision geoms only (the visual ones do not reach Newton's solver), keyed
    ``<body>#<k>``: the k-th collision geom of that body.
    """
    import mujoco

    bodies = {}
    for b in range(1, mj.nbody):
        bodies[_strip_entity(mj.body(b).name)] = {
            "mass": float(mj.body_mass[b]),
            "inertia": mj.body_inertia[b].tolist(),
            "ipos": mj.body_ipos[b].tolist(),
            "gravcomp": float(mj.body_gravcomp[b]),
        }
    joints = {}
    for j in range(mj.njnt):
        dof = mj.jnt_dofadr[j]
        joints[_strip_entity(mj.joint(j).name)] = {
            "type": int(mj.jnt_type[j]),
            "range": mj.jnt_range[j].tolist(),
            "limited": bool(mj.jnt_limited[j]),
            "armature": float(mj.dof_armature[dof]),
            "damping": float(mj.dof_damping[dof]),
            "frictionloss": float(mj.dof_frictionloss[dof]),
            "stiffness": float(mj.jnt_stiffness[j]),
        }
    geoms, count = {}, {}
    for g in range(mj.ngeom):
        if mj.geom_contype[g] == 0 and mj.geom_conaffinity[g] == 0:
            continue
        body = _strip_entity(mj.body(mj.geom_bodyid[g]).name) if mj.geom_bodyid[g] else "world"
        k = count.get(body, 0)
        count[body] = k + 1
        geoms[f"{body}#{k}"] = {
            "type": int(mj.geom_type[g]),
            "friction": mj.geom_friction[g].tolist(),
            "solref": mj.geom_solref[g].tolist(),
            "solimp": mj.geom_solimp[g].tolist(),
            "condim": int(mj.geom_condim[g]),
            "priority": int(mj.geom_priority[g]),
            "contype": int(mj.geom_contype[g]),
            "conaffinity": int(mj.geom_conaffinity[g]),
            "margin": float(mj.geom_margin[g]),
            "gap": float(mj.geom_gap[g]),
        }
    actuators = {}
    for a in range(mj.nu):
        joint = _strip_entity(mj.joint(mj.actuator_trnid[a, 0]).name)
        actuators[joint] = {
            "gainprm": mj.actuator_gainprm[a, :3].tolist(),
            "biasprm": mj.actuator_biasprm[a, :3].tolist(),
            "forcerange": mj.actuator_forcerange[a].tolist() if mj.actuator_forcelimited[a] else None,
            "is_motor": bool(mj.actuator_biasprm[a, 1] == 0.0),
        }
    pairs = [
        (_strip_entity(mj.body(mj.exclude_signature[i] >> 16).name), _strip_entity(mj.body(mj.exclude_signature[i] & 0xFFFF).name))
        for i in range(mj.nexclude)
    ]
    del mujoco
    return {"bodies": bodies, "joints": joints, "geoms": geoms, "actuators": actuators, "exclude": pairs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Mimic-Panda-Allegro-Apple")
    args = parser.parse_args()

    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from mjlab_hand.mimic.arm import ARMS, make_arm_hand_entity_cfg
    from mjlab_hand.mimic.hands import HANDS
    from mjlab_hand.registration import register_tasks

    register_tasks()
    cfg = load_env_cfg(args.task)
    rl_cfg = load_rl_cfg(args.task)
    motion_files = cfg.commands["motion"].motion_files
    motions = [np.load(f) for f in motion_files]
    if "arm_joint_pos" not in motions[0]:
        raise SystemExit("Only arm-mounted (Mimic-Panda-*) tasks are ported.")
    objects = {str(m["object"]) for m in motions}
    bases = {(tuple(np.round(m["base_pos"], 6)), tuple(np.round(m["base_quat"], 6))) for m in motions}
    if len(objects) != 1 or len(bases) != 1:
        raise SystemExit(f"{args.task}: motions use objects {objects} and {len(bases)} bases; need one of each.")
    hand = HANDS[str(motions[0]["hand"])]
    arm = ARMS[str(motions[0]["arm"])] if "arm" in motions[0] else ARMS["panda"]  # older Panda files have no "arm"

    final = OUT_ROOT / args.task
    out = OUT_ROOT / f".{args.task}.tmp"  # swapped in only once the export has succeeded
    if out.exists():
        shutil.rmtree(out)
    (out / "meshes").mkdir(parents=True)
    (out / "motions").mkdir()

    # Models, from the specs mjlab builds the scene from.
    robot = Entity(make_arm_hand_entity_cfg(hand, arm)).spec
    actuators = _position_actuators(robot)
    _explicit_inertials(robot)
    _strip(robot)
    xml = _name_attached_defaults(_finish_xml(_export_xml(robot, out, "robot")), hand.palm_body)
    (out / "robot.xml").write_text(xml.replace("</mujoco>", actuators + "</mujoco>"))
    _roundtrip_check(out / "robot.xml", robot.compile())

    entities = cfg.scene.entities
    obj = Entity(entities["object"]).spec
    _explicit_inertials(obj)
    _strip(obj)
    (out / "object.xml").write_text(_finish_xml(_export_xml(obj, out, "object")))
    _roundtrip_check(out / "object.xml", obj.compile())
    table = Entity(entities["table"]).spec
    _strip(table)
    (out / "table.xml").write_text(_finish_xml(table.to_xml()))

    for f in motion_files:
        shutil.copy(f, out / "motions" / Path(f).name)

    # Live values from a one-env mjlab env.
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    ent = env.scene["hand"]
    mj = env.sim.mj_model
    cmd = env.command_manager.get_term("motion")
    act = env.action_manager.get_term("hand")
    obs_terms = {}
    for group, group_cfg in cfg.observations.items():
        obs_terms[group] = [
            {
                "name": name,
                "noise": None if term.noise is None else [term.noise.n_min, term.noise.n_max],
                "dim": int(np.prod(dims)),
            }
            for name, term, dims in zip(
                env.observation_manager.active_terms[group],
                [group_cfg.terms[n] for n in env.observation_manager.active_terms[group]],
                env.observation_manager.group_obs_term_dim[group],
            )
        ]
    meta = {
        "task": args.task,
        "hand": hand.name,
        "arm": arm.name,
        "palm_body": hand.palm_body,
        "keypoint_bodies": list(hand.keypoint_bodies),
        "keypoint_geom_pattern": hand.keypoint_geom_pattern,
        "object": objects.pop(),
        "motion_files": [Path(f).name for f in motion_files],
        "joint_names": list(ent.joint_names),
        "body_names": list(ent.body_names),
        "default_joint_pos": ent.data.default_joint_pos[0].tolist(),
        "soft_joint_pos_limits": ent.data.soft_joint_pos_limits[0].tolist(),
        "joint_limits": ent.data.joint_pos_limits[0].tolist(),
        "arm_actuators": [
            {"joints": list(a.target_names_expr), "stiffness": a.stiffness, "damping": a.damping,
             "effort_limit": a.effort_limit}
            for a in arm.actuators
        ],
        "table_contact_excluded_bodies": list(
            cfg.scene.sensors[1].primary.exclude if len(cfg.scene.sensors) > 1 else ()
        ),
        "env": {
            "decimation": cfg.decimation,
            "episode_length_s": cfg.episode_length_s,
            "max_episode_length": env.max_episode_length,
            "scale_rewards_by_dt": cfg.scale_rewards_by_dt,
            "sim": _jsonable(cfg.sim),
        },
        "command": _jsonable(cmd.cfg),
        "action": _jsonable(act.cfg),
        "observations": obs_terms,
        "rewards": {name: {"weight": t.weight, "params": _jsonable(t.params)} for name, t in cfg.rewards.items()},
        "terminations": {
            name: {"time_out": t.time_out, "params": _jsonable(t.params)} for name, t in cfg.terminations.items()
        },
        "events": {name: {"mode": t.mode, "params": _jsonable(t.params)} for name, t in cfg.events.items()},
        "ppo": _jsonable(rl_cfg),
        "mj_params": _mj_params(mj),
        "mj_model": {
            "opt": {"timestep": float(mj.opt.timestep), "gravity": mj.opt.gravity.tolist(),
                    "impratio": float(mj.opt.impratio), "cone": int(mj.opt.cone),
                    "iterations": int(mj.opt.iterations), "ls_iterations": int(mj.opt.ls_iterations),
                    "integrator": int(mj.opt.integrator)},
        },
    }
    (out / "mjlab.json").write_text(json.dumps(meta, indent=1))
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    print(f"Wrote {final}")


if __name__ == "__main__":
    main()
