"""Compare the MuJoCo model Newton builds for a mimic task with the one mjlab compiled.

    uv run python scripts/check_mimic_model.py --task Mimic-Panda-Allegro-Apple

Reads ``assets/mimic/<task>/mjlab.json`` (written by ``export_mimic_from_mjlab.py``)
and checks, element by element, every body's mass, inertia, centre of mass and
gravcomp; every joint's range, armature, damping, frictionloss and stiffness;
every collision geom's friction, solref, solimp, condim, priority, margin and
gap; the finger actuators' gains and force limits; and the solver options.
Prints each mismatch and exits non-zero if there is any.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from newton_hand.mimic.config import MimicEnvCfg
from newton_hand.mimic.scene import build_model, load_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Mimic-Panda-Allegro-Apple")
    args = parser.parse_args()

    import newton

    cfg = MimicEnvCfg(task=args.task, num_envs=2)
    meta = load_meta(args.task)
    ref = meta["mj_params"]
    model, ix, _ = build_model(cfg, meta, "cuda:0")
    s = cfg.sim
    solver = newton.solvers.SolverMuJoCo(
        model, use_mujoco_contacts=True, solver="newton", integrator=s.integrator, cone=s.cone,
        impratio=s.impratio, iterations=s.iterations, ls_iterations=s.ls_iterations,
        njmax=s.njmax, nconmax=s.nconmax,
    )
    mj = solver.mj_model
    labels = {"body": model.body_label, "shape": model.shape_label, "joint": model.joint_label}
    body_map = solver.mjc_body_to_newton.numpy()[0]
    geom_map = solver.mjc_geom_to_newton_shape.numpy()[0]
    jnt_map = solver.mjc_jnt_to_newton_jnt.numpy()[0]

    def short(label: str) -> str:
        return label.rsplit("/", 1)[-1]

    errors: list[str] = []

    def check(what: str, a, b, tol: float = 1e-5) -> None:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, atol=tol, rtol=1e-4):
            errors.append(f"{what}: newton {np.round(a, 6).tolist()} vs mjlab {np.round(b, 6).tolist()}")

    seen = set()
    for b in range(1, mj.nbody):
        name = short(labels["body"][body_map[b]]) if body_map[b] >= 0 else None
        if name is None or name not in ref["bodies"]:
            continue
        seen.add(name)
        r = ref["bodies"][name]
        check(f"body {name} mass", mj.body_mass[b], r["mass"])
        check(f"body {name} inertia", mj.body_inertia[b], r["inertia"], 1e-8)
        if r["mass"] > 0:  # a massless frame body's "COM" is just where Newton put it
            check(f"body {name} ipos", mj.body_ipos[b], r["ipos"])
        check(f"body {name} gravcomp", mj.body_gravcomp[b], r["gravcomp"])
    missing = {n for n in ref["bodies"] if ref["bodies"][n]["mass"] > 0} - seen - {"mocap_base"}
    if missing:
        errors.append(f"bodies with mass missing in Newton: {sorted(missing)}")

    for j in range(mj.njnt):
        name = short(labels["joint"][jnt_map[j]])
        if name not in ref["joints"]:
            errors.append(f"joint {name} not in mjlab")
            continue
        r, dof = ref["joints"][name], mj.jnt_dofadr[j]
        check(f"joint {name} type", mj.jnt_type[j], r["type"])
        if r["limited"]:
            check(f"joint {name} range", mj.jnt_range[j], r["range"])
        for field, value in (("armature", mj.dof_armature[dof]), ("damping", mj.dof_damping[dof]),
                             ("frictionloss", mj.dof_frictionloss[dof]), ("stiffness", mj.jnt_stiffness[j])):
            check(f"joint {name} {field}", value, r[field])
    if mj.njnt != len(ref["joints"]):
        errors.append(f"{mj.njnt} joints in Newton, {len(ref['joints'])} in mjlab")

    count: dict[str, int] = {}
    for g in range(mj.ngeom):
        shape = geom_map[g]
        if shape < 0:
            continue
        body_idx = model.shape_body.numpy()[shape]
        body = short(labels["body"][body_idx]) if body_idx >= 0 else "world"
        if body == "world" and mj.geom_type[g] == 0:
            body = "terrain"  # the ground plane
        k = count.get(body, 0)
        count[body] = k + 1
        key = f"{body}#{k}"
        if key not in ref["geoms"]:
            errors.append(f"geom {key} ({short(labels['shape'][shape])}) not in mjlab")
            continue
        r = ref["geoms"][key]
        for field, value in (("type", mj.geom_type[g]), ("friction", mj.geom_friction[g]), ("solref", mj.geom_solref[g]),
                             ("solimp", mj.geom_solimp[g]), ("condim", mj.geom_condim[g]),
                             ("priority", mj.geom_priority[g]), ("margin", mj.geom_margin[g]), ("gap", mj.geom_gap[g])):
            check(f"geom {key} {field}", value, r[field])
    if sum(count.values()) != len(ref["geoms"]):
        errors.append(f"{sum(count.values())} collision geoms in Newton, {len(ref['geoms'])} in mjlab")

    finger_acts = {n: a for n, a in ref["actuators"].items() if not a["is_motor"]}
    for a in range(mj.nu):
        joint = short(labels["joint"][jnt_map[mj.actuator_trnid[a, 0]]])
        r = finger_acts.pop(joint, None)
        if r is None:
            errors.append(f"actuator on {joint} not a mjlab position actuator")
            continue
        check(f"actuator {joint} gain", mj.actuator_gainprm[a, :3], r["gainprm"])
        check(f"actuator {joint} bias", mj.actuator_biasprm[a, :3], r["biasprm"])
        if r["forcerange"] is not None:
            check(f"actuator {joint} forcerange", mj.actuator_forcerange[a], r["forcerange"])
    if finger_acts:
        errors.append(f"position actuators missing in Newton: {sorted(finger_acts)}")

    opt = meta["mj_model"]["opt"]
    # Not the timestep: Newton passes dt to every step (MimicSimCfg.dt, checked against mjlab's).
    for field in ("impratio", "cone", "iterations", "ls_iterations", "integrator"):
        check(f"opt {field}", getattr(mj.opt, field), opt[field])
    check("opt gravity", mj.opt.gravity, opt["gravity"])

    print(f"Checked {len(seen)} bodies, {mj.njnt} joints, {sum(count.values())} collision geoms, {mj.nu} actuators.")
    for e in errors:
        print("MISMATCH", e)
    if errors:
        sys.exit(1)
    print("All match.")


if __name__ == "__main__":
    main()
