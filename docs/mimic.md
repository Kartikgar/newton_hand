# Mimic tasks on Newton (port of mjlab_hand's `Mimic-Panda-*` and `Mimic-FR3-*`)

A Franka arm (Panda or FR3) stands on a wide table and carries a dexterous hand (Allegro or Wuji). The policy has
to reproduce a human GRAB demonstration, retargeted to the hand: reach the object, grasp it and move
it as the human did. This is the Newton port of mjlab_hand's DeepMimic-style mimic tasks. For the
task itself (data, design choices, measured behaviour), see mjlab_hand's
`docs/mimic_grab_imitation.md`. This page covers how the port works and how we know it matches.

## Quick start

```bash
# 1. Export the task from mjlab_hand (once, and again whenever mjlab_hand's task changes):
cd ~/mjlab_hand && uv run python ~/newton_hand/scripts/export_mimic_from_mjlab.py --task Mimic-Panda-Allegro-Apple

# 2. Check that Newton's model is mjlab's:
cd ~/newton_hand && uv run python scripts/check_mimic_model.py --task Mimic-Panda-Allegro-Apple

# 3. Train:
CUDA_VISIBLE_DEVICES=0 uv run train-mimic --env.task Mimic-Panda-Allegro-Apple \
  --env.num-envs 4096 --agent.max-iterations 20000 \
  --agent.logger wandb --agent.wandb-project mjlab-amp \
  --agent.run-name mimic-panda-allegro-apple-newton --video True
```

Exported tasks: `Mimic-Panda-Allegro-Apple`, `Mimic-Panda-Wuji`, `Mimic-FR3-Allegro-Apple`,
`Mimic-FR3-Wuji` and `Mimic-Vega-Wuji` (the Wuji tasks are apple lifts too). All five pass
`check_mimic_model.py`. `Mimic-Vega-Wuji` puts the Wuji on the right arm of a Dexmate Vega 1U, in
prpl-dexmate's lab geometry, its clip played at half speed to stay within the Vega's joint speed
limits (see mjlab_hand's `mimic/arm.py`). In the paired rollout below, its palms stay 0.05 cm apart
and the arm and wrist reward terms agree within 1%. Only
tasks whose motions share one object and one arm base can be exported. The multi-object
`Mimic-Panda-Allegro` needs per-world object variants, which aren't ported.

## How "the same" is guaranteed

Nothing about the model or the task is retyped by hand without a check against mjlab.

| What | Where it comes from | Checked by |
|---|---|---|
| Robot, object and table models: masses, inertias, joint ranges, armature, damping, gravcomp, collision flags, friction, solref/solimp, condim, priority, finger actuator gains | `export_mimic_from_mjlab.py` writes the specs mjlab builds its scene from, *after* mjlab applied its collision and actuator config | The exporter recompiles its own XML in MuJoCo and compares it with mjlab's compiled spec. `check_mimic_model.py` then compares Newton's compiled MuJoCo model with mjlab's, element by element. Both currently report **all match** |
| Joint order, default pose, soft limits, arm PD gains | `mjlab.json`, read from a live mjlab env | Used directly |
| Reward weights and stds, termination thresholds, command and action settings, sim settings, PPO settings | `mimic/config.py` dataclasses, mirroring mjlab's | `check_against_mjlab` compares every one with `mjlab.json` at startup and prints any difference (a command-line override shows up there) |
| Observation terms, sizes, order and noise | `MimicEnv._compute_obs` | Checked against `mjlab.json` on the first observation; a mismatch raises |
| Dynamics | the whole env | `parity_mimic.py` replays a paired mjlab rollout (below) |

**Three export fixes worth knowing about:**
1. **Explicit inertials.** MuJoCo computes the Allegro's link masses from geom density, and Newton
   computes them differently (fingertips 2.9 g instead of 7.5 g). The exporter writes MuJoCo's
   compiled mass, COM and inertia into every body.
2. **Default classes.** MuJoCo's XML export put the Wuji's own default class in an unnamed block,
   so in the XML the Wuji joints inherited the Panda's `armature 0.1, damping 1`. The exporter
   names the arm's and hand's default blocks and points the palm's `childclass` at the hand's.
3. **`solref` damping.** MuJoCo's export drops a solref's default damping ratio of 1, and Newton
   reads the missing value as 0. The exporter spells it out, as for the grasp task.

## Paired rollout (`parity_mimic.py`)

`mjlab_mimic_trace.py` (run in mjlab_hand) and `parity_mimic.py` (run here) use an identical
setup:
- 16 envs, spread over the clip at fixed start frames;
- no reset noise and no observation noise;
- fingertip friction fixed at 1;
- no early terminations;
- a scripted "oracle" wrist that steps toward the reference palm, with the fingers on the reference.

Measured on `Mimic-Panda-Allegro-Apple`, 150 steps. On `Mimic-FR3-Allegro-Apple` the palms stay
0.05 cm apart over the whole rollout, and the hand and arm reward terms agree within 2%.

| | mjlab | Newton |
|---|---|---|
| Palm error vs reference, mean / p95 | 0.85 / 2.03 cm | 0.87 / 2.03 cm |
| Newton vs mjlab palm position, first 10 steps / all 150 | — | 0.03 / 0.17 cm |
| Rewards: root, keypoint, joint, velocity, action-rate, joint-limit terms | — | within 1% of mjlab |
| Object terms | 15.5–17.6 (×1e-3 per step, two mjlab runs) | 18.2–20.0 |

The object numbers differ even between two identical mjlab runs. With an oracle wrist and the
reference fingers, the apple slips chaotically. Some start frames also have the retargeted fingers
inside the apple, and both sims fling it out (20–27 rad/s). The two repos run different MuJoCo-Warp
versions (mjlab_hand pins mujoco 3.8 / warp 1.12; Newton 1.6 needs 3.12 / 1.17), so contact-rich
moments can't agree to the last digit.

## How the env maps to mjlab

`src/newton_hand/mimic/env.py` is one vectorized env. Each piece is named after the mjlab term it
replaces, and the step runs them in mjlab's order:

1. action;
2. 4 physics substeps;
3. terminations;
4. rewards;
5. resets;
6. command update (metrics, then the reference advances a frame);
7. observations.

- **Speed.** About 180k env-steps/s at 4096 envs for both arms (0.65 s per PPO iteration). Training
  videos skip reloading the viewer's shapes on resets (`shapes_change_on_reset`), which took 6.5 s each.
- **Arm control.** mjlab's arm actuators are `IdealPdActuator`s, whose torque is computed in torch
  every substep: `clip(kp (q* − q) + kd (q̇* − q̇), ±limit)`. Here that is a warp kernel inside the
  captured CUDA graph, writing `control.joint_f`. The exported XML drops the arm's `<motor>`s. The
  gains are mjlab's (`mjlab.json`):

  | Joints | kp | kd | Torque limit |
  |---|---|---|---|
  | 1–2 | 1000 | 40 | 87 N·m |
  | 3–4 | 750 | 30 | 87 N·m |
  | 5–7 | 300 | 10 | 12 N·m |

- **Fingers.** MuJoCo position actuators with the exported kp/kv, driven through
  `control.joint_target_q`.
- **Differential IK.** Same damped-least-squares IK, nullspace posture pull and step clamp. The palm
  Jacobian uses hinge axes and anchors computed from the child bodies' poses, where mjlab reads
  MuJoCo's `xaxis`/`xanchor`.
- **Collision.** MuJoCo-Warp's own detection (`use_mujoco_contacts=True`), as mjlab, and unlike
  the grasp port's default. It is also the only mode in which Newton reports contact forces, which
  `undesired_contacts` needs.
- **Contact sensors.**
  - `keypoint_contacts` (fingertip geom touches the object) reuses the grasp port's pad kernel.
  - `table_contacts` is a new kernel. Each substep, it sums the contact forces between each hand
    body (fingertip bodies excluded) and the table. It keeps the largest magnitude over the 4
    substeps, like mjlab's force history.
- **Start frames and episode length.** Adaptive start sampling (BeyondMimic's 1 s failure-weighted
  bins) and the 10 s episode cap are ported from mjlab_hand's `MimicCommand`. The settings are checked
  against `mjlab.json`, and the episode length is read from it. In a zero-action test, the per-bin
  failure averages and the shift in starts match mjlab's.
- **Reset velocities.** mjlab writes the motion's object velocity as the body-origin velocity.
  Newton's free joint takes the COM velocity, so the reset converts with `v + ω × r_com`. This was
  checked to match the reference exactly.
- **Logging.** The same keys as mjlab: `Episode_Reward/*` (episode sum / episode length in s),
  `Episode_Termination/*` and `Metrics/motion/*`.

## Known differences

- **Kinematics freshness.** mjlab's terminations and rewards read body poses one substep stale;
  here they are fresh. The grasp port makes the same choice.
- **MuJoCo-Warp version.** See above.
- **Not ported:**
  - multi-object tasks;
  - the floating-hand tasks (`Mimic-Allegro`, `Mimic-Wuji`);
  - the reference "ghost" in videos;
  - a `play` CLI for mimic checkpoints.

## Code map

| Path | What it does |
|---|---|
| `scripts/export_mimic_from_mjlab.py` | Writes `assets/mimic/<task>/`: `robot.xml`, `object.xml`, `table.xml`, `meshes/`, `motions/`, `mjlab.json` |
| `scripts/check_mimic_model.py` | Newton's compiled model vs mjlab's, element by element |
| `scripts/mjlab_mimic_trace.py`, `scripts/parity_mimic.py` | The paired rollout |
| `src/newton_hand/mimic/config.py` | Settings, plus `check_against_mjlab` |
| `src/newton_hand/mimic/scene.py` | Builds the model; per-world index tables |
| `src/newton_hand/mimic/env.py` | **The task.** Start reading at `MimicEnv.step` |
| `src/newton_hand/mimic/kernels.py` | Arm ideal PD; table contact forces |
| `src/newton_hand/mimic/motions.py` | Motion library (mjlab_hand's `MotionLibrary`) |
| `src/newton_hand/cli/train_mimic.py` | `train-mimic` |
