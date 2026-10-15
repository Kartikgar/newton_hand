# newton_hand onboarding

This guide is for someone joining the project. It explains what the task is, how one episode runs, and exactly what the policy sees, does and is rewarded for. It also covers how training works and what we've learned so far about the physics. Read it top to bottom once; after that the section headings work as a reference.

The README covers setup commands and the parity tables in short form. This doc is the long version.

---

## 1. What this repo is

`newton_hand` trains a **dexterous robot hand** (the 26-DOF Wuji hand: a 6-DOF floating base plus 5 fingers × 4 joints) to interact with objects on a table. The training uses **unsupervised skill discovery (METRA)**, so nobody hand-designs a "grasp" reward. Each episode the policy gets a random **skill vector z**. It's rewarded for changing *how its fingertips touch the object* in the direction z points. Over training, different z values turn into different ways of touching and holding the object.

It's a port of the task `Grasp-METRA-Hover-Lift-Contacts-Push-RandomObject-Wuji` from our main repo, **mjlab_hand**, which runs on mjlab with MuJoCo-Warp. This repo runs the same task on **Newton**, NVIDIA's GPU physics engine. We ported it for two reasons:

- **Physics.** mjlab's collision detection gives a lying cylinder (such as our pen) only one contact point against the table, so pens sink into the table. Newton's collision gives two (see §10).
- **Speed.** About 3.8× faster per training iteration on the same GPU (§10).

It's a separate repo because the two stacks pin incompatible library versions. Newton 1.6 needs warp ≥ 1.17 and mujoco 3.12, while mjlab_hand pins warp 1.12 and mujoco 3.8.

**The physics solver is still MuJoCo.** Newton's `SolverMuJoCo` wraps MuJoCo-Warp 3.12. By default, Newton's own `CollisionPipeline` finds the contacts and hands them to that solver. So "Newton physics" here means Newton's collision detection plus MuJoCo's contact solver and integrator.

---

## 2. Quick start

```bash
cd ~/newton_hand
uv sync                       # Python 3.13, Newton 1.6.1, torch 2.11 (cu128), rsl-rl-lib 5.2.0

# Train (4096 envs, logs to W&B, a video clip every 2000 iterations)
CUDA_VISIBLE_DEVICES=0 uv run train-newton \
  --env.num-envs 4096 --agent.max-iterations 100000 \
  --agent.logger wandb --agent.wandb-project mjlab-amp \
  --agent.run-name <descriptive-name> \
  --video True --video-interval 48000

# Resume (max-iterations then counts *additional* iterations)
  ... --agent.resume True --agent.load-run <run folder> --agent.load-checkpoint model_<N>.pt

# Watch a checkpoint: live window, or a headless mp4
uv run play-newton --checkpoint logs/rsl_rl/<exp>/<run>/model_<N>.pt
CUDA_VISIBLE_DEVICES=0 uv run play-newton --checkpoint ... --video renders/play.mp4 --steps 500
```

- **Config.** Every setting lives in `src/newton_hand/config.py` and can be set from the command line. Booleans take a value (`--video True`). Examples:
  - `--env.object.shape-weights 0 0 1` trains on pens only.
  - `--env.sim.collision mujoco` switches back to MuJoCo's collision.
  - `--env.reward.drop-penalty 0` turns off the drop penalty.
- **Outputs.** Logs and checkpoints go to `logs/rsl_rl/<experiment>/<date>_<run-name>/`. Every `model_<N>.pt` has a matching `metra_phi_<N>.pt` (the METRA network, §6.1). You need both to resume.
- **GPUs.** On prism-whitney use GPU 0 or 1. GPU 2 belongs to someone else unless you're told otherwise.
- **Run names** follow mjlab_hand's convention:

  | Token | Meaning |
  |---|---|
  | `z8` | 8-D skills |
  | `w1` | METRA weight 1 |
  | `h5` | 5cm hover |
  | `pd22` | 22cm palm radius |
  | `drop10` | drop penalty 10 |
  | `lift5` | lift at 5s |
  | `cone45` | lift within 45° of vertical |
  | `spd05-20` | lift speed 0.05–0.2 m/s |
  | `f3to7` | 3–7N pushes |
  | `p1.5` | 1.5s push period |
  | `cyl8x9`, `sph8`, `pen1.3x14` | object sizes in cm |
  | `m160` | 160g |
  | `pen1000kgm3` | pen density |
  | `pinert…` | pen inertia floor |
  | `newtoncoll` | Newton collision |

---

## 3. One episode, start to finish

An episode is **10 s = 500 control steps** of 20ms. Each control step is 4 physics substeps of 5ms.

| Time | What happens |
|---|---|
| **Reset** | An object is drawn: a cylinder, sphere or pen. It's placed on the table at a random orientation and the hand is placed hovering about 5cm above it. A random skill z is drawn. (Details in §8.) |
| **0–5 s** | The policy controls both the **base** (moving the whole hand) and the **fingers**. This is when it reaches, touches and closes on the object. |
| **5 s** | A **scripted lift** takes over the base. It moves the hand away at 0.05–0.2 m/s in a random direction within 45° of vertical. Base actions stop counting, and the policy controls only the fingers. If it's holding the object, the object comes along. |
| **While the object is in the air** | Once the object is 3cm above where it settled, it gets a random **3–7N push** lasting 0.1s, every 1.5s. The object weighs 1.6N, so these are strong. |
| **Ends** | At 10s (time-out), or earlier if the object falls, gets more than 22cm from the palm, the state goes NaN, or the object spins above 1000 rad/s. |

Nothing rewards lifting directly. The lift tests whether the contacts the skill produced amount to a real grasp. A drop costs −10.

---

## 4. The scene

**Hand.** The Wuji right hand, loaded from `src/newton_hand/assets/wuji/wuji_right_floating.xml`.
- **Origin of the file.** It was exported from mjlab *after* mjlab applied its collision and actuator setup, so the gains, friction, solref and so on are exactly what mjlab trains with.
- **Base.** The palm hangs off the world on 6 joints: three slides and three hinges.
- **Actuators.** Every joint has a position actuator.
  - The base is stiff: kp 300 N/m, force limit 200N on the slides; kp 60, limit 80 N·m on the hinges.
  - The fingers are weak: kp 0.2–0.7, about 0.15–0.65 N·m of torque.
  - **Consequence: the base can press the object into the table with up to about 200N.** This matters for §10.
- **Collision shapes.** The hand's collision meshes are converted to **convex hulls**, which is what MuJoCo always uses (§11). The visual meshes keep their detail.
- **Fingertip pads.** The 5 contacts the task cares about are the `right_finger{1..5}_link4_collision_1` geoms: condim 6, priority 1, solref 0.01.

**Table.** A 60 × 60 × 4cm box. Its top is at z = 0.42m.
- Stiff contact: solref 0.005, solimp (0.99, 0.99, 0.001).
- Friction 1.0, torsional 0.005, rolling 0.0001.

**Object.** One free body carrying one cylinder shape and one sphere shape. At each reset:
- **Active shape:** it gets its size.
- **Inactive shape:** it shrinks to a 0.1mm speck at the centre.
- **Pen:** the cylinder shape is rotated 90° so the pen lies along the body's x axis.
- **Contact settings:** condim 6, friction 1.0, priority 1, and the table's stiff solref/solimp.

Default sizes are **fixed** (only the shape is drawn):

| Shape | Size | Mass |
|---|---|---|
| Cylinder | 8cm across × 9cm tall | 160g |
| Sphere | 8cm across | 160g |
| Pen | 1.3cm across × 14cm long | 18.6g (density 1000 kg/m³) |

`RandomObjectCfg` holds the full randomization ranges, so you can switch them back on. The inertia is that of a uniform solid. **Pens now use their true inertia** (`min_inertia = 0`); see §10.5 for why this changed.

**Worlds.** `N` copies of the scene are built with `ModelBuilder.replicate` as separate MuJoCo worlds, all at the same origin. They never collide with each other, and every world's frame is its env frame. You never need an env-origin correction.

**Physics settings** (all copied from mjlab):

| Setting | Value |
|---|---|
| Timestep | 5ms |
| Substeps per control step | 4 (decimation) |
| Integrator | implicitfast |
| Solver | Newton, 50 iterations, 50 line-search iterations |
| Friction cone | elliptic |
| impratio | 10 |
| `nconmax` / `njmax` | 400 / 1000 |

The 4-substep loop is captured as one CUDA graph.

---

## 5. Actions (26-D, each in [−1, 1])

| Index | Meaning | Full-scale value |
|---|---|---|
| 0–2 | Base translation velocity | 0.2 m/s |
| 3–5 | Base rotation velocity | π/4 rad/s |
| 6–25 | Finger joint velocities (5 fingers × 4) | 2.5 rad/s |

- **Actions are velocities.** Each step, an action is integrated into a joint position target (`target += a · vmax · dt`) and clamped to the joint limits.
- **Smoothing.** The target is smoothed with an EMA (α = 0.3) before going to the position actuators.
- **After 5s** the base part of the action is ignored. The base target instead moves along the episode's lift velocity, and the whole lift stops if any slide would pass its limit.

Code: `GraspEnv._process_action` in `envs/grasp_env.py`.

---

## 6. Rewards

The total reward is the sum of three terms:

| Term | Formula | Weight | When |
|---|---|---|---|
| **metra** | `(phi(s') − phi(s)) · z` | 1.0 | every step, except the first step of an episode |
| **smoothness** | `−0.1 · Σ |finger joint velocity|` | 1e-4 | every step |
| **object_dropped** | `−1` | 10 | once, when the episode ends by `object_fallen` or `palm_object_distance` |

**NaN guard.** NaN rewards from a diverged world are replaced with 0 before they reach PPO.

**Logging.** W&B logs `Episode_Reward/<term>` as the per-episode sum divided by 10s.

### 6.1 METRA and what phi sees

METRA learns a network **phi** that maps a state to an 8-D embedding. The policy is rewarded for moving phi along its skill direction z. phi is trained at the same time to make that movement as large as possible, subject to `‖phi(s') − phi(s)‖² ≤ 1` per step.

**phi's input is contacts only: 35 numbers.** For each of the 5 fingertip pads:

| Feature | Size | Meaning |
|---|---|---|
| touching | 1 | 1 if the pad has a penetrating contact with the object, else 0 |
| region centre | 3 | centre of the surface region the pad touches, in the object's frame |
| region normal | 3 | outward normal of that region, in the object's frame |

All 7 values are zero when a pad isn't touching.

**Regions.**
- Each shape's surface is split into **64 regions**: 512 surface samples, clustered with farthest-point sampling and then k-means, at a reference size.
- A contact point maps to its nearest region, after scaling by the object's current size.
- The surface sampling seed matches mjlab_hand, so region ids mean the same thing in both repos.
- Because contacts snap to regions, a pad sliding slightly within a region doesn't change phi's input.

So a skill rewards changes in **which fingers touch, and where on the object they touch**. phi never sees joint angles, poses or velocities.

**phi network and training:**

| Setting | Value |
|---|---|
| Architecture | 35 → 512 → 512 → 8 MLP |
| Input scaling | region centres ×10, so they're on the order of 1 like the flags and normals |
| Constraint | dual variable λ, initial value 30 |
| Learning rate | 1e-4 for both phi and λ |
| Replay | the last 50 steps × num_envs transitions |
| Update schedule | every 24 env steps (once per PPO iteration), 32 gradient steps with batch 4096 |

**Consistency rules:**
- `update_every` equals PPO's `num_steps_per_env`, so all rewards in one PPO batch come from the same phi.
- A transition never crosses an episode boundary.
- **Contact sensor:** `sim/contacts.py` is a warp kernel over Newton's contact buffer. It counts penetrating pad–object contacts, and a pad's touch point is the mean of its contact positions.

Code: `metra/reward.py`, `metra/phi.py`, `GraspEnv._compute_contact_features`.

---

## 7. Terminations

| Term | Condition | Type | Drop penalty? |
|---|---|---|---|
| `time_out` | 500 steps (10s) | time-out | no |
| `object_spin` | object angular speed > 1000 rad/s | time-out (guards against divergence) | no |
| `nan_term` | any hand or object state is non-finite | terminal | no |
| `object_fallen` | object centre below z = 0.2m (20cm under the table reference) | terminal | **yes** |
| `palm_object_distance` | palm more than 22cm from the object | terminal | **yes** |

- **Time-outs vs terminals.** Time-outs are passed to PPO as `time_outs`, so their values are bootstrapped. Terminal episodes are not bootstrapped.
- **Logging.** W&B logs `Episode_Termination/<term>` as counts per reset batch.

---

## 8. Resets and domain randomization

### Every reset, per world

| What | Distribution |
|---|---|
| Hand joints (all 26, base included) | home pose + U(−0.1, 0.1), clamped to 90% of each joint range |
| Object shape | cylinder / sphere / pen, equal odds (`shape_weights`) |
| Object size and mass | fixed by default; ranges available in `RandomObjectCfg` |
| Object position | table centre ± 5cm in x and y; 2mm above the table at its exact lowest point |
| Object orientation | roll and pitch U(±1.57), yaw U(±3.14). **Pens can start upright** and then fall over. |
| Hand placement | palm moved so it's 5cm above the object's highest point, ±1cm noise per axis, then shifted 14cm in the palm frame so the fingers (not the wrist) are over the object |
| Skill z | uniform on the 8-D unit sphere, held for the whole episode |
| Lift | direction uniform on the cap within 45° of vertical; speed U(0.05, 0.2) m/s |

**Reproducibility.** The object draw uses a hash RNG keyed on (seed, env, episode), the same as mjlab. A given seed produces the same object sequence in both repos.

### Once at startup, per world

Friction on the fingertip pads:

| Friction | Distribution |
|---|---|
| Sliding | U(0.5, 1.0) |
| Torsional | log-uniform(1e-4, 2e-2) |
| Rolling | log-uniform(1e-5, 5e-3) |

### During the episode

- **Pushes:** 3–7N, random direction, 0.1s long, every 1.5s while the object is more than 3cm above its rest height. The clock restarts whenever it touches down.
- **Observation noise:** actor observations only, uniform. The half-widths are in §9.

---

## 9. Observations

The **actor gets 177 numbers** and the **critic gets 178**. They are the same terms in the same order as mjlab. The critic's copy is noise-free and adds the object's mass. Conventions:
- Positions are in the world frame, which is the env frame.
- Quaternions are (w, x, y, z).
- Velocities are scaled ×0.2.
- "Rel hand" means expressed in the palm frame.

| # | Term | Dims | Actor noise (± uniform) |
|---|---|---|---|
| 1 | `hand_dof_pos`: finger joints minus home | 20 | 0.01 |
| 2 | `hand_dof_vel`: finger joint velocities | 20 | 1.5 |
| 3 | `hand_root_pos`: palm position | 3 | 0.005 |
| 4 | `hand_root_quat` | 4 | |
| 5 | `hand_root_linvel` (×0.2) | 3 | |
| 6 | `hand_root_angvel` (×0.2) | 3 | |
| 7 | `object_pos` | 3 | 0.01 |
| 8 | `object_quat` | 4 | |
| 9 | `object_pos_rel_hand` | 3 | 0.01 |
| 10 | `object_quat_rel_hand` | 4 | |
| 11 | `object_linvel` (×0.2) | 3 | |
| 12 | `object_angvel` (×0.2) | 3 | |
| 13 | `keypoint_pos_rel`: 5 fingertips, palm frame | 15 | |
| 14 | `actions`: last action | 26 | |
| 15 | `skill`: z | 8 | |
| 16 | `contact_regions`: phi's 35 inputs (§6.1) | 35 | |
| 17 | `keypoint_pos_object`: 5 fingertips, object frame | 15 | |
| 18 | `object_shape`: one-hot shape (3), radius and half-length in dm (2) | 5 | |
| | **Actor total** | **177** | |
| 19 | `object_mass` (critic only, in units of 100g) | 1 | |
| | **Critic total** | **178** | |

**Velocity convention.** Velocities are those of the body *origin*, as in mjlab. Newton stores centre-of-mass velocity, so `_link_velocity` converts.

**Non-finite observations raise an error** instead of being passed on (mjlab's `nan_policy="error"`).

---

## 10. What we've learned (results so far)

### 10.1 Parity with mjlab

With MuJoCo collision (`--env.sim.collision mujoco`) and zero actions, the Newton env matches mjlab:

| Check | mjlab | Newton |
|---|---|---|
| Observation sizes | 177 / 178 | 177 / 178 (same terms, same order) |
| Palm–object offset after reset | (−0.1395, −0.0012, 0.0971) | (−0.1392, −0.0009, 0.0971) |
| Object rest heights (cylinder / sphere / pen) | 0.4597 / 0.4600 / 0.4253 | 0.4598 / 0.4600 / 0.4253 |

**Sim-to-sim check.** mjlab's `model_63000` checkpoint (the `drop10` run), loaded unchanged and run on 1000 envs. "Held" means the object is still more than 5cm up at the end:

| Sim | Cylinder | Sphere | Pen | Total |
|---|---|---|---|---|
| mjlab | 201/338 | 280/361 | 0/301 | 481 |
| Newton, MuJoCo collision | 173 | 276 | 1 | 450 |
| Newton, Newton collision | 148 | 274 | 2 | 424 |

The policy transfers well. The small drop under Newton collision fits the contact model being different from the one it trained on.

### 10.2 Why pens sink into the table under MuJoCo collision

- **One contact point.** MuJoCo-Warp sends a cylinder lying on a box through its general convex collision routine, which returns **one contact point**.
- **That point lands at a pen end.** One point can carry the pen's weight but can't stop it rotating about that point. The pen rocks and digs in.
- **Measured depth:** about 2mm median and 8–10mm at the 95th percentile during rollouts, even when the hand barely touches the pen.
- **Newton's collision** returns several points (one at each end of the pen), so the pen sits at its true height.
- **MuJoCo's multi-contact flag doesn't help.** `enable_multiccd`, exposed as `--env.sim.multiccd`, only works for box and mesh pairs, not cylinders.

To confirm, we ran a bare scene with only the table and an 18.6g pen, pressed down at its centre:

| Pen | Collision | Contacts | Rest depth | 2N press | 20N press |
|---|---|---|---|---|---|
| cylinder | MuJoCo (± multiccd) | 1 | 0.54mm | 6.6mm | falls through the table |
| cylinder | Newton | 2 | 0.01mm | 0.02mm | 0.14mm |
| 32-sided mesh prism | MuJoCo + multiccd | 2 | 0.00mm | 0.01mm | 0.14mm |

Once MuJoCo is given two contacts it matches Newton exactly, which shows the problem was the contact count all along. Fixing it inside mjlab would mean a mesh pen with multiccd on, or a capsule pen.

### 10.3 Fingers sink deeper into the pen under Newton collision

**Effect.** This is the flip side of §10.2. The table no longer gives way, so when the base presses the pen down (it can push about 200N), the fingers absorb the squeeze instead.

**Comparison.** 18.6g pen, `model_47000` policy:

| | MuJoCo collision | Newton collision |
|---|---|---|
| Touching steps with a finger deeper than the pen's radius | 0.2% | 3.8% |
| Steps where a finger passes fully through the pen | 0 | ~70 |
| Touching steps with more than 50N of hand force on the pen | 2.6% | 9.7% |

**Ruled out:**
- Contact parameters, which are identical in both modes.
- Contact depth: on identical states, Newton's detector and MuJoCo's report the same depths.
- Stale contacts: collision runs every substep.

**Cause.** The hand–pen contacts are soft (mixed solref 0.0075, solimp 0.945–0.97).

**A fix that works (not applied yet).** Making hand–pen contacts as stiff as the table's cuts deep-penetration steps from 3.8% to 0.05%, and pass-through steps from about 70 to 4. Making it permanent is an open decision (§13).

### 10.4 The video "floating pen" was a rendering bug

- **What happened.** Newton's viewer reads shape sizes and transforms only once, when the model is loaded. After a reset changed the object, videos drew stale geometry, for example a pen drawn upright while the simulated one lay flat.
- **Status.** Fixed for training videos (`rl/video.py` reloads the shapes on reset). Not yet fixed in `cli/play.py`.
- **mjlab is not affected.** Its renderer copies per-env `geom_size` and `geom_quat` before every frame (we checked frame by frame).
- **Floating doesn't happen physically.** We checked numerically: no pen hangs unsupported.

### 10.5 Pen inertia and NaNs

- **Background.** mjlab floors every axis of the pen's inertia at 1e-5 kg·m². Its true long-axis inertia is 3.9e-7, and that axis used to spin up until MuJoCo 3.8 diverged.
- **The test.** Here we removed the floor and checked:
  - six 1000-env × 1000-step rollouts, using both the trained policy and random actions, in both collision modes;
  - a 300-iteration training run from scratch.
- **Results:**
  - **No NaNs anywhere.**
  - Pens occasionally spin up to about 1000 rad/s, and the `object_spin` guard ends about 0.6% of rollout episodes as time-outs. In training this was rarer, at most 0.08 per iteration.
- **Turning the floor back on:** `--env.object.min-inertia 1e-5`.

### 10.6 Speed

| | Per iteration (4096 envs, 24 steps) |
|---|---|
| Newton | **1.9s** |
| mjlab | **7.2s** |

Both were measured on the same GPU under the same load. Physics accounts for only part of the gap:
- **Contact features.** mjlab computes them 3 times per step (actor obs, critic obs and the METRA reward), at about 35ms each. Each pass loops over shapes with boolean masks that wait on the GPU. Newton computes them once with a single lookup.
- **Overhead.** mjlab's per-term manager overhead costs time too.
- **Physics.** Newton's physics is faster as well: about 25ms vs 48ms for the 4 substeps.

---

## 11. Gotchas (things that cost us time)

1. **Newton's default contact gap is 10cm**, and it gets forwarded to MuJoCo-Warp. That made every pair within 10cm a contact candidate, overflowing the broadphase and silently dropping real contacts. `scene.py` sets `rigid_gap = 0`.
2. **A `mujoco:solref` set from code is ignored** unless that shape's `mujoco:solref_mode` is 1 (RAW). Otherwise Newton derives solref from its own `ke`/`kd`.
3. **MuJoCo's XML export writes `solref="0.01"`** without the damping ratio, and Newton reads the missing value as 0, an undamped contact. The exported MJCF spells out `"0.01 1"`.
4. **Newton collides MJCF mesh geoms as triangle meshes.** That was 20× slower and gave wrong contacts. Call `approximate_meshes("convex_hull")` to match MuJoCo.
5. **Newton 1.6 lays out `control.joint_target_q` like `joint_q`** (by coordinate), not by DOF.
6. **Conventions differ:**
   - Newton quaternions are (x, y, z, w), but everything outside warp kernels here uses (w, x, y, z) (`math_utils.py`).
   - `body_qd` is centre-of-mass velocity, linear part first.
   - `body_f` is a world-frame wrench at the centre of mass.
7. **Viewer:**
   - Headless rendering needs `pyglet.options["headless"] = True` before importing the viewer.
   - A **second** `ViewerGL` in the same process renders black, so reuse one.
   - The viewer won't see object changes unless you call `set_model` again (§10.4).
8. **wandb ≥ 0.26 breaks rsl-rl 5.2.0** (`Settings(start_method=...)` was removed). It's pinned to 0.25.

---

## 12. Code map

| Path | What it does |
|---|---|
| `src/newton_hand/config.py` | Every task, sim and PPO setting, as dataclasses. Each default notes which mjlab constant it reproduces. |
| `src/newton_hand/sim/scene.py` | Builds the Newton model: hand from MJCF, table, object; per-world index tables (`SceneIndex`). |
| `src/newton_hand/sim/contacts.py` | Warp kernel for fingertip-pad vs object contacts (touching + mean position). |
| `src/newton_hand/envs/grasp_env.py` | **The task**: reset, actions, push, rewards, terminations, observations. Start reading here. |
| `src/newton_hand/objects/random_primitive.py` | Shape/size/mass draw, inertia, surface regions, region lookups. |
| `src/newton_hand/objects/regions.py` | Farthest-point sampling + k-means region clustering (copied from mjlab). |
| `src/newton_hand/metra/` | `phi.py` (network + dual-constraint update) and `reward.py` (reward, replay, update schedule). |
| `src/newton_hand/rl/runner.py` | rsl-rl `VecEnv` wrapper; runner that saves and loads `metra_phi_<N>.pt`. |
| `src/newton_hand/rl/video.py` | Training-time video clips (env 0, one episode, uploaded to W&B). |
| `src/newton_hand/cli/train.py`, `cli/play.py` | `train-newton`, `play-newton`. |
| `scripts/replay_mjlab_skills.py` | Replays logged mjlab episodes in Newton, from the same start state with the same skill. |
| `scripts/pen_contact_gifs.py` | Pen close-up GIFs with per-finger force arrows. |

**Where to start reading.** `GraspEnv.step` is about 30 lines and calls everything else in order: action → physics → terminations → contact features → rewards → resets → push → observations.

---

## 13. Open questions and next steps

**Hand–pen contact stiffness (§10.3).** Decide whether to stiffen it permanently, and whether only in newton_hand or in mjlab too. The clean way is to change the hand geoms' own solref/solimp. Raising the object's priority would also override the pads' friction randomization.

**Lift-cancel loophole.** If the policy has pushed a base slide to its limit in the direction the lift needs, the scripted lift is cancelled entirely, so the policy can avoid being lifted. mjlab has the same behaviour. It isn't fixed in either repo.

**`play-newton` rendering.** It needs the same shape reload on reset that `rl/video.py` got (§10.4).

**A first full training run in Newton.** We have sim-to-sim results with mjlab checkpoints but no long Newton-trained run yet. One was just started, with the old 1e-5 inertia floor.

**Repo state.** The repo has **no commits yet**; everything is uncommitted on `main`. Recent uncommitted changes:
- the `multiccd` flag;
- training videos;
- the wandb pin;
- `min_inertia = 0`.

Nobody has created a GitHub remote yet.
