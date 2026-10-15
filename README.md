# newton_hand

The random-object METRA grasp task from
[mjlab_hand](https://github.com/Kartikgar/mjlab_hand), running on the
[Newton](https://github.com/newton-physics/newton) physics engine.

The task is mjlab_hand's `Grasp-METRA-Hover-Lift-Contacts-Push-RandomObject-Wuji`:

- The Wuji hand starts hovering 5cm above an object on a table.
- Every episode, each env draws a **cylinder, sphere or pen**.
- The policy moves the base and the fingers.
- At 5s a scripted lift pulls the base away in a random direction within 45° of vertical.
- While the object is in the air it is pushed with random 3–7N forces.
- The reward is METRA's `(phi(s') - phi(s))^T z` on contacts-only `phi`. That is, per fingertip: touching, plus the touched region's centre and normal.
- There is also a smoothness penalty and a drop penalty.

It is a separate repo because Newton 1.6 needs warp ≥ 1.17 and mujoco 3.12. mjlab_hand pins warp 1.12 and mujoco 3.8.

## Mimic tasks (mjlab_hand's `Mimic-Panda-*` and `Mimic-FR3-*`)

newton_hand also runs mjlab_hand's DeepMimic-style tasks: a Franka Panda or FR3 on a table carries an Allegro or
Wuji hand, and the policy imitates a retargeted GRAB demonstration. The model, gains and rewards are
exported from mjlab_hand and checked against it element by element. See [docs/mimic.md](docs/mimic.md).

```bash
cd ~/mjlab_hand && uv run python ~/newton_hand/scripts/export_mimic_from_mjlab.py --task Mimic-Panda-Allegro-Apple
cd ~/newton_hand && uv run python scripts/check_mimic_model.py --task Mimic-Panda-Allegro-Apple
CUDA_VISIBLE_DEVICES=0 uv run train-mimic --env.task Mimic-Panda-Allegro-Apple --env.num-envs 4096 \
  --agent.logger wandb --agent.run-name mimic-panda-allegro-apple-newton --video True
```

## Setup

```bash
cd ~/newton_hand
uv sync            # Python 3.13, Newton 1.6.1, torch 2.11 (cu128), rsl-rl-lib 5.2.0
```

## Train / play

```bash
# Same command shape as mjlab_hand. Bools take a value (--agent.resume True).
CUDA_VISIBLE_DEVICES=0 uv run train-newton \
  --env.num-envs 4096 --agent.max-iterations 20000 \
  --agent.logger wandb --agent.wandb-project mjlab-amp \
  --agent.run-name metra-random-object-newton-z8

# Resume (max-iterations counts *additional* iterations):
  ... --agent.resume True --agent.load-run <run folder> --agent.load-checkpoint model_<N>.pt

# Videos during training: a close-up of world 0, one whole episode every --video-interval env steps,
# saved to <run>/videos/train/ and uploaded to W&B. Recording costs ~25ms per frame.
  ... --video True --video-interval 48000

# Watch a checkpoint: live window, or headless mp4
uv run play-newton --checkpoint logs/rsl_rl/<exp>/<run>/model_<N>.pt
CUDA_VISIBLE_DEVICES=0 uv run play-newton --checkpoint ... --video renders/play.mp4 --steps 500
```

Every setting is a CLI flag. For example `--env.push.force-range 1.0 3.0`, `--env.push.enabled False`,
`--env.object.shape-weights 0 0 1` (pens only), or `--env.sim.collision mujoco`. See
`src/newton_hand/config.py`; each default there carries a comment saying which mjlab constant it reproduces.
Logs go to `logs/rsl_rl/<experiment>/<date>_<run-name>/`. `metra_phi_<N>.pt` is saved next to every `model_<N>.pt`.

## What matches mjlab_hand

Checked against the mjlab env with zero actions, 96 envs, using MuJoCo collision (`--env.sim.collision mujoco`):

| | mjlab_hand | newton_hand |
|---|---|---|
| Actor / critic obs, action | 177 / 178, 26 | 177 / 178, 26 (same terms, same order) |
| Palm−object offset after reset | (−0.1395, −0.0012, 0.0971) | (−0.1392, −0.0009, 0.0971) |
| Object rest height: cylinder / sphere / pen | 0.4597 / 0.4600 / 0.4253 | 0.4598 / 0.4600 / 0.4253 |

These are the same in both repos, not re-tuned:

- **Physics settings:** 5ms timestep, 4 substeps per control step, implicitfast integrator, Newton solver with 50/50 iterations, elliptic cone, impratio 10.
- **Hand:** collision and actuator setup. The MJCF in `assets/wuji/` is exported from mjlab *after* `FULL_COLLISION` and the position actuators were applied.
- **Contacts:** the table and object use stiff contacts (solref 0.005, solimp 0.99/0.99/0.001). The fingertip pads get friction randomization at startup.
- **Object draw:** uses the same hash RNG and region layouts, so a seed gives the same object sequence and region ids as mjlab.
- **Resets, actions and pushes:** same reset events, action mapping with EMA and scripted lift, push timing, and terminations (22cm palm radius, fall, spin guard).
- **Training:** METRA hyperparameters and PPO config are the same.

## Differences

- **Contact sensing.** mjlab's `keypoint_contacts` sensor is replaced by a warp kernel over Newton's `Contacts` buffer (`sim/contacts.py`). A pad's touch point is the mean of its touching contact points.
- **Collision detection is Newton's by default.** Newton's `CollisionPipeline` finds the contacts and hands them to the MuJoCo solver, which still applies each geom's condim, priority, solref, solimp and friction. This is the setup Newton's own hand examples use. Objects rest exactly on their surfaces with it (pens at 0.4265, where MuJoCo's collision leaves them 1.2mm deep). `--env.sim.collision mujoco` switches back to MuJoCo-Warp's collision detection, as mjlab uses.
- **Hand colliders are convex hulls.** MuJoCo always collides a mesh as its convex hull. Newton keeps the triangle mesh unless told otherwise, and that was 20x slower and ended 3x more episodes early. `scene.py` converts the hand's collision meshes to hulls; the visual meshes keep their detail.
- **Kinematics freshness.** Terminations and rewards read kinematics refreshed after the last substep. mjlab's are one substep stale.
- **`nconmax` is 400, not 200.** Newton keeps more broadphase candidates, and 200 overflowed at 255.
- **Not ported (yet):**
  - the Viser skill GUI;
  - `grasp-taxonomy`, `eval-policy`, the drop test;
  - every other task in mjlab_hand.

## Performance

4096 envs on one RTX PRO 6000 run at about **37ms per env step with Newton collision (~110k env-steps/s)**, 41ms with MuJoCo collision. With MuJoCo collision the split was:
- 25ms physics, captured as a CUDA graph;
- about 16ms for MuJoCo's `set_const`, which must run after each reset changes object masses (mjlab pays the same);
- the rest is observations and reset bookkeeping.

## Sim-to-sim check

mjlab's `model_63000` from the `drop10` fixed-size random-object run, loaded unchanged. 1000 envs, one random skill each, seed 0, 500 steps; held = still running at step 499 with the object more than 5cm up:

| Sim | Cylinder | Sphere | Pen | Total |
|---|---|---|---|---|
| mjlab (trained here) | 201/338 | 280/361 | 0/301 | 481 |
| Newton, MuJoCo collision | 173 | 276 | 1 | 450 |
| Newton, Newton collision | 148 | 274 | 2 | 424 |

## Gotchas found while porting

- Newton's default contact `gap` (10cm) is forwarded to MuJoCo-Warp. It made every pair within 10cm a contact candidate, which overflowed the broadphase and dropped real contacts. `scene.py` sets `rigid_gap = 0`.
- A `mujoco:solref` set from code is ignored unless the shape's `mujoco:solref_mode` is RAW (1). Without it, Newton derives solref from `ke`/`kd`.
- MuJoCo's XML export writes `solref="0.01"` without the default damping ratio, and Newton reads the missing value as 0, which makes the contact undamped. The exported MJCF spells out `0.01 1`.
- Newton 1.6 lays out `control.joint_target_q` like `joint_q` (coordinates), not per DOF.
- Newton's `CollisionPipeline` collides MJCF mesh geoms as triangle meshes; call `approximate_meshes("convex_hull")` to get MuJoCo's behaviour.
- Newton quaternions are `(x, y, z, w)`; everything outside warp kernels here uses mjlab's `(w, x, y, z)` (`math_utils.py`).

## Layout

| Path | Role |
|---|---|
| `config.py` | Every task, sim and PPO setting (tyro CLI) |
| `sim/scene.py` | Builds the Newton model and per-world index tables |
| `sim/contacts.py` | Fingertip-pad vs object contact kernel |
| `envs/grasp_env.py` | The task: reset, action, push, rewards, terminations, observations |
| `objects/random_primitive.py` | Shape/size/mass draw, surface regions, region lookups |
| `metra/` | `phi` (copied from mjlab_hand) and the METRA reward with its replay and updates |
| `rl/runner.py` | rsl-rl `VecEnv` wrapper and the runner that checkpoints `phi` |
| `cli/train.py`, `cli/play.py` | `train-newton`, `play-newton` |
