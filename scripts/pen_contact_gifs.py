"""Pen contact GIFs in Newton, drawn as mjlab_hand's ``render_pen_contacts_gif.py`` draws them.

For each chosen env of an mjlab pen rollout log, start Newton from that env's logged
state (see ``replay_mjlab_skills.start_from_log``), run the mjlab policy, and render
one panel per pen mass, side by side. Drawing follows
``docs/weekly_2026-10-08/pen_data_and_plots_handoff.md`` section 2:

- see-through hand (opacity 0.3), camera 0.38 m away at azimuth 145 deg, elevation
  -22 deg, looking at the midpoint of the pen and the palm;
- per touching hand link, a yellow arrow along its net force on the pen, 2 cm per N
  (capped at 10 cm), ending at its strongest contact, with a dot on that contact;
- the two-line caption: pen height above its step-30 rest height; the sum of the
  per-link force magnitudes, the number of links touching, PUSH;
- every 4th step at 12.5 fps, 480 px wide per panel, 256-colour palette, no dither;
  a panel freezes on its last frame once its episode ends.

Forces come from MuJoCo-Warp's contact list after the last physics substep of each
step (``mujoco_warp.contact_force``), which holds the contacts in either collision mode.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/pen_contact_gifs.py \\
        --log ~/mjlab_hand/docs/weekly_2026-10-08/pen_contacts_drop10_47000.npz \\
        --checkpoint <mjlab run>/model_47000.pt --out-dir renders/pen_contacts_47000
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco_warp
import numpy as np
import torch
import tyro
import warp as wp
from PIL import Image, ImageDraw, ImageFont

from newton_hand.config import EnvCfg, PpoCfg
from newton_hand.envs.grasp_env import GraspEnv
from newton_hand.objects.random_primitive import RandomObjectCfg
from newton_hand.rl.runner import MetraOnPolicyRunner, RslRlVecEnv
sys.path.insert(0, str(Path(__file__).parent))
from replay_mjlab_skills import start_from_log  # noqa: E402

W, H, EVERY = 640, 480, 4
ARROW_M_PER_N, ARROW_MAX_M = 0.02, 0.10
CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION = 0.38, 145.0, -22.0
PEN_RADIUS, PEN_HALF_LENGTH = 0.0065, 0.07


@dataclass
class Args:
    log: str
    checkpoint: str
    envs: tuple[int, ...] = (88, 160, 196, 304, 323)
    masses_g: tuple[float, ...] = (18.6, 180.0)
    steps: int = 500
    collision: str = "newton"
    out_dir: str = "renders/pen_contacts"
    device: str = "cuda:0"
    render_mass_g: float | None = None
    """Internal: render only this mass into ``frames_root`` (one process per mass, since a
    second headless GL viewer in one process renders black)."""
    frames_root: str | None = None


class PenContacts:
    """Per step and world: each hand link's net force on the pen and strongest contact, and the table's."""

    def __init__(self, env: GraspEnv, link_names: list[str]):
        self.env = env
        solver = env.solver
        mj = solver.mj_model
        geom_to_shape = solver.mjc_geom_to_newton_shape.numpy()[0]
        labels = env.model.shape_label

        def leaf(g: int) -> str:
            s = geom_to_shape[g]
            return labels[s].rsplit("/", 1)[-1] if s >= 0 else ""

        self.geom_link = np.full(mj.ngeom, -1)
        self.geom_is_pen = np.zeros(mj.ngeom, bool)
        self.geom_is_table = np.zeros(mj.ngeom, bool)
        for g in range(mj.ngeom):
            name = leaf(g)
            if name in link_names:
                self.geom_link[g] = link_names.index(name)
            self.geom_is_pen[g] = name in ("object_cylinder", "object_sphere")
            self.geom_is_table[g] = name == "table_surface"
        self.num_links = len(link_names)
        d = solver.mjw_data
        self._ids = wp.array(np.arange(d.naconmax, dtype=np.int32), device=env.model.device)
        self._force = wp.zeros(d.naconmax, dtype=wp.spatial_vector, device=env.model.device)

    def read(self) -> dict[str, np.ndarray]:
        solver, n_env = self.env.solver, self.env.num_envs
        d = solver.mjw_data
        mujoco_warp.contact_force(solver.mjw_model, d, self._ids, True, self._force)
        n = int(d.nacon.numpy()[0])
        geom = d.contact.geom.numpy()[:n]
        world = d.contact.worldid.numpy()[:n]
        pos = d.contact.pos.numpy()[:n]
        dist = d.contact.dist.numpy()[:n]
        f12 = self._force.numpy()[:n, :3]  # world frame, the force geom 1 receives from geom 2 is -f12

        L = self.num_links
        out = {
            "hand_force": np.zeros((n_env, L, 3)),
            "hand_found": np.zeros((n_env, L), int),
            "hand_pos": np.zeros((n_env, L, 3)),
            "hand_dist": np.zeros((n_env, L)),  # deepest contact per link, m (negative = overlap)
            "table_force": np.zeros((n_env, 3)),
            "table_dist": np.zeros(n_env),
        }
        best = np.zeros((n_env, L))
        for c in range(n):
            g1, g2 = geom[c]
            pen1, pen2 = self.geom_is_pen[g1], self.geom_is_pen[g2]
            if pen1 == pen2:
                continue
            other = g2 if pen1 else g1
            # MuJoCo's contact force acts on geom 2 along the frame (normal from geom 1 to 2).
            f_on_pen = f12[c] if pen2 else -f12[c]
            w = world[c]
            if self.geom_is_table[other]:
                out["table_force"][w] += f_on_pen
                out["table_dist"][w] = min(out["table_dist"][w], dist[c])
                continue
            link = self.geom_link[other]
            if link < 0:
                continue
            out["hand_force"][w, link] += f_on_pen
            out["hand_found"][w, link] += 1
            out["hand_dist"][w, link] = min(out["hand_dist"][w, link], dist[c])
            mag = float(np.linalg.norm(f_on_pen))
            if mag >= best[w, link]:
                best[w, link] = mag
                out["hand_pos"][w, link] = pos[c]
        return out


def run(args: Args, log, mass_g: float, frames_dir: Path, font) -> dict:
    """One rollout of every chosen env at pen mass ``mass_g``; frames go to ``frames_dir/<k>/``."""
    envs = list(args.envs)
    volume = math.pi * PEN_RADIUS**2 * 2 * PEN_HALF_LENGTH
    density = mass_g / 1000.0 / volume
    cfg = EnvCfg(num_envs=len(envs), seed=0, obs_noise=False, episode_length_s=1e9)
    cfg.metra.train_phi = False
    cfg.sim.collision = args.collision
    cfg.object = RandomObjectCfg(shape_weights=(0.0, 0.0, 1.0), pen_density=(density, density))
    env = GraspEnv(cfg, device=args.device)
    vec = RslRlVecEnv(env, clip_actions=1.0)
    runner = MetraOnPolicyRunner(vec, PpoCfg().to_rsl_rl(), log_dir=None, device=args.device)
    runner.load(str(Path(args.checkpoint).expanduser()), load_cfg={"actor": True})
    policy = runner.get_inference_policy(device=args.device)
    start_from_log(env, log, envs)
    print(f"[pen {mass_g:g} g] density {density:.0f} kg/m^3, mass in sim {env.objects.mass.tolist()}")

    import pyglet

    pyglet.options["headless"] = True
    import newton.viewer

    model = env.model
    hand_shapes = [i for i, label in enumerate(model.shape_label) if "/right_" in label]
    opacity = wp.to_torch(model.shape_opacity)
    opacity[hand_shapes] = 0.3  # render only
    viewer = newton.viewer.ViewerGL(width=W, height=H, headless=True)
    viewer.set_model(model)

    contacts = PenContacts(env, [str(n) for n in log["links"]])
    ix = env.ix
    n = len(envs)
    done = np.zeros(n, bool)
    end = ["running"] * n
    end_step = [args.steps - 1] * n
    rest = np.full(n, np.nan)
    stats = {k: {"max_into_table_mm": 0.0, "hand_force": [], "max_rise_cm": -1e9} for k in envs}
    for k in envs:
        (frames_dir / str(k)).mkdir(parents=True, exist_ok=True)
    frame_count = [0] * n

    el, az = math.radians(CAM_ELEVATION), math.radians(CAM_AZIMUTH)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    obs = vec.get_observations()
    with torch.inference_mode():
        for s in range(1, args.steps):
            obs, _, dones, _ = vec.step(policy(obs))
            newly = dones.bool().cpu().numpy() & ~done
            for i in np.flatnonzero(newly):
                terms = env.last_terminations
                end[i] = next(t for t in ("nan_term", "object_fallen", "palm_object_distance", "object_spin") if bool(terms[t][i]))
                end_step[i] = s
            done |= newly
            data = contacts.read()
            pen_pos = env._pose(ix.object_body)[0].cpu().numpy()
            palm_pos = env._pose(ix.palm_body)[0].cpu().numpy()
            push = env._push_active.cpu().numpy()
            if s == 30:
                rest[:] = pen_pos[:, 2]
            for i, k in enumerate(envs):
                if done[i]:
                    continue
                fmag = np.linalg.norm(data["hand_force"][i], axis=-1)
                st = stats[k]
                st["max_into_table_mm"] = max(st["max_into_table_mm"], -1000 * data["table_dist"][i])
                if fmag.sum() > 0:
                    st["hand_force"].append(float(fmag.sum()))
                if s >= 30:
                    st["max_rise_cm"] = max(st["max_rise_cm"], 100 * (pen_pos[i, 2] - rest[i]))
                if s % EVERY:
                    continue
                # Camera as the mjlab renderer: fixed distance and angles, looking at pen-palm midpoint.
                look = 0.5 * (pen_pos[i] + palm_pos[i])
                eye = look - CAM_DISTANCE * forward
                viewer.set_visible_worlds([i])
                viewer.set_camera(wp.vec3(*eye.tolist()), CAM_ELEVATION, CAM_AZIMUTH)
                viewer.begin_frame(s * env.step_dt)
                viewer.log_state(env.state)
                touching = np.flatnonzero((data["hand_found"][i] > 0) & (fmag >= 0.05))
                if len(touching):
                    tips = data["hand_pos"][i, touching]
                    dirs = data["hand_force"][i, touching] / fmag[touching, None]
                    tails = tips - dirs * np.minimum(ARROW_M_PER_N * fmag[touching], ARROW_MAX_M)[:, None]
                    viewer.log_arrows("forces", wp.array(tails.astype(np.float32), dtype=wp.vec3),
                                      wp.array(tips.astype(np.float32), dtype=wp.vec3), (1.0, 0.85, 0.0))
                    dots = np.tile(np.float32([0.1, 0.1, 0.1]), (len(tips), 1))
                    viewer.log_points("contact_points", wp.array(tips.astype(np.float32), dtype=wp.vec3), 0.004,
                                      wp.array(dots, dtype=wp.vec3))
                else:
                    viewer.log_arrows("forces", None, None, (1.0, 0.85, 0.0), hidden=True)
                    viewer.log_points("contact_points", None, 0.004, None, hidden=True)
                viewer.end_frame()
                img = Image.fromarray(viewer.get_frame().numpy())
                dr = ImageDraw.Draw(img)
                dr.rectangle([0, 0, W, 48], fill=(250, 249, 245))
                height = "" if np.isnan(rest[i]) else f"  |  pen {100 * (pen_pos[i, 2] - rest[i]):+5.1f} cm"
                dr.text((8, 4), f"pen {mass_g:g} g  |  skill {k}  |  t = {s * env.step_dt:4.1f} s{height}",
                        fill=(20, 20, 19), font=font)
                dr.text((8, 26), f"hand force on pen {fmag.sum():5.2f} N  |  links touching {int((data['hand_found'][i] > 0).sum())}"
                        + ("  |  PUSH" if push[i] else "") + f"  |  arrows: 1 N = {100 * ARROW_M_PER_N:.0f} cm",
                        fill=(20, 20, 19), font=font)
                img.save(frames_dir / str(k) / f"{frame_count[i]:04d}.png")
                frame_count[i] += 1
    viewer.close()
    for i, k in enumerate(envs):
        if frame_count[i] == 0:  # ended before the first rendered step
            img = Image.new("RGB", (W, H), (250, 249, 245))
            ImageDraw.Draw(img).text((8, 4), f"pen {mass_g:g} g  |  skill {k}\nepisode ended at step {end_step[i]} ({end[i]})",
                                     fill=(20, 20, 19), font=font)
            img.save(frames_dir / str(k) / "0000.png")

    summary = {}
    for i, k in enumerate(envs):
        st = stats[k]
        alive = not done[i]
        rise = None if not alive else round(100 * float(env._pose(ix.object_body)[0][i, 2].item() - rest[i]), 1)
        summary[k] = {
            "held": bool(alive and rise is not None and rise > 5.0),
            "end": end[i], "end_step": end_step[i], "pen_rise_end_cm": rise,
            "max_pen_rise_cm": round(st["max_rise_cm"], 1) if st["max_rise_cm"] > -1e8 else None,
            "max_into_table_mm": round(st["max_into_table_mm"], 2),
            "hand_force_median_N": round(float(np.median(st["hand_force"])), 2) if st["hand_force"] else None,
            "hand_force_max_N": round(float(np.max(st["hand_force"])), 1) if st["hand_force"] else None,
        }
    return summary


def encode(panel_dirs: list[Path], gif: Path) -> None:
    """Side-by-side panels; a panel whose episode ended holds its last frame."""
    counts = [len(sorted(p.glob("*.png"))) for p in panel_dirs]
    with tempfile.TemporaryDirectory() as tmp:
        for f in range(max(counts)):
            row = [Image.open(p / f"{min(f, c - 1):04d}.png") for p, c in zip(panel_dirs, counts)]
            canvas = Image.new("RGB", (W * len(row) + 8 * (len(row) - 1), H), (255, 255, 255))
            for j, im in enumerate(row):
                canvas.paste(im, (j * (W + 8), 0))
            canvas.save(Path(tmp) / f"{f:04d}.png")
        width = 480 * len(panel_dirs)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", str(1 / (0.02 * EVERY)), "-i", str(Path(tmp) / "%04d.png"),
             "-vf", f"scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256[p];[b][p]paletteuse=dither=none",
             "-loop", "0", str(gif)],
            check=True,
        )


def main() -> None:
    args = tyro.cli(Args, config=(tyro.conf.FlagConversionOff,))
    log = np.load(Path(args.log).expanduser(), mmap_mode="r", allow_pickle=True)
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    if args.render_mass_g is not None:
        root = Path(args.frames_root)
        result = run(args, log, args.render_mass_g, root / f"{args.render_mass_g:g}", font)
        (root / f"{args.render_mass_g:g}.json").write_text(json.dumps(result))
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"log": args.log, "checkpoint": args.checkpoint, "collision": args.collision, "runs": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for mass in args.masses_g:
            cmd = [sys.executable, __file__, "--log", args.log, "--checkpoint", args.checkpoint,
                   "--envs", *map(str, args.envs), "--steps", str(args.steps), "--collision", args.collision,
                   "--device", args.device, "--render-mass-g", str(mass), "--frames-root", tmp]
            subprocess.run(cmd, check=True)
            summary["runs"][f"{mass:g}g"] = json.loads((Path(tmp) / f"{mass:g}.json").read_text())
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        for k in args.envs:
            gif = out / f"pen_contacts_skill{k}.gif"
            encode([Path(tmp) / f"{m:g}" / str(k) for m in args.masses_g], gif)
            print(f"wrote {gif}")
    print(json.dumps(summary["runs"], indent=1))


if __name__ == "__main__":
    main()
