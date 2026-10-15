"""One object whose shape, size and mass are redrawn every episode.

Port of ``mjlab_hand.object.random_primitive``. Each episode, every world holds
one of three solids -- a cylinder, a sphere, or a pen (a thin cylinder lying
along the body's x axis) -- at a size and mass drawn fresh at reset. The object
body carries one cylinder shape and one sphere shape; the reset writes each
world's ``shape_scale``, ``shape_transform``, ``body_mass`` and
``body_inertia`` (see :class:`newton_hand.envs.grasp_env.GraspEnv`):

- the active shape gets its sampled size, and the other shrinks to a 0.1mm
  speck at the centre, where nothing can reach it;
- the pen is the cylinder shape turned to lie along body x, so the body frame
  means the same thing for every shape;
- mass and inertia are those of a uniform solid, as a diagonal inertia. Pens
  take their mass from a density; ``min_inertia`` can floor a pen's inertia.

Regions are built once per shape on a reference-size copy and each world's
contacts are mapped through its current size, so a region keeps its place on
the shape (cap rim, side, pole) whatever the size. Everything here is plain
torch on per-world tensors; nothing touches the simulator.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np
import torch

from newton_hand.math_utils import quat_apply_inverse
from newton_hand.objects.regions import cluster_points_fps_kmeans

SHAPES: tuple[str, ...] = ("cylinder", "sphere", "pen")
CYLINDER, SPHERE, PEN = range(len(SHAPES))

# Reference sizes, (radius, half-length) in metres; the regions are clustered on
# these. Same as mjlab_hand, so region ids mean the same thing in both repos.
REFERENCE_SIZE: dict[str, tuple[float, float]] = {
    "cylinder": (0.04, 0.04),
    "sphere": (0.04, 0.04),
    "pen": (0.0065, 0.07),
}

# Size that hides the inactive shape inside the active one: 30x smaller than the
# thinnest pen's 3mm radius.
HIDDEN_SIZE = 1.0e-4

# The pen's cylinder shape, turned 90 degrees about y: shape z -> body x. (w, x, y, z).
PEN_QUAT_WXYZ = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)

_NUM_POINTS = 512
_NUM_REGIONS = 64
_SEED_PREFIX = "random_primitive"  # mjlab_hand's RANDOM_PRIMITIVE, so the samples match.


@dataclass(frozen=True)
class RandomObjectCfg:
    """Per-episode object draw. Defaults are the Wuji RandomObject task's.

    Sizes and masses are fixed at the middle of each full range; only the shape
    is drawn per episode. The full ranges, for randomizing them again: cylinder
    radius (0.02, 0.06) and half-length (0.02, 0.07), sphere radius (0.02, 0.06),
    pen radius (0.003, 0.01) and half-length (0.05, 0.09), mass (0.02, 0.3),
    pen density (500.0, 1500.0).
    """

    shape_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # cylinder, sphere, pen
    cylinder_radius: tuple[float, float] = (0.04, 0.04)  # 8cm across
    cylinder_half_length: tuple[float, float] = (0.045, 0.045)  # 9cm tall
    sphere_radius: tuple[float, float] = (0.04, 0.04)  # 8cm across
    pen_radius: tuple[float, float] = (0.0065, 0.0065)  # 1.3cm across
    pen_half_length: tuple[float, float] = (0.07, 0.07)  # 14cm long
    mass_range: tuple[float, float] = (0.16, 0.16)  # kg, cylinders and spheres
    pen_density: tuple[float, float] = (1000.0, 1000.0)  # kg/m^3; 18.6g at the size above
    # Optional floor on each axis of a pen's inertia. mjlab floors it at 1e-5 because a
    # thin pen's long axis (3.9e-7 for the default pen) spun up until MuJoCo diverged;
    # 0 keeps the true solid-cylinder inertia.
    min_inertia: float = 0.0  # kg m^2

    def max_cylinder_size(self) -> tuple[float, float]:
        """Largest (radius, half-length) the cylinder shape takes, pens included."""
        r = max(self.cylinder_radius[1], self.pen_radius[1])
        h = max(self.cylinder_half_length[1], self.pen_half_length[1])
        return r, h


@dataclass
class ShapeRegions:
    """A shape's surface regions at its reference size, in the body frame."""

    points: torch.Tensor  # (M, 3)
    point_to_region: torch.Tensor  # (M,)
    centers: torch.Tensor  # (K, 3)
    normals: torch.Tensor  # (K, 3)


_REGION_CACHE: dict[str, ShapeRegions] = {}


def shape_regions(shape: str) -> ShapeRegions:
    """Regions of ``shape`` at :data:`REFERENCE_SIZE`, built once per process.

    The surface sample is seeded from the shape name with the same seed as
    mjlab_hand, so region ids match between the two repos.
    """
    if shape in _REGION_CACHE:
        return _REGION_CACHE[shape]
    import trimesh

    radius, half_length = REFERENCE_SIZE[shape]
    if shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    else:
        mesh = trimesh.creation.cylinder(radius=radius, height=2 * half_length, sections=64)
    pts, face_idx = trimesh.sample.sample_surface(
        mesh, _NUM_POINTS, seed=zlib.crc32(f"{_SEED_PREFIX}/{shape}".encode())
    )
    pts = np.asarray(pts, dtype=np.float32)
    nrm = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    if shape == "pen":
        # Same turn as PEN_QUAT_WXYZ: (x, y, z) -> (z, y, -x).
        pts = np.stack([pts[:, 2], pts[:, 1], -pts[:, 0]], axis=-1)
        nrm = np.stack([nrm[:, 2], nrm[:, 1], -nrm[:, 0]], axis=-1)
    points, normals = torch.from_numpy(pts), torch.from_numpy(nrm)

    cluster = cluster_points_fps_kmeans(points, normals, _NUM_REGIONS)
    normal_sum = torch.zeros_like(cluster.region_centers)
    normal_sum.index_add_(0, cluster.point_to_region, normals)
    regions = ShapeRegions(
        points=points,
        point_to_region=cluster.point_to_region,
        centers=cluster.region_centers,
        normals=normal_sum / normal_sum.norm(dim=-1, keepdim=True).clamp_min(1e-8),
    )
    _REGION_CACHE[shape] = regions
    return regions


def solid_inertia(shape: torch.Tensor, mass: torch.Tensor, radius: torch.Tensor, half_length: torch.Tensor) -> torch.Tensor:
    """Diagonal inertia of a uniform solid in the body frame, ``(N, 3)``."""
    r2, length2 = radius**2, (2.0 * half_length) ** 2
    axial = 0.5 * mass * r2
    transverse = mass * (3.0 * r2 + length2) / 12.0
    ball = (0.4 * mass * r2).unsqueeze(-1).expand(-1, 3)
    cylinder = torch.stack([transverse, transverse, axial], dim=-1)
    pen = torch.stack([axial, transverse, transverse], dim=-1)  # long axis along body x
    return _by_shape(shape, cylinder, ball, pen)


def _by_shape(shape: torch.Tensor, cylinder: torch.Tensor, sphere: torch.Tensor, pen: torch.Tensor) -> torch.Tensor:
    """Pick per row from three ``(N, 3)`` candidates by shape, without a host sync."""
    s = shape.unsqueeze(-1)
    return torch.where(s == SPHERE, sphere, torch.where(s == PEN, pen, cylinder))


class RandomPrimitiveState:
    """Each world's current shape, size and mass, and the per-episode draw.

    Draws are a pure function of ``(seed, world, episode, stream)``, so world
    i's k-th object is the same in every run with this seed, whatever order
    the worlds reset in and whatever else consumes torch's RNG.
    """

    def __init__(self, num_envs: int, device: str, cfg: RandomObjectCfg, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.shape = torch.full((num_envs,), CYLINDER, dtype=torch.long, device=device)
        r, h = REFERENCE_SIZE["cylinder"]
        self.radius = torch.full((num_envs,), r, device=device)
        self.half_length = torch.full((num_envs,), h, device=device)
        self.mass = torch.zeros(num_envs, device=device)
        self.inertia = torch.zeros(num_envs, 3, device=device)
        self._episode = torch.zeros(num_envs, dtype=torch.long, device=device)

    def sample(self, env_ids: torch.Tensor) -> None:
        """Draw a new shape, size and mass for ``env_ids``."""
        cfg = self.cfg
        dev = self.shape.device
        episode = self._episode[env_ids]
        self._episode[env_ids] += 1
        streams = iter(range(8))

        def unit() -> torch.Tensor:
            return _hash_uniform(self.seed, env_ids, episode, next(streams))

        def uniform(lo_hi: tuple[float, float]) -> torch.Tensor:
            lo, hi = lo_hi
            return unit() * (hi - lo) + lo

        weights = torch.tensor(cfg.shape_weights, dtype=torch.float, device=dev)
        cdf = torch.cumsum(weights / weights.sum(), 0)
        shape = torch.searchsorted(cdf, unit().unsqueeze(-1), right=True).squeeze(-1)
        shape = shape.clamp_max(len(SHAPES) - 1)
        # Draw order matches mjlab_hand stream for stream, so a seed gives the same objects.
        cyl_r, sph_r, pen_r = uniform(cfg.cylinder_radius), uniform(cfg.sphere_radius), uniform(cfg.pen_radius)
        radius = torch.where(shape == CYLINDER, cyl_r, torch.where(shape == SPHERE, sph_r, pen_r))
        cyl_h, pen_h = uniform(cfg.cylinder_half_length), uniform(cfg.pen_half_length)
        half_length = torch.where(shape == CYLINDER, cyl_h, torch.where(shape == SPHERE, radius, pen_h))
        # One draw serves both: a pen's density is its mass draw, rescaled.
        mass_unit = unit()
        pen_volume = math.pi * radius**2 * (2.0 * half_length)
        lo, hi = cfg.pen_density
        pen_mass = (mass_unit * (hi - lo) + lo) * pen_volume
        lo, hi = cfg.mass_range
        mass = torch.where(shape == PEN, pen_mass, mass_unit * (hi - lo) + lo)

        inertia = solid_inertia(shape, mass, radius, half_length)
        inertia = torch.where((shape == PEN).unsqueeze(-1), inertia.clamp_min(cfg.min_inertia), inertia)

        self.shape[env_ids] = shape
        self.radius[env_ids] = radius
        self.half_length[env_ids] = half_length
        self.mass[env_ids] = mass
        self.inertia[env_ids] = inertia

    def scale(self) -> torch.Tensor:
        """Per-axis stretch from each world's reference shape to its current one, ``(N, 3)``."""
        ref = self._reference_size()
        r = self.radius / ref[self.shape, 0]
        h = self.half_length / ref[self.shape, 1]
        # Cylinder axis along z, pen along x; a sphere stretches evenly.
        return _by_shape(
            self.shape,
            torch.stack([r, r, h], dim=-1),
            r.unsqueeze(-1).expand(-1, 3),
            torch.stack([h, r, r], dim=-1),
        )

    def _reference_size(self) -> torch.Tensor:
        ref = getattr(self, "_ref", None)
        if ref is None:
            ref = torch.tensor([REFERENCE_SIZE[s] for s in SHAPES], device=self.shape.device)
            self._ref = ref
        return ref

    @torch.no_grad()
    def world_z_extent(self, quat_wxyz: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Highest point above the origin once rotated by ``quat`` (= depth of the lowest).

        Exact support functions of the solids, so the object rests on the table
        at any tilt instead of within a surface sample's error.
        """
        up = torch.tensor([0.0, 0.0, 1.0], device=quat_wxyz.device).expand(len(env_ids), 3)
        d = quat_apply_inverse(quat_wxyz, up)
        shape = self.shape[env_ids]
        r, h = self.radius[env_ids], self.half_length[env_ids]
        axis = torch.where(shape == PEN, d[:, 0], d[:, 2]).abs()
        reach = r * torch.sqrt((1.0 - axis**2).clamp_min(0.0)) + h * axis
        return torch.where(shape == SPHERE, r, reach)

    def _region_tables(self) -> tuple[torch.Tensor, ...]:
        """Every shape's regions stacked, indexed by shape id: points (3, M, 3),
        point_to_region (3, M), centres and normals (3, K, 3). One gather then
        serves all worlds, with no per-shape loop or host sync."""
        tables = getattr(self, "_tables", None)
        if tables is None:
            regions = [shape_regions(name) for name in SHAPES]
            dev = self.shape.device
            tables = tuple(
                torch.stack([getattr(r, field) for r in regions]).to(dev)
                for field in ("points", "point_to_region", "centers", "normals")
            )
            self._tables = tables
        return tables

    @torch.no_grad()
    def nearest_region_ids(self, points_w: torch.Tensor, pos_w: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Region id under each world point, ``(N, P)``, against each world's own shape and size."""
        points, point_to_region, _, _ = self._region_tables()
        num_points = points_w.shape[1]
        local = quat_apply_inverse(
            quat_wxyz.unsqueeze(1).expand(-1, num_points, -1), points_w - pos_w.unsqueeze(1)
        )
        # Back to the reference size, where the regions were clustered.
        local = local / self.scale().unsqueeze(1)
        nearest = torch.cdist(local, points[self.shape]).argmin(-1)  # (N, P)
        return torch.gather(point_to_region[self.shape], 1, nearest)

    @torch.no_grad()
    def region_centers_normals(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Centre and outward normal of each region in ``ids``, ``(N, P, 3)`` each, at each world's size.

        Normals go through the stretch as covectors (divided by it, then
        renormalised), which keeps a stretched cap's normal along the axis and a
        stretched side's normal radial.
        """
        _, _, centers, normals = self._region_tables()
        s = self.scale().unsqueeze(1)
        shape = self.shape.unsqueeze(-1).expand_as(ids)
        c = centers[shape, ids] * s
        n = normals[shape, ids] / s
        return c, n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def shape_params(self) -> torch.Tensor:
        """Observation: one-hot shape and its radius and half-length in decimetres, ``(N, 5)``."""
        onehot = torch.nn.functional.one_hot(self.shape, len(SHAPES)).float()
        return torch.cat([onehot, 10.0 * self.radius[:, None], 10.0 * self.half_length[:, None]], dim=-1)

    def mass_obs(self) -> torch.Tensor:
        """Critic observation: mass in units of 100g, ``(N, 1)``."""
        return 10.0 * self.mass[:, None]


_MASK32 = 0xFFFFFFFF


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """Avalanche a 32-bit value held in int64. Multipliers stay below 2^31 so no product overflows."""
    x = x & _MASK32
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & _MASK32
    x = ((x ^ (x >> 15)) * 0x6B7A3A8B) & _MASK32
    return x ^ (x >> 16)


def _hash_uniform(seed: int, env_ids: torch.Tensor, episode: torch.Tensor, stream: int) -> torch.Tensor:
    """Uniform [0, 1) per world, determined by ``(seed, world, episode, stream)`` alone."""
    x = _mix32(torch.full_like(env_ids, seed) ^ (stream * 0x9E3779B))
    x = _mix32(x ^ env_ids)
    x = _mix32(x ^ episode)
    return (x >> 8).float() / float(1 << 24)  # Top 24 bits: exact in float32.
