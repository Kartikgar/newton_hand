"""Farthest-point-sampling + k-means clustering of a surface sample into regions.

Copied from ``mjlab_hand.object.object`` so the region layouts, and hence
phi's contact features, are built the same way in both repos.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RegionClusterResult:
    point_to_region: torch.Tensor  # (M,) long, values in [0, K-1]
    region_centers: torch.Tensor  # (K, 3) float32, object frame


def _fps_points(points: torch.Tensor, k: int, start_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    n = points.shape[0]
    k = min(k, n)
    dists = torch.full((n,), float("inf"), device=points.device, dtype=points.dtype)
    idx = torch.empty((k,), dtype=torch.long, device=points.device)
    farthest = int(start_idx) % n
    for i in range(k):
        idx[i] = farthest
        dist2 = ((points - points[farthest : farthest + 1]) ** 2).sum(dim=-1)
        dists = torch.minimum(dists, dist2)
        farthest = int(torch.argmax(dists).item())
    return idx, points[idx]


def cluster_points_fps_kmeans(
    points: torch.Tensor,
    normals: torch.Tensor | None,
    num_regions: int,
    max_iters: int = 10,
    use_normal_in_clustering: bool = True,
    normal_weight: float = 0.5,
) -> RegionClusterResult:
    m, _ = points.shape
    k = min(int(num_regions), m)
    pts = points.to(dtype=torch.float32)
    nrms = normals.to(dtype=torch.float32) if normals is not None else None
    use_normals = nrms is not None and use_normal_in_clustering

    fps_idx, centers = _fps_points(pts, k, start_idx=0)
    center_normals = nrms[fps_idx].clone() if use_normals else None

    for _ in range(max_iters):
        if use_normals:
            pos_dist = torch.cdist(pts, centers)
            normal_dist = 1.0 - torch.mm(nrms, center_normals.T).clamp(-1.0, 1.0)
            labels = ((1.0 - normal_weight) * pos_dist + normal_weight * normal_dist).argmin(dim=1)
        else:
            labels = torch.cdist(pts, centers).argmin(dim=1)

        new_centers = torch.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            new_centers[i] = pts[mask].mean(dim=0) if mask.any() else centers[i]
        centers = new_centers

        if use_normals:
            new_cn = torch.zeros(k, 3, device=pts.device, dtype=pts.dtype)
            for i in range(k):
                mask = labels == i
                if mask.any():
                    avg = nrms[mask].mean(dim=0)
                    new_cn[i] = avg / avg.norm().clamp_min(1e-8)
                else:
                    new_cn[i] = center_normals[i]
            center_normals = new_cn

    return RegionClusterResult(point_to_region=labels.to(torch.long), region_centers=centers)
