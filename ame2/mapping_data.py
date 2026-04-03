"""
Warp-raytraced terrain data pipeline for MappingNet training (Paper Sec. V-B).

Generates 4 terrain mesh types, raycasts from random poses using NVIDIA Warp,
and applies 5 augmentations to produce (gt, noisy) elevation grid pairs.

Components:
    TerrainMeshGenerator  - Standalone mesh generation (no Isaac Lab)
    WarpRaycastSampler    - GPU raytracing via wp.mesh_query_ray
    MappingAugmentor      - 5 paper augmentations (Sec. V-B.2)
    WarpMappingDataset    - Top-level interface returning (gt, noisy) batches
"""

from __future__ import annotations

import math
import numpy as np
import torch

try:
    import warp as wp
except ImportError:
    wp = None


# ------------------------------------------------------------------ #
# 1. TerrainMeshGenerator
# ------------------------------------------------------------------ #

class TerrainMeshGenerator:
    """
    Generates (vertices, triangles) numpy arrays for 4 terrain types.

    Type 1: Locomotion training terrains (12 sub-types matching terrains.py)
    Type 2: Randomly stacked boxes
    Type 3: Random heightfields (smoothed noise)
    Type 4: Random floating boxes (overhangs / occlusion geometry)
    """

    # Paper Appendix A proportions for the 12 locomotion sub-types
    LOCO_SUBTYPES = [
        ("rough", 0.05),
        ("stair_down", 0.05),
        ("stair_up", 0.05),
        ("boxes", 0.05),
        ("obstacles", 0.05),
        ("climbing_up", 0.20),
        ("climbing_down", 0.05),
        ("climbing_consecutive", 0.05),
        ("gap", 0.05),
        ("pallets", 0.05),
        ("stones", 0.30),
        ("beam", 0.05),
    ]

    def __init__(
        self,
        patch_size: float = 8.0,
        resolution: float = 0.04,
        terrain_type_weights: tuple[float, ...] = (0.40, 0.20, 0.20, 0.20),
    ):
        self.patch_size = patch_size
        self.resolution = resolution
        self.terrain_type_weights = np.array(terrain_type_weights, dtype=np.float64)
        self.terrain_type_weights /= self.terrain_type_weights.sum()
        n = int(patch_size / resolution)
        self.grid_n = n
        # Pre-compute sub-type cumulative distribution
        names, probs = zip(*self.LOCO_SUBTYPES)
        self._loco_names = names
        self._loco_cdf = np.cumsum(probs)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (vertices [V,3], triangles [F,3]) for a random terrain."""
        t = np.random.choice(4, p=self.terrain_type_weights)
        if t == 0:
            return self._locomotion_terrain()
        elif t == 1:
            return self._stacked_boxes()
        elif t == 2:
            return self._random_heightfield()
        else:
            return self._floating_boxes()

    # -- Type 1: Locomotion training terrains --------------------------

    def _locomotion_terrain(self) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(self._loco_cdf, np.random.rand())
        idx = min(idx, len(self._loco_names) - 1)
        name = self._loco_names[idx]
        difficulty = np.random.uniform(0.0, 1.0)
        hf = self._loco_heightfield(name, difficulty)
        return self._heightfield_to_mesh(hf, self.resolution)

    def _loco_heightfield(self, name: str, d: float) -> np.ndarray:
        n = self.grid_n
        hf = np.zeros((n, n), dtype=np.float32)

        if name == "rough":
            amp = 0.20 * d
            hf = np.random.uniform(-amp, amp, (n, n)).astype(np.float32)

        elif name == "stair_up":
            step_h = 0.05 + 0.35 * d
            num_steps = max(2, int(n / 8))
            for i in range(num_steps):
                r0 = int(i * n / num_steps)
                r1 = int((i + 1) * n / num_steps)
                hf[r0:r1, :] = i * step_h

        elif name == "stair_down":
            step_h = 0.05 + 0.35 * d
            num_steps = max(2, int(n / 8))
            for i in range(num_steps):
                r0 = int(i * n / num_steps)
                r1 = int((i + 1) * n / num_steps)
                hf[r0:r1, :] = (num_steps - 1 - i) * step_h

        elif name == "boxes":
            box_h_max = 0.05 + 0.35 * d
            grid_w = max(3, int(0.45 / self.resolution))
            for r in range(0, n, grid_w):
                for c in range(0, n, grid_w):
                    h = np.random.uniform(0.0, box_h_max)
                    hf[r:min(r + grid_w, n), c:min(c + grid_w, n)] = h

        elif name == "obstacles":
            num_obs = 40
            for _ in range(num_obs):
                w = np.random.randint(3, max(4, int(0.75 / self.resolution)))
                h = np.random.randint(3, max(4, int(0.75 / self.resolution)))
                r0 = np.random.randint(0, max(1, n - h))
                c0 = np.random.randint(0, max(1, n - w))
                oh = np.random.uniform(0.05, 0.20) * d
                hf[r0:r0 + h, c0:c0 + w] = max(hf[r0, c0], oh)

        elif name == "climbing_up":
            depth = 0.10 + 0.90 * d
            border = n // 4
            hf[:, :] = 0.0
            hf[border:n - border, border:n - border] = -depth

        elif name == "climbing_down":
            height = 0.20 + 0.80 * d
            border = n // 4
            hf[border:n - border, border:n - border] = height

        elif name == "climbing_consecutive":
            depth = 0.05 + 0.45 * d
            b1, b2 = n // 5, 2 * n // 5
            hf[b1:n - b1, b1:n - b1] = -depth
            hf[b2:n - b2, b2:n - b2] = -depth * 1.5

        elif name == "gap":
            gap_w = 0.10 + 1.00 * d
            gap_cells = max(1, int(gap_w / self.resolution))
            mid = n // 2
            half = gap_cells // 2
            hf[:, :] = 0.3
            hf[:, max(0, mid - half):min(n, mid + half)] = -1.0

        elif name == "pallets":
            rail_h = 0.05 + 0.25 * d
            rail_w = max(2, int(0.16 / self.resolution))
            spacing = max(rail_w + 2, int(0.5 / self.resolution))
            for c in range(0, n, spacing):
                hf[:, c:min(c + rail_w, n)] = rail_h

        elif name == "stones":
            stone_max_h = 0.25 * d
            hf[:, :] = -1.5  # holes
            # Place random stepping stones
            num_stones = max(20, int(n * n * 0.15))
            for _ in range(num_stones):
                sw = np.random.randint(
                    max(1, int(0.15 / self.resolution)),
                    max(2, int(0.60 / self.resolution)),
                )
                sh = np.random.randint(
                    max(1, int(0.15 / self.resolution)),
                    max(2, int(0.60 / self.resolution)),
                )
                r0 = np.random.randint(0, max(1, n - sh))
                c0 = np.random.randint(0, max(1, n - sw))
                h = np.random.uniform(0, stone_max_h)
                hf[r0:r0 + sh, c0:c0 + sw] = h

        elif name == "beam":
            bar_h = 0.05 + 0.15 * d
            bar_w = max(2, int(0.18 / self.resolution))
            cx, cy = n // 2, n // 2
            num_bars = 4
            for k in range(num_bars):
                angle = k * math.pi / num_bars
                dx = math.cos(angle)
                dy = math.sin(angle)
                for t in range(-n, n):
                    r = int(cy + t * dy)
                    c = int(cx + t * dx)
                    if 0 <= r < n and 0 <= c < n:
                        for bw in range(-bar_w // 2, bar_w // 2 + 1):
                            rr = int(r - bw * dx)
                            cc = int(c + bw * dy)
                            if 0 <= rr < n and 0 <= cc < n:
                                hf[rr, cc] = bar_h

        return hf

    # -- Type 2: Randomly stacked boxes --------------------------------

    def _stacked_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        num_boxes = np.random.randint(5, 25)
        all_verts = []
        all_tris = []
        vert_offset = 0
        for _ in range(num_boxes):
            sx = np.random.uniform(0.2, 1.5)
            sy = np.random.uniform(0.2, 1.5)
            sz = np.random.uniform(0.1, 0.8)
            cx = np.random.uniform(0, self.patch_size)
            cy = np.random.uniform(0, self.patch_size)
            cz = np.random.uniform(0, sz)  # sits on or near ground
            v, t = self._box_mesh(cx, cy, cz, sx, sy, sz)
            all_verts.append(v)
            all_tris.append(t + vert_offset)
            vert_offset += len(v)
        # Add ground plane
        gv, gt_ = self._ground_plane()
        all_verts.append(gv)
        all_tris.append(gt_ + vert_offset)
        return np.concatenate(all_verts, axis=0), np.concatenate(all_tris, axis=0)

    # -- Type 3: Random heightfield ------------------------------------

    def _random_heightfield(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.grid_n
        hf = np.random.randn(n, n).astype(np.float32) * 0.3
        # Smooth with box filter (2 passes)
        k = 5
        from scipy.ndimage import uniform_filter
        hf = uniform_filter(hf, size=k)
        hf = uniform_filter(hf, size=k)
        return self._heightfield_to_mesh(hf, self.resolution)

    # -- Type 4: Floating boxes (overhangs) ----------------------------

    def _floating_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        num_boxes = np.random.randint(3, 15)
        all_verts = []
        all_tris = []
        vert_offset = 0
        for _ in range(num_boxes):
            sx = np.random.uniform(0.3, 2.0)
            sy = np.random.uniform(0.3, 2.0)
            sz = np.random.uniform(0.05, 0.3)
            cx = np.random.uniform(0, self.patch_size)
            cy = np.random.uniform(0, self.patch_size)
            cz = np.random.uniform(0.3, 2.0)  # floating above ground
            v, t = self._box_mesh(cx, cy, cz, sx, sy, sz)
            all_verts.append(v)
            all_tris.append(t + vert_offset)
            vert_offset += len(v)
        # Ground plane with some roughness
        n = self.grid_n
        hf = np.random.randn(n, n).astype(np.float32) * 0.05
        gv, gt_ = self._heightfield_to_mesh(hf, self.resolution)
        all_verts.append(gv)
        all_tris.append(gt_ + vert_offset)
        return np.concatenate(all_verts, axis=0), np.concatenate(all_tris, axis=0)

    # -- Helpers -------------------------------------------------------

    @staticmethod
    def _heightfield_to_mesh(
        heights: np.ndarray, resolution: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert 2D height array to triangle mesh (standard grid triangulation)."""
        rows, cols = heights.shape
        # Vertices
        rs = np.arange(rows, dtype=np.float32) * resolution
        cs = np.arange(cols, dtype=np.float32) * resolution
        cc, rr = np.meshgrid(cs, rs)
        verts = np.stack([cc.ravel(), rr.ravel(), heights.ravel()], axis=-1)
        # Triangles: 2 per cell
        tris = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                i00 = r * cols + c
                i01 = r * cols + (c + 1)
                i10 = (r + 1) * cols + c
                i11 = (r + 1) * cols + (c + 1)
                tris.append([i00, i10, i01])
                tris.append([i01, i10, i11])
        return verts.astype(np.float32), np.array(tris, dtype=np.int32)

    @staticmethod
    def _box_mesh(
        cx: float, cy: float, cz: float,
        sx: float, sy: float, sz: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned box → 8 vertices, 12 triangles."""
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        verts = np.array([
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ], dtype=np.float32)
        tris = np.array([
            [0, 2, 1], [0, 3, 2],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 4, 7], [0, 7, 3],  # left
            [1, 2, 6], [1, 6, 5],  # right
        ], dtype=np.int32)
        return verts, tris

    def _ground_plane(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.patch_size
        verts = np.array([
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
        ], dtype=np.float32)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        return verts, tris


# ------------------------------------------------------------------ #
# 2. WarpRaycastSampler
# ------------------------------------------------------------------ #

def _check_warp():
    if wp is None:
        raise ImportError(
            "warp-lang is required for WarpRaycastSampler. "
            "Install with: pip install 'ame2[warp]'"
        )


# Warp kernel: one thread per ray
if wp is not None:
    @wp.kernel
    def _raycast_kernel(
        mesh_id: wp.uint64,
        ray_origins: wp.array(dtype=wp.vec3),
        ray_dirs: wp.array(dtype=wp.vec3),
        max_t: float,
        hits: wp.array(dtype=float),
    ):
        tid = wp.tid()
        origin = ray_origins[tid]
        direction = ray_dirs[tid]
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3(0.0, 0.0, 0.0)
        face = int(0)
        if wp.mesh_query_ray(mesh_id, origin, direction, max_t, t, u, v, sign, n, face):
            hit_point = origin + direction * t
            hits[tid] = hit_point[2] - origin[2]
        else:
            hits[tid] = 0.0


class WarpRaycastSampler:
    """
    GPU raytracing using NVIDIA Warp to sample ground-truth elevation grids.

    For each sample in a batch:
    - Random pose (x, y, yaw) is drawn over the mesh extent
    - A grid of downward rays is cast from the sensor frame
    - Hit distances are recorded as elevation relative to sensor height
    """

    def __init__(
        self,
        grid_h: int = 31,
        grid_w: int = 51,
        res: float = 0.04,
        cx: float = 1.0,
        cy: float = 0.0,
        sensor_height: float = 0.5,
        max_t: float = 5.0,
        device: str = "cuda:0",
    ):
        _check_warp()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.res = res
        self.cx = cx
        self.cy = cy
        self.sensor_height = sensor_height
        self.max_t = max_t
        self.device = device
        self._mesh = None
        self._mesh_id = None

        # Pre-compute local grid offsets (relative to sensor center)
        js = np.arange(grid_w, dtype=np.float32) - grid_w // 2
        is_ = np.arange(grid_h, dtype=np.float32) - grid_h // 2
        jj, ii = np.meshgrid(js, is_)
        # Local offsets: x forward, y left
        self._local_dx = (jj.ravel() * res + cx).astype(np.float32)
        self._local_dy = (ii.ravel() * res + cy).astype(np.float32)
        self._num_rays = grid_h * grid_w

    def set_mesh(self, vertices: np.ndarray, triangles: np.ndarray):
        """Build Warp mesh + BVH from numpy arrays."""
        _check_warp()
        wp_verts = wp.array(vertices, dtype=wp.vec3, device=self.device)
        wp_tris = wp.array(triangles.ravel(), dtype=wp.int32, device=self.device)
        self._mesh = wp.Mesh(points=wp_verts, indices=wp_tris)
        self._mesh_id = self._mesh.id

    def sample(
        self,
        B: int,
        mesh_extent: tuple[float, float, float, float] | None = None,
    ) -> torch.Tensor:
        """
        Cast rays for B random poses, return (B, 1, grid_h, grid_w) elevation grid.

        mesh_extent: (x_min, x_max, y_min, y_max) for pose sampling.
        """
        if self._mesh is None:
            raise RuntimeError("Call set_mesh() before sample()")

        if mesh_extent is None:
            mesh_extent = (1.0, 7.0, 1.0, 7.0)

        x_min, x_max, y_min, y_max = mesh_extent

        # Random poses
        px = np.random.uniform(x_min, x_max, B).astype(np.float32)
        py = np.random.uniform(y_min, y_max, B).astype(np.float32)
        yaw = np.random.uniform(-math.pi, math.pi, B).astype(np.float32)

        # Build ray origins: for each sample, rotate local grid by yaw and translate
        total_rays = B * self._num_rays
        origins = np.empty((total_rays, 3), dtype=np.float32)
        dirs = np.zeros((total_rays, 3), dtype=np.float32)
        dirs[:, 2] = -1.0  # downward rays

        for b in range(B):
            cos_y = math.cos(yaw[b])
            sin_y = math.sin(yaw[b])
            offset = b * self._num_rays
            # Rotate local offsets
            world_dx = self._local_dx * cos_y - self._local_dy * sin_y
            world_dy = self._local_dx * sin_y + self._local_dy * cos_y
            origins[offset:offset + self._num_rays, 0] = px[b] + world_dx
            origins[offset:offset + self._num_rays, 1] = py[b] + world_dy
            origins[offset:offset + self._num_rays, 2] = self.sensor_height

        # Launch Warp kernel
        wp_origins = wp.array(origins, dtype=wp.vec3, device=self.device)
        wp_dirs = wp.array(dirs, dtype=wp.vec3, device=self.device)
        wp_hits = wp.zeros(total_rays, dtype=float, device=self.device)

        wp.launch(
            kernel=_raycast_kernel,
            dim=total_rays,
            inputs=[self._mesh_id, wp_origins, wp_dirs, self.max_t, wp_hits],
            device=self.device,
        )
        wp.synchronize_device(self.device)

        # Convert to torch
        hits_np = wp_hits.numpy().reshape(B, 1, self.grid_h, self.grid_w)
        return torch.from_numpy(hits_np).to(self.device.replace("cuda:", "cuda:"))


# ------------------------------------------------------------------ #
# 3. MappingAugmentor
# ------------------------------------------------------------------ #

class MappingAugmentor:
    """
    5 paper augmentations (Sec. V-B.2) applied sequentially to GT elevation grids.

    1. Uniform noise with random per-sample magnitude
    2. Random border cropping (zero out border rows/cols)
    3. Simulated occlusions (angular FOV mask + height occlusion)
    4. Random elevation clipping
    5. Random missing cells + outliers
    """

    def __init__(
        self,
        noise_max: float = 0.05,
        crop_max_frac: float = 0.10,
        clip_range: tuple[float, float] = (-2.0, 2.0),
        dropout_max: float = 0.10,
        outlier_max: float = 0.03,
    ):
        self.noise_max = noise_max
        self.crop_max_frac = crop_max_frac
        self.clip_range = clip_range
        self.dropout_max = dropout_max
        self.outlier_max = outlier_max

    def __call__(self, gt: torch.Tensor) -> torch.Tensor:
        """Apply augmentations. Input/output: (B, 1, H, W).

        Occlusion is applied per-sample with 30% probability to avoid
        destroying too much of the input signal.
        """
        x = gt.clone()
        x = self._uniform_noise(x)
        x = self._border_crop(x)
        x = self._simulated_occlusion(x, prob=0.3)
        x = self._missing_and_outliers(x)
        return x

    def _uniform_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Aug 1: Per-sample random magnitude uniform noise."""
        B = x.shape[0]
        mag = torch.rand(B, 1, 1, 1, device=x.device) * self.noise_max
        noise = (torch.rand_like(x) * 2.0 - 1.0) * mag
        return x + noise

    def _border_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Aug 2: Zero out random rows/cols from each border (vectorized)."""
        B, _, H, W = x.shape
        max_h = max(1, int(H * self.crop_max_frac))
        max_w = max(1, int(W * self.crop_max_frac))
        # Random crop sizes per sample
        top = torch.from_numpy(np.random.randint(0, max_h + 1, B)).to(x.device)
        bot = torch.from_numpy(np.random.randint(0, max_h + 1, B)).to(x.device)
        left = torch.from_numpy(np.random.randint(0, max_w + 1, B)).to(x.device)
        right = torch.from_numpy(np.random.randint(0, max_w + 1, B)).to(x.device)
        # Build mask: (B, 1, H, W)
        rows = torch.arange(H, device=x.device).view(1, 1, H, 1)
        cols = torch.arange(W, device=x.device).view(1, 1, 1, W)
        mask = (
            (rows < top.view(B, 1, 1, 1)) |
            (rows >= (H - bot).view(B, 1, 1, 1)) |
            (cols < left.view(B, 1, 1, 1)) |
            (cols >= (W - right).view(B, 1, 1, 1))
        )
        x[mask] = 0.0
        return x

    def _simulated_occlusion(self, x: torch.Tensor, prob: float = 1.0) -> torch.Tensor:
        """Aug 3: Random viewpoint → angular FOV mask (vectorized, no per-sample loop)."""
        B, _, H, W = x.shape
        # Decide which samples get occlusion
        apply = torch.from_numpy(np.random.rand(B) < prob).to(x.device)
        if not apply.any():
            return x

        js = torch.arange(W, device=x.device, dtype=torch.float32) - W // 2
        is_ = torch.arange(H, device=x.device, dtype=torch.float32) - H // 2
        grid_j, grid_i = torch.meshgrid(js, is_, indexing="xy")  # (H, W)
        cell_angles = torch.atan2(grid_i, grid_j)  # (H, W)

        # Vectorized: random view angles and FOVs for all samples
        view_angles = torch.from_numpy(
            np.random.uniform(-math.pi, math.pi, B).astype(np.float32)
        ).to(x.device).view(B, 1, 1)
        fovs = torch.from_numpy(
            np.random.uniform(math.pi * 0.6, math.pi * 1.2, B).astype(np.float32)
        ).to(x.device).view(B, 1, 1)

        # (B, H, W) angle diff
        angle_diff = cell_angles.unsqueeze(0) - view_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        outside_fov = angle_diff.abs() > (fovs / 2)  # (B, H, W)

        # Only apply to selected samples
        mask = outside_fov & apply.view(B, 1, 1)
        x[:, 0][mask] = 0.0
        return x

    def _elevation_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Aug 4: Per-sample random clip range, zero cells outside."""
        B = x.shape[0]
        for b in range(B):
            lo = np.random.uniform(self.clip_range[0], 0.0)
            hi = np.random.uniform(0.0, self.clip_range[1])
            mask = (x[b] < lo) | (x[b] > hi)
            x[b][mask] = 0.0
        return x

    def _missing_and_outliers(self, x: torch.Tensor) -> torch.Tensor:
        """Aug 5: Random dropout + random outlier values (vectorized)."""
        B = x.shape[0]
        # Per-sample random rates broadcast over spatial dims
        drop_rates = torch.from_numpy(
            np.random.uniform(0, self.dropout_max, B).astype(np.float32)
        ).to(x.device).view(B, 1, 1, 1)
        outlier_rates = torch.from_numpy(
            np.random.uniform(0, self.outlier_max, B).astype(np.float32)
        ).to(x.device).view(B, 1, 1, 1)
        # Dropout
        drop_mask = torch.rand_like(x) < drop_rates
        x[drop_mask] = 0.0
        # Outliers
        outlier_mask = torch.rand_like(x) < outlier_rates
        x[outlier_mask] = (torch.rand_like(x)[outlier_mask] * 2.0 - 1.0)
        return x


# ------------------------------------------------------------------ #
# 4. WarpMappingDataset
# ------------------------------------------------------------------ #

class WarpMappingDataset:
    """
    Top-level interface for Warp-raytraced MappingNet training data.

    Pre-generates meshes (4 per terrain type), raycasts from random poses,
    applies augmentations, returns (gt, noisy) pairs matching MappingNet I/O.
    """

    def __init__(
        self,
        device: str = "cuda",
        num_meshes_cached: int = 16,
        regen_interval: int = 500,
        grid_h: int = 31,
        grid_w: int = 51,
        res: float = 0.04,
    ):
        _check_warp()
        wp.init()

        self.device = device
        self.num_meshes_cached = num_meshes_cached
        self.regen_interval = regen_interval
        self._batch_count = 0

        self._terrain_gen = TerrainMeshGenerator()
        self._raycaster = WarpRaycastSampler(
            grid_h=grid_h, grid_w=grid_w, res=res, device=device,
        )
        self._augmentor = MappingAugmentor()

        # Cache: list of (vertices, triangles, mesh_extent)
        self._mesh_cache: list[tuple[np.ndarray, np.ndarray, tuple]] = []
        self._regenerate_cache()

    def _regenerate_cache(self):
        """Generate num_meshes_cached meshes (balanced across 4 types)."""
        self._mesh_cache.clear()
        per_type = max(1, self.num_meshes_cached // 4)
        gen = self._terrain_gen
        for type_idx in range(4):
            # Temporarily force terrain type
            orig_weights = gen.terrain_type_weights.copy()
            w = np.zeros(4)
            w[type_idx] = 1.0
            gen.terrain_type_weights = w
            for _ in range(per_type):
                verts, tris = gen.generate()
                x_min = float(verts[:, 0].min())
                x_max = float(verts[:, 0].max())
                y_min = float(verts[:, 1].min())
                y_max = float(verts[:, 1].max())
                border = 1.0  # stay away from edges
                extent = (
                    x_min + border,
                    max(x_min + border + 0.1, x_max - border),
                    y_min + border,
                    max(y_min + border + 0.1, y_max - border),
                )
                self._mesh_cache.append((verts, tris, extent))
            gen.terrain_type_weights = orig_weights

    def sample_batch(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (gt, noisy) both shaped (B, 1, 31, 51).

        gt: ground-truth elevation from raycasting
        noisy: augmented version simulating real sensor data
        """
        self._batch_count += 1
        if self._batch_count % self.regen_interval == 0:
            self._regenerate_cache()

        # Pick a random cached mesh
        idx = np.random.randint(len(self._mesh_cache))
        verts, tris, extent = self._mesh_cache[idx]
        self._raycaster.set_mesh(verts, tris)

        gt = self._raycaster.sample(B, mesh_extent=extent)
        # Move to torch device
        torch_device = torch.device(self.device)
        gt = gt.to(torch_device)
        noisy = self._augmentor(gt)
        return gt, noisy
