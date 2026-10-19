"""Regular voxel grid. Z = 0 at surface, positive downward. Values live at voxel centres."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LITHOLOGY = ("Marble", "Schist", "Breccia", "Altered host", "Background")
LITH_MARBLE, LITH_SCHIST, LITH_BRECCIA, LITH_ALTERED, LITH_BACKGROUND = range(5)


@dataclass(frozen=True)
class Grid:
    extent: tuple[float, float, float]
    shape: tuple[int, int, int]

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(e / n for e, n in zip(self.extent, self.shape))  # type: ignore[return-value]

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    def axis(self, i: int) -> np.ndarray:
        d = self.spacing[i]
        return (np.arange(self.shape[i]) + 0.5) * d

    @property
    def x(self) -> np.ndarray:
        return self.axis(0)

    @property
    def y(self) -> np.ndarray:
        return self.axis(1)

    @property
    def z(self) -> np.ndarray:
        return self.axis(2)

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, self.z, indexing="ij")

    def points(self) -> np.ndarray:
        """All voxel centres as (N, 3) float32 in C order of (nx, ny, nz)."""
        X, Y, Z = self.mesh()
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)

    def to_index(self, pts: np.ndarray) -> np.ndarray:
        """Continuous voxel index coordinates for interpolation (map_coordinates convention)."""
        sp = np.asarray(self.spacing)
        return (np.asarray(pts, dtype=np.float64) / sp - 0.5).T

    def contains(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        pts = np.atleast_2d(pts)
        ext = np.asarray(self.extent)
        return np.all((pts >= margin) & (pts <= ext - margin), axis=1)

    def deep_mask(self, depth: float = 100.0) -> np.ndarray:
        """Boolean (nx, ny, nz) mask of voxels whose centre is below ``depth``."""
        return np.broadcast_to(self.z > depth, self.shape)
