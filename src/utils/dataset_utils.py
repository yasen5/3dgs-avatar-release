from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from plyfile import PlyData, PlyElement

from src.utils.graphics_utils import BasicPointCloud


def fetchPly(path: str) -> BasicPointCloud:
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path: str, xyz: npt.NDArray[np.floating[Any]], rgb: npt.NDArray[np.floating[Any]]) -> None:
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class AABB(torch.nn.Module):
    coord_max: torch.Tensor
    coord_min: torch.Tensor

    def __init__(self, coord_max: npt.NDArray[np.floating[Any]], coord_min: npt.NDArray[np.floating[Any]]) -> None:
        super().__init__()
        self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        self.register_buffer("coord_min", torch.from_numpy(coord_min).float())

    def normalize(self, x: torch.Tensor, sym: bool = False) -> torch.Tensor:
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)
        if sym:
            x = 2 * x - 1.
        return x

    def unnormalize(self, x: torch.Tensor, sym: bool = False) -> torch.Tensor:
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x: torch.Tensor) -> torch.Tensor:
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self) -> torch.Tensor:
        return self.coord_max - self.coord_min

    def scale(self) -> float:
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)
