"""Mesh loading and repair (STL, OBJ, 3MF via trimesh)."""
from __future__ import annotations

import logging
from pathlib import Path

import trimesh

log = logging.getLogger(__name__)


class MeshError(RuntimeError):
    pass


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load, merge scene geometry, process, and fail loudly if not watertight."""
    path = Path(path)
    if not path.exists():
        raise MeshError(f"mesh not found: {path}")
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):  # pragma: no cover - force="mesh" normally merges
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise MeshError(f"no triangle mesh in {path}")
    mesh.process(validate=True)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        raise MeshError(f"{path.name} is not watertight ({len(mesh.faces)} faces); repair it first")
    if mesh.volume < 0:
        mesh.invert()
    log.info("loaded %s: %d faces, bounds %s", path.name, len(mesh.faces), mesh.bounds.tolist())
    return mesh
