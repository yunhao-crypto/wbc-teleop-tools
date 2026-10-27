"""Shared MuJoCo geom helpers used by retargeting.

Vendored verbatim from GMR (general_motion_retargeting/_geom_utils.py) so the
bridge's `pico` extra no longer needs the full GMR package. Only the two helpers
the retargeter's ground-snap actually calls are kept.
"""

from __future__ import annotations

import mujoco as mj
import numpy as np


def geom_min_z(model, data, geom_id: int) -> float:
    """Lowest z extent of a geom in world coordinates.

    MuJoCo stores box sizes as three local-frame half extents.  Taking their
    maximum (the old fallback) is only correct for a sphere; for the humanoid
    foot boxes it confused the long toe direction with vertical thickness and
    understated the sole height by almost 6 cm.  Compute the world-Z support
    extent for each primitive explicitly.
    """
    geom_type = model.geom_type[geom_id]
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    if geom_type == mj.mjtGeom.mjGEOM_BOX:
        z_extent = float(np.dot(np.abs(xmat[2, :]), size[:3]))
        return float(xpos[2] - z_extent)
    if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
        return float(xpos[2] - size[0])
    if geom_type == mj.mjtGeom.mjGEOM_CAPSULE:
        axis = xmat[:, 2]
        z_extent = float(size[1] * abs(axis[2]) + size[0])
        return float(xpos[2] - z_extent)
    if geom_type == mj.mjtGeom.mjGEOM_CYLINDER:
        radius = float(size[0])
        half_height = float(size[1])
        axis = xmat[:, 2]
        z_extent = (
            half_height * abs(axis[2])
            + radius * np.sqrt(max(0.0, 1.0 - axis[2] * axis[2]))
        )
        return float(xpos[2] - z_extent)
    if geom_type == mj.mjtGeom.mjGEOM_ELLIPSOID:
        z_extent = float(np.sqrt(np.sum((xmat[2, :] * size[:3]) ** 2)))
        return float(xpos[2] - z_extent)
    if geom_type == mj.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        first = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        verts = model.mesh_vert[first:first + count]
        return float(xpos[2] + np.min(verts @ xmat[2, :]))
    # Ground snap is configured only for the primitives above in the shipped
    # models.  Keep a conservative legacy fallback for unusual custom geoms
    # rather than taking down the live retarget loop.
    return float(xpos[2] - np.max(size))


def collect_geom_ids_by_token(model, name_token: str) -> list[int]:
    """Return all geom IDs whose name contains name_token."""
    geom_ids = []
    for geom_id in range(model.ngeom):
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is not None and name_token in geom_name:
            geom_ids.append(geom_id)
    return geom_ids
