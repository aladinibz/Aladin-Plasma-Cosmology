"""
ALADIN Plasma Drive v2.1
Cylindrical Residual Operator  (v3 — shape-safe)

Differences HLLD face fluxes onto cells.
No cp.where with mixed shapes. Velocity cap uses boolean indexing.
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

VERSION = "v4-flux-scale"


def residual_cylindrical(state, geometry, config, fluxes, dt, rho_floor=1e-6):
    g = config.ghost_width
    sl = state.grid.interior_slice("cell")
    vol = geometry.cell_volumes
    xp = cp if HAS_CUPY else np

    def _clean(arr):
        bad = ~xp.isfinite(arr)
        if xp.any(bad):
            arr = arr.copy()
            arr[bad] = 0.0
        return arr

    def div_r(flux_face):
        f = _clean(flux_face[g:-g, g:-g, g:-g])
        A = geometry.area_r
        # both should be (nr+1, nphi, nz); slice to cells
        return (f[1:, :, :] * A[1:, :, :] - f[:-1, :, :] * A[:-1, :, :]) / vol

    def div_phi(flux_face):
        f = _clean(flux_face[g:-g, g:-g, g:-g])
        A = geometry.area_phi
        if A.shape[1] == f.shape[1]:
            return (f[:, 1:, :] * A[:, 1:, :] - f[:, :-1, :] * A[:, :-1, :]) / vol
        else:
            d = f[:, 1:, :] - f[:, :-1, :]
            return d * A[:, :d.shape[1], :] / vol

    def div_z(flux_face):
        f = _clean(flux_face[g:-g, g:-g, g:-g])
        A = geometry.area_z
        if A.shape[2] == f.shape[2]:
            return (f[:, :, 1:] * A[:, :, 1:] - f[:, :, :-1] * A[:, :, :-1]) / vol
        else:
            d = f[:, :, 1:] - f[:, :, :-1]
            return d * A[:, :, :d.shape[2]] / vol

    def update(field_name, flux_key, max_frac=0.5):
        if field_name not in state.fields:
            return
        d = 0.0
        if "r" in fluxes and flux_key in fluxes["r"]:
            d = d + div_r(fluxes["r"][flux_key])
        if "phi" in fluxes and flux_key in fluxes["phi"]:
            d = d + div_phi(fluxes["phi"][flux_key])
        if "z" in fluxes and flux_key in fluxes["z"]:
            d = d + div_z(fluxes["z"][flux_key])
        # pre-update scale: limit |dt * d| to max_frac * |state|
        arr = state.get(field_name)
        proposed = dt * d
        scale = xp.ones_like(proposed)
        ref = xp.maximum(xp.abs(arr[sl]), 1e-12)
        too_big = xp.abs(proposed) > max_frac * ref
        if xp.any(too_big):
            scale = xp.where(too_big, (max_frac * ref) / (xp.abs(proposed) + 1e-30), scale)
        state.get(field_name)[sl] -= proposed * scale

    update("rho",      "rho",      max_frac=0.5)
    update("rho_vr",   "vr",       max_frac=0.5)
    update("rho_vphi", "vphi",     max_frac=0.5)
    update("rho_vz",   "vz",       max_frac=0.5)
    update("E",        "E",        max_frac=0.5)

    # density floor
    if "rho" in state.fields:
        rho = state.get("rho")
        bad = ~xp.isfinite(rho)
        if xp.any(bad):
            rho[bad] = rho_floor
        rho[:] = xp.maximum(rho, rho_floor)

    for name in ("rho_vr", "rho_vphi", "rho_vz", "E"):
        if name in state.fields:
            arr = state.get(name)
            bad = ~xp.isfinite(arr)
            if xp.any(bad):
                arr[bad] = 0.0

    # velocity cap via boolean indexing (no cp.where)
    if "rho" in state.fields:
        rho = state.get("rho")
        v_cap = 1.0
        low = rho <= 2.0 * rho_floor
        if xp.any(low):
            for name in ("rho_vr", "rho_vphi", "rho_vz"):
                if name in state.fields:
                    arr = state.get(name)
                    v = arr[low] / xp.maximum(rho[low], 1e-12)
                    v = xp.clip(v, -v_cap, v_cap)
                    arr[low] = v * rho[low]


if __name__ == "__main__":
    print(f"Cylindrical residual operator ready ({VERSION}).")
    print("Usage: residual_cylindrical(state, geometry, config, fluxes, dt)")
