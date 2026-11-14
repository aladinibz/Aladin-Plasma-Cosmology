"""
uct_zp11_evolution_audit.py
VERSION: 2026-11-14-ZP11
A. Ghost application trace
B. Induction equation check (static + first step)
"""

import cupy as cp

def compute_Jz_last(B, r, dr):
    """Centered curl on last cell using analytic ghost."""
    r_ghost = r[-1] + dr
    B_ghost = 0.5 * r_ghost
    num = r_ghost * B_ghost - r[-2] * B[-2]
    return float(num / (2 * dr * r[-1]))

def zp11():
    print("ZP-11 — EVOLUTION MACHINERY AUDIT\n")

    # ---------- A. Static equilibrium, vr = 0 ----------
    print("=== A. Static case (vr=0) — B must stay unchanged ===")
    for Nr in [32, 64, 128]:
        dr = 1.0 / Nr
        r = (cp.arange(Nr, dtype=cp.float64) + 0.5) * dr
        B = 0.5 * r.copy()
        vr = cp.zeros(Nr)

        Jz_before = compute_Jz_last(B, r, dr)

        # induction update with vr=0 (should be no-op)
        # discrete: dB/dt = -d(vr*B)/dr
        flux = vr * B
        dflux = cp.zeros(Nr)
        for i in range(1, Nr-1):
            dflux[i] = (flux[i+1] - flux[i-1]) / (2*dr)
        # last cell with ghost
        r_g = r[-1] + dr
        flux_g = 0.0 * (0.5 * r_g)   # vr_ghost=0
        dflux[-1] = (flux_g - flux[-2]) / (2*dr)
        B_new = B - 1e-4 * dflux   # dt=1e-4

        Jz_after = compute_Jz_last(B_new, r, dr)
        max_dB = float(cp.abs(B_new - B).max())
        print(f"Nr={Nr:3d}  Jz_before={Jz_before:.10f}  Jz_after={Jz_after:.10f}  max|dB|={max_dB:.3e}")

    # ---------- B. First step with small vr perturbation ----------
    print("\n=== B. First induction step with vr = 1e-3 sin(pi r) ===")
    Nr = 64
    dr = 1.0 / Nr
    r = (cp.arange(Nr, dtype=cp.float64) + 0.5) * dr
    B = 0.5 * r.copy()
    vr = 1e-3 * cp.sin(cp.pi * r)
    dt = 1e-4

    Jz0 = compute_Jz_last(B, r, dr)

    # independent reference: dB/dt = -d(vr B)/dr
    flux = vr * B
    dflux_ref = cp.zeros(Nr)
    for i in range(1, Nr-1):
        dflux_ref[i] = (flux[i+1] - flux[i-1]) / (2*dr)
    r_g = r[-1] + dr
    vr_g = 1e-3 * cp.sin(cp.pi * r_g)
    B_g = 0.5 * r_g
    flux_g = vr_g * B_g
    dflux_ref[-1] = (flux_g - flux[-2]) / (2*dr)

    B_ref = B - dt * dflux_ref
    Jz_ref = compute_Jz_last(B_ref, r, dr)

    print(f"Jz initial          = {Jz0:.10f}")
    print(f"Jz after 1 step ref = {Jz_ref:.10f}")
    print(f"max |dB| reference  = {float(cp.abs(B_ref-B).max()):.3e}")
    print(f"Jz error after step = {Jz_ref-1:.3e}")

    print("\nIf static case keeps Jz=1 and max|dB|=0, induction is consistent at rest.")
    print("If first-step Jz error stays small, the discrete flux form is acceptable.")

if __name__ == "__main__":
    zp11()
