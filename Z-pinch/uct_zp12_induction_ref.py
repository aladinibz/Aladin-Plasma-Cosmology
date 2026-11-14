"""
uct_zp12_induction_ref.py
VERSION: 2026-11-14-ZP12
First-step induction vs analytical dB/dt
"""

import cupy as cp

def analytical_dBdt(r):
    # dB/dt = -5e-4 * (sin(pi r) + pi r cos(pi r))
    return -5e-4 * (cp.sin(cp.pi * r) + cp.pi * r * cp.cos(cp.pi * r))

def numerical_dBdt(Nr, dt=1e-6):
    dr = 1.0 / Nr
    r = (cp.arange(Nr, dtype=cp.float64) + 0.5) * dr
    B = 0.5 * r
    vr = 1e-3 * cp.sin(cp.pi * r)

    flux = vr * B
    dflux = cp.zeros(Nr)
    for i in range(1, Nr-1):
        dflux[i] = (flux[i+1] - flux[i-1]) / (2 * dr)
    # ghost
    r_g = r[-1] + dr
    vr_g = 1e-3 * cp.sin(cp.pi * r_g)
    B_g = 0.5 * r_g
    flux_g = vr_g * B_g
    dflux[-1] = (flux_g - flux[-2]) / (2 * dr)
    dflux[0] = (flux[1] - 0.0) / (2 * dr)   # axis-friendly

    dBdt_num = -dflux
    return r.get(), dBdt_num.get()

def volumes(Nr):
    dr = 1.0 / Nr
    rf = cp.arange(Nr + 1) * dr
    return (rf[1:]**2 - rf[:-1]**2).get()

def zp12():
    print("ZP-12 — ANALYTICAL INDUCTION REFERENCE\n")
    print(f"{'Nr':>6} {'full L2':>12} {'interior L2':>12} {'max|err|':>12}")
    prev_full = None
    for Nr in [32, 64, 128, 256]:
        r, dBdt_num = numerical_dBdt(Nr)
        dBdt_ana = analytical_dBdt(cp.asarray(r)).get()
        err = dBdt_num - dBdt_ana
        V = volumes(Nr)
        full = float(cp.sqrt(cp.sum(err**2 * V) / cp.sum(V)))
        interior = float(cp.sqrt(cp.sum(err[1:-1]**2 * V[1:-1]) / cp.sum(V[1:-1])))
        mx = float(cp.abs(err).max())
        print(f"{Nr:6d} {full:12.3e} {interior:12.3e} {mx:12.3e}")
        if prev_full is not None:
            order = cp.log2(prev_full / full)
            print(f"         observed order (full) ≈ {float(order):.2f}")
        prev_full = full

    print("\nAnalytical target:")
    print("  dB/dt = -5e-4 * (sin(pi r) + pi r cos(pi r))")
    print("If L2 decreases with Nr, the discrete induction matches the analytic derivative.")

if __name__ == "__main__":
    zp12()
