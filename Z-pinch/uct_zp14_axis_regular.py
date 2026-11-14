"""
uct_zp14_axis_regular.py
VERSION: 2026-11-14-ZP14-reg
Axis FV stencil that enforces B_phi ~ r regularity
"""

import cupy as cp

def analytic_dBdt(r):
    return -5e-4 * (cp.sin(cp.pi * r) + cp.pi * r * cp.cos(cp.pi * r))

def exact_cell_average(r_left, r_right, nquad=256):
    rs = cp.linspace(r_left, r_right, nquad)
    drs = float(rs[1] - rs[0])
    vals = analytic_dBdt(rs)
    integ = float(cp.sum(vals * rs) * drs)
    vol = 0.5 * (r_right**2 - r_left**2)
    return integ / vol

def regular_axis_dBdt(Nr):
    dr = 1.0 / Nr
    r = (cp.arange(Nr, dtype=cp.float64) + 0.5) * dr
    B = 0.5 * r
    vr = 1e-3 * cp.sin(cp.pi * r)

    # Face at r = dr
    r_face = dr
    # Enforce B ~ r
    B_face = B[0] * (r_face / r[0])          # = 2 * B[0]
    vr_face = 0.5 * (vr[0] + vr[1])
    flux_face = r_face * vr_face * B_face

    V0 = 0.5 * dr * dr
    return float(-flux_face / V0)

def zp14_reg():
    print("ZP-14 REGULARITY-AWARE AXIS STENCIL\n")
    print(f"{'Nr':>6} {'num':>12} {'exact':>12} {'abs_err':>12} {'order':>8}")
    prev = None
    for Nr in [32, 64, 128, 256, 512]:
        num = regular_axis_dBdt(Nr)
        exact = exact_cell_average(0.0, 1.0/Nr)
        err = abs(num - exact)
        order = float(cp.log2(prev/err)) if prev else 0.0
        print(f"{Nr:6d} {num:12.6e} {exact:12.6e} {err:12.3e} {order:8.2f}")
        prev = err

if __name__ == "__main__":
    zp14_reg()
