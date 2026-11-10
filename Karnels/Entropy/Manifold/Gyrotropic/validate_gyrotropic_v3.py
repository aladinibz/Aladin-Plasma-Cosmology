"""
ALADIN — Gyrotropic Closure Validation Suite v3
Three requested diagnostics:
  1. Per-moment errors (raw + scaled) for every case
  2. Quadrature convergence: 16×12, 32×24, 48×36, 64×48 vs independent reference
  3. ±0.25 heat-flux failure diagnosis (per-feature residual + free params at exit)

Requires gyrotropic_cylindrical_kernel.py in the same directory.
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    raise RuntimeError("CuPy required")

from gyrotropic_cylindrical_kernel import launch_gyro, compute_base_moments

SCALE = np.array([1,1,1,1,1,1,3,2,3,1,4,3], dtype=np.float32)
FEAT_NAMES = [
    "1", "c∥", "c⊥²", "c∥²", "c∥/(1+c∥²)", "c∥ c⊥²/(1+c⊥²)",
    "c⊥⁴", "c∥² c⊥²", "c∥⁴", "c∥³/(1+c∥²)²", "c⊥⁶/(1+c⊥²)", "c∥² c⊥⁴/(1+c⊥²)"
]

# ---------------------------------------------------------------------------
# Independent reference quadrature (fixed, high-res)
# ---------------------------------------------------------------------------
def independent_reference(alpha_par=0.5, alpha_perp=0.5, n_par=96, n_perp=72):
    np_h = np.linspace(-10.0, 10.0, n_par).astype(np.float32)
    wp_h = np.ones_like(np_h) * (20.0 / n_par)
    nq_h = np.linspace(0.02, 20.0, n_perp).astype(np.float32)
    wq_h = np.ones_like(nq_h) * (20.0 / n_perp)

    def phi(cp, c2):
        cp2 = cp*cp
        idp = 1.0/(1.0+cp2)
        ide = 1.0/(1.0+c2)
        return np.array([
            1, cp, c2, cp2, cp*idp, cp*c2*ide,
            c2*c2, cp2*c2, cp2*cp2,
            (cp**3)*idp*idp, (c2**3)*ide, cp2*(c2**2)*ide
        ], dtype=np.float32)

    Z = 0.0
    M = np.zeros(12, dtype=np.float32)
    for ip in range(n_par):
        for iq in range(n_perp):
            cp_ = np_h[ip]
            c2 = nq_h[iq]**2
            w = wp_h[ip] * wq_h[iq] * nq_h[iq]
            wgt = np.exp(-alpha_par*cp_*cp_ - alpha_perp*c2) * w
            Z += wgt
            M += wgt * phi(cp_, c2)
    return M / Z

# ---------------------------------------------------------------------------
# Per-moment error table
# ---------------------------------------------------------------------------
def print_moment_table(name, M_target, M_recon):
    M_tgt = M_target.get() if hasattr(M_target, "get") else np.asarray(M_target)
    M_rec = M_recon.get() if hasattr(M_recon, "get") else np.asarray(M_recon)
    # use first cell
    tgt = M_tgt[0] if M_tgt.ndim > 1 else M_tgt
    rec = M_rec[0] if M_rec.ndim > 1 else M_rec
    raw = np.abs(rec - tgt)
    scaled = raw / SCALE
    print(f"\n--- Per-moment errors: {name} ---")
    print(f"{'feat':<22} {'raw':>12} {'scaled':>12}")
    for i in range(12):
        print(f"{FEAT_NAMES[i]:<22} {raw[i]:12.6e} {scaled[i]:12.6e}")
    print(f"{'MAX raw':<22} {np.max(raw):12.6e}")
    print(f"{'MAX scaled':<22} {np.max(scaled):12.6e}")

# ---------------------------------------------------------------------------
# 1 + 3: Full reconstruction cases with per-moment table
# ---------------------------------------------------------------------------
def run_reconstruction_cases(Ncells=16):
    print("\n" + "="*70)
    print("1 & 3. RECONSTRUCTION CASES + PER-MOMENT ERRORS")
    print("="*70)
    cases = [
        ("isotropic", 0.5, 0.5, 0.0),
        ("mild aniso", 0.4, 0.7, 0.0),
        ("stronger aniso", 0.3, 0.9, 0.0),
        ("q∥ = -0.25", 0.5, 0.5, -0.25),
        ("q∥ = -0.12", 0.5, 0.5, -0.12),
        ("q∥ = +0.00", 0.5, 0.5, 0.0),
        ("q∥ = +0.12", 0.5, 0.5, 0.12),
        ("q∥ = +0.25", 0.5, 0.5, 0.25),
    ]
    for label, ap, ae, qamp in cases:
        if abs(qamp) < 1e-8:
            M = compute_base_moments(ap, ae)
        else:
            M = compute_base_moments(ap, ae).copy()
            M[1] += 0.04 * np.sign(qamp)
            M[9] += qamp * 0.25
            M[5] += qamp * 0.08
        M_target = cp.tile(cp.asarray(M), (Ncells, 1)).astype(cp.float32)
        n_in = cp.ones(Ncells, dtype=cp.float32)
        lambda_io = cp.zeros((Ncells, 12), dtype=cp.float32)
        lambda_io[:, 6] = lambda_io[:, 8] = lambda_io[:, 10] = -0.05
        outs = launch_gyro(M_target, n_in, lambda_io)
        status = outs[-1].get()
        margin = outs[6].get()
        M_recon = outs[7]
        print(f"\n=== {label} ===")
        print(f"status={np.unique(status)}  margin={margin[0]:.4f}")
        print_moment_table(label, M_target, M_recon)
        # for the ±0.25 cases, also show the free parameters that were committed
        if abs(qamp) >= 0.24:
            lam = lambda_io.get()[0]
            print(f"final λ (first cell): {lam}")

# ---------------------------------------------------------------------------
# 2. Quadrature convergence study
# ---------------------------------------------------------------------------
def quadrature_study():
    print("\n" + "="*70)
    print("2. QUADRATURE CONVERGENCE (vs independent 96×72 reference)")
    print("="*70)
    ref = independent_reference(0.5, 0.5)
    print(f"Independent reference moments (first 4): {ref[:4]}")

    # We only have the kernel's current quadrature exposed via compute_base_moments.
    # To study convergence we re-implement the same rule at different resolutions
    # on the host and compare to the independent reference.
    configs = [
        ("16×12", 16, 12),
        ("32×24", 32, 24),
        ("48×36", 48, 36),
        ("64×48", 64, 48),
    ]
    print(f"\n{'config':<10} {'max raw':>12} {'max scaled':>12} {'raw[2]':>12} {'raw[3]':>12}")
    for name, np_, nq_ in configs:
        # simple uniform nodes for the study (not the kernel's hand-tuned ones)
        np_h = np.linspace(-8.0, 8.0, np_).astype(np.float32)
        wp_h = np.ones_like(np_h) * (16.0 / np_)
        nq_h = np.linspace(0.1, 16.0, nq_).astype(np.float32)
        wq_h = np.ones_like(nq_h) * (16.0 / nq_)
        def phi(cp, c2):
            cp2 = cp*cp
            idp = 1.0/(1.0+cp2)
            ide = 1.0/(1.0+c2)
            return np.array([
                1, cp, c2, cp2, cp*idp, cp*c2*ide,
                c2*c2, cp2*c2, cp2*cp2,
                (cp**3)*idp*idp, (c2**3)*ide, cp2*(c2**2)*ide
            ], dtype=np.float32)
        Z = 0.0
        M = np.zeros(12, dtype=np.float32)
        for ip in range(np_):
            for iq in range(nq_):
                cp_ = np_h[ip]
                c2 = nq_h[iq]**2
                w = wp_h[ip] * wq_h[iq] * nq_h[iq]
                wgt = np.exp(-0.5*cp_*cp_ - 0.5*c2) * w
                Z += wgt
                M += wgt * phi(cp_, c2)
        M /= Z
        raw = np.abs(M - ref)
        scaled = raw / SCALE
        print(f"{name:<10} {np.max(raw):12.6e} {np.max(scaled):12.6e} {raw[2]:12.6e} {raw[3]:12.6e}")

    # also show the kernel's own current quadrature vs the independent reference
    M_kernel = compute_base_moments(0.5, 0.5)
    raw_k = np.abs(M_kernel - ref)
    scaled_k = raw_k / SCALE
    print(f"\nKernel current quad vs independent ref:")
    print(f"  max raw    = {np.max(raw_k):.6e}")
    print(f"  max scaled = {np.max(scaled_k):.6e}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("ALADIN Gyrotropic Closure — Validation v3")
    print("Normalization: scaled = |M_k - Mt_k| / scale[k]")
    print("scale = [1,1,1,1,1,1,3,2,3,1,4,3]")
    run_reconstruction_cases()
    quadrature_study()
    print("\n" + "="*70)
    print("END")
