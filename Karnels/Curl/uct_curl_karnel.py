"""
ALADIN Plasma Drive v2.1
PRODUCTION EXPLICIT CYLINDRICAL UCT CURL RawKernel
Colab Pro A100 — pure CUDA via CuPy RawKernel (Python)

Updates face-centered B from edge-centered EMF:
  ∂B/∂t = −∇×E   (cylindrical staggered constrained transport)

EXPLICIT CYLINDRICAL:
  - All metric factors written out (1/r, r·Eφ, face/center radii)
  - Axis-safe (inv_r = 0 when r ≈ 0)
  - No Cartesian assumptions
  - Matches indexing style of uct_emf.py

Drop this file on Colab Pro A100. It compiles the kernel and is ready to launch.
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

# =============================================================================
# EXPLICIT CYLINDRICAL UCT CURL KERNEL (A100 RawKernel)
# =============================================================================
utc_curl_kernel = None
if HAS_CUPY:
    utc_curl_kernel = cp.RawKernel(r"""
extern "C" __global__
void uct_cylindrical_curl(
    const float* __restrict__ emf_r,
    const float* __restrict__ emf_phi,
    const float* __restrict__ emf_z,
    const float* __restrict__ r_faces,
    const float* __restrict__ r_centers,
    const float* __restrict__ dr,
    const float* __restrict__ dphi,
    const float* __restrict__ dz,
    float* __restrict__ Br,
    float* __restrict__ Bphi,
    float* __restrict__ Bz,
    int nr, int nphi, int nz, int g,
    float dt
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // Update Br (radial faces)
    // EXPLICIT: ∂Br/∂t = ∂Eφ/∂z − (1/r) ∂Ez/∂φ
    // ========================================================================
    int n_br = (nr + 1) * nphi * nz;
    if (tid < n_br) {
        int iz   = tid % nz;
        int t    = tid / nz;
        int iphi = t % nphi;
        int ir   = t / nphi;

        int irg = ir + g;
        int jg  = iphi + g;
        int kg  = iz + g;

        // Explicit cylindrical radius on the radial face
        float r = r_faces[ir];
        float inv_r = (r > 1e-8f) ? 1.0f / r : 0.0f;   // axis-safe
        float idphi = 1.0f / fmaxf(dphi[0], 1e-12f);
        float idz   = 1.0f / fmaxf(dz[0], 1e-12f);

        // Explicit ΔEz in φ
        float dEz_dphi = emf_z[(irg * (nphi + 1 + 2 * g) + (jg + 1)) * (nz + 2 * g) + kg]
                       - emf_z[(irg * (nphi + 1 + 2 * g) + jg) * (nz + 2 * g) + kg];

        // Explicit ΔEφ in z
        float dEphi_dz = emf_phi[(irg * (nphi + 2 * g) + jg) * (nz + 1 + 2 * g) + (kg + 1)]
                       - emf_phi[(irg * (nphi + 2 * g) + jg) * (nz + 1 + 2 * g) + kg];

        // Explicit cylindrical curl_r
        float curl_r = idz * dEphi_dz - inv_r * idphi * dEz_dphi;

        int out = (irg * (nphi + 2 * g) + jg) * (nz + 2 * g) + kg;
        Br[out] -= dt * curl_r;
    }

    // ========================================================================
    // Update Bphi (azimuthal faces)
    // EXPLICIT: ∂Bφ/∂t = ∂Ez/∂r − ∂Er/∂z
    // ========================================================================
    int n_bphi = nr * (nphi + 1) * nz;
    int offset_bphi = n_br;
    if (tid >= offset_bphi && tid < offset_bphi + n_bphi) {
        int local = tid - offset_bphi;
        int iz   = local % nz;
        int t    = local / nz;
        int iphi = t % (nphi + 1);
        int ir   = t / (nphi + 1);

        int irg = ir + g;
        int jg  = iphi + g;
        int kg  = iz + g;

        float idr = 1.0f / fmaxf(dr[min(ir, nr - 1)], 1e-12f);
        float idz = 1.0f / fmaxf(dz[0], 1e-12f);

        // Explicit ΔEz in r
        float dEz_dr = emf_z[((irg + 1) * (nphi + 1 + 2 * g) + jg) * (nz + 2 * g) + kg]
                     - emf_z[(irg * (nphi + 1 + 2 * g) + jg) * (nz + 2 * g) + kg];

        // Explicit ΔEr in z
        float dEr_dz = emf_r[(irg * (nphi + 1 + 2 * g) + jg) * (nz + 1 + 2 * g) + (kg + 1)]
                     - emf_r[(irg * (nphi + 1 + 2 * g) + jg) * (nz + 1 + 2 * g) + kg];

        // Explicit cylindrical curl_phi (no 1/r here)
        float curl_phi = idr * dEz_dr - idz * dEr_dz;

        int out = (irg * (nphi + 1 + 2 * g) + jg) * (nz + 2 * g) + kg;
        Bphi[out] -= dt * curl_phi;
    }

    // ========================================================================
    // Update Bz (axial faces)
    // EXPLICIT: ∂Bz/∂t = (1/r) ∂Er/∂φ − (1/r) ∂(r Eφ)/∂r
    // ========================================================================
    int n_bz = nr * nphi * (nz + 1);
    int offset_bz = n_br + n_bphi;
    if (tid >= offset_bz && tid < offset_bz + n_bz) {
        int local = tid - offset_bz;
        int iz   = local % (nz + 1);
        int t    = local / (nz + 1);
        int iphi = t % nphi;
        int ir   = t / nphi;

        int irg = ir + g;
        int jg  = iphi + g;
        int kg  = iz + g;

        // Explicit cylindrical radius at cell center
        float r_c = r_centers[min(max(ir, 0), nr - 1)];
        float inv_r = (r_c > 1e-8f) ? 1.0f / r_c : 0.0f;   // axis-safe
        float idphi = 1.0f / fmaxf(dphi[0], 1e-12f);
        float idr   = 1.0f / fmaxf(dr[min(ir, nr - 1)], 1e-12f);

        // Explicit ΔEr in φ
        float dEr_dphi = emf_r[(irg * (nphi + 1 + 2 * g) + (jg + 1)) * (nz + 1 + 2 * g) + kg]
                       - emf_r[(irg * (nphi + 1 + 2 * g) + jg) * (nz + 1 + 2 * g) + kg];

        // Explicit Δ(r Eφ)/∂r using face radii
        float r_p = r_faces[ir + 1];
        float r_m = r_faces[ir];
        float rEphi_p = r_p * emf_phi[((irg + 1) * (nphi + 2 * g) + jg) * (nz + 1 + 2 * g) + kg];
        float rEphi_m = r_m * emf_phi[(irg * (nphi + 2 * g) + jg) * (nz + 1 + 2 * g) + kg];
        float d_rEphi_dr = rEphi_p - rEphi_m;

        // Explicit cylindrical curl_z
        float curl_z = inv_r * idphi * dEr_dphi - inv_r * idr * d_rEphi_dr;

        int out = (irg * (nphi + 2 * g) + jg) * (nz + 1 + 2 * g) + kg;
        Bz[out] -= dt * curl_z;
    }
}
""", "uct_cylindrical_curl")


def launch_uct_curl(state, dt, geometry, config):
    """Launch the EXPLICIT CYLINDRICAL UCT curl kernel on A100.
    Updates Br, Bphi, Bz in-place from emf_r, emf_phi, emf_z.
    Works with both NumPy and CuPy state arrays (copies back if NumPy).
    """
    if not HAS_CUPY or utc_curl_kernel is None:
        raise RuntimeError("CuPy required for UCT CURL kernel (Colab Pro A100)")

    g = config.ghost_width
    nr, nphi, nz = config.nr, config.nphi, config.nz

    n_total = (nr + 1) * nphi * nz + nr * (nphi + 1) * nz + nr * nphi * (nz + 1)
    block = 128
    grid = (n_total + block - 1) // block

    def to_cp(a, copy_back_key=None):
        if a is None:
            return None, False
        if isinstance(a, cp.ndarray):
            return a, False
        arr = cp.asarray(a, dtype=cp.float32)
        return arr, True

    emf_r, _ = to_cp(state.get("emf_r"))
    emf_phi, _ = to_cp(state.get("emf_phi"))
    emf_z, _ = to_cp(state.get("emf_z"))
    Br, br_copy = to_cp(state.get("Br"))
    Bphi, bp_copy = to_cp(state.get("Bphi"))
    Bz, bz_copy = to_cp(state.get("Bz"))

    dr = to_cp(getattr(geometry, "dr", np.ones(nr, dtype=np.float32)))[0]
    dphi = to_cp(getattr(geometry, "dphi", np.ones(1, dtype=np.float32)))[0]
    dz = to_cp(getattr(geometry, "dz", np.ones(1, dtype=np.float32)))[0]
    r_faces = to_cp(geometry.r_faces)[0]
    r_centers = to_cp(geometry.r_centers)[0]

    utc_curl_kernel(
        (grid,), (block,),
        (
            emf_r, emf_phi, emf_z,
            r_faces, r_centers,
            dr, dphi, dz,
            Br, Bphi, Bz,
            nr, nphi, nz, g,
            np.float32(dt)
        )
    )
    cp.cuda.Stream.null.synchronize()

    # Copy results back if original was NumPy
    if br_copy:
        state.get("Br")[:] = cp.asnumpy(Br)
    if bp_copy:
        state.get("Bphi")[:] = cp.asnumpy(Bphi)
    if bz_copy:
        state.get("Bz")[:] = cp.asnumpy(Bz)


# =============================================================================
# STANDALONE
# =============================================================================
if __name__ == "__main__":
    print("ALADIN EXPLICIT CYLINDRICAL UCT CURL RawKernel (Colab Pro A100)")
    print("Kernel object:", utc_curl_kernel)
    if HAS_CUPY:
        print("Ready. Call launch_uct_curl(state, dt, geometry, config)")
    else:
        print("CuPy not present — kernel is None. Run on Colab Pro A100.")
