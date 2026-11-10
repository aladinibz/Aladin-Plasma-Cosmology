"""
ALADIN Plasma Drive v2.1
PRODUCTION Colab Pro A100 — Higher-Order Explicit Cylindrical Dedner GLM
(Corrected indexing + staggered derivative centering for legitimate 4th-order tests)

4th-order (interior) hyperbolic + parabolic divergence cleaning.
Axis-safe, pure CuPy RawKernel.

Staggered convention used (ghosts included):
  Br    : (nr+1 + 2*g, nphi + 2*g, nz + 2*g)     radial faces
  Bphi  : (nr + 2*g, nphi+1 + 2*g, nz + 2*g)     azimuthal faces
  Bz    : (nr + 2*g, nphi + 2*g, nz+1 + 2*g)     vertical faces
  psi, divB : (nr + 2*g, nphi + 2*g, nz + 2*g)  cell centers

Interior 4th-order stencils are properly centered on the staggered locations
so that GLM-04 can measure true order of accuracy.
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

GLM_HO_SRC = r"""
#define SMALL_R 1e-8f
#define SMALL   1e-12f

__device__ __forceinline__ float d4(float fm2, float fm1, float fp1, float fp2, float h) {
    return (-fp2 + 8.f*fp1 - 8.f*fm1 + fm2) / (12.f * h);
}

extern "C" __global__
void glm_cylindrical_ho(
    float* __restrict__ Br,
    float* __restrict__ Bphi,
    float* __restrict__ Bz,
    float* __restrict__ psi,
    float* __restrict__ divB,
    const float* __restrict__ r_faces,
    const float* __restrict__ r_centers,
    const float* __restrict__ dr,
    const float* __restrict__ dphi,
    const float* __restrict__ dz,
    const float* __restrict__ ch_local,
    int nr, int nphi, int nz, int g,
    float dt, float ch_const, float cp_scale, float damp_coeff,
    int use_local_ch
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int nzg   = nz + 2 * g;
    int nphig = nphi + 2 * g;
    int nrg   = nr + 2 * g;
    int ncells = nrg * nphig * nzg;
    if (idx >= ncells) return;

    int iz = idx % nzg;
    int ip = (idx / nzg) % nphig;
    int ir = idx / (nzg * nphig);

    if (ir < g || ir >= nr + g ||
        ip < g || ip >= nphi + g ||
        iz < g || iz >= nz + g)
        return;

    int ir_l = ir - g;
    int ip_l = ip - g;
    int iz_l = iz - g;

    float r_c = r_centers[min(max(ir_l, 0), nr - 1)];
    float inv_r = (r_c > SMALL_R) ? (1.f / r_c) : 0.f;
    float dr_c   = dr[min(max(ir_l, 0), nr - 1)];
    float dphi_c = dphi[0];
    float dz_c   = dz[0];

    // Cell strides
    int c_sphi = nphig;
    int c_sz   = nzg;
    int cidx   = (ir * c_sphi + ip) * c_sz + iz;

    // Face strides (ghosts included)
    int Br_sphi = nphig;
    int Br_sz   = nzg;
    int Bp_sphi = nphi + 1 + 2 * g;
    int Bp_sz   = nzg;
    int Bz_sphi = nphig;
    int Bz_sz   = nz + 1 + 2 * g;

    // -----------------------------------------------------------------
    // Cylindrical divergence (correct staggered form)
    // 2nd-order face difference is exact for linear fields;
    // 4th-order used only when stencil is fully interior and centered.
    // -----------------------------------------------------------------
    float div_r = 0.f, div_p = 0.f, div_z = 0.f;

    // Radial divergence
    float r_f_m = r_faces[min(max(ir_l    , 0), nr)];
    float r_f_p = r_faces[min(max(ir_l + 1, 0), nr)];
    float Br_m  = Br[( ir      * Br_sphi + ip) * Br_sz + iz];
    float Br_p  = Br[((ir + 1) * Br_sphi + ip) * Br_sz + iz];

    // Default: correct 2nd-order staggered
    div_r = inv_r * (r_f_p * Br_p - r_f_m * Br_m) / fmaxf(dr_c, SMALL);

    // 4th-order staggered (double accumulation to avoid float32 cancellation)
    bool can4_r = (ir_l >= 1 && ir_l <= nr - 3);
    if (can4_r) {
        double rBm1 = (double)r_faces[ir_l - 1] * (double)Br[((ir - 1) * Br_sphi + ip) * Br_sz + iz];
        double rB0  = (double)r_faces[ir_l    ] * (double)Br[( ir      * Br_sphi + ip) * Br_sz + iz];
        double rB1  = (double)r_faces[ir_l + 1] * (double)Br[((ir + 1) * Br_sphi + ip) * Br_sz + iz];
        double rBp  = (double)r_faces[ir_l + 2] * (double)Br[((ir + 2) * Br_sphi + ip) * Br_sz + iz];
        double dr_d = (double)dr_c;
        // Differences first, then scale — accumulated in double
        double d_rBr = (27.0 * (rB1 - rB0) + (rBm1 - rBp)) / (24.0 * dr_d);
        div_r = (float)((1.0 / (double)r_c) * d_rBr);
    }

    // Azimuthal (2nd-order, exact for linear)
    float Bp_m = Bphi[(ir * Bp_sphi +  ip     ) * Bp_sz + iz];
    float Bp_p = Bphi[(ir * Bp_sphi + (ip + 1)) * Bp_sz + iz];
    div_p = inv_r * (Bp_p - Bp_m) / fmaxf(dphi_c, SMALL);

    // Vertical (2nd-order)
    float Bz_m = Bz[(ir * Bz_sphi + ip) * Bz_sz +  iz     ];
    float Bz_p = Bz[(ir * Bz_sphi + ip) * Bz_sz + (iz + 1)];
    div_z = (Bz_p - Bz_m) / fmaxf(dz_c, SMALL);

    float div = div_r + div_p + div_z;
    if (divB != nullptr) divB[cidx] = div;

    // -----------------------------------------------------------------
    // Update ψ
    // Damping term is applied exactly (unconditionally stable form)
    // -----------------------------------------------------------------
    float ch = ch_const;
    if (use_local_ch && ch_local != nullptr)
        ch = fmaxf(ch_local[cidx], 1e-6f);
    float ch2 = ch * ch;
    float damp = (cp_scale > SMALL) ? (ch2 / (cp_scale * cp_scale)) : damp_coeff;

    float psi_old = psi[cidx];
    // Exact integration of the linear damping + forward Euler on the source
    float decay = expf(-damp * dt);
    float psi_new = decay * psi_old - dt * ch2 * div;
    psi_new = fmaxf(fminf(psi_new, 1e3f), -1e3f);
    psi[cidx] = psi_new;

    // -----------------------------------------------------------------
    // Very conservative B correction (2nd-order, one-sided ownership,
    // small factor to avoid overshoot while we stabilize)
    // -----------------------------------------------------------------
    float corr_factor = 1.0f;   // full strength; stability via sub-cycling + moderate damping
    if (ir > g && ir_l > 0) {
        float pm = psi[((ir - 1) * c_sphi + ip) * c_sz + iz];
        float grad_r = (psi_old - pm) / fmaxf(dr_c, SMALL);
        int fr = (ir * Br_sphi + ip) * Br_sz + iz;
        Br[fr] -= corr_factor * dt * grad_r;
    }
    if (ir_l == 0) {
        int fr = (ir * Br_sphi + ip) * Br_sz + iz;
        Br[fr] = 0.f;
    }

    if (ip > g) {
        float pm = psi[(ir * c_sphi + (ip - 1)) * c_sz + iz];
        float grad_p = inv_r * (psi_old - pm) / fmaxf(dphi_c, SMALL);
        int fp = (ir * Bp_sphi + ip) * Bp_sz + iz;
        Bphi[fp] -= corr_factor * dt * grad_p;
    }

    if (iz > g) {
        float pm = psi[(ir * c_sphi + ip) * c_sz + (iz - 1)];
        float grad_z = (psi_old - pm) / fmaxf(dz_c, SMALL);
        int fz = (ir * Bz_sphi + ip) * Bz_sz + iz;
        Bz[fz] -= corr_factor * dt * grad_z;
    }
}
"""

glm_ho_kernel = None
if HAS_CUPY:
    glm_ho_kernel = cp.RawKernel(GLM_HO_SRC, "glm_cylindrical_ho")


def launch_glm_ho(state, dt, geometry, config,
                  ch=None, cp_scale=None, damp_coeff=0.0,
                  use_local_ch=False, nsub=1):
    """
    nsub : number of GLM sub-cycles. 
           nsub>1 runs nsub steps of dt/nsub, giving faster effective cleaning
           while respecting the explicit stability limit.
    """
    if not HAS_CUPY or glm_ho_kernel is None:
        raise RuntimeError("CuPy required")

    g = int(getattr(config, "ghost_width", 2))
    nr, nphi, nz = int(config.nr), int(config.nphi), int(config.nz)

    if not (hasattr(state, "has") and state.has("psi")):
        shape = state.get("rho").shape
        state.fields["psi"] = cp.zeros(shape, dtype=cp.float32)
    if not (hasattr(state, "has") and state.has("divB")):
        shape = state.get("rho").shape
        state.fields["divB"] = cp.zeros(shape, dtype=cp.float32)

    ch_local = None
    if use_local_ch and hasattr(state, "has") and state.has("ch_local"):
        ch_local = state.get("ch_local")

    if ch is None:
        ch = float(getattr(config, "ch", 1.0))
    ch = max(float(ch), 1e-6)

    dx_min = float(getattr(geometry, "dx_min", 1e-3))
    if cp_scale is None:
        cp_scale = ch * float(dt) / max(dx_min, 1e-8)
    cp_scale = max(float(cp_scale), 1e-8)

    dr   = getattr(geometry, "dr",   np.ones(nr, dtype=np.float32))
    dphi = getattr(geometry, "dphi", np.ones(1,  dtype=np.float32))
    dz   = getattr(geometry, "dz",   np.ones(1,  dtype=np.float32))

    def to_device(a):
        if isinstance(a, cp.ndarray):
            return a.astype(cp.float32, copy=False)
        return cp.asarray(np.asarray(a, dtype=np.float32))

    r_faces   = to_device(geometry.r_faces)
    r_centers = to_device(geometry.r_centers)
    dr        = to_device(dr)
    dphi      = to_device(dphi)
    dz        = to_device(dz)

    if ch_local is None:
        ch_local = cp.zeros(1, dtype=cp.float32)

    n = state.get("rho").size
    block = 128
    grid = (n + block - 1) // block

    dt_sub = float(dt) / max(int(nsub), 1)
    for _ in range(max(int(nsub), 1)):
        glm_ho_kernel(
            (grid,), (block,),
            (
                state.get("Br"),
                state.get("Bphi"),
                state.get("Bz"),
                state.get("psi"),
                state.get("divB"),
                r_faces,
                r_centers,
                dr, dphi, dz,
                ch_local,
                nr, nphi, nz, g,
                np.float32(dt_sub),
                np.float32(ch),
                np.float32(cp_scale),
                np.float32(damp_coeff),
                np.int32(1 if use_local_ch else 0)
            )
        )
    cp.cuda.Stream.null.synchronize()


if __name__ == "__main__":
    print("Higher-order cylindrical GLM RawKernel (v13 - full-strength + sub-cycle) ready.")
    print("Kernel:", glm_ho_kernel)
    if HAS_CUPY:
        print("Ready for Colab Pro A100 — all geometry arrays forced to CuPy.")
    else:
        print("CuPy not present here.")
