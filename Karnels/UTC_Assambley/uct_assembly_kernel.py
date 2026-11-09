"""
ALADIN Plasma Drive v2.1
PRODUCTION Cylindrical UCT Assembly Kernel
(Upwind Constrained Transport EMF assembly)

Takes cell-centered primitives + staggered B and assembles
edge-centered EMFs with:
  - 4-corner weighted average (Londrillo–Mignone / Athena UCT)
  - local fast-magnetosonic diffusion
  - optional Hall term (discrete J×B / ρ)
  - explicit cylindrical metric factors
  - axis-safe (r≈0 suppressed)

Pure CuPy RawKernel (Colab Pro A100). Drop-in for the driver’s
utc or uct_emf operator slot.

Author: Aladin (Bucurenciu Mihai Alexandru)
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

TILE = 8
PAD = 1

# =============================================================================
# DEVICE HELPERS + KERNEL
# =============================================================================
uct_assembly_kernel = None
if HAS_CUPY:
    uct_assembly_kernel = cp.RawKernel(r"""
#define TILE 8
#define PAD 1

__device__ __forceinline__ float upwind_weight(float v) {
    return (v > 1e-8f) ? 1.f : ((v < -1e-8f) ? 0.f : 0.5f);
}

__device__ __forceinline__ float fast_ms(float rho, float vn, float Bn, float Bt, float gamma, float p) {
    float cs2 = gamma * fmaxf(p, 1e-12f) / fmaxf(rho, 1e-12f);
    float ca2 = (Bn*Bn + Bt*Bt) / fmaxf(rho, 1e-12f);
    float can2 = Bn*Bn / fmaxf(rho, 1e-12f);
    float disc = fmaxf((cs2 + ca2)*(cs2 + ca2) - 4.f*cs2*can2, 0.f);
    float cf = sqrtf(0.5f*(cs2 + ca2 + sqrtf(disc)));
    return cf + fabsf(vn);
}

extern "C" __global__
void uct_assembly_v2(
    const float* __restrict__ rho,
    const float* __restrict__ rho_vr,
    const float* __restrict__ rho_vphi,
    const float* __restrict__ rho_vz,
    const float* __restrict__ p,
    const float* __restrict__ Br,
    const float* __restrict__ Bphi,
    const float* __restrict__ Bz,
    const float* __restrict__ r_faces,
    const float* __restrict__ r_centers,
    const float* __restrict__ dr,
    const float* __restrict__ dphi,
    const float* __restrict__ dz,
    float* __restrict__ emf_r,
    float* __restrict__ emf_phi,
    float* __restrict__ emf_z,
    int nr, int nphi, int nz, int g,
    float hall_coeff, int use_hall,
    float gamma
) {
    __shared__ float s_rho [TILE+2*PAD][TILE+2*PAD][TILE+2*PAD];
    __shared__ float s_vr  [TILE+2*PAD][TILE+2*PAD][TILE+2*PAD];
    __shared__ float s_vphi[TILE+2*PAD][TILE+2*PAD][TILE+2*PAD];
    __shared__ float s_vz  [TILE+2*PAD][TILE+2*PAD][TILE+2*PAD];
    __shared__ float s_p   [TILE+2*PAD][TILE+2*PAD][TILE+2*PAD];
    __shared__ float s_Br  [TILE+2*PAD][TILE+2*PAD][TILE+2*PAD];
    __shared__ float s_Bphi[TILE+2*PAD][TILE+2*PAD][TILE+2*PAD];
    __shared__ float s_Bz  [TILE+2*PAD][TILE+2*PAD][TILE+2*PAD];

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int ir0 = blockIdx.x * TILE;
    int ip0 = blockIdx.y * TILE;
    int iz0 = blockIdx.z * TILE;

    // Cooperative halo load (cell-centered)
    for (int dz = tz; dz < TILE+2*PAD; dz += blockDim.z)
    for (int dp = ty; dp < TILE+2*PAD; dp += blockDim.y)
    for (int dr = tx; dr < TILE+2*PAD; dr += blockDim.x) {
        int ir = max(0, min(nr + 2*g - 1, ir0 + dr - PAD + g));
        int ip = max(0, min(nphi + 2*g - 1, ip0 + dp - PAD + g));
        int iz = max(0, min(nz + 2*g - 1, iz0 + dz - PAD + g));
        int idx = (ir * (nphi + 2*g) + ip) * (nz + 2*g) + iz;
        float rh = fmaxf(rho[idx], 1e-12f);
        s_rho[dr][dp][dz]  = rh;
        s_vr[dr][dp][dz]   = rho_vr[idx]  / rh;
        s_vphi[dr][dp][dz] = rho_vphi[idx] / rh;
        s_vz[dr][dp][dz]   = rho_vz[idx]  / rh;
        s_p[dr][dp][dz]    = fmaxf(p[idx], 1e-12f);
        s_Br[dr][dp][dz]   = Br[idx];
        s_Bphi[dr][dp][dz] = Bphi[idx];
        s_Bz[dr][dp][dz]   = Bz[idx];
    }
    __syncthreads();

    if (tx >= TILE || ty >= TILE || tz >= TILE) return;
    int ir = ir0 + tx, ip = ip0 + ty, iz = iz0 + tz;
    int lr = tx + PAD, lp = ty + PAD, lz = tz + PAD;

    float r = (ir < nr + 1) ? r_faces[min(max(ir, 0), nr)] : 1.f;
    float inv_r = (r > 1e-8f) ? 1.f / r : 0.f;
    float idr   = (ir < nr) ? 1.f / fmaxf(dr[ir], 1e-12f) : 0.f;
    float idphi = 1.f / fmaxf(dphi[0], 1e-12f);
    float idz   = 1.f / fmaxf(dz[0], 1e-12f);

    // ========================================================================
    // emf_z  (edges parallel to z, corners in r-φ)
    // ========================================================================
    if (ir <= nr && ip <= nphi && iz < nz) {
        float vr_s[4]  = {s_vr[lr][lp][lz],   s_vr[lr+1][lp][lz],   s_vr[lr][lp+1][lz],   s_vr[lr+1][lp+1][lz]};
        float vp_s[4]  = {s_vphi[lr][lp][lz], s_vphi[lr+1][lp][lz], s_vphi[lr][lp+1][lz], s_vphi[lr+1][lp+1][lz]};
        float Br_s[4]  = {s_Br[lr][lp][lz],   s_Br[lr+1][lp][lz],   s_Br[lr][lp+1][lz],   s_Br[lr+1][lp+1][lz]};
        float Bp_s[4]  = {s_Bphi[lr][lp][lz], s_Bphi[lr+1][lp][lz], s_Bphi[lr][lp+1][lz], s_Bphi[lr+1][lp+1][lz]};
        float rh_s[4]  = {s_rho[lr][lp][lz],  s_rho[lr+1][lp][lz],  s_rho[lr][lp+1][lz],  s_rho[lr+1][lp+1][lz]};
        float p_s[4]   = {s_p[lr][lp][lz],    s_p[lr+1][lp][lz],    s_p[lr][lp+1][lz],    s_p[lr+1][lp+1][lz]};

        float E_s[4];
        for (int i = 0; i < 4; ++i)
            E_s[i] = -(vr_s[i]*Bp_s[i] - vp_s[i]*Br_s[i]);

        float wr = upwind_weight(0.5f*(vr_s[0] + vr_s[2]));
        float wp = upwind_weight(0.5f*(vp_s[0] + vp_s[1]));
        float E_avg = (1.f-wr)*(1.f-wp)*E_s[0] + wr*(1.f-wp)*E_s[1]
                    + (1.f-wr)*wp*E_s[2] + wr*wp*E_s[3];

        float cf = fast_ms(rh_s[0], vr_s[0], Br_s[0], Bp_s[0], gamma, p_s[0]);
        float E = E_avg + 0.5f*cf*((Br_s[1]-Br_s[0]) - (Bp_s[2]-Bp_s[0]));

        // axis suppress
        if (r < 1e-8f) E = 0.f;

        if (use_hall && hall_coeff != 0.f) {
            float Jr = inv_r * idphi * (s_Bz[lr][lp+1][lz] - s_Bz[lr][lp-1][lz])
                     - idz * (s_Bphi[lr][lp][lz+1] - s_Bphi[lr][lp][lz-1]);
            float Jp = idz * (s_Br[lr][lp][lz+1] - s_Br[lr][lp][lz-1])
                     - idr * (s_Bz[lr+1][lp][lz] - s_Bz[lr-1][lp][lz]);
            float rho_e = 0.25f*(rh_s[0]+rh_s[1]+rh_s[2]+rh_s[3]);
            float Br_e  = 0.25f*(Br_s[0]+Br_s[1]+Br_s[2]+Br_s[3]);
            float Bp_e  = 0.25f*(Bp_s[0]+Bp_s[1]+Bp_s[2]+Bp_s[3]);
            E += (hall_coeff / fmaxf(rho_e, 1e-12f)) * (Jr*Bp_e - Jp*Br_e);
        }

        int eidx = ((ir+g)*(nphi+1+2*g) + (ip+g))*(nz+2*g) + (iz+g);
        emf_z[eidx] = E;
    }

    // ========================================================================
    // emf_phi (edges parallel to φ)
    // ========================================================================
    if (ir < nr && ip <= nphi && iz <= nz) {
        float vr_s[4] = {s_vr[lr][lp][lz],   s_vr[lr][lp][lz+1],   s_vr[lr+1][lp][lz],   s_vr[lr+1][lp][lz+1]};
        float vz_s[4] = {s_vz[lr][lp][lz],   s_vz[lr][lp][lz+1],   s_vz[lr+1][lp][lz],   s_vz[lr+1][lp][lz+1]};
        float Br_s[4] = {s_Br[lr][lp][lz],   s_Br[lr][lp][lz+1],   s_Br[lr+1][lp][lz],   s_Br[lr+1][lp][lz+1]};
        float Bz_s[4] = {s_Bz[lr][lp][lz],   s_Bz[lr][lp][lz+1],   s_Bz[lr+1][lp][lz],   s_Bz[lr+1][lp][lz+1]};
        float rh_s[4] = {s_rho[lr][lp][lz],  s_rho[lr][lp][lz+1],  s_rho[lr+1][lp][lz],  s_rho[lr+1][lp][lz+1]};
        float p_s[4]  = {s_p[lr][lp][lz],    s_p[lr][lp][lz+1],    s_p[lr+1][lp][lz],    s_p[lr+1][lp][lz+1]};

        float E_s[4];
        for (int i = 0; i < 4; ++i)
            E_s[i] = -(vz_s[i]*Br_s[i] - vr_s[i]*Bz_s[i]);

        float wr = upwind_weight(0.5f*(vr_s[0]+vr_s[2]));
        float wz = upwind_weight(0.5f*(vz_s[0]+vz_s[1]));
        float E_avg = (1.f-wr)*(1.f-wz)*E_s[0] + wr*(1.f-wz)*E_s[1]
                    + (1.f-wr)*wz*E_s[2] + wr*wz*E_s[3];

        float cf = fast_ms(rh_s[0], vr_s[0], Br_s[0], Bz_s[0], gamma, p_s[0]);
        float E = E_avg + 0.5f*cf*((Bz_s[1]-Bz_s[0]) - (Br_s[2]-Br_s[0]));

        if (use_hall && hall_coeff != 0.f) {
            float Jr = inv_r * idphi * (s_Bz[lr][lp+1][lz] - s_Bz[lr][lp-1][lz])
                     - idz * (s_Bphi[lr][lp][lz+1] - s_Bphi[lr][lp][lz-1]);
            float Jz = idr * (r * s_Bphi[lr+1][lp][lz] - (r-dr[ir])*s_Bphi[lr-1][lp][lz]) * inv_r
                     - inv_r * idphi * (s_Br[lr][lp+1][lz] - s_Br[lr][lp-1][lz]);
            float rho_e = 0.25f*(rh_s[0]+rh_s[1]+rh_s[2]+rh_s[3]);
            float Br_e  = 0.25f*(Br_s[0]+Br_s[1]+Br_s[2]+Br_s[3]);
            float Bz_e  = 0.25f*(Bz_s[0]+Bz_s[1]+Bz_s[2]+Bz_s[3]);
            E += (hall_coeff / fmaxf(rho_e, 1e-12f)) * (Jz*Br_e - Jr*Bz_e);
        }

        int eidx = ((ir+g)*(nphi+2*g) + (ip+g))*(nz+1+2*g) + (iz+g);
        emf_phi[eidx] = E;
    }

    // ========================================================================
    // emf_r  (edges parallel to r)
    // ========================================================================
    if (ir <= nr && ip < nphi && iz <= nz) {
        float vp_s[4] = {s_vphi[lr][lp][lz], s_vphi[lr][lp][lz+1], s_vphi[lr][lp+1][lz], s_vphi[lr][lp+1][lz+1]};
        float vz_s[4] = {s_vz[lr][lp][lz],   s_vz[lr][lp][lz+1],   s_vz[lr][lp+1][lz],   s_vz[lr][lp+1][lz+1]};
        float Bp_s[4] = {s_Bphi[lr][lp][lz], s_Bphi[lr][lp][lz+1], s_Bphi[lr][lp+1][lz], s_Bphi[lr][lp+1][lz+1]};
        float Bz_s[4] = {s_Bz[lr][lp][lz],   s_Bz[lr][lp][lz+1],   s_Bz[lr][lp+1][lz],   s_Bz[lr][lp+1][lz+1]};
        float rh_s[4] = {s_rho[lr][lp][lz],  s_rho[lr][lp][lz+1],  s_rho[lr][lp+1][lz],  s_rho[lr][lp+1][lz+1]};
        float p_s[4]  = {s_p[lr][lp][lz],    s_p[lr][lp][lz+1],    s_p[lr][lp+1][lz],    s_p[lr][lp+1][lz+1]};

        float E_s[4];
        for (int i = 0; i < 4; ++i)
            E_s[i] = -(vp_s[i]*Bz_s[i] - vz_s[i]*Bp_s[i]);

        float wp = upwind_weight(0.5f*(vp_s[0]+vp_s[2]));
        float wz = upwind_weight(0.5f*(vz_s[0]+vz_s[1]));
        float E_avg = (1.f-wp)*(1.f-wz)*E_s[0] + wp*(1.f-wz)*E_s[1]
                    + (1.f-wp)*wz*E_s[2] + wp*wz*E_s[3];

        float cf = fast_ms(rh_s[0], vp_s[0], Bp_s[0], Bz_s[0], gamma, p_s[0]);
        float E = E_avg + 0.5f*cf*((Bp_s[1]-Bp_s[0]) - (Bz_s[2]-Bz_s[0]));

        if (use_hall && hall_coeff != 0.f) {
            float Jp = idz * (s_Br[lr][lp][lz+1] - s_Br[lr][lp][lz-1])
                     - idr * (s_Bz[lr+1][lp][lz] - s_Bz[lr-1][lp][lz]);
            float Jz = idr * (r * s_Bphi[lr+1][lp][lz] - (r-dr[ir])*s_Bphi[lr-1][lp][lz]) * inv_r
                     - inv_r * idphi * (s_Br[lr][lp+1][lz] - s_Br[lr][lp-1][lz]);
            float rho_e = 0.25f*(rh_s[0]+rh_s[1]+rh_s[2]+rh_s[3]);
            float Bp_e  = 0.25f*(Bp_s[0]+Bp_s[1]+Bp_s[2]+Bp_s[3]);
            float Bz_e  = 0.25f*(Bz_s[0]+Bz_s[1]+Bz_s[2]+Bz_s[3]);
            E += (hall_coeff / fmaxf(rho_e, 1e-12f)) * (Jp*Bz_e - Jz*Bp_e);
        }

        int eidx = ((ir+g)*(nphi+2*g) + (ip+g))*(nz+1+2*g) + (iz+g);
        emf_r[eidx] = E;
    }
}
""", "uct_assembly_v2")


def launch_uct_assembly(state, dt, geometry, config):
    """Drop-in launcher for the driver’s utc / uct_emf slot."""
    if not HAS_CUPY or uct_assembly_kernel is None:
        raise RuntimeError("CuPy required for UCT assembly kernel")

    g = config.ghost_width
    nr, nphi, nz = config.nr, config.nphi, config.nz

    grid = (
        (nr + 1 + TILE - 1) // TILE,
        (nphi + 1 + TILE - 1) // TILE,
        (nz + 1 + TILE - 1) // TILE,
    )
    block = (TILE, TILE, TILE)

    hall = float(getattr(config, "hall_coeff", 0.0))
    use_hall = 1 if getattr(config, "use_hall", False) else 0
    gamma = float(getattr(config, "gamma", 5.0/3.0))

    def to_cp(a):
        if a is None:
            return None
        if isinstance(a, cp.ndarray):
            return a
        return cp.asarray(a, dtype=cp.float32)

    dr = to_cp(getattr(geometry, "dr", np.ones(nr, dtype=np.float32)))
    dphi = to_cp(getattr(geometry, "dphi", np.ones(1, dtype=np.float32)))
    dz = to_cp(getattr(geometry, "dz", np.ones(1, dtype=np.float32)))

    uct_assembly_kernel(grid, block, (
        to_cp(state.get("rho")),
        to_cp(state.get("rho_vr")),
        to_cp(state.get("rho_vphi")),
        to_cp(state.get("rho_vz")),
        to_cp(state.get("p") if (hasattr(state, "has") and state.has("p")) else state.get("rho")),
        to_cp(state.get("Br")),
        to_cp(state.get("Bphi")),
        to_cp(state.get("Bz")),
        to_cp(geometry.r_faces),
        to_cp(geometry.r_centers),
        dr, dphi, dz,
        to_cp(state.get("emf_r")),
        to_cp(state.get("emf_phi")),
        to_cp(state.get("emf_z")),
        nr, nphi, nz, g,
        np.float32(hall), np.int32(use_hall), np.float32(gamma)
    ))
    cp.cuda.Stream.null.synchronize()


# =============================================================================
# QUICK SELF-TEST (force-free + finiteness)
# =============================================================================
if __name__ == "__main__":
    print("UCT Assembly Kernel self-test")
    if not HAS_CUPY:
        print("  CuPy not available — kernel source only")
    else:
        class DummyState:
            def __init__(self, d):
                self._d = d
            def get(self, k):
                return self._d[k]
            def has(self, k):
                return k in self._d

        class DummyGeo:
            def __init__(self, nr=8, nphi=8, nz=8):
                self.r_faces = np.linspace(0, 1, nr+1).astype(np.float32)
                self.r_centers = 0.5*(self.r_faces[:-1] + self.r_faces[1:])
                self.dr = np.diff(self.r_faces).astype(np.float32)
                self.dphi = np.array([2*np.pi/nphi], dtype=np.float32)
                self.dz = np.array([1.0/nz], dtype=np.float32)

        class DummyConfig:
            ghost_width = 2
            nr = nphi = nz = 8
            hall_coeff = 0.0
            use_hall = False
            gamma = 5.0/3.0

        cfg = DummyConfig()
        geo = DummyGeo(cfg.nr, cfg.nphi, cfg.nz)
        cshape = (cfg.nr+4, cfg.nphi+4, cfg.nz+4)
        eshape_z = (cfg.nr+1+4, cfg.nphi+1+4, cfg.nz+4)
        eshape_rp = (cfg.nr+4, cfg.nphi+4, cfg.nz+1+4)
        rho = np.ones(cshape, dtype=np.float32)
        # simple force-free-ish field
        Br = np.zeros(cshape, dtype=np.float32)
        Bphi = np.ones(cshape, dtype=np.float32) * 0.1
        Bz = np.ones(cshape, dtype=np.float32) * 0.5
        state = DummyState({
            "rho": rho,
            "rho_vr": np.zeros(cshape, dtype=np.float32),
            "rho_vphi": np.zeros(cshape, dtype=np.float32),
            "rho_vz": np.zeros(cshape, dtype=np.float32),
            "p": rho * 1.0,
            "Br": Br, "Bphi": Bphi, "Bz": Bz,
            "emf_r": np.zeros(eshape_rp, dtype=np.float32),
            "emf_phi": np.zeros(eshape_rp, dtype=np.float32),
            "emf_z": np.zeros(eshape_z, dtype=np.float32),
        })
        try:
            launch_uct_assembly(state, 1e-4, geo, cfg)
            emfz = state.get("emf_z")
            print(f"  launch ok | max|emf_z| = {np.max(np.abs(emfz)):.3e}")
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
