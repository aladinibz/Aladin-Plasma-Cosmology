"""
ALADIN EGLM — Velocity Verlet + single EGLM source application
"""

import cupy as cp
import numpy as np

KERNEL_SRC = r'''
extern "C" {

__global__ void eglm_psi_and_divB(
    const float* __restrict__ Br, const float* __restrict__ Bphi, const float* __restrict__ Bz,
    float* __restrict__ psi, float* __restrict__ divB,
    const float* __restrict__ r_faces, const float* __restrict__ r_centers,
    float dr, float dphi, float dz, int nr, int nphi, int nz, int g,
    float ch, float dt
) {
    int ir = blockIdx.x*blockDim.x + threadIdx.x;
    int iphi = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if (ir>=nr || iphi>=nphi || iz>=nz) return;

    int nphi_s = nphi+2*g, nz_s = nz+2*g;
    int nphi_f = nphi+1+2*g, nz_f = nz+1+2*g;
    int irg=ir+g, iphig=iphi+g, izg=iz+g;
    int ic = (irg*nphi_s + iphig)*nz_s + izg;

    float rc = r_centers[ir];
    float inv_rc = (rc>1e-8f)?1.0f/rc:0.0f;

    int Br_lo = (irg*nphi_s + iphig)*nz_s + izg;
    int Br_hi = ((irg+1)*nphi_s + iphig)*nz_s + izg;
    float rlo=r_faces[ir], rhi=r_faces[ir+1];
    float dBr = (rhi*Br[Br_hi] - rlo*Br[Br_lo]) / (dr * fmaxf(rc,1e-8f));

    int Bp_lo = (irg*nphi_f + iphig)*nz_s + izg;
    int Bp_hi = (irg*nphi_f + iphig+1)*nz_s + izg;
    float dBp = (Bphi[Bp_hi]-Bphi[Bp_lo])*inv_rc / dphi;

    int Bz_lo = (irg*nphi_s + iphig)*nz_f + izg;
    int Bz_hi = (irg*nphi_s + iphig)*nz_f + izg+1;
    float dBz = (Bz[Bz_hi]-Bz[Bz_lo]) / dz;

    float db = dBr + dBp + dBz;
    divB[ic] = db;
    if (dt > 0.0f) psi[ic] -= dt * ch*ch * db;
}

__global__ void eglm_face_only(
    const float* __restrict__ psi,
    float* __restrict__ dBr, float* __restrict__ dBphi, float* __restrict__ dBz,
    const float* __restrict__ r_centers,
    float dr, float dphi, float dz, int nr, int nphi, int nz, int g, float dt
) {
    int ir = blockIdx.x*blockDim.x + threadIdx.x;
    int iphi = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if (ir>=nr || iphi>=nphi || iz>=nz) return;

    int nphi_s=nphi+2*g, nz_s=nz+2*g, nphi_f=nphi+1+2*g, nz_f=nz+1+2*g;
    int irg=ir+g, iphig=iphi+g, izg=iz+g;
    int ic=(irg*nphi_s+iphig)*nz_s+izg;
    float rc=r_centers[ir];
    float inv_rc=(rc>1e-8f)?1.0f/rc:0.0f;

    int Br_hi=((irg+1)*nphi_s+iphig)*nz_s+izg;
    int ic_rp=((irg+1)*nphi_s+iphig)*nz_s+izg;
    float dpsi_r=psi[ic_rp]-psi[ic];
    dBr[Br_hi]=(fabsf(dpsi_r)>1e-12f)?-dt*dpsi_r/dr:0.0f;

    int Bp_hi=(irg*nphi_f+iphig+1)*nz_s+izg;
    int ic_pp=(irg*nphi_s+iphig+1)*nz_s+izg;
    float dpsi_p=psi[ic_pp]-psi[ic];
    dBphi[Bp_hi]=(fabsf(dpsi_p)>1e-12f)?-dt*dpsi_p/dphi*inv_rc:0.0f;

    int Bz_hi=(irg*nphi_s+iphig)*nz_f+izg+1;
    int ic_zp=(irg*nphi_s+iphig)*nz_s+izg+1;
    float dpsi_z=psi[ic_zp]-psi[ic];
    dBz[Bz_hi]=(fabsf(dpsi_z)>1e-12f)?-dt*dpsi_z/dz:0.0f;
}

__global__ void eglm_sources_only(
    const float* __restrict__ psi, const float* __restrict__ Br,
    const float* __restrict__ Bphi, const float* __restrict__ Bz,
    const float* __restrict__ divB,
    float* __restrict__ dMom_r, float* __restrict__ dMom_phi,
    float* __restrict__ dMom_z, float* __restrict__ dEnergy,
    const float* __restrict__ r_centers,
    float dr, float dphi, float dz, int nr, int nphi, int nz, int g, float dt
) {
    int ir = blockIdx.x*blockDim.x + threadIdx.x;
    int iphi = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if (ir>=nr || iphi>=nphi || iz>=nz) return;

    int nphi_s=nphi+2*g, nz_s=nz+2*g, nphi_f=nphi+1+2*g, nz_f=nz+1+2*g;
    int irg=ir+g, iphig=iphi+g, izg=iz+g;
    int ic=(irg*nphi_s+iphig)*nz_s+izg;
    float rc=r_centers[ir];
    float inv_rc=(rc>1e-8f)?1.0f/rc:0.0f;

    int Br_lo=(irg*nphi_s+iphig)*nz_s+izg;
    int Br_hi=((irg+1)*nphi_s+iphig)*nz_s+izg;
    int Bp_lo=(irg*nphi_f+iphig)*nz_s+izg;
    int Bp_hi=(irg*nphi_f+iphig+1)*nz_s+izg;
    int Bz_lo=(irg*nphi_s+iphig)*nz_f+izg;
    int Bz_hi=(irg*nphi_s+iphig)*nz_f+izg+1;

    float Br_c=0.5f*(Br[Br_lo]+Br[Br_hi]);
    float Bphi_c=0.5f*(Bphi[Bp_lo]+Bphi[Bp_hi]);
    float Bz_c=0.5f*(Bz[Bz_lo]+Bz[Bz_hi]);
    float db=divB[ic];

    dMom_r[ic]   = -dt * db * Br_c;
    dMom_phi[ic] = -dt * db * Bphi_c;
    dMom_z[ic]   = -dt * db * Bz_c;

    int ic_rp=((irg+1)*nphi_s+iphig)*nz_s+izg;
    int ic_pp=(irg*nphi_s+iphig+1)*nz_s+izg;
    int ic_zp=(irg*nphi_s+iphig)*nz_s+izg+1;
    int ic_rm=((irg-1)*nphi_s+iphig)*nz_s+izg;
    int ic_pm=(irg*nphi_s+iphig-1)*nz_s+izg;
    int ic_zm=(irg*nphi_s+iphig)*nz_s+izg-1;

    float dpsi_dr=(psi[ic_rp]-psi[ic_rm])/(2.0f*dr);
    float dpsi_dphi=(psi[ic_pp]-psi[ic_pm])/(2.0f*dphi);
    float dpsi_dz=(psi[ic_zp]-psi[ic_zm])/(2.0f*dz);
    float BdotG = Br_c*dpsi_dr + Bphi_c*(dpsi_dphi*inv_rc) + Bz_c*dpsi_dz;
    dEnergy[ic] = -dt * BdotG;
}

__global__ void eglm_apply_faces(
    float* __restrict__ Br, float* __restrict__ Bphi, float* __restrict__ Bz,
    const float* __restrict__ dBr, const float* __restrict__ dBphi, const float* __restrict__ dBz,
    int nr, int nphi, int nz, int g
) {
    int ir = blockIdx.x*blockDim.x + threadIdx.x;
    int iphi = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if (ir>=nr || iphi>=nphi || iz>=nz) return;

    int nphi_s=nphi+2*g, nz_s=nz+2*g, nphi_f=nphi+1+2*g, nz_f=nz+1+2*g;
    int irg=ir+g, iphig=iphi+g, izg=iz+g;

    int Br_hi=((irg+1)*nphi_s+iphig)*nz_s+izg;
    int Bp_hi=(irg*nphi_f+iphig+1)*nz_s+izg;
    int Bz_hi=(irg*nphi_s+iphig)*nz_f+izg+1;

    Br[Br_hi]   += dBr[Br_hi];
    Bphi[Bp_hi] += dBphi[Bp_hi];
    Bz[Bz_hi]   += dBz[Bz_hi];
}

} // extern "C"
'''

mod = cp.RawModule(code=KERNEL_SRC)
k_psi    = mod.get_function('eglm_psi_and_divB')
k_face   = mod.get_function('eglm_face_only')
k_src    = mod.get_function('eglm_sources_only')
k_applyf = mod.get_function('eglm_apply_faces')


def run_eglm(Br, Bphi, Bz, psi, mom_r, mom_phi, mom_z, energy,
             r_faces, r_centers, dr, dphi, dz,
             nr, nphi, nz, g=2, ch=1.0, L=0.1, dt=1e-3):
    """Leapfrog on B-ψ, EGLM sources applied exactly once."""
    divB = cp.zeros_like(psi)
    dBr = cp.zeros_like(Br)
    dBphi = cp.zeros_like(Bphi)
    dBz = cp.zeros_like(Bz)
    dMom_r = cp.zeros_like(mom_r)
    dMom_phi = cp.zeros_like(mom_phi)
    dMom_z = cp.zeros_like(mom_z)
    dEnergy = cp.zeros_like(energy)

    threads = (8, 8, 4)
    blocks = ((nr+7)//8, (nphi+7)//8, (nz+3)//4)

    # 1. Initial divergence (no ψ update)
    k_psi(blocks, threads, (
        Br, Bphi, Bz, psi, divB, r_faces, r_centers,
        np.float32(dr), np.float32(dphi), np.float32(dz),
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g),
        np.float32(ch), np.float32(0.0)
    ))
    if dt == 0.0:
        return divB

    half = 0.5 * float(dt)

    # 2. First half-step on faces
    dBr.fill(0); dBphi.fill(0); dBz.fill(0)
    k_face(blocks, threads, (
        psi, dBr, dBphi, dBz, r_centers,
        np.float32(dr), np.float32(dphi), np.float32(dz),
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g), np.float32(half)
    ))
    k_applyf(blocks, threads, (
        Br, Bphi, Bz, dBr, dBphi, dBz,
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g)
    ))

    # 3. Full-step on ψ (once)
    k_psi(blocks, threads, (
        Br, Bphi, Bz, psi, divB, r_faces, r_centers,
        np.float32(dr), np.float32(dphi), np.float32(dz),
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g),
        np.float32(ch), np.float32(dt)
    ))

    # 4. Second half-step on faces
    dBr.fill(0); dBphi.fill(0); dBz.fill(0)
    k_face(blocks, threads, (
        psi, dBr, dBphi, dBz, r_centers,
        np.float32(dr), np.float32(dphi), np.float32(dz),
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g), np.float32(half)
    ))
    k_applyf(blocks, threads, (
        Br, Bphi, Bz, dBr, dBphi, dBz,
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g)
    ))

    # 5. Exponential damping
    if 0 < L < 1e6:
        rate = 1.0 / (L * L)
        psi *= np.exp(-rate * dt)

    # 6. EGLM sources exactly once (final fields)
    k_psi(blocks, threads, (
        Br, Bphi, Bz, psi, divB, r_faces, r_centers,
        np.float32(dr), np.float32(dphi), np.float32(dz),
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g),
        np.float32(ch), np.float32(0.0)
    ))
    k_src(blocks, threads, (
        psi, Br, Bphi, Bz, divB,
        dMom_r, dMom_phi, dMom_z, dEnergy, r_centers,
        np.float32(dr), np.float32(dphi), np.float32(dz),
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g), np.float32(dt)
    ))
    mom_r   += dMom_r
    mom_phi += dMom_phi
    mom_z   += dMom_z
    energy  += dEnergy

    return divB

def run_eglm_source_only(Br, Bphi, Bz, psi, mom_r, mom_phi, mom_z, energy,
                         r_faces, r_centers, dr, dphi, dz,
                         nr, nphi, nz, g=2, dt=1e-3):
    """Launch k_src exactly once. No leapfrog."""
    divB = cp.zeros_like(psi)
    dMom_r = cp.zeros_like(mom_r)
    dMom_phi = cp.zeros_like(mom_phi)
    dMom_z = cp.zeros_like(mom_z)
    dEnergy = cp.zeros_like(energy)

    threads = (8, 8, 4)
    blocks = ((nr+7)//8, (nphi+7)//8, (nz+3)//4)

    k_psi(blocks, threads, (
        Br, Bphi, Bz, psi, divB, r_faces, r_centers,
        np.float32(dr), np.float32(dphi), np.float32(dz),
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g),
        np.float32(1.0), np.float32(0.0)
    ))
    k_src(blocks, threads, (
        psi, Br, Bphi, Bz, divB,
        dMom_r, dMom_phi, dMom_z, dEnergy, r_centers,
        np.float32(dr), np.float32(dphi), np.float32(dz),
        np.int32(nr), np.int32(nphi), np.int32(nz), np.int32(g), np.float32(dt)
    ))
    mom_r   += dMom_r
    mom_phi += dMom_phi
    mom_z   += dMom_z
    energy  += dEnergy
    return divB

print("run_eglm_source_only available.")
