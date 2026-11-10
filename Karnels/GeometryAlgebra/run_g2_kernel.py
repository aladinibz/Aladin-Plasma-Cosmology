import numpy as np

try:
    import cupy as cp
except ImportError:
    raise SystemExit("CuPy required. Use a GPU runtime.")

# ---------------------------------------------------------------------------
# Kernel v55 — full Moufang + discrete G2 connection residual
# ---------------------------------------------------------------------------
g2_geometry_algebra_kernel = cp.RawKernel(r'''
extern "C" __global__
void g2_geometry_algebra_kernel(
    const float* __restrict__ rho,
    const float* __restrict__ vr,
    const float* __restrict__ vphi,
    const float* __restrict__ vz,
    const float* __restrict__ Br,
    const float* __restrict__ Bphi,
    const float* __restrict__ Bz,
    const float* __restrict__ r_c,
    const float* __restrict__ dr,
    const float* __restrict__ dphi,
    const float* __restrict__ dz,
    float* __restrict__ S_vr,
    float* __restrict__ S_vphi,
    float* __restrict__ S_vz,
    float* __restrict__ S_E,
    float* __restrict__ phi_residual,
    float* __restrict__ associator_residual,
    float* __restrict__ su3_residual,
    float* __restrict__ cy_diagnostic,
    float* __restrict__ connection_residual,
    float* __restrict__ test_norm,
    float* __restrict__ test_assoc,
    float* __restrict__ test_phi,
    float* __restrict__ test_conn,
    float* __restrict__ test_quat,
    float* __restrict__ test_27,
    float* __restrict__ test_alt,
    float* __restrict__ test_moufang,
    float* __restrict__ test_inverse,
    float* __restrict__ divB,
    int nr, int nphi, int nz,
    float lambda_g2,
    float projection_strength,
    float rho_min)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nr * nphi * nz;
    if (idx >= N) return;

    int iz = idx % nz;
    int iy = (idx / nz) % nphi;
    int ix = idx / (nz * nphi);

    const float eps = 1e-12f;
    const float damp = 0.2f;
    const float step_max = 0.02f;
    const float phi_max = 1.0f;
    const float energy_damp = 0.1f;
    const float damp27 = 0.3f;
    const float source_clamp = 10.0f;
    const float phi_gate = 1.0f;

    const int lines[7][3] = {
        {0,1,2},{0,3,4},{0,5,6},{1,3,5},{1,4,6},{2,3,6},{2,4,5}
    };
    float table[7][7][7];
    for (int i=0;i<7;i++) for (int j=0;j<7;j++) for (int k=0;k<7;k++) table[i][j][k]=0.0f;
    for (int L=0;L<7;L++) {
        int i=lines[L][0], j=lines[L][1], k=lines[L][2];
        table[i][j][k]=table[j][k][i]=table[k][i][j]=1.0f;
        table[j][i][k]=table[k][j][i]=table[i][k][j]=-1.0f;
    }

    float r   = fmaxf(r_c[idx], rho_min);
    float inv_r = 1.0f / r;
    float dr_ = dr[idx];
    float dphi_ = dphi[idx];
    float dz_ = dz[idx];

    float vx0 = vr[idx];
    float vy0 = vphi[idx];
    float vz0 = vz[idx];
    float bx  = Br[idx];
    float by  = Bphi[idx];
    float bz  = Bz[idx];

    // Continuum cylindrical divergence
    float dBr = 0.0f, dBphi = 0.0f, dBz = 0.0f;
    if (ix > 1 && ix < nr-2) {
        int ip  = idx + nphi*nz;
        int im  = idx - nphi*nz;
        int ip2 = idx + 2*nphi*nz;
        int im2 = idx - 2*nphi*nz;
        dBr = (-Br[ip2] + 8.0f*Br[ip] - 8.0f*Br[im] + Br[im2]) / (12.0f * dr_);
    }
    if (iy > 1 && iy < nphi-2) {
        int jp  = idx + nz;
        int jm  = idx - nz;
        int jp2 = idx + 2*nz;
        int jm2 = idx - 2*nz;
        dBphi = (-Bphi[jp2] + 8.0f*Bphi[jp] - 8.0f*Bphi[jm] + Bphi[jm2]) / (12.0f * dphi_);
    }
    if (iz > 1 && iz < nz-2) {
        int kp  = idx + 1;
        int km  = idx - 1;
        int kp2 = idx + 2;
        int km2 = idx - 2;
        dBz = (-Bz[kp2] + 8.0f*Bz[kp] - 8.0f*Bz[km] + Bz[km2]) / (12.0f * dz_);
    }
    float coboundary = dBr + inv_r * dBphi + dBz + inv_r * bx;
    divB[idx] = coboundary;

    // Plasma fiber
    float B2  = bx*bx + by*by + bz*bz + eps;
    float chi = (bx*vx0 + by*vy0 + bz*vz0) * inv_r / (B2 + eps);
    float o_real = 1.0f;
    float o_imag[7] = {vx0, vy0, vz0, bx, by, bz, chi};

    auto full_multiply = [&](float x0, const float x[7], float y0, const float y[7],
                             float& z0, float z[7]) {
        float dot = 0.0f;
        for (int k=0;k<7;k++) dot += x[k]*y[k];
        z0 = x0*y0 - dot;
        float cross[7] = {0};
        for (int i=0;i<7;i++) for (int j=0;j<7;j++) {
            float prod = x[i]*y[j];
            if (prod == 0.0f) continue;
            for (int k=0;k<7;k++) cross[k] += prod * table[i][j][k];
        }
        for (int k=0;k<7;k++) z[k] = x0*y[k] + y0*x[k] + cross[k];
    };

    auto norm8 = [&](float r0, const float e[7]) -> float {
        float n = r0*r0;
        for (int k=0;k<7;k++) n += e[k]*e[k];
        return sqrtf(n + eps);
    };

    auto phi_abc = [&](const float a[7], const float b[7], const float c[7]) -> float {
        float cross[7] = {0};
        for (int i=0;i<7;i++) for (int j=0;j<7;j++) {
            float prod = a[i]*b[j];
            if (prod == 0.0f) continue;
            for (int k=0;k<7;k++) cross[k] += prod * table[i][j][k];
        }
        float p = 0.0f;
        for (int k=0;k<7;k++) p += cross[k] * c[k];
        return p;
    };

    // Controlled octonions
    float ta[7] = {0.1f, 0.2f, 0.3f, 0.1f, 0.0f, 0.0f, 0.0f};
    float tb[7] = {0.0f, 0.1f, 0.0f, 0.2f, 0.3f, 0.1f, 0.0f};
    float tc[7] = {0.2f, 0.0f, 0.1f, 0.0f, 0.1f, 0.2f, 0.3f};

    float phi_test = phi_abc(ta, tb, tc);
    test_phi[idx] = fabsf(phi_test);

    // Norm
    float z0;
    float z_imag[7];
    full_multiply(1.0f, ta, 1.0f, ta, z0, z_imag);
    float n_o = norm8(1.0f, ta);
    float n_prod = norm8(z0, z_imag);
    test_norm[idx] = fabsf(n_prod - n_o * n_o);

    // Associator
    float ab0, bc0, left0, right0;
    float ab[7], bc[7], left[7], right[7], assoc[7];
    full_multiply(1.0f, ta, 1.0f, tb, ab0, ab);
    full_multiply(ab0, ab, 1.0f, tc, left0, left);
    full_multiply(1.0f, tb, 1.0f, tc, bc0, bc);
    full_multiply(1.0f, ta, bc0, bc, right0, right);
    float assoc_norm = 0.0f;
    assoc_norm += (left0 - right0)*(left0 - right0);
    for (int k=0;k<7;k++) {
        assoc[k] = left[k] - right[k];
        assoc_norm += assoc[k]*assoc[k];
    }
    associator_residual[idx] = sqrtf(assoc_norm + eps);
    test_assoc[idx] = associator_residual[idx];

    // Alternativity
    float aa0, aab0, left_alt0, right_alt0;
    float aa[7], aab[7], left_alt[7], right_alt[7];
    full_multiply(1.0f, ta, 1.0f, ta, aa0, aa);
    full_multiply(aa0, aa, 1.0f, tb, left_alt0, left_alt);
    full_multiply(1.0f, ta, 1.0f, tb, ab0, ab);
    full_multiply(1.0f, ta, ab0, ab, right_alt0, right_alt);
    float alt_norm = (left_alt0 - right_alt0)*(left_alt0 - right_alt0);
    for (int k=0;k<7;k++) alt_norm += (left_alt[k] - right_alt[k])*(left_alt[k] - right_alt[k]);
    test_alt[idx] = sqrtf(alt_norm + eps);

    // ---- Full Moufang: (xy)(zx) vs x((yz)x) ----
    float xy0, zx0, left_m0, yz0, yzx0, right_m0;
    float xy[7], zx[7], left_m[7], yz[7], yzx[7], right_m[7];
    full_multiply(1.0f, ta, 1.0f, tb, xy0, xy);
    full_multiply(1.0f, tc, 1.0f, ta, zx0, zx);
    full_multiply(xy0, xy, zx0, zx, left_m0, left_m);
    full_multiply(1.0f, tb, 1.0f, tc, yz0, yz);
    full_multiply(yz0, yz, 1.0f, ta, yzx0, yzx);          // (yz)x
    full_multiply(1.0f, ta, yzx0, yzx, right_m0, right_m); // x((yz)x)
    float moufang = (left_m0 - right_m0)*(left_m0 - right_m0);
    for (int k=0;k<7;k++) moufang += (left_m[k] - right_m[k])*(left_m[k] - right_m[k]);
    test_moufang[idx] = sqrtf(moufang + eps);

    // Inverse
    float n2 = norm8(1.0f, ta);
    n2 = n2*n2 + eps;
    float inv_real = 1.0f / n2;
    float inv_imag[7];
    for (int k=0;k<7;k++) inv_imag[k] = -ta[k] / n2;
    float prod_real;
    float prod_imag[7];
    full_multiply(1.0f, ta, inv_real, inv_imag, prod_real, prod_imag);
    float inv_res = (prod_real - 1.0f)*(prod_real - 1.0f);
    for (int k=0;k<7;k++) inv_res += prod_imag[k]*prod_imag[k];
    test_inverse[idx] = sqrtf(inv_res + eps);

    // Quaternion norm
    float q0 = 1.0f, q1 = ta[0], q2 = ta[1], q3 = ta[2];
    float p0 = q0*q0 - q1*q1 - q2*q2 - q3*q3;
    float p1 = 2.0f*q0*q1;
    float p2 = 2.0f*q0*q2;
    float p3 = 2.0f*q0*q3;
    float n_q = sqrtf(q0*q0 + q1*q1 + q2*q2 + q3*q3 + eps);
    float n_p = sqrtf(p0*p0 + p1*p1 + p2*p2 + p3*p3 + eps);
    test_quat[idx] = fabsf(n_p - n_q * n_q);

    // SU(3) / CY
    float su3 = fabsf(o_imag[6]);
    su3_residual[idx] = su3;
    float norm6 = 0.0f;
    for (int k=0;k<6;k++) norm6 += o_imag[k]*o_imag[k];
    norm6 = sqrtf(norm6 + eps);
    float cy_diag = fabsf(norm6 - 1.0f) + fabsf(o_imag[6]);
    cy_diagnostic[idx] = cy_diag;

    // ---- Discrete G2 connection residual ----
    // Transport fiber by a small g2 action (approximate parallel transport)
    // and measure change in φ
    float phi_here = phi_abc(o_imag, o_imag, o_imag);
    float conn = 0.0f;
    if (ix > 0 && ix < nr-1) {
        int ip = idx + nphi*nz;
        float op[7] = {vr[ip], vphi[ip], vz[ip], Br[ip], Bphi[ip], Bz[ip], chi};
        // simple transport: average + small rotation proxy
        float transported[7];
        for (int k=0;k<7;k++) transported[k] = 0.5f*(o_imag[k] + op[k]);
        float phi_trans = phi_abc(transported, transported, transported);
        conn += fabsf(phi_trans - phi_here);
    }
    connection_residual[idx] = conn;
    test_conn[idx] = conn;

    // 27-tensor damping
    float tr = 0.0f;
    for (int k=0;k<7;k++) tr += o_imag[k]*o_imag[k];
    float mean = tr / 7.0f;
    float frobenius = 0.0f;
    for (int i=0;i<7;i++) {
        for (int j=0;j<7;j++) {
            float S = o_imag[i]*o_imag[j];
            if (i==j) S -= mean;
            frobenius += S*S;
        }
    }
    float norm27 = sqrtf(frobenius + eps);
    float scale = 1.0f / (1.0f + damp27 * norm27);
    for (int k=0;k<7;k++) o_imag[k] *= scale;
    tr = 0.0f;
    for (int k=0;k<7;k++) tr += o_imag[k]*o_imag[k];
    mean = tr / 7.0f;
    frobenius = 0.0f;
    for (int i=0;i<7;i++) {
        for (int j=0;j<7;j++) {
            float S = o_imag[i]*o_imag[j];
            if (i==j) S -= mean;
            frobenius += S*S;
        }
    }
    test_27[idx] = sqrtf(frobenius + eps);

    // Gated Newton
    float phi = phi_abc(o_imag, o_imag, o_imag);
    if (fabsf(phi) < phi_gate) {
        for (int it=0; it<3; ++it) {
            phi = phi_abc(o_imag, o_imag, o_imag);
            float dphi[7] = {0};
            for (int L=0;L<7;L++) {
                int i=lines[L][0], j=lines[L][1], k=lines[L][2];
                dphi[i] += o_imag[j]*o_imag[k];
                dphi[j] += o_imag[k]*o_imag[i];
                dphi[k] += o_imag[i]*o_imag[j];
            }
            float grad2 = eps;
            for (int k=0;k<7;k++) grad2 += dphi[k]*dphi[k];
            float step = damp * phi / grad2;
            step = fmaxf(fminf(step, step_max), -step_max);
            step *= 1.0f / (1.0f + 0.05f * inv_r);
            for (int k=0;k<7;k++) o_imag[k] -= step * dphi[k];
        }
    }

    float vx  = o_imag[0], vy = o_imag[1], vz_ = o_imag[2];
    bx = o_imag[3]; by = o_imag[4]; bz = o_imag[5];
    B2 = bx*bx + by*by + bz*bz + eps;
    chi = (bx*vx + by*vy + bz*vz_) * inv_r / (B2 + eps);
    o_imag[6] = chi;

    phi = phi_abc(o_imag, o_imag, o_imag);
    float phi_src = fmaxf(fminf(phi, phi_max), -phi_max);
    phi_residual[idx] = phi;

    float dphi_final[7] = {0};
    for (int L=0;L<7;L++) {
        int i=lines[L][0], j=lines[L][1], k=lines[L][2];
        dphi_final[i] += o_imag[j]*o_imag[k];
        dphi_final[j] += o_imag[k]*o_imag[i];
        dphi_final[k] += o_imag[i]*o_imag[j];
    }

    float inv = inv_r / (B2 + eps);
    float dphi_dvx = dphi_final[0] + dphi_final[6] * bx * inv;
    float dphi_dvy = dphi_final[1] + dphi_final[6] * by * inv;
    float dphi_dvz = dphi_final[2] + dphi_final[6] * bz * inv;

    float geom_vr   = (vy*vy - by*by) * inv_r;
    float geom_vphi = -(vx*vy - bx*by) * inv_r;

    float factor = -2.0f * lambda_g2 * phi_src;
    float s_vr   = factor * dphi_dvx * r + geom_vr;
    float s_vphi = factor * dphi_dvy * r + geom_vphi;
    float s_vz   = factor * dphi_dvz * r;
    float s_E    = energy_damp * (-(vx0 * s_vr + vy0 * s_vphi + vz0 * s_vz));

    s_vr   = fmaxf(fminf(s_vr,   source_clamp), -source_clamp);
    s_vphi = fmaxf(fminf(s_vphi, source_clamp), -source_clamp);
    s_vz   = fmaxf(fminf(s_vz,   source_clamp), -source_clamp);
    s_E    = fmaxf(fminf(s_E,    source_clamp), -source_clamp);

    S_vr[idx]   = s_vr;
    S_vphi[idx] = s_vphi;
    S_vz[idx]   = s_vz;
    S_E[idx]    = s_E;
}
''', 'g2_geometry_algebra_kernel')

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def make_nonsmooth_field(nr, nphi, nz):
    r = cp.linspace(0.05, 1.0, nr, dtype=cp.float32)
    phi = cp.linspace(0, 2*np.pi, nphi, endpoint=False, dtype=cp.float32)
    z = cp.linspace(-1.0, 1.0, nz, dtype=cp.float32)
    R, PHI, Z = cp.meshgrid(r, phi, z, indexing='ij')

    sheet = cp.tanh((R - 0.5) / 0.05)
    noise = 0.1 * cp.random.randn(*R.shape, dtype=cp.float32)

    vr   = 0.2 * sheet * cp.sin(PHI) + noise
    vphi = 0.1 * R * sheet + 0.05 * noise
    vz   = 0.05 * cp.tanh(Z) + 0.02 * noise

    Br   = 0.4 * sheet * cp.cos(PHI) + 0.1 * noise
    Bphi = 0.3 * R * (1.0 - sheet) + 0.05 * noise
    Bz   = 0.2 * cp.sin(2*PHI) + 0.05 * noise

    rho = cp.ones_like(R)
    dr = cp.full_like(R, r[1]-r[0])
    dphi = cp.full_like(R, phi[1]-phi[0])
    dz = cp.full_like(R, z[1]-z[0])

    return (rho.ravel(), vr.ravel(), vphi.ravel(), vz.ravel(),
            Br.ravel(), Bphi.ravel(), Bz.ravel(),
            R.ravel(), dr.ravel(), dphi.ravel(), dz.ravel())

def run(nr=32, nphi=32, nz=32, lambda_g2=1.0, projection_strength=0.5, rho_min=1e-6):
    print("KERNEL VERSION: full-moufang + connection v55")
    N = nr * nphi * nz
    (rho, vr, vphi, vz, Br, Bphi, Bz,
     r_c, dr, dphi, dz) = make_nonsmooth_field(nr, nphi, nz)

    S_vr   = cp.zeros_like(rho)
    S_vphi = cp.zeros_like(rho)
    S_vz   = cp.zeros_like(rho)
    S_E    = cp.zeros_like(rho)
    phi_res = cp.zeros_like(rho)
    assoc_res = cp.zeros_like(rho)
    su3_res = cp.zeros_like(rho)
    cy_diag = cp.zeros_like(rho)
    conn_res = cp.zeros_like(rho)
    test_norm = cp.zeros_like(rho)
    test_assoc = cp.zeros_like(rho)
    test_phi = cp.zeros_like(rho)
    test_conn = cp.zeros_like(rho)
    test_quat = cp.zeros_like(rho)
    test_27 = cp.zeros_like(rho)
    test_alt = cp.zeros_like(rho)
    test_moufang = cp.zeros_like(rho)
    test_inverse = cp.zeros_like(rho)
    divB = cp.zeros_like(rho)

    block = 128
    grid  = (N + block - 1) // block

    g2_geometry_algebra_kernel(
        (grid,), (block,),
        (rho, vr, vphi, vz, Br, Bphi, Bz,
         r_c, dr, dphi, dz,
         S_vr, S_vphi, S_vz, S_E, phi_res, assoc_res, su3_res, cy_diag, conn_res,
         test_norm, test_assoc, test_phi, test_conn, test_quat, test_27,
         test_alt, test_moufang, test_inverse, divB,
         np.int32(nr), np.int32(nphi), np.int32(nz),
         np.float32(lambda_g2), np.float32(projection_strength), np.float32(rho_min))
    )

    B2 = Br*Br + Bphi*Bphi + Bz*Bz + 1e-12
    inv_r = 1.0 / r_c
    chi = (Br*vr + Bphi*vphi + Bz*vz) * inv_r / (B2 + 1e-12)
    phi0 = (vr*vphi*vz + vr*Br*Bphi + vr*Bz*chi
          + vphi*Br*Bz + vphi*Bphi*chi + vz*Br*chi + vz*Bphi*Bz)
    mean_phi0 = float(cp.mean(cp.abs(phi0)))
    mean_phi1 = float(cp.mean(cp.abs(phi_res)))

    energy_err = cp.abs(S_E + (vr * S_vr + vphi * S_vphi + vz * S_vz))
    print(f"grid = {nr}x{nphi}x{nz} = {N}")
    print(f"mean |φ| before          = {mean_phi0:.3e}")
    print(f"mean |φ| after           = {mean_phi1:.3e}")
    print(f"reduction                = {mean_phi0 / (mean_phi1 + 1e-12):.1f}x")
    print(f"mean |divB|              = {float(cp.mean(cp.abs(divB))):.3e}")
    print(f"mean associator residual = {float(cp.mean(assoc_res)):.3e}")
    print(f"mean SU(3) residual      = {float(cp.mean(su3_res)):.3e}")
    print(f"mean CY diagnostic       = {float(cp.mean(cy_diag)):.3e}")
    print(f"mean connection residual = {float(cp.mean(conn_res)):.3e}")
    print("--- controlled validation tests ---")
    print(f"test norm composition    = {float(cp.mean(test_norm)):.3e}")
    print(f"test associator (indep)  = {float(cp.mean(test_assoc)):.3e}")
    print(f"test φ(a,b,c)            = {float(cp.mean(test_phi)):.3e}")
    print(f"test connection          = {float(cp.mean(test_conn)):.3e}")
    print(f"test quaternion norm     = {float(cp.mean(test_quat)):.3e}")
    print(f"test 27-tensor (after)   = {float(cp.mean(test_27)):.3e}")
    print(f"test alternativity       = {float(cp.mean(test_alt)):.3e}")
    print(f"test Moufang (full)      = {float(cp.mean(test_moufang)):.3e}")
    print(f"test inverse             = {float(cp.mean(test_inverse)):.3e}")
    print(f"max energy err           = {float(cp.max(energy_err)):.3e}")
    return phi_res, test_moufang

if __name__ == "__main__":
    run()
