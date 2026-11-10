"""
Minimal diagnostic — run this alone after uploading the kernel.
Prints residual, status, and whether the warm-start path is live.
"""
import cupy as cp
from gyrotropic_cylindrical_kernel import launch_gyro, compute_base_moments

print("diagnostic start")
M = compute_base_moments(0.5, 0.5)
print("target M[0:4] =", M[:4])

N = 4
Mt = cp.tile(cp.asarray(M), (N, 1)).astype(cp.float32)
n_in = cp.ones(N, dtype=cp.float32)
lam = cp.zeros((N, 12), dtype=cp.float32)
lam[:, 6] = lam[:, 8] = lam[:, 10] = -0.05

outs = launch_gyro(Mt, n_in, lam)
status = outs[-1].get()
Mrec = outs[7].get()[0]
scale = cp.asarray([1,1,1,1,1,1,3,2,3,1,4,3], dtype=cp.float32).get()
scaled = abs(Mrec - M) / scale
print("status =", status)
print("max scaled residual =", float(scaled.max()))
print("Ppar, Pperp =", outs[2].get()[0], outs[3].get()[0])
print("margin =", outs[6].get()[0])
print("done")
