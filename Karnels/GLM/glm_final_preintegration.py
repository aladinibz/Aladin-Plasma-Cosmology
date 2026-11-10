"""
ALADIN Plasma Drive v2.1
GLM Final Pre-Integration Suite (standalone)

High-value checks before any driver plug-in:
  CW-10  Strong divergence pulse
  CW-11  Weak divergence pulse
  CW-13  Random divergence field
  ST-01  Long-duration stability (800 steps)
  ST-03  High cleaning speed
  CY-01  Radial cleaning
  CY-04  Axis-crossing stability

Run on Colab Pro A100 with the latest glm_cylindrical_ho_kernel.py
"""

import numpy as np
import sys
import os
import importlib

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/content")

try:
    import cupy as cp
except ImportError:
    print("CuPy required")
    sys.exit(1)

if "glm_cylindrical_ho_kernel" in sys.modules:
    del sys.modules["glm_cylindrical_ho_kernel"]
import glm_cylindrical_ho_kernel
importlib.reload(glm_cylindrical_ho_kernel)
from glm_cylindrical_ho_kernel import launch_glm_ho

print("ALADIN GLM — Final Pre-Integration Suite")
print("Kernel:", glm_cylindrical_ho_kernel.glm_ho_kernel)
print("=" * 60)

# ---------------------------------------------------------------------------
class MiniGeometry:
    def __init__(self, n, r_min=0.0, r_max=1.0):
        self.nr = self.nphi = self.nz = n
        self.r_faces = np.linspace(r_min, r_max, n+1).astype(np.float32)
        self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:]).astype(np.float32)
        self.dr = np.diff(self.r_faces).astype(np.float32)
        self.dphi = np.array([2*np.pi / max(n, 1)], dtype=np.float32)
        self.dz = np.array([1.0 / n], dtype=np.float32)
        self.dx_min = float(min(self.dr.min(), self.dz[0]))

class MiniConfig:
    def __init__(self, n, g=2):
        self.nr = self.nphi = self.nz = n
        self.ghost_width = g
        self.ch = 1.0

class MiniState:
    def __init__(self, n, g=2):
        sc  = (n+2*g, n+2*g, n+2*g)
        sbr = (n+1+2*g, n+2*g, n+2*g)
        sbp = (n+2*g, n+1+2*g, n+2*g)
        sbz = (n+2*g, n+2*g, n+1+2*g)
        self.fields = {
            "rho":  cp.ones(sc, dtype=cp.float32),
            "Br":   cp.zeros(sbr, dtype=cp.float32),
            "Bphi": cp.zeros(sbp, dtype=cp.float32),
            "Bz":   cp.zeros(sbz, dtype=cp.float32),
            "psi":  cp.zeros(sc, dtype=cp.float32),
            "divB": cp.zeros(sc, dtype=cp.float32),
        }
    def get(self, name): return self.fields[name]
    def has(self, name): return name in self.fields

def to_cp(a):
    return a if isinstance(a, cp.ndarray) else cp.asarray(a, dtype=cp.float32)

def safe_launch(state, dt, geometry, config, **kw):
    geometry.r_faces   = to_cp(geometry.r_faces)
    geometry.r_centers = to_cp(geometry.r_centers)
    geometry.dr        = to_cp(geometry.dr)
    geometry.dphi      = to_cp(geometry.dphi)
    geometry.dz        = to_cp(geometry.dz)
    return launch_glm_ho(state, dt, geometry, config, **kw)

def max_div(state, g):
    return float(cp.max(cp.abs(state.fields["divB"][g:-g, g:-g, g:-g])))

def all_finite(state):
    return bool(cp.all(cp.isfinite(state.fields["psi"])) and
                cp.all(cp.isfinite(state.fields["Br"])) and
                cp.all(cp.isfinite(state.fields["Bphi"])) and
                cp.all(cp.isfinite(state.fields["Bz"])))

# ---------------------------------------------------------------------------
results = {}

# CW-10 Strong pulse
print("CW-10  Strong divergence pulse (N=64)")
n, g = 64, 2
geo, cfg, st = MiniGeometry(n, 0.1), MiniConfig(n), MiniState(n)
for ir in range(n+1):
    r = float(geo.r_faces[ir])
    st.fields["Br"][ir+g] = 0.15 * np.exp(-((r-0.45)/0.06)**2)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d0 = max_div(st, g)
for _ in range(40):
    safe_launch(st, 0.003, geo, cfg, ch=1.0, cp_scale=1.0, nsub=2)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d1 = max_div(st, g)
red = d0 / max(d1, 1e-30)
print(f"  {d0:.3e} → {d1:.3e}  (×{red:.1f})")
results["CW-10"] = red > 1.2 and all_finite(st)
print(f"  CW-10: {'PASS' if results['CW-10'] else 'FAIL'}")

# CW-11 Weak pulse
print("\nCW-11  Weak divergence pulse (N=64)")
st = MiniState(n)
for ir in range(n+1):
    r = float(geo.r_faces[ir])
    st.fields["Br"][ir+g] = 1e-4 * np.exp(-((r-0.5)/0.1)**2)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d0 = max_div(st, g)
for _ in range(30):
    safe_launch(st, 0.003, geo, cfg, ch=1.0, cp_scale=0.8, nsub=2)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d1 = max_div(st, g)
print(f"  {d0:.3e} → {d1:.3e}")
results["CW-11"] = d1 < d0 * 1.5 and all_finite(st) and d1 < 1e-2
print(f"  CW-11: {'PASS' if results['CW-11'] else 'FAIL'}")

# CW-13 Random field
print("\nCW-13  Random divergence field (N=48)")
n = 48
geo, cfg, st = MiniGeometry(n, 0.1), MiniConfig(n), MiniState(n)
rng = np.random.default_rng(7)
Br_rand = (rng.standard_normal((n+1+4, n+4, n+4)) * 0.01).astype(np.float32)
st.fields["Br"][:] = cp.asarray(Br_rand)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d0 = max_div(st, 2)
for _ in range(25):
    safe_launch(st, 0.002, geo, cfg, ch=0.8, cp_scale=1.0, nsub=2)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d1 = max_div(st, 2)
print(f"  {d0:.3e} → {d1:.3e}")
results["CW-13"] = all_finite(st) and d1 < d0 * 2.0
print(f"  CW-13: {'PASS' if results['CW-13'] else 'FAIL'}")

# ST-01 Long-duration (800 steps)
print("\nST-01  Long-duration stability (800 steps, N=32)")
n = 32
geo, cfg, st = MiniGeometry(n, 0.1), MiniConfig(n), MiniState(n)
for ir in range(n+1):
    st.fields["Br"][ir+2] = 0.02 * np.sin(2*np.pi * ir / n)
finite = True
max_psi = 0.0
for step in range(800):
    safe_launch(st, 0.002, geo, cfg, ch=0.6, cp_scale=1.5, nsub=1)
    if step % 100 == 0:
        if not all_finite(st):
            finite = False
            break
        max_psi = max(max_psi, float(cp.max(cp.abs(st.fields["psi"]))))
print(f"  finite={finite}  max|ψ|={max_psi:.3e}")
results["ST-01"] = finite and max_psi < 10.0
print(f"  ST-01: {'PASS' if results['ST-01'] else 'FAIL'}")

# ST-03 High cleaning speed
print("\nST-03  High cleaning speed (ch=3.0, N=32)")
st = MiniState(n)
for ir in range(n+1):
    r = float(geo.r_faces[ir])
    st.fields["Br"][ir+2] = 0.03 * np.exp(-((r-0.4)/0.08)**2)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d0 = max_div(st, 2)
for _ in range(20):
    safe_launch(st, 0.001, geo, cfg, ch=3.0, cp_scale=0.5, nsub=3)
ok = all_finite(st)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d1 = max_div(st, 2)
print(f"  finite={ok}  {d0:.3e} → {d1:.3e}")
results["ST-03"] = ok and d1 < d0 * 3.0
print(f"  ST-03: {'PASS' if results['ST-03'] else 'FAIL'}")

# CY-01 Radial cleaning
print("\nCY-01  Radial cleaning (N=64, r_min=0.1)")
n = 64
geo, cfg, st = MiniGeometry(n, 0.1), MiniConfig(n), MiniState(n)
for ir in range(n+1):
    r = float(geo.r_faces[ir])
    st.fields["Br"][ir+2] = 0.05 * r * np.exp(-((r-0.6)/0.1)**2)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d0 = max_div(st, 2)
for _ in range(35):
    safe_launch(st, 0.0025, geo, cfg, ch=1.0, cp_scale=1.0, nsub=2)
safe_launch(st, 1e-8, geo, cfg, ch=0.0)
d1 = max_div(st, 2)
print(f"  {d0:.3e} → {d1:.3e}")
results["CY-01"] = d1 < d0 and all_finite(st)
print(f"  CY-01: {'PASS' if results['CY-01'] else 'FAIL'}")

# CY-04 Axis-crossing
print("\nCY-04  Axis-crossing stability (r_min=0, N=48)")
n = 48
geo, cfg, st = MiniGeometry(n, 0.0), MiniConfig(n), MiniState(n)
for ir in range(n+1):
    r = float(geo.r_faces[ir])
    st.fields["Br"][ir+2] = 0.02 * r * np.sin(np.pi * r)
for _ in range(40):
    safe_launch(st, 0.002, geo, cfg, ch=0.8, cp_scale=1.2, nsub=2)
axis_ok = float(cp.max(cp.abs(st.fields["Br"][2]))) < 1e-4
finite = all_finite(st)
print(f"  finite={finite}  max|Br_axis|={float(cp.max(cp.abs(st.fields['Br'][2]))):.3e}")
results["CY-04"] = finite and axis_ok
print(f"  CY-04: {'PASS' if results['CY-04'] else 'FAIL'}")

# ---------------------------------------------------------------------------
print("=" * 60)
passed = sum(results.values())
total = len(results)
print(f"Result: {passed}/{total} passed")
for k, v in results.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
print("=" * 60)
if passed == total:
    print("Final pre-integration checks passed.")
    print("Kernel is safe to plug into the driver behind a flag.")
else:
    print("Some checks failed — review before driver integration.")
print("Driver integration still deferred until you decide.")
