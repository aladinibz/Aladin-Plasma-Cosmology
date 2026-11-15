"""
Aladin_Driver_v3.1_AIL
======================
ALADIN Plasma Drive v3.1 — Driver Core + AIL + full v2 infrastructure
Kernels remain stubs (real CUDA kernels plug in last).

Locked sequence (11 slots):
  gyro → utc → hlldx → hlldy → hlldz → residual → prim → uct_emf → uct_curl → glm → g2

Merged from v2:
  - Full cylindrical geometry (face areas, edge lengths, volumes)
  - GeometricSources (1/r momentum sources every RK stage)
  - SSP-RK3 integrator
  - GravityManager (Miyamoto-Nagai + NFW)
  - MemoryPoolManager + CUDA Graph backend ("GPA")
  - Equilibrium disk IC + checkpoints
  - Extra fields (P_par/P_perp, fluxes, emfs, S_*)

ScienceManager: pure diagnostic (never writes prognostic fields).
AIL hooks: experiment identity, diagnostic capture points.

Units: code units (kpc, Myr, M☉).

Dependencies
------------
    pip install numpy              # required
    pip install cupy-cuda12x       # optional GPU
    pip install gradio_client      # online hook (optional)
"""
from __future__ import annotations
import math
import sys
import os
import json
import hashlib
import sqlite3
import uuid
import csv
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Literal
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

def _pip_install(package: str) -> bool:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as exc:
        print(f"pip install {package} failed: {exc}")
        return False

try:
    import numpy as np
except ImportError:
    print("Installing missing dependency: numpy")
    if _pip_install("numpy"):
        import numpy as np
    else:
        sys.exit("numpy is required. Install with:  pip install numpy")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

PI = math.pi
TWO_PI = 2.0 * PI

Centering = Literal[
    "cell",
    "face_r", "face_phi", "face_z",
    "edge_r", "edge_phi", "edge_z"
]

# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class ALADINConfig:
    nr: int = 64
    nphi: int = 32
    nz: int = 32
    r_min: float = 0.0
    r_max: float = 1.0
    z_min: float = -0.25
    z_max: float = 0.25
    nghost: int = 2
    precision: str = "float32"
    backend: str = "numpy"          # "numpy" or "cupy"
    use_cuda_graph: bool = False    # GPU Graph / "GPA"
    ic_name: str = "zpinch"         # "zpinch", "uniform", "equilibrium_disk"
    seed: int = 42
    gamma: float = 5.0 / 3.0
    cfl: float = 0.3
    max_dt: float = 1e-3
    min_dt: float = 1e-8
    science_interval: int = 5
    density_floor: float = 1e-12
    self_gravity: bool = False
    external_gravity: bool = True
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "artifacts"

    def xp(self):
        return cp if (self.backend == "cupy" and HAS_CUPY) else np

    @property
    def dtype(self):
        return np.float32 if self.precision == "float32" else np.float64

    def validate(self):
        if self.nr <= 0 or self.nphi <= 0 or self.nz <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.r_max <= self.r_min or self.z_max <= self.z_min:
            raise ValueError("invalid radial or axial range")
        if self.cfl <= 0 or self.max_dt <= 0 or self.min_dt <= 0:
            raise ValueError("invalid CFL or timestep limits")
        if self.science_interval <= 0:
            raise ValueError("science_interval must be >= 1")
        if self.nghost < 1:
            raise ValueError("nghost must be >= 1")
        if self.backend == "cupy" and not HAS_CUPY:
            print("WARNING: CuPy unavailable — falling back to NumPy")
            self.backend = "numpy"


# =============================================================================
# 2. GEOMETRY (full cylindrical from v2)
# =============================================================================
class GeometryManager:
    def __init__(self, cfg: ALADINConfig):
        self.cfg = cfg
        self.xp = cfg.xp()
        self.dtype = cfg.dtype
        self.ghost = cfg.nghost
        self.nr, self.nphi, self.nz = cfg.nr, cfg.nphi, cfg.nz
        self._build()

    def _build(self):
        cfg = self.cfg
        xp = self.xp

        self.r_faces = xp.linspace(cfg.r_min, cfg.r_max, cfg.nr + 1, dtype=self.dtype)
        self.phi_faces = xp.linspace(0.0, TWO_PI, cfg.nphi + 1, dtype=self.dtype)
        self.z_faces = xp.linspace(cfg.z_min, cfg.z_max, cfg.nz + 1, dtype=self.dtype)

        self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        self.phi_centers = 0.5 * (self.phi_faces[:-1] + self.phi_faces[1:])
        self.z_centers = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

        self.dr = self.r_faces[1:] - self.r_faces[:-1]
        self.dphi = self.phi_faces[1:] - self.phi_faces[:-1]
        self.dz = self.z_faces[1:] - self.z_faces[:-1]

        r_out = self.r_faces[1:]
        r_in = self.r_faces[:-1]
        self.cell_volumes = (0.5 * (r_out**2 - r_in**2))[:, None, None] * \
                            self.dphi[None, :, None] * self.dz[None, None, :]

        self.axis_mask = (self.r_faces[:-1] == 0.0)
        self.r_c = self.r_centers[:, None, None]
        self.one_over_r = xp.where(self.r_c > 0, 1.0 / self.r_c, 0.0)

        # Face areas
        self.area_r = self.r_faces[:, None, None] * self.dphi[None, :, None] * self.dz[None, None, :]
        self.area_phi = self.dr[:, None, None] * self.dz[None, None, :]
        self.area_z = 0.5 * (self.r_faces[1:]**2 - self.r_faces[:-1]**2)[:, None, None] * self.dphi[None, :, None]

        # Edge lengths
        self.edge_len_r = self.dr[:, None, None]
        self.edge_len_phi = self.r_centers[:, None, None] * self.dphi[None, :, None]
        self.edge_len_z = self.dz[None, None, :]

        # r at kernel locations
        self.r_face_r = self.r_faces[:, None, None]
        self.r_face_phi = self.r_centers[:, None, None]
        self.r_edge_z = self.r_faces[:, None, None]

    def face_areas(self):
        return self.area_r, self.area_phi, self.area_z

    def edge_lengths(self):
        return self.edge_len_r, self.edge_len_phi, self.edge_len_z

    def total_volume(self) -> float:
        return float(self.xp.sum(self.cell_volumes))

    def axis_mask(self):
        return self.r_centers < 0.5 * float(self.dr.mean() if hasattr(self.dr, "mean") else self.dr[0])


class GeometricSources:
    """1/r geometric source terms for cylindrical momentum. Applied every RK stage."""
    def __init__(self, config, geometry):
        self.config = config
        self.geometry = geometry
        self.xp = config.xp()

    def momentum_sources(self, state):
        xp = self.xp
        rho = state.interior("rho")
        vr = state.interior("rho_vr") / (rho + 1e-40)
        vphi = state.interior("rho_vphi") / (rho + 1e-40)
        r = self.geometry.r_centers
        # broadcast radial coordinate to (nr, 1, 1) so it matches (nr, nphi, nz)
        r3 = r[:, None, None]
        r_safe = xp.maximum(r3, 0.5 * float(self.geometry.dr.mean() if hasattr(self.geometry.dr, "mean") else self.geometry.dr[0]))
        S_vr = rho * vphi**2 / r_safe
        S_vphi = -rho * vr * vphi / r_safe
        return S_vr, S_vphi

    def apply_momentum(self, state, dt):
        S_vr, S_vphi = self.momentum_sources(state)
        sl = state.grid.interior_slice("cell")
        state.get("rho_vr")[sl] += dt * S_vr
        state.get("rho_vphi")[sl] += dt * S_vphi


# =============================================================================
# 3. GRID
# =============================================================================
class GridManager:
    def __init__(self, cfg: ALADINConfig, geometry: GeometryManager):
        self.cfg = cfg
        self.geometry = geometry
        self.ghost = cfg.nghost
        self.nr, self.nphi, self.nz = cfg.nr, cfg.nphi, cfg.nz
        self._define_shapes()

    def _define_shapes(self):
        g = self.ghost
        nr, nphi, nz = self.nr, self.nphi, self.nz
        self.shapes = {
            "cell":     (nr + 2*g, nphi + 2*g, nz + 2*g),
            "face_r":   (nr + 1 + 2*g, nphi + 2*g, nz + 2*g),
            "face_phi": (nr + 2*g, nphi + 1 + 2*g, nz + 2*g),
            "face_z":   (nr + 2*g, nphi + 2*g, nz + 1 + 2*g),
            "edge_r":   (nr + 2*g, nphi + 1 + 2*g, nz + 1 + 2*g),
            "edge_phi": (nr + 1 + 2*g, nphi + 2*g, nz + 1 + 2*g),
            "edge_z":   (nr + 1 + 2*g, nphi + 1 + 2*g, nz + 2*g),
        }
        self.interior = {
            "cell": (slice(g, -g), slice(g, -g), slice(g, -g)),
        }

    def shape(self, centering: str = "cell") -> Tuple[int, ...]:
        return self.shapes.get(centering, self.shapes["cell"])

    def interior_slice(self, centering: str = "cell"):
        return self.interior.get(centering, self.interior["cell"])


# =============================================================================
# 4. STATE
# =============================================================================
class StateRegistry:
    def __init__(self, cfg: ALADINConfig, geometry: GeometryManager, grid: GridManager):
        self.cfg = cfg
        self.geometry = geometry
        self.grid = grid
        self.xp = cfg.xp()
        self.fields: Dict[str, Any] = {}
        self._register_defaults()

    def _register_defaults(self):
        cell = self.grid.shape("cell")
        face_r = self.grid.shape("face_r")
        face_phi = self.grid.shape("face_phi")
        face_z = self.grid.shape("face_z")
        edge_r = self.grid.shape("edge_r")
        edge_phi = self.grid.shape("edge_phi")
        edge_z = self.grid.shape("edge_z")
        xp = self.xp
        # conserved / primitives
        for name in ("rho", "rho_vr", "rho_vphi", "rho_vz", "E",
                     "P_par", "P_perp", "q_par", "q_perp"):
            self.fields[name] = xp.zeros(cell, dtype=xp.float32)
        # staggered B
        self.fields["Br"] = xp.zeros(face_r, dtype=xp.float32)
        self.fields["Bphi"] = xp.zeros(face_phi, dtype=xp.float32)
        self.fields["Bz"] = xp.zeros(face_z, dtype=xp.float32)
        # workspaces
        self.fields["flux_r"] = xp.zeros(face_r, dtype=xp.float32)
        self.fields["flux_phi"] = xp.zeros(face_phi, dtype=xp.float32)
        self.fields["flux_z"] = xp.zeros(face_z, dtype=xp.float32)
        self.fields["emf_r"] = xp.zeros(edge_r, dtype=xp.float32)
        self.fields["emf_phi"] = xp.zeros(edge_phi, dtype=xp.float32)
        self.fields["emf_z"] = xp.zeros(edge_z, dtype=xp.float32)
        # G2 / sources
        for name in ("g2_torsion", "g2_residual", "g2_positivity",
                     "S_vr", "S_vphi", "S_vz", "S_E", "divB"):
            self.fields[name] = xp.zeros(cell, dtype=xp.float32)
        # gravity
        if self.cfg.self_gravity or self.cfg.external_gravity:
            self.fields["Phi"] = xp.zeros(cell, dtype=xp.float32)
            self.fields["g_R"] = xp.zeros(cell, dtype=xp.float32)
            self.fields["g_z"] = xp.zeros(cell, dtype=xp.float32)

    def get(self, name: str):
        return self.fields[name]

    def interior(self, name: str):
        sl = self.grid.interior_slice("cell")
        return self.fields[name][sl]


# =============================================================================
# 5. BOUNDARIES
# =============================================================================
class BoundaryManager:
    def __init__(self, cfg: ALADINConfig, geometry: GeometryManager, grid: GridManager):
        self.cfg = cfg
        self.geometry = geometry
        self.grid = grid

    def apply_all(self, state: StateRegistry):
        self._periodic_phi(state)
        self._axis(state)
        self._outer(state)

    def _periodic_phi(self, state: StateRegistry):
        g = self.cfg.nghost
        for name, arr in state.fields.items():
            if arr.ndim == 3 and arr.shape[1] > 2 * g:
                arr[:, :g, :] = arr[:, -2 * g:-g, :]
                arr[:, -g:, :] = arr[:, g:2 * g, :]

    def _axis(self, state: StateRegistry):
        g = self.cfg.nghost
        if "Br" in state.fields:
            state.fields["Br"][g, ...] = 0.0

    def _outer(self, state: StateRegistry):
        g = self.cfg.nghost
        for name, arr in state.fields.items():
            if arr.ndim == 3:
                if arr.shape[0] > 2 * g:
                    arr[-g:, :, :] = arr[-g-1:-1, :, :]
                if arr.shape[2] > 2 * g:
                    arr[:, :, :g] = arr[:, :, g:2*g]
                    arr[:, :, -g:] = arr[:, :, -2*g:-g]


# =============================================================================
# 6. TIME
# =============================================================================
class TimeManager:
    def __init__(self, cfg: ALADINConfig):
        self.cfg = cfg
        self.t = 0.0
        self.step = 0
        self.dt = cfg.max_dt

    def compute_dt(self, state: StateRegistry, geometry: GeometryManager) -> float:
        cfl_dt = self.cfg.cfl * float(geometry.dr.mean() if hasattr(geometry.dr, "mean") else geometry.dr[0])
        self.dt = max(self.cfg.min_dt, min(cfl_dt, self.cfg.max_dt))
        return self.dt

    def advance(self, dt: float):
        self.t += dt
        self.step += 1


# =============================================================================
# 7. KERNEL SLOTS (stubs for now)
# =============================================================================
KERNEL_SEQUENCE = [
    "gyro",       # 1. gyrotropic closure
    "utc",        # 2. UCT assembly
    "hlldx",      # 3. radial HLLD
    "hlldy",      # 4. azimuthal HLLD
    "hlldz",      # 5. axial HLLD
    "residual",   # 6. cylindrical FV residual
    "prim",       # 7. conserved → primitive
    "uct_emf",    # 8. UCT EMF
    "uct_curl",   # 9. UCT curl
    "glm",        # 10. GLM cleaning
    "g2",         # 11. G2 entropy geometry
]
assert len(KERNEL_SEQUENCE) == 11, "Exactly eleven operator slots"


class Operator:
    """Lightweight wrapper so backends can treat stubs and real kernels the same."""
    def __init__(self, name: str, func: Callable, enabled: bool = True):
        self.name = name
        self.func = func
        self.enabled = enabled

    def __call__(self, state, dt, **kwargs):
        return self.func(state, dt, **kwargs)


def make_kernel_stub(name: str) -> Operator:
    def _stub(state: StateRegistry, dt: float, **kwargs):
        return
    return Operator(name, _stub, enabled=True)


def make_prim_operator() -> Operator:
    def _prim(state: StateRegistry, dt: float, **kwargs):
        xp = state.xp
        rho = state.get("rho")
        floor = state.cfg.density_floor
        _ = xp.maximum(rho, floor)
        return
    return Operator("prim", _prim, enabled=True)


# =============================================================================
# 8. GPU BACKEND (CUDA Graph / "GPA" from v2)
# =============================================================================
@dataclass
class KernelDescriptor:
    name: str
    centering: str
    block: int = 128
    shared_bytes: int = 0
    fields: List[str] = None

    def __post_init__(self):
        if self.fields is None:
            self.fields = []


KERNEL_DESCRIPTORS = {
    name: KernelDescriptor(name, "cell", 128, 0, [])
    for name in KERNEL_SEQUENCE
}


class KernelBackend:
    def execute(self, state, dt):
        raise NotImplementedError
    def capture(self, state, dt):
        pass
    def replay(self):
        pass


class NumPyBackend(KernelBackend):
    def __init__(self, operators):
        self.operators = operators
    def execute(self, state, dt):
        for name in KERNEL_SEQUENCE:
            op = self.operators.get(name)
            if op is not None and getattr(op, "enabled", True):
                # support both Operator and plain callable
                if hasattr(op, "func"):
                    op.func(state, dt)
                else:
                    op(state, dt)


class CuPyBackend(KernelBackend):
    def __init__(self, operators, grid_manager=None):
        self.operators = operators
        self.grid = grid_manager
        self.descriptors = KERNEL_DESCRIPTORS
        self.kernels = {}
        self.stream = None
        if HAS_CUPY:
            self.stream = cp.cuda.Stream(non_blocking=True)
            src = r'''
            extern "C" __global__
            void aladin_placeholder(const float* in, float* out, float dt, int n) {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                if (i < n) {
                    out[i] = in[i] + dt;
                }
            }
            '''
            for name in KERNEL_SEQUENCE:
                self.kernels[name] = cp.RawKernel(src, "aladin_placeholder")

    def _grid_size(self, state, centering: str) -> int:
        if self.grid is None:
            return int(state.get("rho").size)
        shape = self.grid.shape(centering)
        return int(np.prod(shape))

    def _launch(self, name: str, state, dt: float):
        desc = self.descriptors[name]
        kernel = self.kernels.get(name)
        if kernel is None:
            return
        n = self._grid_size(state, desc.centering)
        block = desc.block
        grid = (n + block - 1) // block
        arr = state.get("rho")
        dt32 = np.float32(dt)
        if self.stream is not None:
            with self.stream:
                kernel((grid,), (block,), (arr, arr, dt32, n))
        else:
            kernel((grid,), (block,), (arr, arr, dt32, n))

    def execute(self, state, dt):
        if not HAS_CUPY:
            return
        for name in KERNEL_SEQUENCE:
            if name in self.kernels:
                self._launch(name, state, dt)
        if self.stream is not None:
            self.stream.synchronize()


class CUDAGraphBackend(CuPyBackend):
    """CUDA Graph backend with pointer-table indirection (the 'GPA' thing)."""
    def __init__(self, operators, grid_manager=None):
        super().__init__(operators, grid_manager)
        self.graph = None
        self.graph_exec = None
        self.captured = False
        self.ptrs = None
        self.dt_slot = None
        self._field_names = [
            "rho", "rho_vr", "rho_vphi", "rho_vz", "E",
            "Br", "Bphi", "Bz",
            "emf_r", "emf_phi", "emf_z",
        ]

    def _ensure_table(self, state):
        if not HAS_CUPY:
            return
        if self.ptrs is None:
            self.ptrs = cp.zeros(len(self._field_names), dtype=cp.uintp)
        if self.dt_slot is None:
            self.dt_slot = cp.zeros(1, dtype=cp.float32)

    def update_pointers(self, state, dt: float):
        if not HAS_CUPY or self.ptrs is None:
            return
        for i, name in enumerate(self._field_names):
            if name in state.fields:
                self.ptrs[i] = state.get(name).data.ptr
        self.dt_slot.fill(np.float32(dt))

    def capture(self, state, dt: float):
        if not HAS_CUPY:
            print("CUDA Graph: no CuPy — skipped")
            return False
        self._ensure_table(state)
        self.update_pointers(state, dt)
        stream = self.stream or cp.cuda.Stream()
        try:
            self.graph = cp.cuda.graph.Graph()
            with stream:
                with self.graph:
                    super().execute(state, dt)
            self.graph_exec = self.graph.instantiate()
            self.captured = True
            print("CUDA Graph captured")
            return True
        except Exception as e:
            print(f"CUDA Graph capture failed: {e}")
            self.captured = False
            return False

    def replay(self):
        if self.graph_exec is not None and self.stream is not None:
            self.graph_exec.launch(stream=self.stream)
            self.stream.synchronize()

    def execute(self, state, dt):
        if not HAS_CUPY:
            return
        if not self.captured:
            super().execute(state, dt)
            return
        self.update_pointers(state, dt)
        self.replay()


def make_backend(config, operators, grid_manager=None):
    if config.backend == "cupy" and HAS_CUPY:
        if getattr(config, "use_cuda_graph", False):
            return CUDAGraphBackend(operators, grid_manager)
        return CuPyBackend(operators, grid_manager)
    return NumPyBackend(operators)


# =============================================================================
# 9. EXECUTION CONTROLLER
# =============================================================================
class ExecutionController:
    def __init__(self):
        self.operators: Dict[str, Operator] = {}
        self.trace: List[str] = []
        self.real_kernels: set = set()
        for name in KERNEL_SEQUENCE:
            if name == "prim":
                self.operators[name] = make_prim_operator()
            else:
                self.operators[name] = make_kernel_stub(name)
        self.backend = None

    def register(self, name: str, func: Callable):
        if name not in KERNEL_SEQUENCE:
            raise ValueError(f"Unknown kernel slot: {name}")
        if name in self.real_kernels:
            raise ValueError(f"Duplicate registration: {name}")
        self.operators[name] = Operator(name, func, enabled=True)
        self.real_kernels.add(name)

    def step(self, state: StateRegistry, boundary: BoundaryManager, dt: float):
        self.trace = []
        if self.backend is not None:
            self.backend.execute(state, dt)
            return
        for name in KERNEL_SEQUENCE:
            op = self.operators[name]
            self.trace.append(name)
            op(state, dt)
            boundary.apply_all(state)

    def manifest(self) -> dict:
        return {name: ("real" if name in self.real_kernels else "stub")
                for name in KERNEL_SEQUENCE}


# =============================================================================
# 10. INTEGRATOR — SSP-RK3 + geometric sources
# =============================================================================
class Integrator:
    def __init__(self, cfg: ALADINConfig, state: StateRegistry):
        self.cfg = cfg
        self.state = state
        self._backup: Dict[str, Any] = {}
        self._has_backup = False
        xp = state.xp
        self.rk_buffers = {k: xp.copy(v) for k, v in state.fields.items()}

    def step(self, execution: ExecutionController, boundary: BoundaryManager,
             geo_sources: GeometricSources, dt: float):
        xp = self.state.xp
        self._backup = {k: xp.copy(v) for k, v in self.state.fields.items()}
        self._has_backup = True

        # Stage 1
        boundary.apply_all(self.state)
        execution.step(self.state, boundary, dt)
        geo_sources.apply_momentum(self.state, dt)
        U1 = {k: xp.copy(v) for k, v in self.state.fields.items()}

        # Stage 2
        for k, v in U1.items():
            self.state.fields[k][...] = v
        boundary.apply_all(self.state)
        execution.step(self.state, boundary, dt)
        geo_sources.apply_momentum(self.state, dt)
        for k in self.state.fields:
            self.state.fields[k][...] = 0.75 * self._backup[k] + 0.25 * self.state.fields[k]

        # Stage 3
        boundary.apply_all(self.state)
        execution.step(self.state, boundary, dt)
        geo_sources.apply_momentum(self.state, dt)
        for k in self.state.fields:
            self.state.fields[k][...] = (1.0/3.0) * self._backup[k] + (2.0/3.0) * self.state.fields[k]

        boundary.apply_all(self.state)

    def rollback(self):
        if not self._has_backup:
            return
        for k, v in self._backup.items():
            self.state.fields[k][...] = v
        self._has_backup = False


# =============================================================================
# 11. GRAVITY (Miyamoto-Nagai + NFW)
# =============================================================================
class GravityManager:
    def __init__(self, config: ALADINConfig, geometry=None):
        self.config = config
        self.geometry = geometry
        self.M_disk = 5.0e10
        self.a_disk = 3.0
        self.b_disk = 0.3
        self.M_halo = 1.0e12
        self.r_s = 20.0
        self.G = 4.498e-12

    def set_geometry(self, geometry):
        self.geometry = geometry

    def update(self, state: StateRegistry):
        if not (self.config.self_gravity or self.config.external_gravity):
            return
        if self.geometry is None or "Phi" not in state.fields:
            return
        xp = state.xp
        g = self.geometry
        # (nr, 1, 1) and (1, 1, nz) → broadcast to interior shape
        R = g.r_centers[:, None, None]
        z = g.z_centers[None, None, :]
        Phi = state.get("Phi")
        sl = state.grid.interior_slice("cell")
        target_shape = Phi[sl].shape
        a, b = self.a_disk, self.b_disk
        denom = xp.sqrt(R**2 + (a + xp.sqrt(z**2 + b**2))**2)
        Phi_MN = -self.G * self.M_disk / denom
        r = xp.sqrt(R**2 + z**2) + 1e-8
        c = 10.0
        factor = math.log(1 + c) - c / (1 + c)
        Phi_NFW = -self.G * self.M_halo / r * xp.log(1 + r / self.r_s) / factor
        Phi_tot = Phi_MN + Phi_NFW
        # expand axisymmetric potential to full (nr, nphi, nz)
        Phi_tot = xp.broadcast_to(Phi_tot, target_shape)
        Phi[sl] = Phi_tot.astype(Phi.dtype)
        if "g_R" in state.fields and "g_z" in state.fields:
            dPhi_dR = xp.gradient(Phi[sl], g.r_centers, axis=0)
            state.get("g_R")[sl] = -dPhi_dR
            dPhi_dz = xp.gradient(Phi[sl], g.z_centers, axis=2)
            state.get("g_z")[sl] = -dPhi_dz


# =============================================================================
# 12. CHECKPOINTS
# =============================================================================
class CheckpointManager:
    def __init__(self, config: ALADINConfig):
        self.config = config
        self.dir = config.checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def save(self, state: StateRegistry, time_mgr: TimeManager, tag: str = "latest"):
        path = os.path.join(self.dir, f"{tag}.npz")
        meta_path = os.path.join(self.dir, f"{tag}.json")
        data = {}
        for name, arr in state.fields.items():
            data[name] = arr.get() if hasattr(arr, "get") else arr
        np.savez(path, **data)
        meta = {
            "time": time_mgr.t,
            "step": time_mgr.step,
            "dt": time_mgr.dt,
            "seed": self.config.seed,
            "ic_name": self.config.ic_name,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Checkpoint: {path}")

    def load(self, state: StateRegistry, time_mgr: TimeManager, tag: str = "latest") -> bool:
        path = os.path.join(self.dir, f"{tag}.npz")
        meta_path = os.path.join(self.dir, f"{tag}.json")
        if not os.path.exists(path):
            print(f"No checkpoint at {path}")
            return False
        data = np.load(path)
        xp = state.xp
        for name in state.fields:
            if name in data:
                state.fields[name][:] = xp.asarray(data[name])
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            time_mgr.t = meta.get("time", 0.0)
            time_mgr.step = meta.get("step", 0)
            time_mgr.dt = meta.get("dt", time_mgr.dt)
        print(f"Loaded checkpoint {path}")
        return True


# =============================================================================
# 13. DIAGNOSTICS / HEALTH / RESULTS (minimal + ScienceManager hooks)
# =============================================================================
class DiagnosticsManager:
    def __init__(self):
        self.history: Dict[str, List] = {}

    def measure(self, state: StateRegistry, geometry: GeometryManager, t: float, step: int) -> dict:
        xp = state.xp
        rho = state.interior("rho")
        vol = geometry.cell_volumes
        mass = float(xp.sum(rho * vol))
        return {"t": t, "step": step, "mass": mass}


class HealthManager:
    def check(self, state: StateRegistry) -> bool:
        xp = state.xp
        for name in ("rho", "rho_vr", "rho_vphi", "rho_vz", "E"):
            if name in state.fields:
                arr = state.interior(name)
                if not bool(xp.all(xp.isfinite(arr))):
                    return False
                if name == "rho" and bool(xp.any(arr <= 0)):
                    return False
        for name in ("Br", "Bphi", "Bz"):
            if name in state.fields:
                arr = state.get(name)
                if not bool(xp.all(xp.isfinite(arr))):
                    return False
        return True


class ResultsManager:
    def __init__(self, cfg: ALADINConfig):
        self.cfg = cfg

    def write_diagnostics(self, history: List[dict]):
        print(f"Results: {len(history)} steps recorded")


# =============================================================================
# 14. INITIAL CONDITIONS
# =============================================================================
def set_zpinch(state: StateRegistry, geometry: GeometryManager, **kwargs):
    xp = state.xp
    state.get("rho")[:] = 1.0
    state.get("E")[:] = 0.05
    r = geometry.r_centers
    Bphi = state.get("Bphi")
    g = state.cfg.nghost
    n = min(Bphi.shape[0] - g, r.shape[0])
    for i in range(n):
        Bphi[i + g, ...] = 0.5 * float(r[i])
    print("Z-pinch IC: rho=1, E=0.05, Bphi=0.5*r")


def set_uniform(state: StateRegistry, geometry: GeometryManager, **kwargs):
    state.get("rho")[:] = 1.0
    state.get("E")[:] = 1.0
    print("Uniform IC")


def set_equilibrium_disk(state: StateRegistry, geometry: GeometryManager, gravity=None, **kwargs):
    xp = state.xp
    if gravity is None:
        state.get("rho")[:] = 1.0
        return
    gravity.update(state)
    sl = state.grid.interior_slice("cell")
    R = geometry.r_centers[:, None, None]
    z = geometry.z_centers[None, None, :]
    Phi = state.get("Phi")[sl]
    Phi_mid = Phi[:, :, Phi.shape[2]//2:Phi.shape[2]//2+1]
    cs = 0.1
    H = 0.1
    Sigma0 = 1.0
    scale = cs**2
    rho = Sigma0 / (math.sqrt(2 * math.pi) * H) * xp.exp(-(Phi - Phi_mid) / scale)
    rho = xp.maximum(rho, 1e-6)
    state.get("rho")[sl] = rho.astype(state.get("rho").dtype)
    if "g_R" in state.fields:
        g_R = state.get("g_R")[sl]
        lnrho = xp.log(rho + 1e-12)
        dlnrho_dR = xp.gradient(lnrho, geometry.r_centers, axis=0)
        vphi2 = R * (-g_R + cs**2 * dlnrho_dR)
        vphi2 = xp.maximum(vphi2, 0.0)
        vphi = xp.sqrt(vphi2)
        state.get("rho_vphi")[sl] = (rho * vphi).astype(state.get("rho_vphi").dtype)
    gamma = 5.0/3.0
    P = rho * cs**2
    state.get("E")[sl] = (P / (gamma - 1.0)).astype(state.get("E").dtype)
    print(f"Equilibrium disk IC: Sigma0={Sigma0}, H={H}, cs={cs}")


IC_MAP = {
    "zpinch": set_zpinch,
    "uniform": set_uniform,
    "equilibrium_disk": set_equilibrium_disk,
}


# =============================================================================
# 15. SCIENCE MANAGER (unchanged from v3 — pure diagnostic)
# =============================================================================
class ScienceManager:
    """
    Pure reader. Call after a step or on demand.
    Required public surface for the three experiments:
      force_budget, rotation_budget, gravity_vs_plasma, g2_monitor, record
    """
    def __init__(self, config, geometry, gravity=None, energy_is_total: bool = True,
                 mu0: float = 1.0, density_floor: float = 1e-12, science_interval: int = 10):
        self.config = config
        self.geometry = geometry
        self.gravity = gravity
        self.xp = config.xp() if callable(getattr(config, "xp", None)) else config.xp
        self.energy_is_total = energy_is_total
        self.mu0 = mu0
        self.density_floor = density_floor
        self.science_interval = max(1, int(science_interval))

    def _xp(self, a):
        return self.xp.asarray(a)

    def _phi_average(self, arr, midplane: bool = True):
        xp = self.xp
        if midplane:
            iz = arr.shape[2] // 2
            return xp.mean(arr[:, :, iz], axis=1)
        return xp.mean(arr, axis=(1, 2))

    def _phi_average_weighted(self, arr, rho, midplane: bool = True):
        xp = self.xp
        if midplane:
            iz = arr.shape[2] // 2
            num = xp.mean(rho[:, :, iz] * arr[:, :, iz], axis=1)
            den = xp.mean(rho[:, :, iz], axis=1)
        else:
            num = xp.mean(rho * arr, axis=(1, 2))
            den = xp.mean(rho, axis=(1, 2))
        return num / xp.maximum(den, self.density_floor)

    def _cell_centered_B(self, state):
        Br = state.interior("Br")
        Bphi = state.interior("Bphi")
        Bz = state.interior("Bz")
        Br_c = 0.5 * (Br[:-1] + Br[1:])
        Bphi_c = 0.5 * (Bphi[:, :-1] + Bphi[:, 1:])
        Bz_c = 0.5 * (Bz[:, :, :-1] + Bz[:, :, 1:])
        return Br_c, Bphi_c, Bz_c

    def _thermal_pressure(self, state, return_stats: bool = False):
        xp = self.xp
        sl = state.grid.interior_slice("cell")
        rho = state.get("rho")[sl]
        E = state.get("E")[sl]
        gamma = 5.0 / 3.0
        if not self.energy_is_total:
            P = (gamma - 1.0) * E
            stats = {"min_e_int": float(xp.min(E)), "n_negative": 0, "max_negative_dev": 0.0}
            return (P, stats) if return_stats else P
        vr = state.get("rho_vr")[sl] / xp.maximum(rho, self.density_floor)
        vphi = state.get("rho_vphi")[sl] / xp.maximum(rho, self.density_floor)
        vz = state.get("rho_vz")[sl] / xp.maximum(rho, self.density_floor)
        ke = 0.5 * rho * (vr**2 + vphi**2 + vz**2)
        Br, Bphi, Bz = self._cell_centered_B(state)
        me = (Br**2 + Bphi**2 + Bz**2) / (2.0 * self.mu0)
        e_int = E - ke - me
        e_int_safe = xp.maximum(e_int, 0.0)
        P = (gamma - 1.0) * e_int_safe
        stats = {
            "min_e_int": float(xp.min(e_int)),
            "n_negative": int(xp.sum(e_int < 0)),
            "max_negative_dev": float(xp.min(e_int)),
        }
        return (P, stats) if return_stats else P

    def _axis_mask(self, r):
        dr = self._xp(self.geometry.dr)
        return r < 0.5 * float(dr.mean() if hasattr(dr, "mean") else dr[0])

    def jz(self, state):
        xp = self.xp
        g = self.geometry
        _, Bphi_c, _ = self._cell_centered_B(state)
        Bphi_r = self._phi_average(Bphi_c, midplane=True)
        r = self._xp(g.r_centers)
        axis = self._axis_mask(r)
        jz = xp.gradient(r * Bphi_r, r) / xp.maximum(r, 1e-40) / self.mu0
        jz_axis = 2.0 * (Bphi_r / xp.maximum(r, 1e-40)) / self.mu0
        jz = xp.where(axis, jz_axis, jz)
        return r, jz

    def _jphi(self, state):
        xp = self.xp
        Br_c, _, Bz_c = self._cell_centered_B(state)
        Br_m = self._phi_average(Br_c, midplane=True)
        Bz_m = self._phi_average(Bz_c, midplane=True)
        r = self._xp(self.geometry.r_centers)
        dBz_dr = xp.gradient(Bz_m, r)
        jphi = (0.0 - dBz_dr) / self.mu0
        return jphi

    def lorentz_radial(self, state, mode: str = "validated_jz_only",
                       averaging: str = "axisymmetric_mean_field"):
        if mode not in ("validated_jz_only", "full_axisymmetric_lorentz"):
            raise ValueError(f"Unknown lorentz mode: {mode}")
        if averaging not in ("axisymmetric_mean_field", "azimuthal_mean_local_force"):
            raise ValueError(f"Unknown averaging: {averaging}")
        xp = self.xp
        r, jz = self.jz(state)
        _, Bphi_c, Bz_c = self._cell_centered_B(state)
        if averaging == "axisymmetric_mean_field":
            Bphi_r = self._phi_average(Bphi_c, midplane=True)
            Bz_r = self._phi_average(Bz_c, midplane=True)
            f = -jz * Bphi_r
            if mode == "full_axisymmetric_lorentz":
                jphi = self._jphi(state)
                f = jphi * Bz_r - jz * Bphi_r
        else:
            Bphi_r = self._phi_average(Bphi_c, midplane=True)
            f = -self._phi_average(jz[:, None, None] * Bphi_c, midplane=True)
            if mode == "full_axisymmetric_lorentz":
                f = -jz * Bphi_r
        return r, f / self.mu0

    def force_budget(self, state, midplane: bool = True,
                     lorentz_mode: str = "validated_jz_only",
                     averaging: str = "axisymmetric_mean_field"):
        xp = self.xp
        g = self.geometry
        sl = state.grid.interior_slice("cell")
        rho = state.get("rho")[sl]
        vphi = state.get("rho_vphi")[sl] / xp.maximum(rho, self.density_floor)
        P, pstats = self._thermal_pressure(state, return_stats=True)
        rho_r = self._phi_average(rho, midplane)
        vphi2_weighted = self._phi_average_weighted(vphi**2, rho, midplane)
        P_r = self._phi_average_weighted(P, rho, midplane)
        vphi_mean = self._phi_average(vphi, midplane)
        r = self._xp(g.r_centers)
        axis = self._axis_mask(r)
        centrifugal = vphi2_weighted / xp.maximum(r, 1e-40)
        centrifugal = xp.where(axis, 0.0, centrifugal)
        centrifugal_mean = vphi_mean**2 / xp.maximum(r, 1e-40)
        centrifugal_mean = xp.where(axis, 0.0, centrifugal_mean)
        if self.gravity is not None and hasattr(self.gravity, "radial_acceleration"):
            g_R = self.gravity.radial_acceleration(state)
            gravity = -self._phi_average(g_R, midplane)
        elif self.gravity is not None and "g_R" in state.fields:
            g_R = state.get("g_R")[sl]
            gravity = -self._phi_average(g_R, midplane)
        else:
            gravity = xp.zeros_like(r)
        thermal = -xp.gradient(P_r, r) / xp.maximum(rho_r, self.density_floor)
        _, f_L = self.lorentz_radial(state, mode=lorentz_mode, averaging=averaging)
        magnetic = f_L / xp.maximum(rho_r, self.density_floor)
        residual = centrifugal + thermal + magnetic - gravity
        return {
            "r": r,
            "centrifugal": centrifugal,
            "centrifugal_mean": centrifugal_mean,
            "gravity": gravity,
            "thermal": thermal,
            "magnetic": magnetic,
            "residual": residual,
            "rho": rho_r,
            "vphi": vphi_mean,
            "P": P_r,
            "pressure_stats": pstats,
            "lorentz_mode": lorentz_mode,
            "averaging": averaging,
        }

    def rotation_budget(self, state, reference_budget: Optional[Dict] = None):
        xp = self.xp
        b = self.force_budget(state)
        plasma = b["magnetic"] + b["thermal"]
        total = (
            xp.abs(b["centrifugal"])
            + xp.abs(b["gravity"])
            + xp.abs(plasma)
            + 1e-40
        )
        out = {
            "r": b["r"],
            "vphi": b["vphi"],
            "f_grav": b["gravity"],
            "f_mag": b["magnetic"],
            "f_therm": b["thermal"],
            "f_cent": b["centrifugal"],
            "plasma_fraction_signed": plasma / total,
            "plasma_fraction_abs": xp.abs(plasma) / total,
            "mag_fraction_abs": xp.abs(b["magnetic"]) / total,
            "therm_fraction_abs": xp.abs(b["thermal"]) / total,
            "residual": b["residual"],
            "pressure_stats": b["pressure_stats"],
        }
        if reference_budget is not None:
            out["delta_vphi"] = b["vphi"] - reference_budget["vphi"]
        return out

    def gravity_vs_plasma(self, state):
        xp = self.xp
        b = self.force_budget(state)
        plasma_acc = b["magnetic"] + b["thermal"]
        positive = plasma_acc > 0
        v_eq = xp.sqrt(xp.maximum(b["r"] * plasma_acc, 0.0))
        return {
            "r": b["r"],
            "v_full": b["vphi"],
            "v_plasma_equivalent": v_eq,
            "plasma_acc": plasma_acc,
            "gravity_acc": b["gravity"],
            "support_positive": positive,
            "note": "Mode A: equivalent velocity from current forces only; not a self-consistent equilibrium",
        }

    def g2_monitor(self, state):
        xp = self.xp
        rep = {"available": False}
        if getattr(state, "g2_computed", False) and "g2_torsion" in state.fields:
            rep["available"] = True
            t = state.interior("g2_torsion")
            rep["torsion_norm"] = float(xp.max(xp.abs(t)))
        if "g2_residual" in state.fields:
            rep["projection_residual"] = float(xp.max(xp.abs(state.interior("g2_residual"))))
        if "g2_positivity" in state.fields:
            rep["min_positivity"] = float(xp.min(state.interior("g2_positivity")))
        for name in ("S_vr", "S_vphi", "S_E"):
            if name in state.fields:
                s = state.interior(name)
                rep[f"{name}_max"] = float(xp.max(xp.abs(s)))
                rep[f"{name}_mean"] = float(xp.mean(xp.abs(s)))
        return rep

    def energies(self, state):
        xp = self.xp
        sl = state.grid.interior_slice("cell")
        rho = state.get("rho")[sl]
        vol = self._xp(self.geometry.cell_volumes)
        vr = state.get("rho_vr")[sl] / xp.maximum(rho, self.density_floor)
        vphi = state.get("rho_vphi")[sl] / xp.maximum(rho, self.density_floor)
        vz = state.get("rho_vz")[sl] / xp.maximum(rho, self.density_floor)
        ke = float(xp.sum(0.5 * rho * (vr**2 + vphi**2 + vz**2) * vol))
        Br, Bphi, Bz = self._cell_centered_B(state)
        me = float(xp.sum((Br**2 + Bphi**2 + Bz**2) / (2.0 * self.mu0) * vol))
        P, _ = self._thermal_pressure(state, return_stats=True)
        te = float(xp.sum(P / (5.0 / 3.0 - 1.0) * vol))
        return {"kinetic": ke, "magnetic": me, "thermal": te}

    def record(self, state, diagnostics, step: int, time: float = 0.0):
        if step % self.science_interval != 0:
            return
        h = diagnostics.history
        try:
            b = self.force_budget(state)
            max_res = float(self.xp.max(self.xp.abs(b["residual"])))
            plasma = b["magnetic"] + b["thermal"]
            total = (
                self.xp.abs(b["centrifugal"])
                + self.xp.abs(b["gravity"])
                + self.xp.abs(plasma)
                + 1e-40
            )
            frac = float(self.xp.mean(self.xp.abs(plasma) / total))
            g2 = self.g2_monitor(state)
            en = self.energies(state)
            sample = {
                "max_residual": max_res,
                "plasma_fraction_mean": frac,
                "E_magnetic": en["magnetic"],
                "E_kinetic": en["kinetic"],
                "E_thermal": en["thermal"],
                "n_negative_eint": b["pressure_stats"]["n_negative"],
            }
            if g2.get("available"):
                sample["g2_torsion"] = g2.get("torsion_norm", 0.0)
            for k, v in sample.items():
                h.setdefault(k, []).append(v)
        except Exception as exc:
            if "science_errors" not in h:
                h["science_errors"] = []
            h["science_errors"].append(f"step {step}: {type(exc).__name__}: {exc}")

    def stability_report(self, state, diagnostics=None, residual_tol: float = 1e-2,
                         eint_negative_tol: int = 0, density_floor_tol: float = None):
        xp = self.xp
        report = {"stable": True, "reasons": [], "metrics": {}}
        for name in ("rho", "rho_vr", "rho_vphi", "rho_vz", "E", "Br", "Bphi", "Bz"):
            if name not in state.fields:
                continue
            arr = state.interior(name)
            if not bool(xp.all(xp.isfinite(arr))):
                report["stable"] = False
                report["reasons"].append(f"non-finite values in {name}")
        b = self.force_budget(state)
        max_res = float(xp.max(xp.abs(b["residual"])))
        scale = float(xp.mean(
            xp.abs(b["centrifugal"]) +
            xp.abs(b["gravity"]) +
            xp.abs(b["thermal"]) +
            xp.abs(b["magnetic"]) +
            1e-40
        ))
        rel_res = max_res / scale
        report["metrics"]["max_residual"] = max_res
        report["metrics"]["rel_residual"] = rel_res
        report["metrics"]["residual_scale"] = scale
        if not math.isfinite(rel_res) or rel_res > residual_tol:
            report["stable"] = False
            report["reasons"].append(f"rel residual {rel_res:.3e} > {residual_tol:.3e}")
        n_neg = b["pressure_stats"]["n_negative"]
        report["metrics"]["n_negative_eint"] = n_neg
        if n_neg > eint_negative_tol:
            report["stable"] = False
            report["reasons"].append(f"{n_neg} cells with negative internal energy")
        if density_floor_tol is None:
            density_floor_tol = self.density_floor
        rho = state.interior("rho")
        n_low = int(xp.sum(rho < density_floor_tol))
        report["metrics"]["n_below_density_floor"] = n_low
        if n_low > 0:
            report["stable"] = False
            report["reasons"].append(f"{n_low} cells below density floor")
        g2 = self.g2_monitor(state)
        if g2.get("available"):
            report["metrics"]["g2_torsion"] = g2.get("torsion_norm", 0.0)
            if g2.get("min_positivity", 1.0) < 0.0:
                report["stable"] = False
                report["reasons"].append("G2 positivity violation")
        if diagnostics is not None and "max_residual" in diagnostics.history:
            hist = diagnostics.history["max_residual"]
            if len(hist) >= 3:
                recent = hist[-1]
                earlier = sorted(hist[:-1])[len(hist) // 2]
                if earlier > 0 and recent > 5.0 * earlier:
                    report["stable"] = False
                    report["reasons"].append("residual growing rapidly")
        report["metrics"]["pressure_stats"] = b["pressure_stats"]
        return report

    def completion_report(self, state, time: float, diagnostics=None, t_final: float = None,
                          residual_tol: float = 1e-3, residual_hold_records: int = 5,
                          energy_change_tol: float = 1e-4, energy_floor: float = 1e-8,
                          energy_hold_records: int = 3, plasma_fraction_tol: float = None,
                          plasma_hold_records: int = 3):
        residual_hold_records = max(1, residual_hold_records)
        energy_hold_records = max(2, energy_hold_records)
        if plasma_hold_records is not None:
            plasma_hold_records = max(2, plasma_hold_records)
        xp = self.xp
        report = {
            "finished": False,
            "time_reached": False,
            "numerically_converged": False,
            "scientifically_converged": False,
            "valid": True,
            "reasons": [],
            "metrics": {"time": time},
        }
        b = self.force_budget(state)
        res = b["residual"]
        if not bool(xp.all(xp.isfinite(res))):
            report["valid"] = False
            report["reasons"].append("non-finite residual")
        for name in ("rho", "E", "Br", "Bphi", "Bz"):
            if name in state.fields:
                arr = state.interior(name)
                if not bool(xp.all(xp.isfinite(arr))):
                    report["valid"] = False
                    report["reasons"].append(f"non-finite values in {name}")
        if b["pressure_stats"]["n_negative"] > 0:
            report["valid"] = False
            report["reasons"].append("negative internal energy")
        max_res = float(xp.max(xp.abs(res))) if report["valid"] else float("inf")
        report["metrics"]["max_residual"] = max_res
        if t_final is not None and time >= t_final:
            report["time_reached"] = True
            report["finished"] = True
            report["reasons"].append(f"time {time:.4e} >= t_final {t_final:.4e}")
        num_ok = report["valid"]
        num_reasons = []
        current_res_ok = max_res < residual_tol
        hist_res_ok = False
        if diagnostics is not None and "max_residual" in diagnostics.history:
            hist = diagnostics.history["max_residual"]
            if len(hist) >= residual_hold_records:
                recent = hist[-residual_hold_records:]
                hist_res_ok = all(r < residual_tol for r in recent)
        if not (current_res_ok and hist_res_ok):
            num_ok = False
            num_reasons.append("residual hold not satisfied")
        report["metrics"]["residual_current_ok"] = current_res_ok
        report["metrics"]["residual_hold_ok"] = hist_res_ok
        energy_ok = True
        if diagnostics is not None:
            keys = ("E_magnetic", "E_kinetic", "E_thermal")
            if all(k in diagnostics.history for k in keys):
                lens = [len(diagnostics.history[k]) for k in keys]
                n = min(lens)
                if n >= energy_hold_records:
                    totals = [
                        diagnostics.history["E_magnetic"][i]
                        + diagnostics.history["E_kinetic"][i]
                        + diagnostics.history["E_thermal"][i]
                        for i in range(-energy_hold_records, 0)
                    ]
                    if not all(math.isfinite(t) for t in totals):
                        energy_ok = False
                        num_reasons.append("non-finite energy")
                    elif abs(totals[-2]) < energy_floor:
                        energy_ok = False
                        num_reasons.append("energy near floor")
                    else:
                        rels = [
                            abs(totals[i] - totals[i - 1]) / (abs(totals[i - 1]) + 1e-40)
                            for i in range(1, len(totals))
                        ]
                        energy_ok = all(r < energy_change_tol for r in rels)
                        report["metrics"]["energy_rel_change"] = rels[-1] if rels else None
                else:
                    energy_ok = False
                    num_reasons.append("insufficient energy history")
            else:
                energy_ok = False
                num_reasons.append("no energy history")
        else:
            energy_ok = False
            num_reasons.append("no diagnostics")
        if not energy_ok:
            num_ok = False
        if plasma_fraction_tol is not None:
            pf_ok = False
            if diagnostics is not None and "plasma_fraction_mean" in diagnostics.history:
                pf = diagnostics.history["plasma_fraction_mean"]
                if len(pf) >= plasma_hold_records:
                    recent = pf[-plasma_hold_records:]
                    diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
                    pf_ok = all(d < plasma_fraction_tol for d in diffs)
            if not pf_ok:
                num_ok = False
                num_reasons.append("plasma-fraction hold not satisfied")
            report["metrics"]["plasma_fraction_hold_ok"] = pf_ok
        report["numerically_converged"] = bool(num_ok and report["valid"])
        if report["numerically_converged"]:
            report["reasons"].append("numerical holds satisfied")
        else:
            report["reasons"].extend(num_reasons)
        if not report["reasons"]:
            report["reasons"].append("no completion criterion met")
        return report

    def jz_profile(self, state):
        return self.jz(state)

    def extreme_cell_report(self, state, ir: int):
        xp = self.xp
        sl = state.grid.interior_slice("cell")
        rho = state.get("rho")[sl]
        report = {
            "ir": int(ir),
            "r": float(self.geometry.r_centers[ir]),
            "rho": float(xp.mean(rho[ir])),
        }
        return report


# =============================================================================
# 16. AIL (unchanged from v3)
# =============================================================================
class ValidationStatus(str, Enum):
    QUARANTINE = "quarantine"
    NUMERICAL_PASS = "numerical_pass"
    SCIENTIFIC_PASS = "scientific_pass"
    REJECTED = "rejected"

class SourceType(str, Enum):
    SYNTHETIC = "synthetic"
    REAL = "real"

class AIL:
    def __init__(self, enabled: bool = True, db_path: str = "aladin_ail.db"):
        self.enabled = enabled
        self.db_path = Path(db_path)
        self.current_id: Optional[str] = None
        self.experiments: Dict[str, dict] = {}
        if self.enabled:
            self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    family TEXT,
                    source TEXT,
                    parameters TEXT,
                    diagnostics TEXT,
                    features TEXT,
                    validation_status TEXT,
                    validator_id TEXT,
                    validation_criteria_version TEXT,
                    timestamp TEXT,
                    submission_hash TEXT UNIQUE,
                    driver_version TEXT,
                    n_snapshots INTEGER
                )
            """)
            conn.commit()

    def _hash(self, payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def start_experiment(self, family: str, parameters: dict,
                         driver_version: str = "v3.1_AIL",
                         source: str = SourceType.REAL.value) -> str:
        if not self.enabled:
            return ""
        exp_id = f"{family.upper()}-{uuid.uuid4().hex[:8].upper()}"
        self.current_id = exp_id
        rec = {
            "experiment_id": exp_id,
            "family": family,
            "source": source,
            "driver_version": driver_version,
            "parameters": parameters,
            "diagnostics": [],
            "features": {},
            "validation_status": ValidationStatus.QUARANTINE.value,
            "validator_id": None,
            "validation_criteria_version": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_snapshots": 0,
        }
        self.experiments[exp_id] = rec
        print(f"AIL Registry: {exp_id} created (quarantine)")
        return exp_id

    def capture(self, step: int, t: float, diagnostics: dict):
        if not self.enabled or not self.current_id:
            return
        snap = {"step": step, "t": float(t), **{k: float(v) if isinstance(v, (int, float)) else v
                                                for k, v in diagnostics.items()}}
        self.experiments[self.current_id]["diagnostics"].append(snap)

    def extract_features(self, diagnostics: dict) -> dict:
        wanted = [
            "Bphi_max", "Jz_max", "magnetic_energy", "E_magnetic",
            "kinetic_energy", "E_kinetic", "E_thermal", "mass_drift",
            "rho_max", "rho_min", "vr_max", "max_residual",
            "plasma_fraction_mean", "n_negative_eint",
        ]
        out = {}
        for k in wanted:
            if k in diagnostics and isinstance(diagnostics[k], (int, float)):
                out[k] = float(diagnostics[k])
        if "E_magnetic" in out and "E_B" not in out:
            out["E_B"] = out["E_magnetic"]
        if "E_kinetic" in out and "E_K" not in out:
            out["E_K"] = out["E_kinetic"]
        return out

    def finalize(self, final_diagnostics: Optional[dict] = None):
        if not self.enabled or not self.current_id:
            return
        rec = self.experiments[self.current_id]
        if final_diagnostics:
            rec["features"] = self.extract_features(final_diagnostics)
        rec["n_snapshots"] = len(rec["diagnostics"])
        payload = {
            "experiment_id": rec["experiment_id"],
            "family": rec["family"],
            "source": rec["source"],
            "parameters": rec["parameters"],
            "diagnostics": rec["diagnostics"],
        }
        sub_hash = self._hash(payload)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments (
                    experiment_id, family, source, parameters, diagnostics, features,
                    validation_status, validator_id, validation_criteria_version,
                    timestamp, submission_hash, driver_version, n_snapshots
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec["experiment_id"], rec["family"], rec["source"],
                json.dumps(rec["parameters"]), json.dumps(rec["diagnostics"]),
                json.dumps(rec["features"]), rec["validation_status"],
                rec["validator_id"], rec["validation_criteria_version"],
                rec["timestamp"], sub_hash, rec["driver_version"], rec["n_snapshots"]
            ))
            conn.commit()
        print(f"AIL Dataset: {self.current_id} finalized "
              f"(status={rec['validation_status']}, snapshots={rec['n_snapshots']})")

    def set_validation(self, status: str, validator_id: str = "ScienceManager",
                       criteria_version: str = "v1"):
        if not self.enabled or not self.current_id:
            return
        if status not in [s.value for s in ValidationStatus]:
            return
        rec = self.experiments[self.current_id]
        rec["validation_status"] = status
        rec["validator_id"] = validator_id
        rec["validation_criteria_version"] = criteria_version
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiments SET validation_status=?, validator_id=?,
                validation_criteria_version=? WHERE experiment_id=?
            """, (status, validator_id, criteria_version, self.current_id))
            conn.commit()
        print(f"AIL Validation: {self.current_id} → {status} by {validator_id}")

    def pattern_summary(self) -> dict:
        if not self.enabled:
            return {}
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT family, validation_status, COUNT(*) as n
                FROM experiments GROUP BY family, validation_status
            """).fetchall()
        return {"patterns": [{"family": r[0], "status": r[1], "count": r[2]} for r in rows]}

    def advise_next(self) -> dict:
        if not self.enabled or not self.current_id:
            return {"recommendation": None}
        return {
            "observation": f"Experiment {self.current_id} completed in quarantine.",
            "recommendation": "Run additional Z-pinch cases with varied Bphi amplitude (0.3–0.8).",
            "reason": "Insufficient parameter coverage for reliable pattern discovery.",
            "action": "Propose additional runs. Researcher approval required.",
            "automatic_execution": False,
        }

    def report(self) -> dict:
        if not self.enabled or not self.current_id:
            return {}
        rec = self.experiments[self.current_id]
        return {
            "experiment_id": rec["experiment_id"],
            "family": rec["family"],
            "driver_version": rec["driver_version"],
            "validation_status": rec["validation_status"],
            "n_snapshots": rec["n_snapshots"],
            "features": rec["features"],
            "advisor": self.advise_next(),
            "patterns": self.pattern_summary(),
            "note": "Measured results and diagnostics only. Predictions require separate validated models.",
        }

    def summary(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "n_experiments": len(self.experiments),
            "current": self.current_id,
            "statuses": {eid: rec["validation_status"] for eid, rec in self.experiments.items()},
            "db": str(self.db_path),
        }


# =============================================================================
# 17. ONLINE HOOK
# =============================================================================
try:
    from gradio_client import Client
    HAS_GRADIO_CLIENT = True
except ImportError:
    print("Installing missing dependency: gradio_client")
    if _pip_install("gradio_client"):
        try:
            from gradio_client import Client
            HAS_GRADIO_CLIENT = True
        except Exception:
            HAS_GRADIO_CLIENT = False
            Client = None
    else:
        HAS_GRADIO_CLIENT = False
        Client = None

class OnlineHook:
    def __init__(self, space: str = "aladinibz/PlasmaCosmology", enabled: bool = False):
        self.space = space
        self.enabled = enabled and HAS_GRADIO_CLIENT
        self._client = None

    def _get_client(self):
        if not self.enabled:
            return None
        if self._client is None:
            try:
                self._client = Client(self.space)
            except Exception:
                self._client = None
        return self._client

    def push(self, payload: dict) -> bool:
        if not self.enabled:
            return False
        client = self._get_client()
        if client is None:
            return False
        try:
            raw = json.dumps(payload, default=str)
            client.predict(raw, api_name="/receive_from_driver")
            return True
        except Exception:
            return False


# =============================================================================
# 18. MEMORY POOL
# =============================================================================
class MemoryPoolManager:
    def __init__(self, config: ALADINConfig):
        self.config = config
        self.enabled = HAS_CUPY and config.backend == "cupy"
        self.device_pool = None
        self.pinned_pool = None
        if self.enabled:
            self.device_pool = cp.get_default_memory_pool()
            self.pinned_pool = cp.get_default_pinned_memory_pool()

    def warmup(self, nbytes: int = 64 * 1024 * 1024):
        if not self.enabled:
            return
        tmp = cp.empty(nbytes // 4, dtype=cp.float32)
        del tmp
        self.device_pool.free_all_blocks()

    def free(self):
        if not self.enabled:
            return
        self.device_pool.free_all_blocks()
        self.pinned_pool.free_all_blocks()

    def report(self) -> str:
        if not self.enabled:
            return "MemoryPool: disabled (NumPy)"
        used = self.device_pool.used_bytes()
        total = self.device_pool.total_bytes()
        return (f"MemoryPool: used={used/1024**2:.1f} MiB  "
                f"total={total/1024**2:.1f} MiB")


# =============================================================================
# 19. DRIVER
# =============================================================================
class ALADINDriver:
    def __init__(self, cfg: ALADINConfig, ail_enabled: bool = True, online_enabled: bool = False):
        cfg.validate()
        self.cfg = cfg
        self.pool = MemoryPoolManager(cfg)
        self.pool.free()
        self.geometry = GeometryManager(cfg)
        self.geo_sources = GeometricSources(cfg, self.geometry)
        self.grid = GridManager(cfg, self.geometry)
        self.state = StateRegistry(cfg, self.geometry, self.grid)
        self.boundary = BoundaryManager(cfg, self.geometry, self.grid)
        self.time = TimeManager(cfg)
        self.execution = ExecutionController()
        self.backend = make_backend(cfg, self.execution.operators, self.grid)
        self.execution.backend = self.backend
        self.integrator = Integrator(cfg, self.state)
        self.diagnostics = DiagnosticsManager()
        self.health = HealthManager()
        self.results = ResultsManager(cfg)
        self.gravity = GravityManager(cfg, self.geometry)
        self.checkpoints = CheckpointManager(cfg)
        # ScienceManager
        self.science = ScienceManager(cfg, self.geometry, gravity=self.gravity,
                                      density_floor=cfg.density_floor,
                                      science_interval=cfg.science_interval)
        if callable(getattr(self.science, "xp", None)):
            self.science.xp = cfg.xp()
        # AIL
        self.ail = AIL(enabled=ail_enabled)
        self.online = OnlineHook(enabled=online_enabled)
        # IC
        ic_func = IC_MAP.get(cfg.ic_name, set_uniform)
        ic_func(self.state, self.geometry, gravity=self.gravity)
        self.lifecycle = "CREATED"
        params = {
            "nr": cfg.nr, "nphi": cfg.nphi, "nz": cfg.nz,
            "ic_name": cfg.ic_name, "cfl": cfg.cfl, "gamma": cfg.gamma,
        }
        self.ail.start_experiment(family=cfg.ic_name, parameters=params, source="synthetic")
        if self.ail.enabled and self.ail.current_id:
            self.ail.experiments[self.ail.current_id]["kernel_manifest"] = dict(self.execution.manifest())
            self.ail.experiments[self.ail.current_id]["manifest_frozen"] = True
        print(f"Aladin_Driver_v3.1_AIL | {cfg.nr}x{cfg.nphi}x{cfg.nz} | {cfg.backend} | ic={cfg.ic_name}")
        print(f"Sequence: {KERNEL_SEQUENCE}")
        print("Kernels: stubs (real kernels not loaded)")
        print(self.pool.report())
        if self.ail.enabled:
            print(f"AIL: experiment started → {self.ail.current_id}")
        if self.online.enabled:
            print(f"Online hook: enabled → {self.online.space}")
        else:
            print("Online hook: disabled")
        if cfg.use_cuda_graph:
            print("CUDA Graph backend: enabled")

    def register_kernel(self, name: str, func: Callable):
        if name not in KERNEL_SEQUENCE:
            raise ValueError(f"Unknown kernel slot: {name}")
        self.execution.register(name, func)
        print(f"Registered real kernel: {name}")

    def run(self, nsteps: int = 10):
        self.lifecycle = "RUNNING"
        history = []
        for i in range(nsteps):
            dt = self.time.compute_dt(self.state, self.geometry)
            try:
                self.gravity.update(self.state)
                self.integrator.step(self.execution, self.boundary, self.geo_sources, dt)
            except Exception as exc:
                self.integrator.rollback()
                self.lifecycle = "FAILED"
                print(f"Kernel exception — rolled back: {exc}")
                break
            if not self.health.check(self.state):
                self.integrator.rollback()
                self.lifecycle = "FAILED"
                print("Health failed — rolled back")
                break
            self.time.advance(dt)
            meas = self.diagnostics.measure(self.state, self.geometry, self.time.t, self.time.step)
            xp = self.state.xp
            rho_int = self.state.interior("rho")
            Bphi = self.state.get("Bphi")
            meas.update({
                "rho_max": float(xp.max(rho_int)),
                "rho_min": float(xp.min(rho_int)),
                "Bphi_max": float(xp.max(xp.abs(Bphi))),
            })
            if self.science is not None:
                try:
                    b = self.science.force_budget(self.state)
                    en = self.science.energies(self.state)
                    g2 = self.science.g2_monitor(self.state)
                    stab = self.science.stability_report(self.state, self.diagnostics)
                    r, jz = self.science.jz(self.state)
                    meas.update({
                        "max_residual": float(xp.max(xp.abs(b["residual"]))),
                        "plasma_fraction_mean": float(xp.mean(
                            xp.abs(b["magnetic"] + b["thermal"]) /
                            (xp.abs(b["centrifugal"]) + xp.abs(b["gravity"]) +
                             xp.abs(b["magnetic"] + b["thermal"]) + 1e-40)
                        )),
                        "E_magnetic": en["magnetic"],
                        "E_kinetic": en["kinetic"],
                        "E_thermal": en["thermal"],
                        "n_negative_eint": b["pressure_stats"]["n_negative"],
                        "Jz_max": float(xp.max(xp.abs(jz))),
                        "stable": stab["stable"],
                    })
                    if g2.get("available"):
                        meas["g2_torsion"] = g2.get("torsion_norm", 0.0)
                except Exception as exc:
                    meas["science_error"] = str(exc)
            history.append(meas)
            self.ail.capture(self.time.step, self.time.t, meas)
            if self.science is not None and (i + 1) % self.cfg.science_interval == 0:
                self.science.record(self.state, self.diagnostics, self.time.step, self.time.t)
            if (i + 1) % self.cfg.checkpoint_interval == 0:
                self.checkpoints.save(self.state, self.time)
            if (i + 1) % 5 == 0 or i == 0:
                print(f"Step {i+1} t={self.time.t:.4e} mass={meas['mass']:.4e}")
        self.results.write_diagnostics(history)
        if self.lifecycle != "FAILED":
            self.lifecycle = "COMPLETED"
        final = history[-1] if history else {}
        self.ail.finalize(final)
        if self.ail.enabled:
            rep = self.ail.report()
            print(f"AIL Report: {json.dumps(rep, indent=2, default=str)}")
        if self.online.enabled and self.ail.current_id:
            payload = {
                "family": self.cfg.ic_name,
                "experiment_id": self.ail.current_id,
                "source": "real",
                "parameters": {
                    "nr": self.cfg.nr, "nphi": self.cfg.nphi, "nz": self.cfg.nz,
                    "ic_name": self.cfg.ic_name,
                },
                "diagnostics": final,
            }
            ok = self.online.push(payload)
            print(f"Online push: {'ok' if ok else 'failed (ignored)'}")
        print(f"Lifecycle: {self.lifecycle}")
        if self.science is not None:
            print(f"Science history keys: {list(self.diagnostics.history.keys())}")
        if self.ail.enabled:
            print(f"AIL summary: {self.ail.summary()}")
        return history


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    cfg = ALADINConfig(
        nr=64, nphi=32, nz=32,
        backend="numpy",
        ic_name="zpinch",
        use_cuda_graph=False,   # set True when CuPy is available
    )
    driver = ALADINDriver(cfg, ail_enabled=True, online_enabled=False)
    driver.run(20)
    print("Aladin_Driver_v3.1_AIL ready (kernels still stubs).")
    print("Next: register real kernels via driver.register_kernel(name, func)")
    print("CUDA Graph: set cfg.use_cuda_graph = True + backend='cupy'")
