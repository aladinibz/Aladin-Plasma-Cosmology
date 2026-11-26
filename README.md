<div align="center">

# ALADIN Plasma Cosmology

**Native cylindrical MHD for Z-pinch and J × B plasma dynamics, from the laboratory to astrophysical scales**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![CUDA / CuPy](https://img.shields.io/badge/CUDA-CuPy%20RawKernel-76B900.svg)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--4457--557X-A6CE39.svg)](https://orcid.org/0009-0006-4457-557X)
![Open Science](https://img.shields.io/badge/Open%20Science-ByteWise%20%C2%B7%20ByByte-0A7E8C.svg)
[![Command Center](https://img.shields.io/badge/Command%20Center-Hugging%20Face-blueviolet.svg)](https://huggingface.co/spaces/aladinibz/PlasmaCosmology)

[Overview](#overview) · [Z-pinch](#the-z-pinch) · [Z-pinch benchmark](#native-cylindrical-z-pinch--benchmark-status) · [J × B](#the-j--b-force) · [History](#historical-foundations) · [Driver](#aladin-native-cylindrical-mhd-driver) · [Results](#quantitative-results) · [Kernels](#the-eleven-kernels--purpose-and-archival-references) · [Verification](#verification) · [Open Science](#aladin--open-science-by-design) · [Programs](#research-programs) · [Lab](#the-aladin-lab)

</div>

---

## Overview

Almost all visible matter in the universe is plasma: an electrically conducting medium that carries currents and responds to magnetic fields. In plasma, electromagnetic forces do not just act on matter; they **organise** it. Currents and magnetic fields can shape plasma into columns, filaments and flux ropes, and those structures appear across an enormous range of scales, from laboratory discharges to the solar corona, planetary magnetospheres and the interstellar medium.

**Plasma is often called the fourth state of matter**, alongside solids, liquids and gases. Unlike an ordinary neutral gas, plasma contains freely moving charged particles—electrons and ions—which respond collectively to electric and magnetic fields. These interactions allow plasma to support electric currents, electromagnetic waves, instabilities and complex structures that cannot be described by ordinary gas dynamics alone.

What makes plasma particularly fascinating is its **collective behaviour**. Charged particles generate electromagnetic fields, and those fields influence the motion of other particles. This continuous interaction creates a feedback system between matter, electric currents and magnetic fields. Under suitable conditions, plasma can develop organised structures, transport energy, accelerate particles, reconnect magnetic field lines and undergo powerful instabilities.

Plasma is not confined to distant astrophysical environments. It appears in **lightning, auroras, fluorescent lamps, fusion experiments and industrial discharges**. Beyond Earth, it dominates the observable material of the Sun and other stars, fills much of interplanetary and interstellar space, and plays a central role in the dynamics of galaxies and their surrounding environments.

Despite this abundance, plasma is not a single simple substance. Its behaviour depends on density, temperature, ionisation, magnetic-field strength, collisions and characteristic length and time scales. Some plasmas are well described by **magnetohydrodynamics (MHD)**, where the medium is treated as a conducting fluid. Others require kinetic descriptions that follow the motion and interactions of individual particle populations.

**One of the central questions of plasma physics is how electric currents and magnetic fields generate, maintain and destabilise organised structures.**

This question leads directly to two foundations of ALADIN Plasma Cosmology: the **Z-pinch**, where an electric current produces a magnetic field capable of compressing plasma, and the **Lorentz force, J × B**, which describes the electromagnetic force density acting on a current-carrying medium.

From these established physical principles, ALADIN develops numerical experiments to investigate plasma equilibrium, current-driven instabilities, magnetic evolution and the possible roles of plasma dynamics in astrophysical systems.

**ALADIN** studies this organising role of electromagnetism. Its core is two closely linked pieces of physics:

- **The Z-pinch**: the basic configuration in which an axial current confines its own plasma.
- **The J × B force**: the Lorentz force through which currents and magnetic fields self-organise plasma into coherent structures.

To investigate them, the project develops the **ALADIN Driver**: a GPU-accelerated MHD solver in native cylindrical coordinates $(r,\phi,z)$, built from **eleven computational kernels**. Its purpose is to test plasma-cosmology ideas through controlled numerical experiments on pinches, instabilities, field-aligned currents and magnetized astrophysical environments.

---

## The Z-pinch

An axial current $J_z$ generates an azimuthal magnetic field $B_\phi$, and the interaction between the two compresses the plasma toward the axis:

$$
\mathbf B = B_\phi(r)\,\hat{\boldsymbol\phi}, \qquad \mathbf J = J_z(r)\,\hat{\mathbf z}.
$$

The Z-pinch is one of the oldest and most fundamental configurations in plasma physics. It is the reference system for:

- **radial force balance** between magnetic compression and plasma pressure,
- **equilibrium** of current-carrying columns,
- **instabilities** that reshape them: the $m=0$ **sausage** mode and the $m=1$ **kink** mode,
- the behaviour of cylindrical geometry near the **axis** and the **outer boundary**.

It also serves as ALADIN's primary numerical benchmark.

## Native cylindrical Z-pinch — benchmark status

**Z-PINCH BENCHMARK · INDIVIDUAL PASS · 2,000-STEP EQUILIBRIUM TEST**

Analytical equilibrium and reported Driver time-evolution tests.

The normalized Z-pinch provides a common reference for the cylindrical magnetic field, discrete current reconstruction, radial Lorentz force, pressure balance and stationary evolution. For $\mu_0=1$:

$$
B_\phi(r)=\frac r2,\qquad J_z=1,\qquad f_r=-\frac r2,\qquad
P(r)=P_0-\frac{r^2}{4},\qquad \nabla\cdot\mathbf B=0.
$$

| Tested property | Analytical reference | Reported benchmark outcome |
|---|---|---|
| Axial current | $J_z=(1/r)\,d(rB_\phi)/dr=1$ | Reconstructed near floating-point precision |
| Radial magnetic force | $(\mathbf J\times\mathbf B)_r=-r/2$ | Reconstructed near floating-point precision |
| Magnetostatic pressure balance | $dP/dr=-r/2$ | Analytical equilibrium used as the reference state |
| Magnetic divergence | $\nabla\cdot\mathbf B=0$ | Divergence-free analytical reference |
| Stationary Driver evolution | Equilibrium preserved in time | Reported preservation over **2,000 time steps** |

This benchmark links the geometric and electromagnetic operators to an integrated equilibrium-preservation test. It is complemented by the separately reported $m=0$ sausage and $m=1$ kink growth-rate benchmarks, which test perturbations of the current-carrying plasma. The Z-pinch equilibrium result is specific to the stated setup and does not imply stability against all perturbations.


---

## The J × B force

The force density on a current-carrying plasma is the Lorentz force

$$
\boxed{\;\mathbf f_L = \mathbf J \times \mathbf B\;}, \qquad \mathbf J = \frac{1}{\mu_0}\,\nabla\times\mathbf B .
$$

**The J × B force is a central mechanism in magnetized-plasma dynamics.** It concentrates currents into filaments, holds pinches together, twists flux ropes, and drives the structures that emerge in magnetized plasma. In the Z-pinch it takes its simplest form:

$$
J_z = \frac{1}{\mu_0\, r}\frac{d}{dr}\bigl(r B_\phi\bigr), \qquad f_r = -J_z B_\phi ,
$$

an inward radial force. In a static equilibrium it is balanced by the pressure gradient, $dP/dr = f_r$.

In MHD, reconstructing current from the magnetic field requires an accurate discrete curl $\nabla\times\mathbf B$. This is particularly important near the cylindrical axis, where geometric consistency affects the Lorentz force and Hall terms.

### Reference benchmark

The normalised analytical Z-pinch ($\mu_0 = 1$):

$$
B_\phi = \tfrac12 r \quad\Longrightarrow\quad J_z = 1, \qquad f_r = -\tfrac12 r, \qquad \nabla\cdot\mathbf B = 0, \qquad P(r) = P_0 - \tfrac14 r^2 .
$$

| Observable | Analytical value | Tests |
|---|---|---|
| Azimuthal field | $B_\phi = 0.5\,r$ | cylindrical field representation |
| Axial current | $J_z = 1$ | discrete curl |
| Radial Lorentz force | $f_r = -0.5\,r$ | J × B reconstruction |
| Divergence | $\nabla\cdot\mathbf B = 0$ | magnetic constraint |
| Pressure balance | $dP/dr = -0.5\,r$ | force equilibrium |

---

## Historical foundations

*From the first experiments with electric currents to modern plasma laboratories and space missions.*

ALADIN builds on more than two centuries of work on currents, magnetic forces and pinches.

**The J × B force**

| Year | Scientist / program | Contribution |
|---|---|---|
| 1820–1826 | **André-Marie Ampère** | Forces between electric currents; the foundation of electrodynamics |
| 1820 | **Jean-Baptiste Biot & Félix Savart** | The magnetic field produced by a current |
| 1865 | **James Clerk Maxwell** | Unified equations of electricity and magnetism |
| 1889 | **Oliver Heaviside** | The magnetic force on a moving charge |
| 1892–1895 | **Hendrik Lorentz** | The complete force law $\mathbf F = q(\mathbf E + \mathbf v\times\mathbf B)$, whose continuum form is $\mathbf J\times\mathbf B$ |

**The pinch**

| Year | Scientist / program | Contribution |
|---|---|---|
| 1790 | **Martinus van Marum** | High-current discharges through wires: the first observed pinch effects |
| 1905 | **J. A. Pollock & S. Barraclough** | Explained a lightning-crushed copper tube as magnetic compression by the current: the first explained Z-pinch |
| 1934 | **Willard H. Bennett** | The Bennett relation: equilibrium of a self-confined current-carrying plasma |
| 1937 | **Lewi Tonks** | Named the "pinch effect" |
| 1954 | **Martin Kruskal & Martin Schwarzschild** | Theory of pinch instabilities, including the kink and sausage modes |
| 1950s | **ZETA, Perhapsatron and Kurchatov programs** | Z-pinches as the first large fusion experiments |

**Plasma in space**

| Year | Scientist / program | Contribution |
|---|---|---|
| 1942–1970s | **Hannes Alfvén** | Founder of magnetohydrodynamics (Nobel Prize in Physics, 1970); argued that currents and magnetic forces shape cosmic plasma |
| 1986 | **Anthony Peratt** | Simulations of interacting current filaments forming galaxy-like structures |

**Plasma laboratories today**

The same physics is studied experimentally every day. A few facilities closest to ALADIN's focus:

| Facility | Location | Relevance |
|---|---|---|
| **Z Machine** — Sandia National Laboratories | New Mexico, USA | The world's most powerful pulsed-power Z-pinch, driving currents of tens of megaamperes |
| **MAGPIE** — Imperial College London | London, UK | Pulsed-power laboratory astrophysics: Z-pinch-driven jets, shocks and magnetized flows |
| **Zap Energy** | Washington, USA | Sheared-flow-stabilised Z-pinch, aiming at fusion with a pinch that resists its instabilities |
| **Bellan Plasma Lab** — Caltech | California, USA | Laboratory flux ropes and jets that mimic solar and astrophysical structures |
| **MRX** — Princeton Plasma Physics Laboratory | New Jersey, USA | Magnetic reconnection, the process that releases magnetic energy in plasma |
| **LAPD** — Basic Plasma Science Facility, UCLA | California, USA | Large-scale plasma device for Alfvén waves and current-carrying plasma |
| **Wendelstein 7-X** — Max Planck IPP | Greifswald, Germany | Stellarator for long-duration magnetic confinement |
| **ITER** | Cadarache, France | International tokamak, the largest magnetic-confinement fusion experiment under construction |

And in space, missions such as **NASA's Magnetospheric Multiscale (MMS)** and the **Parker Solar Probe** measure currents, fields and reconnection directly in natural plasma.

---

## ALADIN Native-Cylindrical MHD Driver

**Current development version: v4.0.2-dev · Native cylindrical geometry · 11 CUDA kernel slots**

The Driver is a modular MHD solver written in **Python and CUDA** (CuPy `RawKernel`) for NVIDIA GPUs. It coordinates geometry, state, boundary conditions, time integration and diagnostics, and delegates each physical operation to a dedicated kernel.

**Design**

- **Native cylindrical coordinates.** Metric factors, radial face areas and cylindrical cell volumes are part of every operator:
  $$\Delta V = \frac{r_{i+1/2}^2 - r_{i-1/2}^2}{2}\,\Delta\phi\,\Delta z .$$
- **Staggered magnetic fields.** Face-centred $\mathbf B$ and edge-centred EMFs, in a constrained-transport layout.
- **GPU execution.** Each heavy operation is its own CUDA kernel, run on L4- and A100-class hardware.
- **Diagnostics.** Current density, Lorentz force, magnetic and kinetic energy, $\nabla\cdot\mathbf B$, conservation and residuals.
- **Modularity.** Every kernel can be tested in isolation before it is coupled into the full solver.

**Version:** v4.0.2-dev · **Platform:** Python, CUDA and CuPy `RawKernel` · **Hardware:** NVIDIA A100 · **Stage:** integrated numerical verification.

### Numerical architecture

The solver advances conserved MHD variables on a cylindrical finite-volume grid. Magnetic fields are represented on staggered faces, while edge-centred electromotive forces support constrained-transport updates. Radial, azimuthal and axial flux operators are handled separately, with geometric source terms and axis-specific reconstruction included in the cylindrical discretization.

The modular execution path combines primitive-variable recovery, directional MHD fluxes, conserved-state residuals, non-ideal electromotive forces, magnetic-field updates and divergence control. The framework also provides diagnostics for current density, Lorentz force, magnetic energy, conservation and numerical residuals.

**Selected solver benchmarks**

| Benchmark | Reported result | Qualification |
|---|---|---|
| **Z-pinch equilibrium** | Analytical current and Lorentz-force reconstruction near floating-point precision; reported 2,000-step equilibrium preservation | **Individual benchmark PASS** |
| **Sausage instability (m = 0)** | Linear growth-rate error **0.8289%** against an independent linear eigen-solver (finite differences), confirmed to 7 digits by a second, spectral oracle | Individual benchmark PASS |
| **Kink instability (m = 1)** | Finest-grid linear growth-rate error **0.7337%**; refinement order **2.967**; GPU = CPU bit-identical | **PASS** (K01 run 3, all five criteria); run 2 FAIL on CPU/GPU equality kept in the record |
| **Hall-MHD wave (DRV-03)** | Both polarities at $32\times64\times128$: error **0.047%** (order **2.05**) and **0.081%** (order **1.96**); endurance runs of **245,016** and **454,761** steps on an A100 with **zero recovery events** and div B at round-off | **PASS** (DRV-03 parts B and C) |
| **Resistive magnetic decay (D03)** | **0.18014%** relative error at **32 × 64 × 128** after **53,122 steps**; reported CPU/GPU difference **0** | Accuracy and compared-output reproducibility PASS; **full DRV-04R Part A FAIL** because order **1.794 < 1.8** |
| **Manufactured solutions through the axis (MMS-01, RES-02)** | Ideal, resistive and Hall MHD with an exact solution crossing the axis in full 3D: L1 order **≥ 1.86** for all 8 variables from $32\times64\times64$ to $64\times128\times128$ ($v_r$ **2.00**); negative control detects a missing term; GPU = CPU bit-identical | **PASS** (RES-02, 5/5) with RESIDUAL v01.1; MMS-01 **FAIL** on v01 ($v_r$ order 1.65) kept — it found the axis defect |
| **Non-axisymmetric re-run on RESIDUAL v01.1 (DRV-05)** | Kink m = 1: **0.690%**, order **3.03**; Hall waves: **0.0814%** (order **1.957**, 454,759 steps on an L4) and **0.0466%** (order **2.048**); resistive decay D03 **0.180%** (order 1.794) and Alfvén D06 **0.0111%** (order 1.614) | Parts K, B, C **PASS**; part A **FAIL** on order (D03, D06), kept, and diagnosed (D06: time and space errors cancelling under the axis CFL); GPU = CPU bit-identical in all four parts |
| **Current at the axis for every azimuthal mode (AXJ-01, CURL-08)** | Exact Bessel fields $A_z=J_m(kr)\cos m\phi$, $m=0\ldots4$, and $B_\phi=J_1(kr)\to J_z=kJ_0(kr)$: axis-ring order **3.00** for the kink mode $m=1$ (was **1.02** in v1.5), **2.99** for $m=3$; every even mode second order | **PASS** with UCT-CURL v1.6 (CURL-08, 11/11, GPU = CPU); the v1.5 **FAIL** that found the defect is kept |
| **Lundquist force-free field through the axis (LUND-01)** | $B_z=J_0(kr)$, $B_\phi=J_1(kr)$, $\mathbf J=k\mathbf B$: current order **2.000** (axis **2.00**); ideal equilibrium drift **1.5e-4** after 10 Alfvén times (order **2.83**); Hall equilibrium drift **4.7e-5** (order **2.59**) | **PASS**, all three parts, GPU = CPU · archive pending (AXIS-01 release) |
| **Azimuthal modes m = 0…4 through the axis (AXM-01, AXM-02)** | Growing modes: sausage $m=0$ **2.1%** (order **2.30**), kink $m=1$ at $k=4\pi$ **4.0%** (order **2.86**). Stable modes $m=2,3,4$: damping ≤ **1.2%** of ω, mode purity ≤ **0.06%**, frequency within **0.22–0.43%** | Growing modes **PASS**; stable modes **FAIL** on convergence order: a frequency bias of about **+0.25–0.3%** that does not refine away (both oracles agree to 8 digits; cause under investigation, AXIS-02) · archive pending |

---

## Quantitative results

*Measured errors, convergence behavior, test conditions, and reported campaign outcomes.*

**Results updated 23 November 2026.** The following measurements characterize the numerical solver on defined benchmark problems.

### DRV-03 — joint refinement of the wave and decay tests (22–23 November 2026)

DRV-01 refined only $z$ at a fixed $4\times8$ cross-section, and D03, D04 and D06 failed there. DRV-03 refines $r$, $\phi$ and $z$ together up to a new level, $32\times64\times128$, that had never been run, with **DRV-01's original thresholds unchanged** (error $\le 1\%$, order $\ge 1.8$, GPU = CPU 0). All parts ran on an NVIDIA A100; levels 1–3 were previewed in a post-hoc addendum (A1) and are disclosed as such.

| Part | Test | Error at $32\times64\times128$ | Order (level 3→4) | GPU = CPU | Verdict |
|---|---|---:|---:|---:|---|
| A | D03 resistive decay | 0.1846% | 1.799 | 0 | **FAIL** (order) |
| A | D06 ideal Alfvén wave | 0.0112% | 1.620 | 0 | **FAIL** (order) |
| C | D04 Hall wave, branch −1 | 0.0466% | 2.046 | 0 | **PASS** |
| B | D04 Hall wave, branch +1 | 0.0815% | 1.959 | 0 | **PASS** |

**Long-run endurance (Hall-MHD, A100).** The two finest Hall-wave runs are the longest the Driver has completed:

| Run | Steps | Wall time | Recovery events | div B at the end | Final error |
|---|---:|---:|---:|---:|---:|
| D04, branch −1 ($32\times64\times128$) | **245,016** | 4.95 h | **0** | $2.3\times10^{-13}$ | 0.047% |
| D04, branch +1 ($32\times64\times128$) | **454,761** | 9.31 h | **0** | $3.5\times10^{-13}$ | 0.081% |

Both completed without a single local fallback, time-step halving or non-finite value, with $\nabla\cdot\mathbf B$ held at round-off (it starts at $6\times10^{-19}$) and second-order accuracy at the end. Bit-identity with the CPU was checked on the two coarser levels in full and on the first 200 steps of this level; a full CPU run of 454,761 steps at this size was not attempted.

All three tests are now well inside the 1% accuracy threshold, so the DRV-01 accuracy failures came from the coarse cross-section. D03 and D06 still miss the order threshold at this resolution and remain FAIL. The A100 joins the L4 and T4 as hardware on which the Driver reproduces the CPU leg bit for bit.

**Post-hoc diagnosis of the open items (not a change of verdict).** The resistive near-miss led to a defect in UCT-CURL's axis reconstruction, fixed and certified in v1.5 (below). The DRV-01 D13 Hall-heating failure was traced to the test itself: for its off-axis flux tube the Hall term is curl-free, so the reference Poynting divergence is exactly zero and the test compared two vanishing discretisation errors. The absolute spurious Hall heating converges at second order (5.95e-3, 2.13e-3, 5.47e-4, 1.41e-4 at 32², 64², 128², 256²). The D13 FAIL stays in the record; a well-posed Hall-energy test is planned.

### DRV-04R Part A — D03 resistive decay (23 November 2026)

**Driver:** ALADIN v4.0.2-dev · **GPU:** NVIDIA A100-SXM4-40GB · **Result:** **FAIL** under the frozen acceptance criteria. The reported GPU measurements are:

| Grid $(N_r\times N_\phi\times N_z)$ | Relative error | Steps | Runtime |
|---|---:|---:|---:|
| $4\times8\times16$ | 0.95286% | 32 | 2.1 s |
| $8\times16\times32$ | 1.63350% | 208 | 9.7 s |
| $16\times32\times64$ | 0.62464% | 3,321 | 156.2 s |
| $32\times64\times128$ | **0.18014%** | **53,122** | **3,926.9 s** |

**Frozen acceptance verdict:** finest-grid relative error $1.8014\times10^{-3}\leq0.01$ — **PASS**; reported convergence order $1.794<1.8$ — **FAIL**; reported CPU/GPU comparison discrepancy $0$ against a zero tolerance — **PASS** for the compared outputs. **Overall Part A: FAIL.** The coarsest two errors are not monotonic; the two finest grids show a substantial reduction, but do not establish asymptotic second-order convergence on their own.

Relative to the earlier reported D03 result of approximately **0.1846%**, the v1.5 run improves the finest-grid error to **0.1801%**. This is consistent with the axis-ring correction improving a localized contribution while the bulk resistive discretization dominates the global error. Per-ring second-order convergence is separately reported; it does not override the global D03 failure. A $64\times128\times256$ refinement remains **proposed, not completed**. Its reported estimated cost is approximately 850,000 steps and 20 A100 hours; the estimate is not a measured runtime.

### DRV-04R Part C — D04_b-1 re-run with UCT-CURL v1.5 (23 November 2026)

**Driver:** ALADIN v4.0.2-dev · **GPU:** NVIDIA A100-SXM4-40GB · **Status:** **not completed, no verdict.** The finest level was lost to a Colab runtime failure, and the campaign was computed with RESIDUAL v01, which MMS later found defective next to the axis (below); it is superseded by the re-run of the non-axisymmetric Driver tests on RESIDUAL v01.1 (DRV-05).

| Grid $(N_r\times N_\phi\times N_z)$ | Relative error | Steps | Runtime |
|---|---:|---:|---:|
| $4\times8\times16$ | 3.4148% | 148 | 7.4 s |
| $8\times16\times32$ | 0.81548% | 958 | 44.8 s |
| $16\times32\times64$ | 0.19259% | 15,315 | 738.0 s |

The first three levels agree with DRV-03 (UCT-CURL v1.4) to the printed digits: the v1.5 axis correction barely touches this Hall wave.

### MMS-01 and RES-02 — manufactured solutions through the axis, and RESIDUAL v01.1 (23–24 November 2026)

**What was added.** The Driver gained a new verification layer: the **method of manufactured solutions** (MMS) in full 3D. A smooth exact solution — density, pressure, velocity and vector potential all moving in $x, y, z, t$, with an oblique field $B_0 = (0.4, 0.3, 1.0)$ crossing the axis — is imposed with an exact source; the numerical error must then shrink at the scheme's design order. The source enters the conserved variables at every Runge–Kutta stage and the induction equation as an EMF, so constrained transport keeps div B at round-off. Its machinery is self-tested against finite differences to $7\times10^{-13}$, and with MMS switched off the Driver is **bit-identical** to the certified version (six cases, CPU and GPU).

**What it found (MMS-01, kept FAIL).** On RESIDUAL v01, seven of eight variables converged at second order, but $v_r$ only at **1.65**, and its error in the ring of cells touching the axis **grew** with refinement. The truncation error located the cause in the azimuthal momentum: an error growing as $1/\Delta r$ (0.54, 1.1, 2.2, 4.4 at $n_r$ = 8 … 64). The angular-momentum form evaluated its radial term at the volume-mean radius $\bar r$ and its $\phi$ term at $r_c$; for a field crossing the axis their $1/r$ parts no longer cancel. Axisymmetric tests cannot see this.

**The fix — RESIDUAL v01.1 (one line).** Both terms at the same radius $\bar r$. Axisymmetric runs (the README Z-pinch, Sod) are bit-identical to v01; angular momentum is conserved to round-off.

**RES-02 — certification of v01.1: 5/5 PASS** against thresholds approved before any run (GPU legs on NVIDIA L4 and A100; CPU legs bit-identical):

| Part | Physics | L1 order $32\to64$, lowest of 8 variables | $v_r$ | max div B | Verdict |
|---|---|---:|---:|---:|---|
| I | ideal | **1.952** | **2.001** | $1.4\times10^{-14}$ | **PASS** |
| R | resistive, $\eta = 0.02$ | **1.855** | **1.972** | $1.2\times10^{-13}$ | **PASS** |
| H | Hall, 0.02 (82,278 steps at the finest level) | **1.859** | **1.910** | $2.7\times10^{-13}$ | **PASS** |
| N | negative control ($\eta$ left out of the source) | B order **0.61** (must be ≤ 1) | — | $2.6\times10^{-14}$ | **PASS** |
| 0 | self-test · axisymmetric bit-identity · angular momentum · axis-ring error bounded | — | — | — | **PASS** |

The $v_r$ error in the axis ring at $64\times128\times128$ dropped **43-fold** and now shrinks with refinement; every GPU state of the levels 8–32 equals the CPU leg's to the last bit. The report discloses the previews, both GPU attempts of the Hall part and the rebuild of its finest-level record from the checksummed run-1 bundle.

**Archive:** [RESIDUAL v01.1, Zenodo record 24498958](https://zenodo.org/records/24498958).

**Consequence.** Every non-axisymmetric Driver result computed with v01 (Alfvén and Hall waves, resistive decay, the m = 1 kink, the D13 flux tube) changes slightly; they are being re-run on v01.1 (DRV-05). Axisymmetric results are unchanged to the bit.

### UCT-CURL v1.5 — cylindrical-axis correction (reported verification)

UCT-CURL v1.5 corrects the near-axis conversion of azimuthal face-averaged radial magnetic fields to point values before Cartesian-axis reconstruction. Reported dedicated **CURL-07: 9/9 criteria PASS**, including tested CPU/GPU output agreement. At the finest reported uniform-field axis test, the spurious-current error decreased from approximately $6.42\times10^{-3}$ (v1.4) to $1.741\times10^{-6}$ (v1.5), about **3,700-fold**. The reported convergence order for that axis test is approximately **2.998**, not a claim that the entire CURL operator is third-order. In a separate axis-ring resistive-decay diagnostic, the reported relative rate error changed from **−1.369%** to **+0.0178%** at $64\times128\times256$. The normalized Z-pinch and annular-grid regression results were reported preserved. These measurements characterize the tested cylindrical-curl configurations.

**Archive:** [UCT-CURL v1.5, Zenodo record 24465267](https://zenodo.org/records/24465267). 

### UCT-CURL v1.6 — the axis current of odd azimuthal modes (25 November 2026)

A new benchmark set, **AXJ-01**, feeds UCT-CURL exact current fields through the axis: power laws ($B_\phi=r^3, r^5$), $\sin kr$, the Bessel pair $B_\phi=J_1(kr)\to J_z=kJ_0(kr)$, and the Bessel eigenmodes $A_z=J_m(kr)\cos m\phi$ for $m=1\ldots4$, which carry the exact axis regularity of every azimuthal mode. Its criteria were hashed before it ran. On v1.5 it **failed for the odd modes**: the current in the ring touching the axis converged at first order for $m=1$ (the kink) and $m=3$ (orders 1.02, 1.05), while even modes were second order. The cause: $B_\phi$ is a radial average on its face (what constrained transport carries) but was rotated together with point values of $B_r$; the $O(\Delta r^2)$ mismatch was divided by $r\Delta\phi$. v1.6 converts $B_\phi$ to its point value first (three lines, the radial twin of the v1.5 fix). The axis-ring current is now **third order** for $m=1$ and $m=3$ (error at 128 × 256: $3.1\times10^{-2}\to2.0\times10^{-4}$ for the kink mode). **CURL-08: 11/11 PASS**; the README Z-pinch and every annulus grid are bit-identical to v1.5; 93 arrays identical on GPU (L4) and CPU. Disclosed: the AXJ grids had been run on a scratch prototype before CURL-08 was frozen.

**Archive:** [UCT-CURL v1.6, Zenodo record 24532856](https://zenodo.org/records/24532856).

### DRV-05 — the non-axisymmetric Driver tests on RESIDUAL v01.1 (24–25 November 2026)

| Part | Test | Result | Verdict |
|---|---|---|---|
| K | Kink $m=1$ growth vs ORACLE-2 | **0.690%** (v01: 0.734%), order **3.03**, purity 5e-5 | **PASS** |
| A | D03 resistive decay · D06 Alfvén wave | **0.180%**, order 1.794 · **0.0111%**, order 1.614 | **FAIL** (order), kept |
| B | Hall wave, D04 $b=+1$ | **0.0814%**, order **1.957**, 454,759 steps on an L4 (11.7 h, 33 checkpoints) | **PASS** |
| C | Hall wave, D04 $b=-1$ | **0.0466%**, order **2.048**, 245,017 steps | **PASS** |

All four parts are GPU = CPU bit-identical. Post-hoc diagnosis of D06 (not a change of verdict): doubling $z$ cuts its error 4×, doubling $r$ changes nothing and doubling $\phi$ makes it worse — time and space errors cancelling under the axis time step, not a solver defect. D03 stays open.

### Linear sausage and kink instability growth

The Driver's measured linear growth rates are compared with separately computed eigenvalue references. The results below distinguish the **reference growth rate** from the **measured numerical growth rate**.

| Benchmark | Mode / axial wavenumber | Independent reference growth | Finest reported Driver growth | Relative error | Assessment |
|---|---|---:|---:|---:|---|
| Sausage (DRV-01 D07) | $m=0$, $k=4\pi$ | $0.2824957824$ | See DRV-01 report | **0.8289%** | Individual benchmark PASS |
| Kink (DRV-02 K01) | $m=1$, $k=2\pi$ | $0.4681760091$ | **0.464741** | **0.7337%** | **PASS** (run 3, all five criteria) |

#### Kink K01: three-grid GPU convergence

| Grid $(N_r\times N_\phi\times N_z)$ | Measured growth | Relative error | Mode-contamination metric | Steps | Runtime |
|---|---:|---:|---:|---:|---:|
| $16\times8\times16$ | 0.289205 | 38.23% | $4.41\times10^{-3}$ | 1,973 | 85.1 s |
| $32\times16\times32$ | 0.441313 | 5.738% | $8.69\times10^{-4}$ | 7,896 | 305.0 s |
| $64\times32\times64$ | **0.464741** | **0.7337%** | $9.82\times10^{-5}$ | 31,597 | 1,608.4 s |

The reported K01 refinement order is **2.967** between the two finest levels (a benchmark-specific estimate, not a global solver order). Growth-rate error decreases monotonically, and the reported mode-contamination metric falls with refinement. These observations support the correct linear $m=1$ growth dynamics **for this benchmark and fitting procedure**; they do not by themselves establish all nonlinear kink dynamics or universal third-order accuracy.

**K01 run history:** run 1 crashed while building its initial state (a harness bug, fixed as change K-a). Run 2 passed the four physics criteria (growth error $\leq5\%$, observed order $\geq1.5$, monotonic error reduction, contamination metric $\leq10^{-2}$) but **FAILED** the frozen CPU/GPU *bitwise-equality* criterion. Across the 24 compared arrays, the maximum reported absolute CPU/GPU difference was $6.50\times10^{-11}$ for the $16\times8\times16$ case, $7.90\times10^{-12}$ for $32\times16\times32$, and $1.31\times10^{-14}$ for a separate 500-step segment. The uploaded K01 archive includes the CPU leg and source; the GPU numbers above are transcribed from the separately supplied GPU run log.

### K01 Run 3: frozen-state CPU/GPU reproducibility, and the final PASS

Run 3 archive: `ALADIN_driver_v40_K01_run3.tar.gz` · reported SHA-256 `f267205d731eeec93a5d20a0105ed5cc1d5ebf07783889eef018b59b041e2885`. This is a reported archive identifier, not an independently verified file hash in this README.

| Comparison | Differing values | Maximum absolute difference | Scope |
|---|---:|---:|---|
| Initial state rebuilt on Run 3 machine versus frozen CPU-leg initial state | 18,198 | $1.1\times10^{-16}$ | Initial-state construction |
| GPU versus CPU from the **same frozen state** | **0** | **0** | $16\times8\times16$, full 1,973 steps; compared outputs |
| GPU versus CPU from the **same frozen state** | **0** | **0** | $64\times32\times64$, first 500 steps; compared outputs |

The probe confirmed the cause: the initial state is built from a LAPACK eigenvector and a BLAS interpolation whose last bits differ between machines. With the initial states frozen on the CPU-leg machine (change K-b, disclosed before run 3), **K01 run 3 PASSED all five criteria**, with the same growth rates and **0 differing values** between GPU and CPU. The run-2 FAIL stays in the record. The full K01 campaign, its run history and the evidence bundles are published in the ALADIN Driver v4.0.1 report (DRV-02).

### K01 reproducibility and physical setup (DRV-02)

The accompanying `ALADIN_driver_v40_K01.tar.gz` research archive contains the Driver source, the K01 test harness (`drv/tests/kink01.py`), the independent Chebyshev spectral reference (`drv/tests/oracle_cheb_pinch.py`), and the CPU-leg results (`drv/tests/kink01_cpu_results.json`, `drv/tests/kink01_cpu.out`). **The archive supplied here documents the CPU leg; the three-grid GPU measurements in the table above originate from a separately reported GPU run .**

The benchmark evolves the normalized cylindrical equilibrium $\rho=1$, $B_\phi=r/2$, $J_z=1$, $P(r)=1-r^2/4$, $\mu_0=1$, $\gamma=5/3$, inside a **conducting wall at $r=1$**, with periodic axial boundary conditions and $z$-domain length 1. The initial $m=1$ kink perturbation has axial wavenumber $k=2\pi$ and amplitude $\varepsilon=10^{-5}$; its magnetic perturbation is constructed from a vector potential to respect the discrete magnetic constraint. The reported growth rate is a linear fit to $\log A(t)$ over **$t\in[2,15]$**. The *mode-contamination metric* is the fraction of kinetic energy outside the targeted Fourier-mode pair $(m,n)=\pm(1,1)$ at $t=15$; smaller is better.

The archive's **CPU results** reproduce the first two growth-rate measurements: $0.2892051903$ on $16\times8\times16$ (1,973 steps) and $0.4413132821$ on $32\times16\times32$ (7,896 steps). Its CPU result object is explicitly labelled `COMPLETED (no checks declared)`, **not** a CPU campaign PASS. The GPU leg reaches $0.464741$ on $64\times32\times64$. Under the frozen thresholds, run 2 failed the exact CPU/GPU equality check and run 3, from frozen initial states, passed all five. 

**Planned higher-resolution refinement:** preserving the existing grid aspect ratio and doubling all three grid dimensions would produce $128\times64\times128$ and $256\times128\times256$ runs, subject to GPU memory and time budgets. These are proposed tests, not achieved benchmarks.

### Driver DRV-01: coupled verification snapshot

The reported DRV-01 campaign achieved **9 PASS / 4 FAIL** across 13 tests. Passing tests include static pinch equilibrium, closed-system conservation, a Sod shock benchmark, sausage growth, checkpoint restart, recovery, ablation sensitivity, a tested CPU/GPU final-state comparison, and performance/memory reporting. The four failures (**resistive diffusion, Hall-wave convergence, ideal Alfvén-wave convergence, and resistive/Hall energy consistency**) stay in the record. Their follow-up is reported above: DRV-03 brought all three wave and decay errors well inside 1% and passed the Hall wave on both polarities; resistive decay and the Alfvén wave still miss the convergence-order threshold; the Hall-energy test was found to be ill-posed. Further detail is in the DRV-03 section.

**Independent spectral oracle:** the reported Chebyshev-collocation calculation gives sausage growth $0.2824957824$ at $m=0,k=4\pi$ and kink growth $0.4681760091$ at $m=1,k=2\pi$, stable to the displayed digits under the reported spectral refinements. 

**Next qualification:** A100 high-resolution studies, further eigenfunction comparisons, CPU/GPU numerical reproducibility diagnostics, and resolution-controlled coupled tests. Further high-resolution runs are **planned, not completed results**; the next aspect-ratio-preserving grids are $128\times64\times128$ and $256\times128\times256$.

---

## The eleven kernels — purpose and archival references

*What each numerical component does, with links to the corresponding published records.*

The eleven Driver execution slots are mapped to eight individually archived subsystem families. The **evidence notes** identify distinctive capabilities or reported test results for each slot; they are not claims of completed end-to-end qualification. Each link below identifies the corresponding Zenodo record (DOI prefix `10.5281/zenodo.`). **An archival DOI establishes a citable research release, not automatically full Driver-integrated qualification.** Versions and acceptance evidence should be checked against each release's source hash and verification report.

| # | Driver kernel | Purpose and evidence | Zenodo archive |
|:--:|---|---|---|
| 01 | **PRIM** | Recovers density, velocity, thermal pressure and other primitive quantities from conserved MHD variables; applies admissibility checks and floors; low-β and FP32 cancellation remain challenging. | [10.5281/zenodo.24403809](https://doi.org/10.5281/zenodo.24403809) |
| 02 | **GYRO / Entropy** | Reconstructs gyrotropic kinetic closure quantities, including parallel/perpendicular pressure and validity/fallback diagnostics; coupling anisotropic stresses to evolution is a separate milestone. Reported 19/19 CPU and 22/23 GPU checks (one harness-parser issue); ~93% closure success across 4,000 mixtures. | [10.5281/zenodo.24383948](https://doi.org/10.5281/zenodo.24383948) |
| 03 | **G2 / Geometry Algebra** | Computes read-only octonion/G₂ and MHD-gradient invariant diagnostics; introduces no additional force in the default model. 122/122 checks and second-order convergence on selected cylindrical gradient tests; diagnostic, not a new force. | [10.5281/zenodo.24376742](https://doi.org/10.5281/zenodo.24376742) · [G2 v01 self-test: 10.5281/zenodo.24369578](https://doi.org/10.5281/zenodo.24369578) |
| 04 | **HLLD-R** | Resolves approximate MHD Riemann problems across radial faces, including cylindrical radial-direction fluxes. cylindrical MHD wave/flux handling with fallback and health diagnostics; shared HLLD release. | [10.5281/zenodo.24408600](https://doi.org/10.5281/zenodo.24408600) (shared HLLD release) |
| 05 | **HLLD-φ** | Resolves MHD fluxes across azimuthal faces with cylindrical geometry and direction-aware state handling. Direction-aware φ fluxes needed for non-axisymmetric evolution, including kink modes; shared HLLD release. | [10.5281/zenodo.24408600](https://doi.org/10.5281/zenodo.24408600) (shared HLLD release) |
| 06 | **HLLD-Z** | Resolves MHD fluxes across axial faces and contributes to the three-dimensional finite-volume update. Completes the three-direction MHD flux system; shared HLLD release. | [10.5281/zenodo.24408600](https://doi.org/10.5281/zenodo.24408600) (shared HLLD release) |
| 07 | **RESIDUAL** | Combines directional flux divergences and cylindrical geometric/source terms into the conserved-state time derivative. v01: 11/11 GPU and 7/7 CPU checks; near-second-order annular convergence and a stationary Z-pinch. v01.1 corrects the angular-momentum term next to the axis, found by MMS (RES-02: 5/5 PASS). | [v01.1: 10.5281/zenodo.24498958](https://doi.org/10.5281/zenodo.24498958) · [v01: 10.5281/zenodo.24404520](https://doi.org/10.5281/zenodo.24404520) |
| 08 | **UCT-EMF** | Builds edge-centred electromotive forces for constrained transport, with ideal, Hall and resistive contributions as configured. Hall/resistive edge EMFs, reported stationary pinch and axis-closure improvements; perturbed nonideal tests remain important. | [10.5281/zenodo.24405701](https://doi.org/10.5281/zenodo.24405701) |
| 09 | **UCT-CURL** | Reconstructs current density from magnetic fields in cylindrical geometry; v1.3 completed 50 tests, v1.4 added bit-identical GPU = CPU evaluation, v1.5 corrects the near-axis reconstruction (CURL-07: 9/9 PASS), and v1.6 makes the axis current third order for odd azimuthal modes (CURL-08: 11/11 PASS). | [v1.6: 10.5281/zenodo.24532856](https://doi.org/10.5281/zenodo.24532856) · [v1.5: 10.5281/zenodo.24465267](https://doi.org/10.5281/zenodo.24465267) · [v1.4: 10.5281/zenodo.24417336](https://doi.org/10.5281/zenodo.24417336) |
| 10 | **E-GLM** | Propagates and damps numerical magnetic-divergence errors through an extended generalized Lagrange multiplier cleaning subsystem. reported divergence monitoring and near-roundoff projection in selected configurations. | [10.5281/zenodo.24395791](https://doi.org/10.5281/zenodo.24395791) |
| 11 | **UCT-ASSEMBLY** | Applies the edge-EMF curl to staggered magnetic faces and assembles the constrained-transport magnetic update, including the axis closure. It ships and is certified inside the UCT-EMF release (Driver slot "08+11"). | [10.5281/zenodo.24405701](https://doi.org/10.5281/zenodo.24405701) (shared UCT-EMF release) |

**Evidence mapping.** The eight Zenodo subsystem records above are drawn from the accompanying `ALADIN_8_Kernel_Subsystem_READMEs.zip` index. The G2 v01 self-test DOI is retained from the previous project README. A single HLLD release is cited for all three directional execution slots; this is not three independent archived publications. UCT-ASSEMBLY (slot 11) has no record of its own: it is part of the UCT-EMF release, as the Driver's kernel registry states.

**Kernel and solver results.** The archived subsystem tests and integrated Driver benchmarks are presented separately, with individual results and acceptance status in the tables above.

---

## Verification

*Subsystem checks and integrated acceptance are reported separately; both successful and failed tests remain visible.*

Each kernel is intended to pass a dedicated test campaign before final acceptance and integration. Its source is frozen by SHA-256 hash, its results are compared against independent analytical references, with CPU/GPU and precision coverage documented per campaign rather than assumed for every kernel.

Experiment archives include source versions, configurations, raw measurements and reproduction instructions.

**UCT-CURL v1.3** completed five campaigns (50 tests): second-order convergence against independent analytical, exact-discrete and DEC references, including Bessel force-free fields, the Z-pinch and helical perturbations, on CPU and GPU in FP32 and FP64. The campaign reports also record its known limitations: it is not a mimetic operator (div J ≠ 0 at truncation level) and grid-scale checkerboard modes of B are invisible to it. v1.4 added a bit-identical GPU = CPU evaluation, and v1.5 corrects the near-axis reconstruction for fields that cross the axis (CURL-07, 9/9 PASS; summarised in [Quantitative results](#quantitative-results)); full coupled qualification remains incomplete. A DEC-based successor, `uct_curl_v02`, with edge-centred currents and exact discrete identities, is planned.

**G2 v01** ([10.5281/zenodo.24369578](https://doi.org/10.5281/zenodo.24369578)) passes 122/122 self-test checks on CPU and GPU: octonion identities, Z-pinch and helical reference fields against an independent implementation, second-order convergence against an analytic oracle, rotation and Galilean invariance, and negative controls that fire as designed. Its main exact result: in the default mode the pulled-back G₂ 3-form equals Re det ∇(u + i b), a combination of the classical Elsässer gradient invariants. An independent MHD study (turbulence and a kinking Z-pinch) found no additional predictive power in the regimes tested; these negative results are published with the kernel. Behaviour inside the coupled Driver is future work.

**Manufactured solutions (MMS).** The Driver can run any problem with an exact source term, so a chosen smooth solution becomes the exact answer and the convergence order of the full coupled solver can be measured directly — ideal, resistive and Hall MHD, through the axis, in 3D. The source manager is off by default and proved bit-neutral when off; a negative control (a term deliberately left out of the source) shows that the test detects a missing term. MMS-01 found a real axis defect in RESIDUAL v01 on its first run; RES-02 certified the fix (see [Quantitative results](#quantitative-results)).

Subsystem verification and coupled solver testing are documented independently.

## ALADIN — Open Science by Design

*How this project works, and the ethics behind it.*

Open Science makes research accessible, transparent, reproducible and independently verifiable. Its principle is simple: **scientific conclusions should rest on evidence that others can examine, reproduce and challenge, not on trust in the researcher.** ALADIN tests unconventional ideas in the open, which only means something if the standard of evidence is higher, not lower. That standard is built into the instrument itself as six pillars, one for each way an experiment can go wrong.

| Pillar | Responsibility | Question it answers | Threat it addresses |
|---|---|---|---|
| **MMS** | Mathematical verification | Are we solving the equations correctly? | Incorrect numerical mathematics |
| **ByteWise** | Scientific integrity | Are the results judged fairly? | Moved or manipulated acceptance criteria |
| **ByByte** | Numerical reproducibility | Does the experiment give the same answer again, to the last bit? | Hidden, irreproducible numerics |
| **MakeReport** | Evidence transparency | Can another researcher see exactly what happened? | Undocumented experiments, concealed failures |
| **Bundle** | Evidence preservation | Does the evidence survive without the original machine? | Lost or altered evidence |
| **REC** | Computational continuity | Can an interrupted experiment continue without changing its result? | Corrupted checkpoints, silent recovery errors |

### 1. MMS — verify the mathematics
A known solution is manufactured, its source terms are derived exactly, and the code must reproduce it at the expected order. ALADIN uses this alongside analytical benchmarks, independent oracles, convergence studies and negative controls (a test that must fail when a term is removed). **RES-02** is the example: ideal, resistive and Hall MHD with an exact solution crossing the axis in full 3D; its predecessor **MMS-01 failed** and found a real defect next to the axis (RESIDUAL v01 → v01.1). MMS verifies the implementation; it does not by itself show that the physical model describes nature.

### 2. ByteWise — protect the acceptance criteria
- Pass/fail thresholds are written down and frozen **before** a test runs; the test files are hashed (SHA-256), so nobody, the author included, can move a threshold afterwards.
- Anything seen before the freeze (a preview, a plumbing run) is disclosed in the campaign's header. After a run starts, only the harness may be fixed, never the thresholds, and every change is disclosed with its reason.
- Development experiments and formal campaigns are kept distinct. Example: the stable modes of **AXM-01/AXM-02** have frequency errors below 0.5 %, damping and mode purity that pass, and bit-identical GPU = CPU — and still **FAIL**, because the frozen convergence order is not met. The FAIL stays; the cause is being investigated.

### 3. ByByte — reproduce the numbers, bit for bit
- The Driver must give **bit-identical results** on CPU and GPU in FP64, with a tolerance of exactly zero, within each campaign's documented scope. It does, on NVIDIA L4, T4 and A100.
- When it did not, the cause was hunted down and fixed, and every story is published: a `x ** 2` that CUDA and NumPy round differently; initial states that LAPACK builds differently on two machines; a start-up divergence projection that uses NumPy's FFT on the host even in a GPU run; an end time derived from a LAPACK eigenvalue. The lesson: every number that shapes a run is either computed by the certified kernels or frozen from one machine.
- Bit identity does not prove correctness — two implementations can reproduce the same mistake. That is why MMS and ByByte complement each other.

### 4. MakeReport — document the complete experiment
Every formal campaign produces a structured record: the mathematics, the source versions and hashes, the hardware, the numerical parameters, the frozen thresholds, the execution history, every measurement, the verdict, and the failures and limitations. A report explains what happened, not just PASS or FAIL. **A failed experiment gets the same documentation as a successful one.**

### 5. Bundle — preserve and distribute the evidence
Each test runs inside a bundle manager that seals the configuration, raw measurements, log and verdict with SHA-256 checksums. Each release ships its code, campaigns, bundles and report, archived on Zenodo with a DOI, so any number in this README can be traced to the files that produced it — independently of the Colab session or GPU that made them.

### 6. REC — protect the interrupted experiment
Recovery is a tested claim, not an assumption. Long GPU runs checkpoint every 20 minutes and back up to Google Drive; a preflight refuses to start if any file needed at the end is missing; saved work cannot be overwritten by accident; results are packed and downloaded automatically. **REC-01** (26 November 2026) proves the continuity: a resumed run follows the uninterrupted one step for step, to the last bit, for ideal, resistive and Hall MHD, after three hard kills, on CPU and GPU (**15/15 PASS**); damaged, truncated or foreign checkpoints are refused. Its first run **failed** — a crash *during* a save destroyed the only checkpoint — and the checkpoint manager now writes one atomic file per checkpoint with its hash inside and keeps the previous one. Real Colab disconnects and a browser closed by accident have since been recovered this way, bit-identically.

*Bundle protects the evidence of the experiment. REC protects the continuity of the experiment itself.*

### Failures stay in the record
- A FAIL is never deleted, renamed or re-run until it passes. It stays FAIL, with what it measured and why.
- Mistakes are published too: a test later found to be ill-posed (DRV-01 D13, Hall heating) keeps its FAIL with the explanation, and kernel defects found by the Driver were fixed in new, separately certified versions: UCT-CURL v1.4 near the axis (→ v1.5) and for odd azimuthal modes (AXJ-01 FAIL kept → v1.6), and RESIDUAL v01, whose axis error was found by manufactured solutions (MMS-01 FAIL kept → v01.1).
- Negative results are published with the work: the G₂ diagnostic study found no extra predictive power in the regimes tested, and says so.

### Established physics and open hypotheses are kept apart
- The Driver and its benchmarks are tested against exact solutions and independent references; those results stand on their own.
- Cosmological ideas are treated as **hypotheses under test**. They are claimed only through predictions written down before the experiment, which the experiment could have proved wrong.

### What these pillars are, and are not
The individual practices are established methods; ALADIN's contribution is to integrate them into one explicit methodology, built into the instrument rather than added after the experiment. They are internal ALADIN standards, not external certifications, and they do not replace independent replication, peer review, physical validation or comparison with established codes — they make those easier and more meaningful.

- The code is MIT-licensed; reports, data and bundles are on Zenodo; anyone can reproduce a result from its bundle.
- AI assistance is declared openly (see [The ALADIN Lab](#the-aladin-lab)); research direction and scientific responsibility remain with the author.
- Independent reproduction by others is welcome and is the strongest check of all.

> **MMS** verifies · **ByteWise** protects · **ByByte** compares · **MakeReport** documents · **Bundle** preserves · **REC** restores
>
> Open Source · Open Methods · Open Results · Open Failures · Open Evidence
>
> **Never ask the scientific community to believe a result that you could instead give them the means to verify.**

---

## ALADIN Online Command Center

**[huggingface.co/spaces/aladinibz/PlasmaCosmology](https://huggingface.co/spaces/aladinibz/PlasmaCosmology)**

The Driver connects to an online research environment through its **Adaptive Intelligence Layer (AIL)**. Experiments upload selected diagnostics, which can then be visualised, compared across runs and analysed with AI assistance to identify trends, anomalies and follow-up experiments.

---

## Research programs

| Program | Focus | Directory |
|---|---|---|
| **Z-pinch** | Equilibrium, confinement and benchmarks | [`Z-pinch/`](Z-pinch/) |
| **J × B** | Lorentz-force dynamics and plasma self-organisation | [`JxB/`](JxB/) |
| **Kink** | $m=1$ current-driven instability | [`Kink/`](Kink/) |
| **Sausage** | $m=0$ pinch instability | [`Sausage/`](Sausage/) |
| **Birkeland** | Field-aligned current systems | [`Birkeland/`](Birkeland/) |
| **Galactic Disk** | Magnetized disks and galactic rotation | [`GalacticDisk/`](GalacticDisk/) |
| **Gravity** | Gravitational forces compared with electromagnetic ones | [`Gravity/`](Gravity/) |

---

## Repository structure

```text
Aladin-Plasma-Cosmology/
├── AladinDriver/                  # Driver framework
├── Karnels/                       # Kernels and their test campaigns
├── Karnel/Entropy/Manifold/Gyro/  # Gyro, entropy and manifold research
├── Z-pinch/                       # Z-pinch studies and benchmarks
├── JxB/                           # Lorentz-force studies
├── Kink/  Sausage/                # Pinch instabilities
├── Birkeland/                     # Field-aligned currents
├── GalacticDisk/  Gravity/        # Astrophysical applications
├── CITATION.cff
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/aladinibz/Aladin-Plasma-Cosmology.git
cd Aladin-Plasma-Cosmology
```

Requires Python ≥ 3.10 and NumPy; GPU runs need CuPy and an NVIDIA GPU with a matching CUDA 12.x driver. Each experiment directory includes its own launcher and documentation.

**Status:** active development. The coupled Driver reproduces its CPU results bit for bit on NVIDIA L4, T4 and A100 GPUs. Individual kernel campaigns and coupled Driver benchmarks have produced quantitative results (see [Quantitative results](#quantitative-results)); full qualification remains incomplete, with known failures disclosed above. High-resolution A100 qualification and further releases are planned.

---

## The ALADIN Lab

**Mihai Alexandru Bucurenciu (ALADIN)**: founder and lead researcher

**AI research team**

| Member | Role |
|---|---|
| **SuperGrok** (xAI) | exploration and research ideas |
| **ChatGPT** (OpenAI) | scientific review and verification standards |
| **Claude** (Anthropic) | adversarial test campaigns, evidence bundles and technical documentation |

Research direction and scientific responsibility remain with the lead researcher.

---

## Citation

```bibtex
@software{aladin_plasma_cosmology_2026,
  author  = {Bucurenciu, Mihai Alexandru},
  title   = {ALADIN Plasma Cosmology},
  year    = {2026},
  url     = {https://github.com/aladinibz/Aladin-Plasma-Cosmology},
  license = {MIT}
}
```

**Mihai Alexandru Bucurenciu (Aladin)** · [ORCID 0009-0006-4457-557X](https://orcid.org/0009-0006-4457-557X) · aladinibz@proton.me
Archival releases and kernel verification reports on Zenodo.

Released under the [MIT License](LICENSE).
