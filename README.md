<div align="center">

# ALADIN Plasma Cosmology

**Native cylindrical MHD for Z-pinch and J × B plasma dynamics, from the laboratory to astrophysical scales**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![CUDA / CuPy](https://img.shields.io/badge/CUDA-CuPy%20RawKernel-76B900.svg)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--4457--557X-A6CE39.svg)](https://orcid.org/0009-0006-4457-557X)
[![Command Center](https://img.shields.io/badge/Command%20Center-Hugging%20Face-blueviolet.svg)](https://huggingface.co/spaces/aladinibz/PlasmaCosmology)

[Overview](#overview) · [Results](#quantitative-results) · [Z-pinch](#the-z-pinch) · [J × B](#the-j--b-force) · [History](#historical-foundations) · [Driver](#the-aladin-driver) · [Kernels](#the-eleven-kernels) · [Verification](#verification) · [Programs](#research-programs) · [Lab](#the-aladin-lab)

</div>

---

## Overview

Almost all visible matter in the universe is plasma: an electrically conducting medium that carries currents and responds to magnetic fields. In plasma, electromagnetic forces do not just act on matter; they **organise** it. Currents draw plasma into columns, filaments and flux ropes, and those structures appear at every scale, from laboratory discharges to the solar corona, planetary magnetospheres and the interstellar medium.

**ALADIN** studies this organising role of electromagnetism. Its core is two closely linked pieces of physics:

- **The Z-pinch**: the basic configuration in which an axial current confines its own plasma.
- **The J × B force**: the Lorentz force through which currents and magnetic fields self-organise plasma into coherent structures.

To investigate them, the project develops the **ALADIN Driver**: a GPU-accelerated MHD solver in native cylindrical coordinates $(r,\phi,z)$, built from **eleven computational kernels**. Its purpose is to test plasma-cosmology ideas through controlled numerical experiments on pinches, instabilities, field-aligned currents and magnetized astrophysical environments.

---

## Quantitative results

**Reported results as of 22 November 2026.** These are numerical verification results for specified benchmark problems, **not** a validation of a galactic-plasma cosmological model. Results marked *PASS* below refer to their **individual physics criteria**; the complete Driver campaign has additional requirements.

### Native cylindrical Z-pinch: analytical reference

For the normalized equilibrium $\mu_0=1$, $B_\phi=r/2$, $J_z=1$, $f_r=-r/2$, $P=P_0-r^2/4$, and $\nabla\cdot\mathbf B=0$. This is a known analytical benchmark, not a newly discovered force law. ALADIN's reported tests reconstruct the radial $\mathbf J\times\mathbf B$ force to near floating-point precision and demonstrate static equilibrium preservation over 2,000 time steps in the specified Driver test.

### Linear sausage and kink instability growth

The Driver's measured linear growth rates are compared with separately computed eigenvalue references. The results below distinguish the **reference growth rate** from the **measured numerical growth rate**.

| Benchmark | Mode / axial wavenumber | Independent reference growth | Finest reported Driver growth | Relative error | Assessment |
|---|---|---:|---:|---:|---|
| Sausage (DRV-01 D07) | $m=0$, $k=4\pi$ | $0.2824957824$ | See DRV-01 report | **0.8289%** | Individual benchmark PASS |
| Kink (DRV-02 K01) | $m=1$, $k=2\pi$ | $0.4681760091$ | **0.464741** | **0.7337%** | Growth-rate criterion PASS; full K01 FAIL |

#### Kink K01: three-grid GPU convergence

| Grid $(N_r\times N_\phi\times N_z)$ | Measured growth | Relative error | Mode-contamination metric | Steps | Runtime |
|---|---:|---:|---:|---:|---:|
| $16\times8\times16$ | 0.289205 | 38.23% | $4.41\times10^{-3}$ | 1,973 | 85.1 s |
| $32\times16\times32$ | 0.441313 | 5.738% | $8.69\times10^{-4}$ | 7,896 | 305.0 s |
| $64\times32\times64$ | **0.464741** | **0.7337%** | $9.82\times10^{-5}$ | 31,597 | 1,608.4 s |

The reported K01 refinement order is **2.967** between the two finest levels. Growth-rate error decreases monotonically, and the reported mode-contamination metric falls with refinement. These observations support the correct linear $m=1$ growth dynamics **for this benchmark and fitting procedure**; they do not by themselves establish all nonlinear kink dynamics or universal third-order accuracy.

**K01 qualification disclosure:** four individual criteria pass (growth error $\leq5\%$, observed order $\geq1.5$, monotonic error reduction, contamination metric $\leq10^{-2}$), while the complete K01 verdict is **FAIL** because the frozen CPU/GPU *bitwise-equality* criterion was not met. Across the 24 compared arrays, the maximum reported absolute CPU/GPU difference was $6.50\times10^{-11}$ for the $16\times8\times16$ case, $7.90\times10^{-12}$ for $32\times16\times32$, and $1.31\times10^{-14}$ for a separate 500-step segment. Small absolute differences do not alone prove relative numerical equivalence; the exact-equality failure remains recorded. The uploaded K01 archive includes the CPU leg and source; the GPU numbers above are transcribed from the separately supplied GPU run log.

### K01 Run 3: frozen-state CPU/GPU reproducibility (reported, pending complete MakeReport)

Run 3 archive: `ALADIN_driver_v40_K01_run3.tar.gz` · reported SHA-256 `f267205d731eeec93a5d20a0105ed5cc1d5ebf07783889eef018b59b041e2885`. This is a reported archive identifier, not an independently verified file hash in this README.

| Comparison | Differing values | Maximum absolute difference | Scope |
|---|---:|---:|---|
| Initial state rebuilt on Run 3 machine versus frozen CPU-leg initial state | 18,198 | $1.1\times10^{-16}$ | Initial-state construction |
| GPU versus CPU from the **same frozen state** | **0** | **0** | $16\times8\times16$, full 1,973 steps; compared outputs |
| GPU versus CPU from the **same frozen state** | **0** | **0** | $64\times32\times64$, first 500 steps; compared outputs |

These reported results indicate that machine-dependent initial-state construction may explain the earlier bitwise mismatch: the tested GPU evolution reproduces the CPU outputs exactly when started from the same frozen state. **This does not retroactively turn the original K01 FAIL into a PASS.** The complete fine-grid trajectory, full compared-array inventory, source/hash verification and frozen acceptance verdict still require a complete MakeReport.

### K01 reproducibility and physical setup (DRV-02)

The accompanying `ALADIN_driver_v40_K01.tar.gz` research archive contains the Driver source, the K01 test harness (`drv/tests/kink01.py`), the independent Chebyshev spectral reference (`drv/tests/oracle_cheb_pinch.py`), and the CPU-leg results (`drv/tests/kink01_cpu_results.json`, `drv/tests/kink01_cpu.out`). **The archive supplied here documents the CPU leg; the three-grid GPU measurements in the table above originate from a separately reported GPU run and should be deposited with its complete MakeReport and machine-readable output.**

The benchmark evolves the normalized cylindrical equilibrium $\rho=1$, $B_\phi=r/2$, $J_z=1$, $P(r)=1-r^2/4$, $\mu_0=1$, $\gamma=5/3$, inside a **conducting wall at $r=1$**, with periodic axial boundary conditions and $z$-domain length 1. The initial $m=1$ kink perturbation has axial wavenumber $k=2\pi$ and amplitude $\varepsilon=10^{-5}$; its magnetic perturbation is constructed from a vector potential to respect the discrete magnetic constraint. The reported growth rate is a linear fit to $\log A(t)$ over **$t\in[2,15]$**. The *mode-contamination metric* is the fraction of kinetic energy outside the targeted Fourier-mode pair $(m,n)=\pm(1,1)$ at $t=15$; smaller is better.

The archive's **CPU results** reproduce the first two growth-rate measurements: $0.2892051903$ on $16\times8\times16$ (1,973 steps) and $0.4413132821$ on $32\times16\times32$ (7,896 steps). Its CPU result object is explicitly labelled `COMPLETED (no checks declared)`, **not** a CPU campaign PASS. The reported GPU leg reaches $0.464741$ on $64\times32\times64$; the overall K01 acceptance remains **FAIL** under the frozen exact CPU/GPU equality threshold, despite passing the four stated physics checks. Future comparisons should retain the original failure and additionally report scientifically justified relative/normalized numerical differences, growth-rate agreement, and reproducibility evidence.

**Planned higher-resolution refinement:** preserving the existing grid aspect ratio and doubling all three grid dimensions would produce $128\times64\times128$ and $256\times128\times256$ runs, subject to GPU memory and time budgets. These are proposed tests, not achieved benchmarks.

### Driver DRV-01: coupled verification snapshot

The reported DRV-01 campaign achieved **9 PASS / 4 FAIL** across 13 tests. Passing tests include static pinch equilibrium, closed-system conservation, a Sod shock benchmark, sausage growth, checkpoint restart, recovery, ablation sensitivity, a tested CPU/GPU final-state comparison, and performance/memory reporting. Outstanding failures involve **resistive diffusion, Hall-wave convergence, ideal Alfvén-wave convergence, and resistive/Hall energy consistency**. These failures must be resolved or explicitly excluded from any broader solver-qualification claim. A successful pinch test does not override them.

**Independent spectral oracle:** the reported Chebyshev-collocation calculation gives sausage growth $0.2824957824$ at $m=0,k=4\pi$ and kink growth $0.4681760091$ at $m=1,k=2\pi$, stable to the displayed digits under the reported spectral refinements. Full physical independence also requires scrutiny of the oracle's equations, normalization, axis parity, boundary conditions, and eigenpair residuals.

**Next qualification:** A100 high-resolution studies, further eigenfunction comparisons, CPU/GPU numerical reproducibility diagnostics, and resolution-controlled coupled tests. Further high-resolution runs are **planned, not completed results**; the next aspect-ratio-preserving grids are $128\times64\times128$ and $256\times128\times256$.

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

---

## The J × B force

The force density on a current-carrying plasma is the Lorentz force

$$
\boxed{\;\mathbf f_L = \mathbf J \times \mathbf B\;}, \qquad \mathbf J = \frac{1}{\mu_0}\,\nabla\times\mathbf B .
$$

**J × B is the self-organising mechanism of plasma.** It concentrates currents into filaments, holds pinches together, twists flux ropes, and drives the structures that emerge in magnetized plasma. In the Z-pinch it takes its simplest form:

$$
J_z = \frac{1}{\mu_0\, r}\frac{d}{dr}\bigl(r B_\phi\bigr), \qquad f_r = -J_z B_\phi ,
$$

an inward radial force. In a static equilibrium it is balanced by the pressure gradient, $dP/dr = f_r$.

Because J × B is built directly from the current, an accurate discrete curl $\nabla\times\mathbf B$ is the foundation of everything ALADIN computes.

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

ALADIN builds on more than two centuries of work on currents, magnetic forces and pinches.

**The J × B force**

| Year | | Contribution |
|---|---|---|
| 1820–1826 | **André-Marie Ampère** | Forces between electric currents; the foundation of electrodynamics |
| 1820 | **Jean-Baptiste Biot & Félix Savart** | The magnetic field produced by a current |
| 1865 | **James Clerk Maxwell** | Unified equations of electricity and magnetism |
| 1889 | **Oliver Heaviside** | The magnetic force on a moving charge |
| 1892–1895 | **Hendrik Lorentz** | The complete force law $\mathbf F = q(\mathbf E + \mathbf v\times\mathbf B)$, whose continuum form is $\mathbf J\times\mathbf B$ |

**The pinch**

| Year | | Contribution |
|---|---|---|
| 1790 | **Martinus van Marum** | High-current discharges through wires: the first observed pinch effects |
| 1905 | **J. A. Pollock & S. Barraclough** | Explained a lightning-crushed copper tube as magnetic compression by the current: the first explained Z-pinch |
| 1934 | **Willard H. Bennett** | The Bennett relation: equilibrium of a self-confined current-carrying plasma |
| 1937 | **Lewi Tonks** | Named the "pinch effect" |
| 1954 | **Martin Kruskal & Martin Schwarzschild** | Theory of pinch instabilities, including the kink and sausage modes |
| 1950s | **ZETA, Perhapsatron and Kurchatov programs** | Z-pinches as the first large fusion experiments |

**Plasma in space**

| Year | | Contribution |
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

## The ALADIN Driver

The Driver is a modular MHD solver written in **Python and CUDA** (CuPy `RawKernel`) for NVIDIA GPUs. It coordinates geometry, state, boundary conditions, time integration and diagnostics, and delegates each physical operation to a dedicated kernel.

**Design**

- **Native cylindrical coordinates.** Metric factors, radial face areas and cylindrical cell volumes are part of every operator:
  $$\Delta V = \frac{r_{i+1/2}^2 - r_{i-1/2}^2}{2}\,\Delta\phi\,\Delta z .$$
- **Staggered magnetic fields.** Face-centred $\mathbf B$ and edge-centred EMFs, in a constrained-transport layout.
- **GPU execution.** Each heavy operation is its own CUDA kernel, run on L4- and A100-class hardware.
- **Diagnostics.** Current density, Lorentz force, magnetic and kinetic energy, $\nabla\cdot\mathbf B$, conservation and residuals.
- **Modularity.** Every kernel can be tested in isolation before it is coupled into the full solver.

---

## The eleven kernels — purpose and archival references

The eleven Driver execution slots are mapped to eight individually archived subsystem families. The **evidence notes** identify distinctive capabilities or reported test results for each slot; they are not claims of completed end-to-end qualification. Each link below identifies the corresponding Zenodo record (DOI prefix `10.5281/zenodo.`). **An archival DOI establishes a citable research release, not automatically full Driver-integrated qualification.** Versions and acceptance evidence should be checked against each release's source hash and verification report.

| # | Driver kernel | Purpose and evidence | Zenodo archive |
|:--:|---|---|---|
| 01 | **PRIM** | Recovers density, velocity, thermal pressure and other primitive quantities from conserved MHD variables; applies admissibility checks and floors. admissibility checks and floors; low-β and FP32 cancellation remain challenging. | [10.5281/zenodo.24403809](https://doi.org/10.5281/zenodo.24403809) |
| 02 | **GYRO / Entropy** | Reconstructs gyrotropic kinetic closure quantities, including parallel/perpendicular pressure and validity/fallback diagnostics; coupling anisotropic stresses to evolution is a separate milestone. reported 19/19 CPU and 22/23 GPU checks (one harness-parser issue); ~93% closure success across 4,000 mixtures. | [10.5281/zenodo.24383948](https://doi.org/10.5281/zenodo.24383948) |
| 03 | **G2 / Geometry Algebra** | Computes read-only octonion/G₂ and MHD-gradient invariant diagnostics; introduces no additional force in the default model. 122/122 checks and second-order convergence on selected cylindrical gradient tests; diagnostic, not a new force. | [10.5281/zenodo.24376742](https://doi.org/10.5281/zenodo.24376742) · [G2 v01 self-test: 10.5281/zenodo.24369578](https://doi.org/10.5281/zenodo.24369578) |
| 04 | **HLLD-R** | Resolves approximate MHD Riemann problems across radial faces, including cylindrical radial-direction fluxes. cylindrical MHD wave/flux handling with fallback and health diagnostics; shared HLLD release. | [10.5281/zenodo.24408600](https://doi.org/10.5281/zenodo.24408600) (shared HLLD release) |
| 05 | **HLLD-φ** | Resolves MHD fluxes across azimuthal faces with cylindrical geometry and direction-aware state handling. direction-aware φ fluxes needed for non-axisymmetric evolution, including kink modes; shared HLLD release. | [10.5281/zenodo.24408600](https://doi.org/10.5281/zenodo.24408600) (shared HLLD release) |
| 06 | **HLLD-Z** | Resolves MHD fluxes across axial faces and contributes to the three-dimensional finite-volume update. completes the three-direction MHD flux system; shared HLLD release. | [10.5281/zenodo.24408600](https://doi.org/10.5281/zenodo.24408600) (shared HLLD release) |
| 07 | **RESIDUAL** | Combines directional flux divergences and cylindrical geometric/source terms into the conserved-state time derivative. reported 11/11 GPU and 7/7 CPU checks; near-second-order annular convergence and a stationary Z-pinch. | [10.5281/zenodo.24404520](https://doi.org/10.5281/zenodo.24404520) |
| 08 | **UCT-EMF** | Builds edge-centred electromotive forces for constrained transport, with ideal, Hall and resistive contributions as configured. Hall/resistive edge EMFs, reported stationary pinch and axis-closure improvements; perturbed nonideal tests remain important. | [10.5281/zenodo.24405701](https://doi.org/10.5281/zenodo.24405701) |
| 09 | **UCT-CURL** | Reconstructs current density from the magnetic field using cylindrical discrete-curl operators; supports Lorentz-force and Hall-term calculations. cylindrical curl tested against Z-pinch, Bessel and helical references; 50 tests across five campaigns in v1.3. | [10.5281/zenodo.24417336](https://doi.org/10.5281/zenodo.24417336) |
| 10 | **E-GLM** | Propagates and damps numerical magnetic-divergence errors through an extended generalized Lagrange multiplier cleaning subsystem. reported divergence monitoring and near-roundoff projection in selected configurations. | [10.5281/zenodo.24395791](https://doi.org/10.5281/zenodo.24395791) |
| 11 | **UCT-ASSEMBLY** | Applies the edge-EMF curl to staggered magnetic faces and assembles the constrained-transport magnetic update. assembles magnetic face updates from edge EMFs; separate DOI and stand-alone qualification not established in supplied index. | **No separate DOI identified in the supplied eight-subsystem index**; part of the Driver workflow, subject to separate evidence mapping. |

**Evidence mapping.** The eight Zenodo subsystem records above are drawn from the accompanying `ALADIN_8_Kernel_Subsystem_READMEs.zip` index. The G2 v01 self-test DOI is retained from the previous project README. A single HLLD release is cited for all three directional execution slots; this is not three independent archived publications. The UCT-ASSEMBLY row deliberately does not claim a dedicated DOI without an identifiable record.

**Qualification status.** The archived kernel implementations and their test campaigns must be distinguished from coupled Driver acceptance. The Driver's Z-pinch, sausage and kink benchmark results are documented in the verification sections below, including any outstanding numerical and CPU/GPU reproducibility failures. All experiments require a full MakeReport with configuration, source hash, acceptance thresholds, raw measurements and reproducibility instructions.

---

## Verification

Each kernel is intended to pass a dedicated test campaign before final acceptance and integration. Its source is frozen by SHA-256 hash, its results are compared against independent analytical references, with CPU/GPU and precision coverage documented per campaign rather than assumed for every kernel.

The project requires a **MakeReport / TestReport Bundle** for every experiment and campaign: source and hashes, configuration, raw results, figures, verdicts, recorded limitations and a one-command reproduction script. Anyone can rerun the work and check it.

**UCT-CURL v1.3** has completed five campaigns (50 tests): second-order convergence against independent analytical, exact-discrete and DEC references, including Bessel force-free fields, the Z-pinch and helical perturbations, on CPU and GPU in FP32 and FP64. The campaign reports also record its known limitations: it is not a mimetic operator (div J ≠ 0 at truncation level) and grid-scale checkerboard modes of B are invisible to it. Full integration testing waits for the remaining kernels. A DEC-based successor, `uct_curl_v02`, with edge-centred currents and exact discrete identities, is planned.

**G2 v01** ([10.5281/zenodo.24369578](https://doi.org/10.5281/zenodo.24369578)) passes 122/122 self-test checks on CPU and GPU: octonion identities, Z-pinch and helical reference fields against an independent implementation, second-order convergence against an analytic oracle, rotation and Galilean invariance, and negative controls that fire as designed. Its main exact result: in the default mode the pulled-back G₂ 3-form equals Re det ∇(u + i b), a combination of the classical Elsässer gradient invariants. An independent MHD study (turbulence and a kinking Z-pinch) found no additional predictive power in the regimes tested; these negative results are published with the kernel. Behaviour inside the coupled Driver is future work.

A kernel is marked *Verified* only after its campaign is complete. A kernel-level result is not a full-solver result: coupled time-evolution benchmarks follow once all kernels are integrated.

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

**Status:** active development. Individual kernel campaigns and coupled Driver benchmarks have produced quantitative results (see [Quantitative results](#quantitative-results)); full qualification remains incomplete, with known failures disclosed above. High-resolution A100 qualification and further releases are planned.

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
